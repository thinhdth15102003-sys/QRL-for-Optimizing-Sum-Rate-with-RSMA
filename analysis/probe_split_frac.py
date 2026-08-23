"""
probe_split_frac.py
-------------------
What private-power split fraction f does AO actually choose, and what does the
policy execute?

`PowerMLP.power_priv_frac` is the target f that `--power-fairness` blends the
executed allocation toward. The code comment records the measured optimum as
0.68 at K5/M1 and 0.28 at K10/M2 under a demand-filling C_k -- a steep fall with
K -- yet every trained run so far pins f = 0.80, the value that was optimal only
under an EQUAL C_k. If that bias matters it should matter most at K=15, where
the private budget is spread over the most users.

AO settles this without any training: `_best_power` grids f over `splits` and
keeps the argmax of J, so AO's selection IS the empirical optimum for that state
under AO's own (equal-power, equal-common) allocation. Comparing it to the
fraction the policy actually executes says whether the prior is pulling the
policy away from the optimum, and in which direction.

Reports, per case:
  AO f*        distribution of the split AO selects (median + IQR)
  policy f     the private fraction the policy's PowerMLP actually executes
  J(f) curve   AO's J as a function of f, so the flatness of the optimum is
               visible -- a sharp optimum badly missed is a lever; a flat one is not

Usage:
  python analysis/probe_split_frac.py --run results/result_299/best --states 60
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state)
from analysis.probe_ao_rate import AORatePolicy
from analysis.probe_vs_ao import make_J, _AO_SWEEPS
from analysis.phase_oracle import oracle_phase_search
from analysis.oracle_alloc import phi_from_idx, _est_channels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle-phase", dest="oracle_phase", action="store_true",
                    help="replace the policy's phase by the J-optimal one (same "
                         "search AO uses, decided on g-hat) before sweeping f. "
                         "Tests whether a low f only pays once the common stream "
                         "actually works.")
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Jt = make_J(env, cfg, True)
    sw = _AO_SWEEPS.get(cfg.K, 1)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=sw)
    splits = np.asarray(ao.splits, dtype=float)

    ao_f, pol_f = [], []
    Jf = np.zeros(len(splits))
    # Counterfactual on the POLICY's own routing/phase: keep its learned shape
    # inside the private and common vectors, change ONLY the split between them.
    Jpol = np.zeros(len(splits))
    Qpol = np.zeros(len(splits))
    Jpol_now = 0.0
    env.reset(seed=a.seed)
    for _ in range(a.states):
        obs = env._get_obs()
        blocked = _compute_irs_favored(env)

        # policy: read the executed private fraction straight off the allocation
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, ainfo = actor.forward(s_t, greedy=True)
        rep = ainfo["z_t"]
        active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
        pidx, _, _ = phase_net.forward(
            _build_phase_state(env.channels, phi, cfg, rep), active, greedy=True)
        Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
        h = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
        hs = float(np.mean(np.abs(h))) + 1e-9
        sp = np.concatenate([h.real / hs, h.imag / hs])
        if getattr(power_net, "scale_feats", False):
            from train import _power_scale_feats
            cfg.power_feat_scale = getattr(power_net, "feat_scale", 1.0)
            sp = np.concatenate([sp, _power_scale_feats(h, cfg)])
        if getattr(power_net, "mag_feats", None):
            from train import _power_mag_feats
            sp = np.concatenate([sp, _power_mag_feats(h, power_net.mag_feats)])
        if power_net.d_s > sp.shape[0]:
            sp = np.concatenate([sp, rep])
        wc_p, wp_p, _, _ = power_net.forward(sp, ids)
        pol_f.append(float(np.sum(wp_p)) / float(np.sum(wp_p) + np.sum(wc_p)))

        if a.oracle_phase:
            # Same search AO runs, on the POLICY's routing and power, decided on
            # g-hat. Scored inside by the same J the rest of the probe uses.
            def _score(ph, _wp=wp_p, _wc=wc_p, _ids=ids, _phi=phi):
                return Jt(_phi, phi_from_idx(ph, cfg), _wp, _wc, _ids)
            pidx_o, _, _ = oracle_phase_search(
                _est_channels(env.channels), phi, cfg, _score,
                n_sweeps=sw, seed_pidx=pidx)
            Phi = phi_from_idx(pidx_o, cfg)

        Jpol_now += Jt(phi, Phi, wp_p, wc_p, ids)
        sp_p = float(np.sum(wp_p)) or 1.0
        sc_p = float(np.sum(wc_p)) or 1.0
        for i, f in enumerate(splits):
            wp2 = wp_p / sp_p * cfg.P_S * f
            wc2 = wc_p / sc_p * cfg.P_S * (1.0 - f)
            Jpol[i] += Jt(phi, Phi, wp2, wc2, ids)
            Qpol[i] += Jt.qos

        # AO: its own routing/phase, then sweep f and record both argmax and curve
        act = ao.act(obs, env)
        aphi = act["assignment"]
        aids = sorted({int(x) for x in aphi if x > 0})
        aPhi = env.phase_model.build_phi(
            env.phase_model.index_to_phase(act["phase_idx"]))
        js = []
        for f in splits:
            wp = np.full(cfg.K, cfg.P_S * f / cfg.K)
            wc = np.full(len(aids) + 1, cfg.P_S * (1.0 - f) / (len(aids) + 1))
            js.append(Jt(aphi, aPhi, wp, wc, aids))
        js = np.asarray(js)
        Jf += js
        ao_f.append(float(splits[int(js.argmax())]))

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    Jf /= a.states
    ao_f, pol_f = np.asarray(ao_f), np.asarray(pol_f)
    print("=" * 74)
    print(f"  SPLIT FRACTION f · {a.run} · K={cfg.K} M={cfg.M} · {a.states} states")
    print(f"  prior in the run: power_priv_frac = "
          f"{getattr(power_net, 'power_priv_frac', float('nan')):.2f}"
          f"   (blend weight alpha_priv = "
          f"{getattr(power_net, 'power_fairness_priv', float('nan')):.2f})")
    print("=" * 74)
    q = np.percentile(ao_f, [25, 50, 75])
    print(f"  AO chooses f*      median {q[1]:.2f}   IQR [{q[0]:.2f}, {q[2]:.2f}]")
    q = np.percentile(pol_f, [25, 50, 75])
    print(f"  policy executes f  median {q[1]:.2f}   IQR [{q[0]:.2f}, {q[2]:.2f}]")
    print("\n  AO's mean J as a function of f  (how sharp is the optimum?)")
    best = Jf.max()
    for f, j in zip(splits, Jf):
        bar = "#" * int(max(0, 40 * (j - Jf.min()) / (best - Jf.min() + 1e-12)))
        mark = "  <- argmax" if j == best else ""
        print(f"    f={f:.2f}   J={j:.4f}  {bar}{mark}")

    Jpol /= a.states
    Qpol /= a.states
    Jpol_now /= a.states
    print(f"\n  COUNTERFACTUAL on the policy's OWN routing/phase/shape "
          f"(only the split moves)")
    print(f"    as executed (f={np.median(pol_f):.2f})   J={Jpol_now:.4f}")
    bi = int(Jpol.argmax())
    for f, j, q in zip(splits, Jpol, Qpol):
        mark = "  <- best" if j == Jpol[bi] else ""
        print(f"    f={f:.2f}   J={j:.4f}   QoS {100*q:5.1f}%{mark}")
    print(f"\n    headroom from the split ALONE: {Jpol[bi] - Jpol_now:+.4f} J"
          f"   (AO is at {Jf.max():.4f})")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
