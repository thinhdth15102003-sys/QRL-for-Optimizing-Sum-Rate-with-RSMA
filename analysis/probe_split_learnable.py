"""
probe_split_learnable.py
------------------------
The split step is the biggest rung on the Case-3 ladder (+0.113 J), but only
when f is chosen PER STATE. A single constant f captures much less. So before
spending a training run on it: can a head actually learn the per-state choice,
or is f* driven by something the policy cannot see?

Three questions, all answered without training:

  SPREAD        how much does f* actually move across states? If it is nearly
                constant, `--power-priv-frac <that value>` is the whole lever and
                no architectural change is needed.

  PRICE OF A    J at the best SINGLE constant f, against J at the per-state f*.
  CONSTANT      This is exactly the part of the rung a fixed prior can never
                reach, i.e. what a learned, state-conditioned head would buy.

  LEARNABILITY  Spearman correlation of f* against features the policy already
                observes at inference (IRS-favoured count, per-IRS load, mean and
                spread of the effective channel). A signal that correlates with
                nothing observable cannot be learned no matter the capacity --
                in that case the honest lever is the constant, not the head.

f* is selected on the ESTIMATED channel (as AO does) and scored on the true one.

Usage:
  python analysis/probe_split_learnable.py --run results/result_299/best \
      --seeds 42 7 --states 60
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
from analysis.oracle_alloc import phi_from_idx, _est_channels
from analysis.phase_oracle import oracle_phase_search
from analysis.probe_ao_rate import AORatePolicy
from analysis.probe_vs_ao import make_J, _AO_SWEEPS


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--states", type=int, default=60)
    a = ap.parse_args()

    fstar, feats, Jgrid = [], [], []
    for seed in a.seeds:
        cfg = load_training_cfg(a.run)
        env = ISTNEnv(cfg, seed=seed, n_steps_ep=a.states + 5)
        actor, phase_net, power_net, ck_net = load_agents(a.run, seed=seed)
        demand = np.full(cfg.K, cfg.D_k_bps_hz)
        Jt, Jd = make_J(env, cfg, True), make_J(env, cfg, False)
        sw = _AO_SWEEPS.get(cfg.K, 1)
        ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                          phase_sweeps=sw)
        fracs = np.asarray(ao.splits, dtype=float)

        env.reset(seed=seed)
        for _ in range(a.states):
            obs = env._get_obs()
            blocked = _compute_irs_favored(env)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, ainfo = actor.forward(s_t, greedy=True)
            rep = ainfo["z_t"]
            active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
            pidx, _, _ = phase_net.forward(
                _build_phase_state(env.channels, phi, cfg, rep), active,
                greedy=True)
            h = env.rate_computer.effective_channels_all(
                phi, env.phase_model.build_phi(
                    env.phase_model.index_to_phase(pidx)), env.channels)
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

            # the ladder's winning configuration: oracle phase, then split
            def _sc(ph, _wp=wp_p, _wc=wc_p, _ids=ids, _a=phi):
                return Jd(_a, phi_from_idx(ph, cfg), _wp, _wc, _ids)
            pid2, _, _ = oracle_phase_search(_est_channels(env.channels), phi,
                                             cfg, _sc, n_sweeps=sw,
                                             seed_pidx=pidx)
            Phi = phi_from_idx(pid2, cfg)

            sp_p = float(np.sum(wp_p)) or 1.0
            sc_p = float(np.sum(wc_p)) or 1.0
            row, bv, bf = [], -np.inf, float("nan")
            for f in fracs:
                w1 = wp_p / sp_p * cfg.P_S * f
                w2 = wc_p / sc_p * cfg.P_S * (1.0 - f)
                v = Jd(phi, Phi, w1, w2, ids)          # select on g-hat
                row.append(Jt(phi, Phi, w1, w2, ids))  # score on true g
                if v > bv:
                    bv, bf = v, float(f)
            Jgrid.append(row)
            fstar.append(bf)

            hm = np.abs(h)
            loads = [int(np.sum(phi == m)) for m in range(1, cfg.M + 1)]
            feats.append([
                float(np.sum(blocked)),                    # IRS-favoured count
                float(np.mean(phi > 0)),                   # fraction routed
                float(np.max(loads)) if loads else 0.0,    # worst per-IRS load
                float(np.mean(hm)),                        # mean |h_eff|
                float(np.std(hm) / (np.mean(hm) + 1e-12)),  # channel spread
            ])

            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    fstar = np.asarray(fstar)
    feats = np.asarray(feats)
    Jgrid = np.asarray(Jgrid)
    fr = np.asarray(fracs)

    print("=" * 74)
    print(f"  SPLIT LEARNABILITY · {a.run} · {len(fstar)} states "
          f"({len(a.seeds)} seeds)")
    print("=" * 74)
    q = np.percentile(fstar, [10, 25, 50, 75, 90])
    print(f"  f*  p10 {q[0]:.2f}  p25 {q[1]:.2f}  median {q[2]:.2f}  "
          f"p75 {q[3]:.2f}  p90 {q[4]:.2f}   (sd {fstar.std():.3f})")
    vals, cnt = np.unique(fstar, return_counts=True)
    top = np.argsort(cnt)[::-1][:3]
    print("  most common f*: " + ", ".join(
        f"{vals[i]:.2f} ({100*cnt[i]/len(fstar):.0f}%)" for i in top))

    per_state = Jgrid[np.arange(len(fstar)), [int(np.argmin(np.abs(fr - f)))
                                              for f in fstar]].mean()
    const = Jgrid.mean(axis=0)
    ib = int(const.argmax())
    print(f"\n  J at per-state f*        {per_state:.4f}")
    print(f"  J at best CONSTANT f     {const[ib]:.4f}   (f = {fr[ib]:.2f})")
    print(f"  price of using a constant  {per_state - const[ib]:+.4f} J"
          f"   <- what a state-conditioned head would buy")

    names = ["IRS-favoured count", "fraction routed", "worst per-IRS load",
             "mean |h_eff|", "channel spread"]
    print("\n  LEARNABILITY — Spearman(f*, observable feature)")
    for i, n in enumerate(names):
        r = _spearman(feats[:, i], fstar)
        flag = "  <- usable" if abs(r) >= 0.3 else ""
        print(f"    {n:<22}{r:+.3f}{flag}")
    print("\n  |rho| < 0.3 everywhere means f* is not predictable from what the")
    print("  policy sees, so the lever is the CONSTANT, not a bigger head.")
    print("=" * 74)


if __name__ == "__main__":
    sys.exit(main())
