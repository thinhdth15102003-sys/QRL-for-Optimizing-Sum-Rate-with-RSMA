"""
probe_cumulative_ceiling.py
---------------------------
How far can this policy be pushed, and does the top of that ladder clear AO?

`probe_gap_split.py` swaps ONE component at a time and leaves the rest at the
policy's choice. That is the right tool for attribution, but it systematically
UNDERSTATES the reachable total whenever the axes interact -- and at Case 3 they
interact hard: retuning the split f is worth +0.018 J under the policy's own
phase and +0.068 J once the phase is fixed, because a common-heavy allocation
only pays after the common stream actually works (probe_split_frac.py).

So this probe applies the fixes CUMULATIVELY, in the order the dependency runs:

    policy            as executed
    + oracle phase    J-optimal per-element phase (same search AO uses, on g-hat)
    + best split f    the split grid, given that phase
    + route ascent    coordinate ascent on the assignment, given both
    + power beta      concentration grid (beta<=1, the capped one -- the
                      uncapped grid selects on g-hat and loses on true g)
    AO                the baseline at its per-case plateau

Every rung keeps the rungs above it. The last policy rung is therefore the
ceiling of this recipe family under perfect sub-solvers, and the question the
probe exists to answer is whether that ceiling is above or below AO. If it is
below, no amount of training on these axes wins the case and the honest move is
to say so.

Everything is decided on g-hat and scored on true g, with the same oracle C_k
applied to every rung and to AO (C_k is sum-rate-neutral -- see
memory/ao-baseline-structure).

Usage:
  python analysis/probe_cumulative_ceiling.py --run results/result_299/best \
      --seeds 42 7 --states 40
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
from analysis.probe_gap_split import best_power, route_ascent

RUNGS = ("policy", "+oracle phase", "+best split f", "+route ascent",
         "+power beta", "AO")


def run_seed(run_dir, seed, n_states):
    cfg = load_training_cfg(run_dir)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_states + 5)
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    # TWO views, and the distinction is the whole validity of this probe.
    # Jd = DESIGN view (estimated channel) — every oracle SELECTS with this,
    #      because that is all AO gets: its _obj calls compute_sum_rate with the
    #      default use_true=False.
    # Jt = TRUE view — used only to SCORE the choice that was already made.
    # Selecting with Jt would let the oracles peek at the realised channel and
    # inflates every rung; an earlier version of this probe did exactly that.
    Jt = make_J(env, cfg, True)
    Jd = make_J(env, cfg, False)
    sw = _AO_SWEEPS.get(cfg.K, 1)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=sw)
    fracs = np.asarray(ao.splits, dtype=float)

    acc = {k: [] for k in RUNGS}
    qos = {k: [] for k in RUNGS}
    fstar = []
    env.reset(seed=seed)
    for _ in range(n_states):
        obs = env._get_obs()
        blocked = _compute_irs_favored(env)
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

        def rec(k, a_, P_, wp_, wc_, ids_):
            acc[k].append(Jt(a_, P_, wp_, wc_, ids_))
            qos[k].append(Jt.qos)

        rec("policy", phi, Phi, wp_p, wc_p, ids)

        # 1. oracle phase, on the policy's routing and power
        def _score(ph, _wp=wp_p, _wc=wc_p, _ids=ids, _a=phi):
            return Jd(_a, phi_from_idx(ph, cfg), _wp, _wc, _ids)
        pid2, _, _ = oracle_phase_search(_est_channels(env.channels), phi, cfg,
                                         _score, n_sweeps=sw, seed_pidx=pidx)
        Phi = phi_from_idx(pid2, cfg)
        rec("+oracle phase", phi, Phi, wp_p, wc_p, ids)

        # 2. best split, keeping the learned shape inside private and common
        sp_p = float(np.sum(wp_p)) or 1.0
        sc_p = float(np.sum(wc_p)) or 1.0
        bv, bwp, bwc, bf = -np.inf, wp_p, wc_p, float("nan")
        for f in fracs:
            w1 = wp_p / sp_p * cfg.P_S * f
            w2 = wc_p / sc_p * cfg.P_S * (1.0 - f)
            v = Jd(phi, Phi, w1, w2, ids)
            if v > bv:
                bv, bwp, bwc, bf = v, w1, w2, float(f)
        wp_p, wc_p = bwp, bwc
        fstar.append(bf)
        rec("+best split f", phi, Phi, wp_p, wc_p, ids)

        # 3. routing coordinate ascent (the combinatorial step AO also runs)
        phi, ids = route_ascent(Jd, phi, Phi, ids, cfg, wp_p, wc_p)
        wc_p = np.full(len(ids) + 1, float(wc_p.sum()) / (len(ids) + 1))
        rec("+route ascent", phi, Phi, wp_p, wc_p, ids)

        # 4. private-power concentration (capped beta grid)
        h2 = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
        wpo, wco = best_power(Jd, phi, Phi, ids, cfg, h2)
        rec("+power beta", phi, Phi, wpo, wco, ids)

        act = ao.act(obs, env)
        aid = sorted({int(x) for x in act["assignment"] if x > 0})
        rec("AO", act["assignment"], phi_from_idx(act["phase_idx"], cfg),
            act["w_p"], act["w_c_vec"], aid)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
    return ({k: float(np.mean(v)) for k, v in acc.items()},
            {k: float(np.mean(v)) for k, v in qos.items()},
            float(np.median(fstar)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7])
    ap.add_argument("--states", type=int, default=40)
    a = ap.parse_args()

    J = {k: [] for k in RUNGS}
    Q = {k: [] for k in RUNGS}
    fs = []
    for s in a.seeds:
        j, q, f = run_seed(a.run, s, a.states)
        for k in RUNGS:
            J[k].append(j[k])
            Q[k].append(q[k])
        fs.append(f)
        print(f"    seed {s:<3} " + "  ".join(f"{k}={j[k]:.4f}" for k in RUNGS),
              flush=True)

    print("\n" + "=" * 76)
    print(f"  CUMULATIVE CEILING · {a.run} · {a.states} states x "
          f"{len(a.seeds)} seeds · median f* = {np.median(fs):.2f}")
    print("=" * 76)
    print(f"  {'rung':<18}{'J':>9}{'sd':>8}{'QoS':>8}{'step':>9}{'vs AO':>10}")
    print("  " + "-" * 62)
    ao = float(np.mean(J["AO"]))
    prev = None
    for k in RUNGS:
        v = np.array(J[k])
        step = "" if prev is None or k == "AO" else f"{v.mean()-prev:+9.4f}"
        print(f"  {k:<18}{v.mean():>9.4f}"
              f"{v.std(ddof=1) if len(v) > 1 else 0:>8.4f}"
              f"{100*np.mean(Q[k]):>7.1f}%{step:>9}"
              f"{'' if k == 'AO' else format(v.mean()-ao, '+10.4f')}")
        if k != "AO":
            prev = v.mean()
    print("=" * 76)
    top = float(np.mean(J["+power beta"]))
    print(f"  ceiling of this recipe family: {top:.4f} vs AO {ao:.4f}"
          f"  ->  {top-ao:+.4f}")
    print("  A negative number here means NO amount of training on these axes")
    print("  wins the case; say so rather than spending GPU on it.")
    print("=" * 76)


if __name__ == "__main__":
    sys.exit(main())
