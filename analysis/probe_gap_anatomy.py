"""
probe_gap_anatomy.py
--------------------
Where does the policy's margin over AO come from, and where does it leak?

The mean margin is already known (Case 2: DNN +0.051 +- 0.011 over four seeds,
VQC +0.044 to +0.061). A mean says nothing about whether that margin is spread
evenly or is a large win on most states dragged down by a recoverable minority.
Only the second case gives somewhere to aim.

Three things the per-state view answers that the mean cannot:

  WIN RATE     the fraction of states the policy actually wins, and the size of
               its losses when it loses. A 60% win rate with heavy losses is a
               different problem from a 95% win rate with light ones.

  LOSS CLASS   what distinguishes the states where AO wins. Correlates the
               per-state delta against blocked count, IRS load, QoS, and channel
               spread. A structure here is an addressable target; no structure
               means the leak is diffuse and no per-class fix exists.

  CEILING      the oracle swaps run one at a time hide interactions. This runs
               them in combination as well, so the true joint ceiling and the
               size of the interaction term are both visible. Case 2's singles
               already showed assign and power going NEGATIVE (oracles chosen on
               g-hat lose on g), which makes the joint number the honest one.

Conventions per PROBES.md: J not R_tot, decide on g-hat and score on g, AO at its
measured plateau setting (rounds=1, phase_sweeps=1, n_split=11).
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
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx, _est_channels
from analysis.phase_oracle import oracle_phase_idx, oracle_phase_search
from analysis.probe_ao_rate import AORatePolicy

BETAS = (0.0, 0.25, 0.5, 1.0)


def make_J(env, cfg, use_true):
    rc, Dk = env.rate_computer, cfg.D_k_bps_hz
    lam = float(getattr(cfg, "lambda_D", 1.5))
    eps = float(getattr(cfg, "epsilon_qp", 1e-3))

    def J(phi, Phi, wp, wc, ids):
        p = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                     active_irs_ids=ids)
        _, Ck = oracle_ck_met(p["R_private"], p["R_c_group"], p["groups"], Dk)
        for g, m in p["groups"].items():
            m = np.asarray(list(m), dtype=int)
            if m.size:
                left = float(p["R_c_group"].get(int(g), 0.0)) - float(Ck[m].sum())
                if left > 1e-12:
                    Ck[m] += left / m.size
        o = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=Ck,
                                active_irs_ids=ids, use_true=use_true)
        R = np.asarray(o["R_private"]) + np.asarray(o["C_k"])
        s = np.maximum(0.0, Dk - R) / (Dk + eps)
        J.qos = float(np.sum(R >= Dk)) / len(R)
        return float(o["sum_rate"]) - lam * float((s ** 2).sum())
    J.qos = float("nan")
    return J


def best_power(Jd, phi, Phi, ids, cfg, h, splits):
    g = np.abs(h) ** 2 + 1e-30
    best = None
    for b in BETAS:
        w = g ** b
        w = w / w.sum()
        for f in splits:
            wp = w * cfg.P_S * f
            wc = np.full(len(ids) + 1, cfg.P_S * (1.0 - f) / (len(ids) + 1))
            v = Jd(phi, Phi, wp, wc, ids)
            if best is None or v > best[0]:
                best = (v, wp, wc)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Jd, Jt = make_J(env, cfg, False), make_J(env, cfg, True)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=1)
    splits = np.linspace(0.05, 0.95, 11)

    rec = {k: [] for k in ("pol", "ao", "ph", "pw", "ph+pw", "all")}
    feat = {k: [] for k in ("blk", "irs", "qos", "hspread", "K")}

    env.reset(seed=a.seed)
    for _ in range(a.states):
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

        rec["pol"].append(Jt(phi, Phi, wp_p, wc_p, ids))
        q_pol = Jt.qos

        est = _est_channels(env.channels)
        seed_p = np.asarray(oracle_phase_idx(est, phi, cfg, n_sweeps=0), dtype=int)
        po, _, _ = oracle_phase_search(
            est, phi, cfg,
            lambda p: Jd(phi, phi_from_idx(p, cfg), wp_p, wc_p, ids),
            n_sweeps=1, seed_pidx=seed_p.copy())
        Phi_o = phi_from_idx(np.asarray(po, dtype=int), cfg)
        rec["ph"].append(Jt(phi, Phi_o, wp_p, wc_p, ids))

        wpo, wco = best_power(Jd, phi, Phi, ids, cfg, h, splits)
        rec["pw"].append(Jt(phi, Phi, wpo, wco, ids))

        h2 = env.rate_computer.effective_channels_all(phi, Phi_o, env.channels)
        wpo2, wco2 = best_power(Jd, phi, Phi_o, ids, cfg, h2, splits)
        rec["ph+pw"].append(Jt(phi, Phi_o, wpo2, wco2, ids))
        rec["all"].append(rec["ph+pw"][-1])

        act = ao.act(obs, env)
        aid = sorted({int(x) for x in act["assignment"] if x > 0})
        rec["ao"].append(Jt(act["assignment"],
                            phi_from_idx(act["phase_idx"], cfg),
                            act["w_p"], act["w_c_vec"], aid))

        feat["blk"].append(int(np.sum(np.asarray(blocked) > 0)))
        feat["irs"].append(int(np.sum(np.asarray(phi) > 0)))
        feat["qos"].append(q_pol)
        ah = np.abs(h)
        feat["hspread"].append(float(ah.max() / max(ah.min(), 1e-30)))
        feat["K"].append(cfg.K)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    P, A = np.array(rec["pol"]), np.array(rec["ao"])
    d = P - A
    print("=" * 78)
    print(f"  GAP ANATOMY · {a.run} · K={cfg.K} M={cfg.M} · {a.states} states "
          f"· env seed {a.seed}")
    print("=" * 78)
    print(f"  policy {P.mean():.4f}   AO {A.mean():.4f}   "
          f"delta {d.mean():+.4f} +- {d.std(ddof=1)/np.sqrt(len(d)):.4f}")
    print(f"\n  WIN RATE  policy wins {100*np.mean(d>0):.1f}% of states")
    w, l = d[d > 0], d[d <= 0]
    if len(w):
        print(f"    wins  n={len(w):3d}  mean +{w.mean():.4f}  worst +{w.min():.4f}"
              f"  best +{w.max():.4f}")
    if len(l):
        print(f"    losses n={len(l):3d}  mean {l.mean():.4f}  worst {l.min():.4f}")
        print(f"    total won {w.sum():+.2f} vs total lost {l.sum():+.2f}"
              f"   -> losses erase {100*abs(l.sum())/max(w.sum(),1e-9):.0f}% of the wins")

    print("\n  LOSS CLASS  correlation of per-state delta with state features:")
    for k in ("blk", "irs", "qos", "hspread"):
        v = np.array(feat[k], dtype=float)
        if v.std() < 1e-12:
            print(f"    {k:<9} constant")
            continue
        r = float(np.corrcoef(v, d)[0, 1])
        # Terciles by RANK, not by value: several of these features are discrete
        # (blocked count, IRS count) and a median split on them can put every
        # sample on one side, which silently produced an empty slice and a nan.
        order = np.argsort(v, kind="stable")
        t = len(order) // 3
        lo_i, hi_i = order[:t], order[-t:] if t else order
        print(f"    {k:<9} corr {r:+.3f}   delta|bottom-third {d[lo_i].mean():+.4f}"
              f"   delta|top-third {d[hi_i].mean():+.4f}"
              f"   (range {v.min():.3g}..{v.max():.3g})")

    # Where the oracle gain LIVES. The pooled number hides whether a swap helps
    # everywhere a little or rescues the minority of states that carry the whole
    # deficit. Only the second is a target: on Case 2 the losing fifth erases
    # 70-92% of everything the winning majority earns, so a swap that only pays
    # on states already being won cannot move the total.
    print("\n  WHERE THE ORACLE GAIN LIVES  (mean gain over policy, by outcome)")
    win_m, los_m = d > 0, d <= 0
    print(f"    {'swap':<12}{'on WINS':>12}{'on LOSSES':>12}{'pooled':>10}")
    base = P.mean()
    for k, lbl in (("ph", "phase"), ("pw", "power"), ("ph+pw", "phase+power")):
        v = np.array(rec[k]) - P
        gw = v[win_m].mean() if win_m.any() else float("nan")
        gl = v[los_m].mean() if los_m.any() else float("nan")
        print(f"    {lbl:<12}{gw:>+12.4f}{gl:>+12.4f}{v.mean():>+10.4f}")
    print(f"    {'(AO itself)':<12}{-d[win_m].mean():>+12.4f}"
          f"{-d[los_m].mean():>+12.4f}{-d.mean():>+10.4f}")

    print("\n  CEILING  oracle swaps, singly and jointly (vs policy):")
    for k, lbl in (("ph", "phase"), ("pw", "power"), ("ph+pw", "phase+power")):
        v = np.array(rec[k])
        print(f"    {lbl:<12}{v.mean():.4f}  {v.mean()-base:+.4f}")
    inter = (np.array(rec["ph+pw"]).mean() - base) - \
            ((np.array(rec["ph"]).mean() - base) + (np.array(rec["pw"]).mean() - base))
    print(f"    interaction term {inter:+.4f}")
    print(f"    joint ceiling vs AO: {np.array(rec['ph+pw']).mean()-A.mean():+.4f}")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
