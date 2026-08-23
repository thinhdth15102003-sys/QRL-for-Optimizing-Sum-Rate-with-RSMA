"""The phase gap: is the teacher aiming wrong, or is the head not following?

Phase is the only axis left with positive headroom (+0.098 J mean over three env
seeds; assignment and power both came out NEGATIVE once oracles were selected on
g-hat and paid on g). Two very different causes need separating, because they
have opposite fixes:

  TARGET   the teacher supplies oracle_phase_idx, the closed-form alignment that
           maximises |h_eff|. If that target is itself well short of the J-oracle,
           the head is being taught to reach the wrong place and a better teacher
           is the lever.

  TRACKING the target is right but the head does not reproduce it. Then the lever
           is capacity or optimisation -- for Case 2 the head maps 5K+rep = 74
           inputs onto M*N*L = 192 per-element logits through [64,64], which is a
           lot of decisions from a small trunk.

Measures J for head / teacher-target / J-oracle on the SAME states (decide on
g-hat, score on true g), plus per-element agreement between all three. If the
teacher target already scores near the J-oracle and the head does not match it,
the answer is TRACKING.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state)
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx, _est_channels
from analysis.phase_oracle import oracle_phase_idx, oracle_phase_search


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/result_275")
    ap.add_argument("--states", type=int, default=120)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweeps", type=int, default=2)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Jd, Jt = make_J(env, cfg, False), make_J(env, cfg, True)

    keys = ("head", "teacher target", "J-oracle")
    J = {k: [] for k in keys}
    Q = {k: [] for k in keys}
    m_ht, m_ho, m_to, n_act = [], [], [], []

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
        pidx = np.asarray(pidx, dtype=int)
        Phi_h = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
        h = env.rate_computer.effective_channels_all(phi, Phi_h, env.channels)

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

        est = _est_channels(env.channels)
        tgt = np.asarray(oracle_phase_idx(est, phi, cfg), dtype=int)
        orc, _, _ = oracle_phase_search(
            est, phi, cfg,
            lambda ph: Jd(phi, phi_from_idx(ph, cfg), wp_p, wc_p, ids),
            n_sweeps=a.sweeps, seed_pidx=tgt.copy())
        orc = np.asarray(orc, dtype=int)

        for k, p in (("head", pidx), ("teacher target", tgt), ("J-oracle", orc)):
            J[k].append(Jt(phi, phi_from_idx(p, cfg), wp_p, wc_p, ids))
            Q[k].append(Jt.qos)

        # agreement only over ACTIVE IRS rows -- inactive rows are unconstrained
        rows = [g - 1 for g in ids] if ids else []
        if rows:
            hh, tt, oo = pidx[rows], tgt[rows], orc[rows]
            m_ht.append(float((hh == tt).mean()))
            m_ho.append(float((hh == oo).mean()))
            m_to.append(float((tt == oo).mean()))
            n_act.append(len(rows))

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    print("=" * 76)
    print(f"  PHASE ANATOMY · {a.run} · K={cfg.K} M={cfg.M} N={cfg.N} "
          f"L={len(cfg.phase_levels)} · {a.states} states")
    print("  decide on g-hat, score on true g")
    print("=" * 76)
    base = float(np.mean(J["head"]))
    print(f"  {'':<18}{'J':>9}{'+/-':>8}{'QoS':>8}{'vs head':>10}")
    for k in keys:
        v = np.array(J[k])
        print(f"  {k:<18}{v.mean():>9.4f}{v.std()/np.sqrt(len(v)):>8.4f}"
              f"{100*np.mean(Q[k]):>7.1f}%{v.mean()-base:>+10.4f}")
    print("-" * 76)
    tgt_gain = float(np.mean(J["teacher target"])) - base
    orc_gain = float(np.mean(J["J-oracle"])) - base
    if orc_gain > 1e-9:
        print(f"  teacher target captures {100*tgt_gain/orc_gain:.0f}% of the "
              f"J-oracle's gain over the head")
    print(f"\n  per-element agreement (active IRS rows only, "
          f"{np.mean(n_act):.1f} of {cfg.M} rows active)")
    print(f"    head    vs teacher : {100*np.mean(m_ht):.1f}%")
    print(f"    head    vs J-oracle: {100*np.mean(m_ho):.1f}%")
    print(f"    teacher vs J-oracle: {100*np.mean(m_to):.1f}%")
    print(f"    chance             : {100.0/len(cfg.phase_levels):.1f}%")
    print("=" * 76)


if __name__ == "__main__":
    main()
