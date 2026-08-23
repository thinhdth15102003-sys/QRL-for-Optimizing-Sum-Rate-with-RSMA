"""Where is r275 losing, and how far is it from AO?

Every policy and every oracle swap is evaluated on the SAME states: the user
walk is independent of the action, so the trajectory can be generated once and
replayed, making this a paired comparison rather than five different worlds.

Each swap replaces ONE component with its oracle and leaves the other two at the
policy's own choice, so the three numbers are attributable. All decisions are
made on the estimated channel and every score is paid on the true one -- the
beta sweep showed that mixing those two views is exactly how the power teacher
came to prefer allocations that lose.

The power oracle uses the CAPPED grid (beta <= 1). The default (0,1,4,16) picks
beta 4-16 on the estimate, which measured J 1.375-1.392 on the true channel
against 1.577 at beta=1 -- quoting that as "power headroom" would be quoting an
artefact of the estimate.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import (load_training_cfg, load_agents, _compute_irs_favored,
                   _get_active_irs, _get_active_irs_ids, _build_phase_state,
                   _build_ck_state)
from analysis.oracle_alloc import oracle_ck_met, phi_from_idx, _est_channels
from analysis.phase_oracle import oracle_phase_idx, oracle_phase_search
from analysis.probe_ao_rate import AORatePolicy

BETAS = (0.0, 0.25, 0.5, 1.0)
FRACS = np.linspace(0.05, 0.95, 11)


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
        J.rtot = float(o["sum_rate"])
        return float(o["sum_rate"]) - lam * float((s ** 2).sum())
    J.qos = J.rtot = float("nan")
    return J


def best_power(Jd, phi, Phi, ids, cfg, h):
    """beta<=1 grid x split grid, selected on the DESIGN view."""
    g = np.abs(h) ** 2 + 1e-30
    best = None
    for b in BETAS:
        w = g ** b
        w = w / w.sum()
        for f in FRACS:
            wp = w * cfg.P_S * f
            wc = np.full(len(ids) + 1, cfg.P_S * (1.0 - f) / (len(ids) + 1))
            v = Jd(phi, Phi, wp, wc, ids)
            if best is None or v > best[0]:
                best = (v, wp, wc)
    return best[1], best[2]


def route_ascent(Jd, phi0, Phi, ids0, cfg, wp, wc):
    """Coordinate ascent on the assignment, design view."""
    a = np.array(phi0, dtype=int)
    best = Jd(a, Phi, wp, wc, ids0)
    for k in range(cfg.K):
        for c in range(cfg.M + 1):
            t = a.copy()
            t[k] = c
            ids = sorted({int(x) for x in t if x > 0})
            wc2 = np.full(len(ids) + 1, float(wc.sum()) / (len(ids) + 1))
            v = Jd(t, Phi, wp, wc2, ids)
            if v > best:
                best, a = v, t
    return a, sorted({int(x) for x in a if x > 0})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/result_275")
    ap.add_argument("--states", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Jd, Jt = make_J(env, cfg, False), make_J(env, cfg, True)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=1)

    keys = ("policy", "+oracle assign", "+oracle phase", "+oracle power", "AO")
    res = {k: [] for k in keys}
    qos = {k: [] for k in keys}

    def rec(k, phi, Phi, wp, wc, ids):
        res[k].append(Jt(phi, Phi, wp, wc, ids))
        qos[k].append(Jt.qos)

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

        rec("policy", phi, Phi, wp_p, wc_p, ids)

        a2, ids2 = route_ascent(Jd, phi, Phi, ids, cfg, wp_p, wc_p)
        wc2 = np.full(len(ids2) + 1, float(wc_p.sum()) / (len(ids2) + 1))
        rec("+oracle assign", a2, Phi, wp_p, wc2, ids2)

        pid2, _, _ = oracle_phase_search(
            _est_channels(env.channels), phi, cfg,
            lambda ph: Jd(phi, phi_from_idx(ph, cfg), wp_p, wc_p, ids),
            n_sweeps=1, seed_pidx=np.asarray(
                oracle_phase_idx(_est_channels(env.channels), phi, cfg), dtype=int))
        rec("+oracle phase", phi, phi_from_idx(pid2, cfg), wp_p, wc_p, ids)

        wpo, wco = best_power(Jd, phi, Phi, ids, cfg, h)
        rec("+oracle power", phi, Phi, wpo, wco, ids)

        act = ao.act(obs, env)
        aid = sorted({int(x) for x in act["assignment"] if x > 0})
        rec("AO", act["assignment"], phi_from_idx(act["phase_idx"], cfg),
            act["w_p"], act["w_c_vec"], aid)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    print("=" * 74)
    print(f"  GAP SPLIT · {a.run} · K={cfg.K} M={cfg.M} · {a.states} paired states")
    print("  decide on g-hat, score on true g · power oracle uses beta<=1 grid")
    print("=" * 74)
    base = float(np.mean(res["policy"]))
    ao_J = float(np.mean(res["AO"]))
    print(f"  {'':<18}{'J':>9}{'+/-':>7}{'QoS':>8}{'gain':>9}")
    for k in keys:
        v = np.array(res[k])
        print(f"  {k:<18}{v.mean():>9.4f}{v.std()/np.sqrt(len(v)):>7.4f}"
              f"{100*np.mean(qos[k]):>7.1f}%"
              f"{(v.mean()-base) if k != 'policy' else 0:>+9.4f}")
    print("-" * 74)
    gaps = {k: max(0.0, float(np.mean(res[k])) - base)
            for k in ("+oracle assign", "+oracle phase", "+oracle power")}
    tot = sum(gaps.values())
    print("  Gap con lai:", "  ".join(
        f"{k.replace('+oracle ','')} {100*v/tot:.0f}%" for k, v in gaps.items())
        if tot > 0 else "  (khong con gap do duoc)")
    print(f"  Khoang cach toi AO: {base - ao_J:+.4f} J "
          f"({'VUOT AO' if base > ao_J else 'THUA AO'})   "
          f"QoS {100*(np.mean(qos['policy'])-np.mean(qos['AO'])):+.1f} pt")
    print("=" * 74)


if __name__ == "__main__":
    main()
