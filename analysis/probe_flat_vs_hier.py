"""
probe_flat_vs_hier.py
---------------------
Is the flat-PPO baseline actually BETTER than the hierarchical actor, or is it
just standing somewhere else on the same rate/QoS trade-off?

`probe_vs_ao.py` says flat wins on J at Case 1 (+0.947 vs +0.648 against AO) while
serving 77.2% of users against 99.3%. J alone cannot separate those two readings,
because J's shortfall penalty is quadratic and therefore very soft just under the
threshold: a user at 0.9*D_k costs ~0.015, which is nearly free. A policy that
shaves every user to just under D_k can bank the freed power as sum-rate and still
look good on J. So J at one lambda is exactly the wrong instrument here.

Four measurements, all on the SAME paired states, decide-on-g-hat / score-on-true-g,
and with the same oracle C_k rule applied to every arm (as in probe_vs_ao.make_J):

  RATE vs QoS   R_tot and binary QoS side by side. If one policy wins BOTH it
                Pareto-dominates and the comparison is settled. If it wins one and
                loses the other, they are two points on a frontier and the ranking
                is a question of which operating point the paper claims.

  SERVICE       the distribution of R_k / D_k, not the binary pass/fail. A policy
                whose failures sit at 0.95*D_k is shaving; one whose failures sit
                at 0.3*D_k is starving users. These are very different things and
                the binary QoS column hides both.

  LAMBDA SWEEP  J recomputed at several lambda_D. The crossing point tells you how
                much the ranking depends on the arbitrary penalty weight. A ranking
                that flips between lambda 1.5 and 3 is not a result.

  EQUAL-QoS     the honest headline: among states where BOTH policies serve the same
                number of users, who gets more sum-rate? This removes the trade-off
                entirely and asks the one question that survives any lambda choice.

Usage:
  python analysis/probe_flat_vs_hier.py \
      --runs flat:results/result_301 hier:results/result_238 \
      --seeds 42 7 21 --states 100
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
from analysis.oracle_alloc import oracle_ck_met

LAMBDAS = (1.5, 3.0, 5.0, 10.0, 20.0)


def _rates(env, cfg, phi, Phi, wp, wc, ids):
    """True-channel per-user rate under the shared oracle C_k rule."""
    rc = rc_of(env)
    p = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                 active_irs_ids=ids)
    _, Ck = oracle_ck_met(p["R_private"], p["R_c_group"], p["groups"],
                          cfg.D_k_bps_hz)
    for g, m in p["groups"].items():
        m = np.asarray(list(m), dtype=int)
        if m.size:
            left = float(p["R_c_group"].get(int(g), 0.0)) - float(Ck[m].sum())
            if left > 1e-12:
                Ck[m] += left / m.size
    o = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=Ck,
                            active_irs_ids=ids, use_true=True)
    rp, ck = np.asarray(o["R_private"]), np.asarray(o["C_k"])
    return rp + ck, float(o["sum_rate"]), rp, float(np.mean(ck))


def rc_of(env):
    return env.rate_computer


def collect(run_dir, seed, n_states):
    cfg = load_training_cfg(run_dir)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_states + 5)
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    is_flat = phase_net is None
    flat_prev = None

    R_all, rtot = [], []
    env.reset(seed=seed)
    for _ in range(n_states):
        obs = env._get_obs()
        blocked = _compute_irs_favored(env)
        if is_flat:
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            s_t = actor.extract_state(obs, demand, blocked, h_eff=h_now,
                                      prev_rates=flat_prev, update_norm=False)
            a, _, fi = actor.forward(s_t, greedy=True)
            phi, ids = a["phi"], a["active_ids"]
            Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(a["phase_idx"]))
            wp, wc = fi["w_p"], fi["w_c_vec"]
        else:
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, ainfo = actor.forward(s_t, greedy=True)
            rep = ainfo["z_t"]
            active, ids = _get_active_irs(phi), _get_active_irs_ids(phi)
            pidx, _, _ = phase_net.forward(
                _build_phase_state(env.channels, phi, cfg, rep), active,
                greedy=True)
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
            wc, wp, _, _ = power_net.forward(sp, ids)

        R, tot, rp, ckm = _rates(env, cfg, phi, Phi, wp, wc, ids)
        if is_flat:
            # train_flat.rollout feeds back (R_private, mean C_k), not the total.
            flat_prev = (rp, ckm)
        R_all.append(R)
        rtot.append(tot)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
    return np.array(R_all), np.array(rtot), cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="LABEL:run_dir ...")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 21])
    ap.add_argument("--states", type=int, default=100)
    a = ap.parse_args()

    got = {}
    for spec in a.runs:
        lbl, d = spec.split(":", 1)
        Rs, Ts, cfg = [], [], None
        for s in a.seeds:
            R, T, cfg = collect(d, s, a.states)
            Rs.append(R)
            Ts.append(T)
        got[lbl] = (np.concatenate(Rs), np.concatenate(Ts), cfg)

    Dk = list(got.values())[0][2].D_k_bps_hz
    eps = float(getattr(list(got.values())[0][2], "epsilon_qp", 1e-3))

    print("=" * 78)
    print(f"  FLAT vs HIERARCHICAL · {a.states} states x {len(a.seeds)} seeds"
          f" · D_k={Dk} · score on true g")
    print("=" * 78)
    print(f"  {'policy':<10}{'R_tot':>9}{'QoS':>8}{'med R/D':>10}"
          f"{'<0.5D':>8}{'0.9-1D':>9}{'>=1D':>8}")
    for lbl, (R, T, _) in got.items():
        ratio = R.ravel() / Dk
        print(f"  {lbl:<10}{T.mean():>9.4f}{100*np.mean(R >= Dk):>7.1f}%"
              f"{np.median(ratio):>10.3f}"
              f"{100*np.mean(ratio < 0.5):>7.1f}%"
              f"{100*np.mean((ratio >= 0.9) & (ratio < 1.0)):>8.1f}%"
              f"{100*np.mean(ratio >= 1.0):>7.1f}%")

    print("\n  J AT SEVERAL lambda_D   (the paper uses 1.5)")
    print(f"  {'policy':<10}" + "".join(f"{('λ=%g' % l):>11}" for l in LAMBDAS))
    for lbl, (R, T, _) in got.items():
        row = ""
        for lam in LAMBDAS:
            s = np.maximum(0.0, Dk - R) / (Dk + eps)
            J = T - lam * (s ** 2).sum(axis=1)
            row += f"{J.mean():>11.4f}"
        print(f"  {lbl:<10}{row}")

    labels = list(got)
    if len(labels) == 2:
        A, B = labels
        Ra, Ta, _ = got[A]
        Rb, Tb, _ = got[B]
        na = (Ra >= Dk).sum(axis=1)
        nb = (Rb >= Dk).sum(axis=1)
        m = na == nb
        print(f"\n  EQUAL-QoS STATES  ({m.sum()}/{len(m)} = {100*m.mean():.1f}%"
              f" of states serve the same number of users)")
        if m.sum():
            print(f"    {A:<10} R_tot {Ta[m].mean():.4f}")
            print(f"    {B:<10} R_tot {Tb[m].mean():.4f}"
                  f"    -> {A} {'+' if Ta[m].mean() > Tb[m].mean() else ''}"
                  f"{Ta[m].mean() - Tb[m].mean():+.4f}")
        print(f"\n  PARETO   {A} serves more users in {100*np.mean(na > nb):.1f}%"
              f" of states, fewer in {100*np.mean(na < nb):.1f}%")
        both = np.mean((Ta > Tb) & (na > nb))
        print(f"           {A} wins BOTH rate and QoS in {100*both:.1f}% of states"
              f"  (Pareto dominance would be ~100%)")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
