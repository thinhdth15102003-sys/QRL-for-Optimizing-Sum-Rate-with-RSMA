"""Rate--QoS Pareto frontier with each policy plotted as a point on the same axes.

Correctness requirements this probe is built around:

  1. ONE state pool. The frontier and every policy are evaluated on the identical
     states and the identical pre-drawn sigma^2 realisations. Measuring them in
     separate runs would put the curve and the points in different samples, and
     a gap of the size being claimed here is smaller than that discrepancy.

  2. The frontier is the CONSTRAINED optimum, not a lambda sweep. Sweeping lambda
     traces only the convex hull of the achievable set — any concave stretch is
     skipped and the curve is drawn above points that are actually attainable.
     Here, for each service level n, it is  max sum-R  s.t.  #met >= n, plus a
     robust-margin sweep that reaches the QoS = 100% end which the #met
     constraint cannot (nominal selection still loses users to the noise draw).

  3. Same search as the paper's ceiling. Candidates come from
     probe_v_star.build_candidates, so the curve is the same object cited in
     docs/Pareto-Frontier.md rather than a re-implementation that might search
     a different set.

  4. The curve is an ACHIEVABLE LOWER BOUND on the true frontier: power is a rich
     grid rather than a solver, and at Case-2 size the assignment is a hill-climb
     rather than exhaustive. The true frontier can only lie above it, so a policy
     found above the curve would indicate a bug, not a discovery.

Writes a .npz of per-state records; probe_pareto_plot.py draws from that.
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from CSI.baselines import RandomPolicy, GreedyPolicy
from RL import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP
from train import (_build_phase_state, _build_ck_state, _get_active_irs,
                   _get_active_irs_ids, _compute_irs_favored, _build_power_state)
from analysis.oracle_alloc import phi_from_idx
from analysis.probe_ao_rate import AORatePolicy
from analysis.probe_v_star import (build_candidates, rescore_noisy, Cand,
                                   _oracle_assignment, oracle_phase_idx,
                                   eval_candidate)

DELTA_GRID = [0.0, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15]


def rescore(a, Phi, w_p, w_c, active, C_k, ch, cfg, rate, sig, Dk, eps):
    """Score a fixed action over the pre-drawn sigma^2 list.

    Takes Phi directly rather than a phase index because the blind baselines do
    not emit `phase_idx` at all — env._apply_action leaves self.Phi untouched, so
    they run on whatever the environment currently has applied. Scoring them
    against a phase they never chose would flatter them.

    Numerically identical to probe_v_star.rescore_noisy when Phi comes from that
    candidate's own pidx; asserted at start-up in main().
    """
    sR = pen = qos = feas = 0.0
    for s2 in sig:
        out = rate.compute_sum_rate(assignment=a, Phi=Phi, channels=ch,
                                    w_p=w_p, w_c_vec=w_c, C_k=C_k,
                                    active_irs_ids=active, sigma2=s2)
        R_tot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
        short = np.maximum(0.0, Dk - R_tot)
        sR += out['sum_rate']
        pen += float(np.sum((short / (Dk + eps)) ** 2))
        qos += float(np.mean(R_tot >= Dk))
        feas += float(out['feasible'])
    n = len(sig)
    return {'sumR': sR / n, 'pen': pen / n, 'qos': qos / n, 'feas': feas / n}


class HierPolicy:
    """A trained assignment actor plus its three sub-actors, as one act()."""

    def __init__(self, run_dir, cfg, seed=0):
        d = os.path.join(run_dir, 'agents')
        ac = json.load(open(os.path.join(d, 'actor_config.json')))
        self.mode = ac.get('mode', 'quantum')
        self.actor = (ClassicalActor if self.mode == 'classical'
                      else QuantumActor).from_dir(d, seed=seed)
        if self.mode != 'classical':
            self.actor.n_shots = P.n_shots_eval
        self.ph = PhaseMLP.from_dir(d, seed=seed)
        self.pw = PowerMLP.from_dir(d, seed=seed)
        self.ck = CkMLP.from_dir(d, seed=seed)
        # power-fairness lives in hyperparameters.json, not power_config.json;
        # omitting it silently changes the executed allocation
        hp_dir = os.path.abspath(d)
        for _ in range(5):
            hp_dir = os.path.dirname(hp_dir)
            hp = os.path.join(hp_dir, 'hyperparameters.json')
            if os.path.isfile(hp):
                pn = json.load(open(hp)).get('power_net', {})
                self.pw.power_fairness = float(pn.get('power_fairness', 0.0))
                self.pw.power_fairness_priv = float(
                    pn.get('power_fairness_priv', pn.get('power_fairness', 0.0)))
                self.pw.power_priv_frac = float(pn.get('power_priv_frac', 0.8))
                break
        self.cfg = cfg
        self.dem = np.full(cfg.K, cfg.D_k_bps_hz)

    def act(self, obs, env, ch):
        cfg = self.cfg
        s = self.actor.extract_state(obs, self.dem, _compute_irs_favored(env))
        phi, _, inf = self.actor.forward(s, greedy=True)
        rep = inf['z_t']
        act_irs, act_ids = _get_active_irs(phi), _get_active_irs_ids(phi)
        pidx, _, _ = self.ph.forward(
            _build_phase_state(ch, phi, cfg, rep), act_irs, greedy=True)
        Phi = phi_from_idx(pidx, cfg)
        h = env.rate_computer.effective_channels_all(phi, Phi, ch)
        wc, wp, _, _ = self.pw.forward(_build_power_state(h, rep), act_ids)
        part = env.rate_computer.compute_rates_partial(
            phi, Phi, ch, wp, wc, active_irs_ids=act_ids)
        C_k, _, _ = self.ck.forward(
            _build_ck_state(self.dem, part['R_private'], part['R_c_group'], phi, cfg),
            phi, part['R_c_group'])
        return dict(a=phi, Phi=Phi, w_p=wp, w_c=wc, active=act_ids, C_k=C_k)


def act_to_exec(action, env, cfg):
    """A policy's action dict -> the action the environment would actually apply.

    Mirrors env._apply_action: a missing `phase_idx` or `C_k` means the env keeps
    the value it already holds, so the blind baselines inherit the current Phi
    and C_k rather than choosing one.
    """
    a = np.asarray(action['assignment'], dtype=int)
    active = sorted({int(x) for x in a if x > 0})
    if 'phase_idx' in action:
        Phi = phi_from_idx(np.asarray(action['phase_idx'], dtype=int), cfg)
    else:
        Phi = env.Phi
    C_k = action.get('C_k', env.C_k)
    return dict(a=a, Phi=Phi, w_p=action['w_p'], w_c=action['w_c_vec'],
                active=active, C_k=C_k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, required=True)
    ap.add_argument('--M', type=int, required=True)
    ap.add_argument('--P-S', dest='ps', type=float, default=50.0)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.5)
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--noise-draws', type=int, default=16)
    ap.add_argument('--hier', nargs='*', default=[], help='label=run_dir')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M, P_S_dBm=a.ps, R_LoS_km=a.rlos)
    rate = RateComputer(cfg)
    K, M, Dk, eps = cfg.K, cfg.M, cfg.D_k_bps_hz, cfg.epsilon_qp
    L = len(cfg.phase_levels)
    n_asg = (M + 1) ** K
    large = n_asg * (L ** M) > 2100
    assignments = [np.array(x, dtype=int) for x in
                   itertools.product(range(M + 1), repeat=K)] if not large else None

    # env seeded EXACTLY as probe_v_star does, so the sanity row below can be
    # compared against that probe's own output to prove the pools coincide
    env = ISTNEnv(cfg=cfg, seed=a.seed + 777, n_steps_ep=a.steps, reward_noise_avg=1)
    ao = AORatePolicy(cfg, rate, rounds=2, n_split=11, route_ascent=True, phase_sweeps=1)
    greedy, rnd = GreedyPolicy(cfg), RandomPolicy(cfg, rng=np.random.default_rng(0))
    pols = {lbl: HierPolicy(rd, cfg) for lbl, rd in
            (s.split('=', 1) for s in a.hier)}

    print(f"⚙ K={K} M={M} P_S={a.ps} R_LoS={a.rlos} D_k={Dk} "
          f"| assignments={n_asg} [{'LOCAL-SEARCH' if large else 'EXHAUSTIVE'}]"
          f" | policies={list(pols)}", flush=True)

    fr_n = {n: [] for n in range(1, K + 1)}     # #met-constrained
    fr_d = {d: [] for d in DELTA_GRID}          # robust-margin
    mods = {lbl: [] for lbl in list(pols) + ['AO', 'Greedy', 'Random']}
    san = []
    ns = 0

    for ep in range(a.episodes):
        obs = env.reset(seed=a.seed + ep)
        for st in range(a.steps):
            if st % a.stride == 0:
                ch = {k: (v.copy() if hasattr(v, 'copy') else v)
                      for k, v in env.channels.items()}
                sig = [env.channel_model.sample_noise_sigma2()
                       for _ in range(a.noise_draws)]
                cands = build_candidates(ch, env.user_pos, env.irs_pos,
                                         cfg, rate, assignments, large)

                if ns == 0:      # one-time check that the two scorers agree
                    b0 = cands[0]
                    r1 = rescore_noisy(b0, b0.C_g, ch, cfg, rate, sig, Dk, eps)
                    r2 = rescore(b0.a, phi_from_idx(b0.pidx, cfg), b0.w_p, b0.w_c,
                                 b0.active, b0.C_g, ch, cfg, rate, sig, Dk, eps)
                    d = max(abs(r1[k] - r2[k]) for k in r1)
                    assert d < 1e-12, f"scorers disagree by {d:.3e}"
                    print(f"  ✓ local rescore == probe_v_star.rescore_noisy "
                          f"(max diff {d:.1e})", flush=True)

                for n in range(1, K + 1):
                    sub = [c for c in cands if c.nmet_g >= n]
                    if sub:
                        b = max(sub, key=lambda c: c.sumR)
                        r = rescore_noisy(b, b.C_g, ch, cfg, rate, sig, Dk, eps)
                        fr_n[n].append((r['sumR'], r['qos']))
                for dl in DELTA_GRID:
                    sub = [c for c in cands if c.margin >= dl]
                    if sub:
                        b = max(sub, key=lambda c: c.sumR)
                    else:
                        # No candidate clears the margin in this state. Dropping
                        # it would average the point over the SUBSET of states
                        # where a high margin happens to be attainable — the
                        # easy ones — and bias the high-QoS end of the frontier.
                        # Take the best margin the state can reach instead, so
                        # every point is an average over all states.
                        b = max(cands, key=lambda c: (c.margin, c.sumR))
                    r = rescore_noisy(b, b.C_m, ch, cfg, rate, sig, Dk, eps)
                    fr_d[dl].append((r['sumR'], r['qos'], 1.0 if sub else 0.0))

                for lbl, pol in pols.items():
                    e = pol.act(obs, env, ch)
                    r = rescore(e['a'], e['Phi'], e['w_p'], e['w_c'], e['active'],
                                e['C_k'], ch, cfg, rate, sig, Dk, eps)
                    mods[lbl].append((r['sumR'], r['qos']))
                for lbl, pol in (('AO', ao), ('Greedy', greedy), ('Random', rnd)):
                    act = pol.act(obs) if lbl == 'Random' else pol.act(obs, env)
                    e = act_to_exec(act, env, cfg)
                    r = rescore(e['a'], e['Phi'], e['w_p'], e['w_c'], e['active'],
                                e['C_k'], ch, cfg, rate, sig, Dk, eps)
                    mods[lbl].append((r['sumR'], r['qos']))

                # sanity row, identical recipe to probe_v_star's — the cross-check
                a_or = np.asarray(_oracle_assignment(ch, env.user_pos, env.irs_pos),
                                  dtype=int)
                act_or = sorted({int(x) for x in a_or if x > 0})
                est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
                       'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
                p_or = oracle_phase_idx(est, a_or, cfg)
                c_or = eval_candidate(a_or, phi_from_idx(p_or, cfg), p_or, ch, cfg,
                                      rate, act_or, 0.2, 0.0, 'eq', Dk, eps)
                r = rescore_noisy(c_or, c_or.C_wf, ch, cfg, rate, sig, Dk, eps)
                san.append((r['sumR'], r['qos']))
                ns += 1
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
            obs = env._get_obs()
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{a.episodes}  states={ns}", flush=True)

    np.savez(a.out, K=K, M=M, n_states=ns,
             sanity=np.array(san),
             **{f"frn_{n}": np.array(v) for n, v in fr_n.items()},
             **{f"frd_{i}": np.array(v) for i, v in enumerate(fr_d.values())},
             delta_grid=np.array(DELTA_GRID),
             **{f"mod_{lbl}": np.array(v) for lbl, v in mods.items()})

    s = np.array(san)
    print(f"\n  states={ns}   sanity: R_tot {s[:,0].mean():.4f} · QoS {s[:,1].mean()*100:.1f}%"
          f"   <-- must match probe_v_star's sanity row on the same seed/stride")
    print("  #met-constrained frontier:")
    for n in range(K, 0, -1):
        v = np.array(fr_n[n])
        if len(v):
            print(f"    n>={n:<3} R_tot {v[:,0].mean():.4f}  QoS {v[:,1].mean()*100:5.1f}%"
                  f"  ({len(v)}/{ns} states attainable)")
    print("  robust-margin frontier  (all states averaged; 'strict%' = states "
          "where the margin was actually attainable):")
    for dl in DELTA_GRID:
        v = np.array(fr_d[dl])
        if len(v):
            print(f"    d>={dl:<6} R_tot {v[:,0].mean():.4f}  QoS {v[:,1].mean()*100:5.1f}%"
                  f"  strict {v[:,2].mean()*100:5.1f}%")
    print("  policies:")
    for lbl, v in mods.items():
        v = np.array(v)
        print(f"    {lbl:<12} R_tot {v[:,0].mean():.4f}  QoS {v[:,1].mean()*100:5.1f}%")
    print(f"  -> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
