"""
probe_assignment_quality.py
---------------------------
ASSIGNMENT-QUALITY analysis — bakes the "route blocked users to their building's
IRS" principle into a routine, re-usable check.

Principle (confirmed empirically, see docs/Common-Knowledge [N]):
  • A blocked user sits IN its building's footprint → strong g_RU to THAT IRS and
    a dead (×β_block) direct link → its best link is its building's IRS.
  • A non-blocked user has a good direct link → its best link is direct.
  ⟹ ORACLE assignment = {blocked → nearest/own-building IRS, non-blocked → direct}.
  ⟹ optimal #IRS-routed ≈ #blocked;  per-IRS load ≈ #blocked / M.
  OVER-assign (non-blocked → IRS) = bad: crowds the IRS group, no upside.
  UNDER-assign (blocked → direct)  = bad: leaves a dead link on the table.

Whether routing ALL blocked users is loss-free depends on per-IRS load vs the
closed-form capacity  n*(K,G) = R_c,sat / (D_k − R_priv,sat)  (Common-Knowledge [N]):
  load ≤ n*  → ~100% (Case 1)   |   load ≈ n* → marginal (Case 2)   |
  load > n*  → structural QoS shortfall (Case 3).

Reports the oracle assignment's QoS ceiling, the per-IRS load, the capacity n*,
and the load-vs-capacity verdict. With --ckpt, also scores the trained policy's
assignment: over-/under-/mis-assignment rates and the QoS gap to the oracle.

Usage:
  python analysis/probe_assignment_quality.py --K 10 --M 2 --P 50
  python analysis/probe_assignment_quality.py --ckpt results/result_N/checkpoints/ep_00400
"""

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx


def _capacity(K, G, Dk, rho=0.2):
    """Closed-form saturated per-IRS capacity n* (Common-Knowledge [N])."""
    b = (1 - rho) / K
    a = rho / (G + 1)
    Rp = np.log2(1 + b / (1 - b - a))
    Rc = -np.log2(1 - a)
    n_star = (Rc / (Dk - Rp)) if Dk > Rp else float('inf')
    return n_star, Rp, Rc


def _oracle_assignment(ch, user_pos, irs_pos):
    """blocked → nearest IRS (its building), non-blocked → direct (0)."""
    K, M = user_pos.shape[0], irs_pos.shape[0]
    blk = ch['su_blocked']
    d = np.linalg.norm(user_pos[:, None, :] - irs_pos[None, :, :], axis=2)  # (K,M)
    nearest = d.argmin(axis=1) + 1
    return np.where(blk, nearest, 0).astype(int)


def _qos_of(assignment, ch, cfg, rate):
    """QoS-met fraction + Σ R_tot under oracle phase + reference power."""
    K, M, N = cfg.K, cfg.M, cfg.N
    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    pidx = oracle_phase_idx(est, assignment, cfg)
    Phi = np.zeros((M, N, N), dtype=complex)
    di = np.arange(N)
    Phi[:, di, di] = np.exp(1j * cfg.phase_levels[pidx])
    active = sorted(set(int(a) for a in assignment if a > 0))
    G = max(len(active), 0)
    w_p = np.full(K, cfg.P_S * 0.8 / K)
    w_c_vec = np.full(G + 1, cfg.P_S * 0.2 / (G + 1))
    out = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                w_p=w_p, w_c_vec=w_c_vec,
                                active_irs_ids=active, sigma2=cfg.sigma2)
    Rtot = out['R_private'] + out['C_k']
    return float(np.mean(Rtot >= cfg.D_k_bps_hz)), float(np.sum(Rtot))


def score_ckpt(ckpt, args):
    """Load a trained policy and score its assignment vs the oracle
    (blocked→its-IRS): over-/under-/mis-assign rates + QoS gap to oracle.
    Auto-derives the run's config from its hyperparameters.json."""
    import json
    from probe_critic_ceiling import make_checkpoint_policy

    d = os.path.abspath(ckpt)
    hp = None
    for _ in range(4):
        d = os.path.dirname(d)
        cand = os.path.join(d, 'hyperparameters.json')
        if os.path.isfile(cand):
            hp = json.load(open(cand)); break
    if hp is None:
        raise SystemExit(f"hyperparameters.json not found near {ckpt}")
    s = hp['system']
    cfg = make_config(K=s['K'], M=s['M'], N=s['N'], P_S_dBm=s['P_S_dBm'],
                      R_LoS_km=s['R_LoS_km'], D_k_bps_hz=s['D_k_bps_hz'])
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=16)
    policy = make_checkpoint_policy(ckpt, cfg)
    K, M, Dk = cfg.K, cfg.M, cfg.D_k_bps_hz

    tot = tot_blk = over = under = mis = correct = 0
    agent_irs, q_agent, q_oracle = [], [], []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            ch = {k: (v.copy() if hasattr(v, 'copy') else v)
                  for k, v in env.channels.items()}
            blk = ch['su_blocked'].astype(bool)
            a = policy(env)
            assign = np.asarray(a['assignment'])
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
            agent_irs.append(int(np.sum(assign > 0)))
            for k in range(K):
                tot += 1
                if blk[k]:
                    tot_blk += 1
                    if assign[k] == 0:          under += 1
                    elif assign[k] == a_or[k]:  correct += 1
                    else:                       mis += 1
                elif assign[k] > 0:             over += 1
            q_oracle.append(_qos_of(a_or, ch, cfg, rate)[0])
            _, _, _, info = env.step(a)
            q_agent.append(float(np.mean(np.asarray(info['R_tot']) >= Dk)))

    n_nonblk = tot - tot_blk
    print("=" * 80)
    print(f"  ASSIGNMENT-QUALITY (ckpt vs oracle) · {ckpt}")
    print(f"  Case K={K} M={M} N={cfg.N} P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} D_k={Dk}")
    print("=" * 80)
    print(f"  blocked / step           : {tot_blk/tot*K:.2f} of {K}  ({tot_blk/tot*100:.0f}%)")
    print(f"  agent #IRS-routed / step : {np.mean(agent_irs):.2f}   "
          f"(oracle target ≈ {tot_blk/tot*K:.2f})")
    print("  " + "-" * 76)
    print(f"  blocked → correct IRS    : {100*correct/max(tot_blk,1):5.1f}%")
    print(f"  blocked → WRONG IRS (mis): {100*mis/max(tot_blk,1):5.1f}%")
    print(f"  blocked → direct (UNDER) : {100*under/max(tot_blk,1):5.1f}%  ← dead links wasted")
    print(f"  non-block → IRS  (OVER)  : {100*over/max(n_nonblk,1):5.1f}%  ← crowds IRS, dilutes common")
    print("  " + "-" * 76)
    qa, qo = np.mean(q_agent)*100, np.mean(q_oracle)*100
    print(f"  QoS  agent (full policy)        : {qa:5.1f}%")
    print(f"  QoS  oracle {{blocked→IRS + oracle-φ}} : {qo:5.1f}%")
    print(f"  GAP to oracle                   : {qo-qa:+5.1f} pp  "
          f"= optimization/credit headroom [N]")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None,
                    help='score a trained checkpoint vs the oracle (auto-derives config)')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=2)
    ap.add_argument('--N', type=int, default=24)
    ap.add_argument('--P', type=float, default=None, help='P_S_dBm override (default per-case)')
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.ckpt:
        score_ckpt(args.ckpt, args)
        return

    ov = dict(K=args.K, M=args.M, N=args.N, R_LoS_km=args.rlos)
    if args.P is not None:
        ov['P_S_dBm'] = args.P
    cfg = make_config(**ov)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)
    K, M, Dk = cfg.K, cfg.M, cfg.D_k_bps_hz

    blk_frac, n_irs_load, q_oracle, q_alldir, q_allirs = [], [], [], [], []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            ch = env.channels
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
            blk_frac.append(float(np.mean(ch['su_blocked'])))
            # per-IRS load under oracle routing
            loads = [int(np.sum(a_or == (m + 1))) for m in range(M)]
            n_irs_load.append(np.mean([l for l in loads if l > 0]) if any(loads) else 0.0)
            q_oracle.append(_qos_of(a_or, ch, cfg, rate)[0])
            q_alldir.append(_qos_of(np.zeros(K, int), ch, cfg, rate)[0])
            q_allirs.append(_qos_of(np.array([(k % M) + 1 for k in range(K)]), ch, cfg, rate)[0])
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    bf = np.mean(blk_frac)
    load = np.mean(n_irs_load)
    n_blocked = bf * K
    G_active = M
    n_star, Rp, Rc = _capacity(K, G_active, Dk)

    print("=" * 80)
    print(f"  ASSIGNMENT-QUALITY (oracle blocked→IRS) · K={K} M={M} N={cfg.N} "
          f"P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} D_k={Dk}")
    print("=" * 80)
    print(f"  blocked users            : {bf*100:.1f}%  (≈ {n_blocked:.1f} of {K})")
    print(f"  ⟹ optimal #IRS-routed    : ≈ {n_blocked:.1f}   (IRS/K ≈ Blk/K = {bf:.2f})")
    print(f"  ⟹ per-IRS load (Blk/M)   : {load:.2f} users/IRS")
    print(f"  per-IRS capacity n*(K,G={G_active}) : {n_star:.2f}   "
          f"(R_priv,sat={Rp:.3f} {'≥' if Rp>=Dk else '<'} D_k={Dk}, R_c,sat={Rc:.3f})")
    margin = (n_star - load) if np.isfinite(n_star) else float('inf')
    if not np.isfinite(n_star) or load <= n_star - 0.5:
        verdict = "load < capacity → routing all blocked is LOSS-FREE (≈100% achievable)"
    elif load <= n_star + 0.5:
        verdict = "load ≈ capacity → MARGINAL (tail users fail; structural ~partial ceiling)"
    else:
        verdict = "load > capacity → STRUCTURAL SHORTFALL (cannot serve all blocked)"
    print(f"  load-vs-capacity verdict : {verdict}")
    print("  " + "-" * 76)
    print(f"  QoS ceiling (oracle blocked→IRS) : {np.mean(q_oracle)*100:5.1f}%")
    print(f"     vs all-direct                 : {np.mean(q_alldir)*100:5.1f}%   "
          f"(under-assign floor)")
    print(f"     vs all-IRS (over-assign)      : {np.mean(q_allirs)*100:5.1f}%   "
          f"(over-assign: non-blocked crammed onto IRS)")
    print("=" * 80)
    print("  READING: oracle ≫ all-direct ⟹ blocked users MUST be routed to IRS (don't under-assign).")
    print("           oracle ≥ all-IRS    ⟹ cramming non-blocked onto IRS doesn't help (don't over-assign).")
    print("           oracle < 100% with load≥n* ⟹ ceiling is STRUCTURAL (per-IRS sharing), not agent fault.")
    print("=" * 80)


if __name__ == '__main__':
    main()
