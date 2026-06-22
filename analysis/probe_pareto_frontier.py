"""
probe_pareto_frontier.py
------------------------
The (QoS, R_tot) PARETO FRONTIER at oracle routing + oracle phase, P_S fixed.

Sweeps a private-power concentration exponent β:
    w_p,k ∝ |h_eff,k|^(2β)     (normalized to 0.8·P_S; common = 0.2·P_S equal)
  β = 0  → EQUAL power  (max QoS: everyone gets the same → most users clear D_k)
  β > 0  → CONCENTRATE on strong links (max R_tot, but weak users drop below D_k)
  β < 0  → favour weak links (usually wasteful — over-invests below-D_k users)

For each β reports QoS%, Σ R_tot, and reward = ΣR − λ_D·Σ(shortfall/D_k)².
The frontier shows where QoS↔R_tot becomes a pure tradeoff, and which β maximises
the training reward (the point the agent SHOULD sit on — "just enough for the
servable, dump the rest on strong links, drop the hopeless within the λ_D margin").

Usage:
  python analysis/probe_pareto_frontier.py --P 50 --K 10 --M 2 --lambda-D 1.5
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx
from analysis.probe_assignment_quality import _oracle_assignment


def eval_beta(assignment, ch, cfg, rate, beta):
    K, M, N = cfg.K, cfg.M, cfg.N
    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    pidx = oracle_phase_idx(est, assignment, cfg)
    Phi  = np.zeros((M, N, N), dtype=complex)
    di   = np.arange(N); Phi[:, di, di] = np.exp(1j * cfg.phase_levels[pidx])
    active = sorted(set(int(a) for a in assignment if a > 0)); G = len(active)
    h_eff = rate.effective_channels_all(assignment, Phi, ch)        # (K,) complex
    g = np.abs(h_eff) ** 2 + 1e-30
    w = g ** beta
    w_p     = w / w.sum() * (cfg.P_S * 0.8)
    w_c_vec = np.full(G + 1, cfg.P_S * 0.2 / (G + 1))
    out  = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                 w_p=w_p, w_c_vec=w_c_vec, active_irs_ids=active,
                                 sigma2=cfg.sigma2)
    Rtot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
    qos  = float(np.mean(Rtot >= cfg.D_k_bps_hz))
    sumR = float(np.sum(Rtot))
    short = np.maximum(0.0, cfg.D_k_bps_hz - Rtot) / cfg.D_k_bps_hz
    reward = sumR - cfg.lambda_D * float(np.sum(short ** 2))
    return qos, sumR, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--P', type=float, default=50.0)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=2)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--lambda-D', dest='lam', type=float, default=1.5)
    ap.add_argument('--episodes', type=int, default=25)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    cfg  = make_config(K=args.K, M=args.M, P_S_dBm=args.P, R_LoS_km=args.rlos,
                       lambda_D=args.lam)
    rate = RateComputer(cfg)
    env  = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)
    betas = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    acc = {b: {'q': [], 'r': [], 'rew': []} for b in betas}
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            ch   = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in env.channels.items()}
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
            for b in betas:
                q, r, rew = eval_beta(a_or, ch, cfg, rate, b)
                acc[b]['q'].append(q); acc[b]['r'].append(r); acc[b]['rew'].append(rew)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)
    print("=" * 74)
    print(f"  (QoS, R_tot) PARETO FRONTIER · oracle routing+phase · K={args.K} M={args.M} "
          f"P_S={args.P} λ_D={args.lam}")
    print("=" * 74)
    print(f"  {'β (pwr conc.)':>13} | {'QoS%':>6} | {'Σ R_tot':>8} | {'reward':>8}")
    print("  " + "-" * 52)
    best_rew, best_b = -1e9, None
    for b in betas:
        q = np.mean(acc[b]['q']) * 100; r = np.mean(acc[b]['r']); rew = np.mean(acc[b]['rew'])
        tag = "  ← EQUAL (max QoS)" if b == 0 else ("  ← concentrate" if b >= 4 else "")
        print(f"  {b:>13.1f} | {q:>5.1f}% | {r:>8.3f} | {rew:>8.3f}{tag}")
        if rew > best_rew: best_rew, best_b = rew, b
    print("  " + "-" * 52)
    print(f"  reward-OPTIMAL β = {best_b}  (this is the (QoS,R_tot) point the agent SHOULD sit on @λ_D={args.lam})")
    print("=" * 74)
    print("  Read: QoS falls + R_tot rises as β↑ = the TRADEOFF region. If QoS stays flat")
    print("  while R_tot rises, equal-power is over-investing weak links (room to push R_tot free).")


if __name__ == "__main__":
    main()
