"""
probe_irs_overload.py
---------------------
IRS-group OVERLOAD tradeoff (oracle phase + equal power, one IRS m=1). Answers:
  Q1 BLOCKED saturation: how many BLOCKED users can share ONE IRS before the
     system reward stops rising (each blocked user has NO direct → must use IRS)?
  Q2 FREE overload: with the blocked users on the IRS, how many extra FREE
     (non-blocked) users joining the same group before total quality/reward drops?

Single sweep n = #users forced onto IRS #1, ordered by IRS-over-direct margin →
BLOCKED users come first (their direct is blocked → huge margin), then FREE users;
the rest stay direct. Per n we report the GLOBAL metrics (all K users) so the
tradeoff is visible end-to-end:
  QoS%   = fraction of ALL K users meeting D_k
  ΣR     = total sum-rate
  reward = ΣR − λ_D · (#users missing D_k)        (training-objective proxy)
plus the blocked/free split INSIDE the group (does adding free users dilute the
common stream and start dropping the blocked users?).

Mechanism (CSI/rate.py): C_k = R_c_group / n_members, R_c_group = log2(1+min_k
SINR_c) → the WEAKEST group member caps the common rate, and it is split ÷n. So
adding users (esp. free users who gave up a good direct link) dilutes everyone.
CPU-only channel math; safe alongside GPU training.

Usage:
  python analysis/probe_irs_overload.py --K 10 --M 1 --P 50        # 6 blocked + 4 free on one IRS
  python analysis/probe_irs_overload.py --K 5  --M 1 --P 50        # Case 1 (3 blocked + 2 free)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx


def _irs_margin(ch, cfg):
    """Per-user IRS(m=1)-optimal gain / direct gain (estimated channels)."""
    N = cfg.N
    g_dir = np.abs(ch['g_SU_hat']) ** 2
    coeff = ch['beta'][0] * np.abs(ch['g_SR_hat'][0]) * N
    g_irs = (coeff * np.abs(ch['g_RU_hat'][0])) ** 2
    return g_irs / (g_dir + 1e-30)


def _eval(ch, cfg, rate, irs_users):
    """Assign irs_users → IRS#1 (oracle phase + equal power), rest direct.
    Returns R_tot (K,) and met (K,) bool over ALL users."""
    K = cfg.K
    assignment = np.zeros(K, dtype=int)
    if len(irs_users):
        assignment[irs_users] = 1
    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    phase_idx = oracle_phase_idx(est, assignment, cfg)
    phases = cfg.phase_levels[phase_idx]
    Phi = np.zeros((cfg.M, cfg.N, cfg.N), dtype=complex)
    di = np.arange(cfg.N)
    Phi[:, di, di] = np.exp(1j * phases)
    w_p = np.full(K, cfg.P_S * 0.8 / K)
    w_c_vec = np.full(2, cfg.P_S * 0.2 / 2)
    out = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                w_p=w_p, w_c_vec=w_c_vec,
                                active_irs_ids=[1], sigma2=cfg.sigma2)
    R_tot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
    return R_tot, (R_tot >= cfg.D_k_bps_hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=1)
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--P', type=float, default=50.0, help='P_S_dBm override')
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--lambda-D', dest='lam', type=float, default=1.5)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ov = dict(K=args.K, M=args.M, P_S_dBm=args.P, R_LoS_km=args.rlos)
    if args.N is not None:
        ov['N'] = args.N
    cfg = make_config(**ov)
    rate = RateComputer(cfg); K = cfg.K; Dk = cfg.D_k_bps_hz; lam = args.lam

    snaps = []
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            snaps.append({k: (v.copy() if hasattr(v, 'copy') else v)
                          for k, v in env.channels.items()})
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    ns = list(range(1, K + 1))
    agg = {n: [] for n in ns}
    nblk_list = []
    for ch in snaps:
        blk = ch['su_blocked'].astype(bool)
        nblk_list.append(int(blk.sum()))
        order = np.argsort(-_irs_margin(ch, cfg))
        for n in ns:
            irs_users = np.sort(order[:n])
            R_tot, met = _eval(ch, cfg, rate, irs_users)
            grp = np.zeros(K, dtype=bool); grp[irs_users] = True
            agg[n].append((int(met.sum()), float(R_tot.sum()),
                           float(R_tot.sum()) - lam * (K - int(met.sum())),
                           int((grp & blk).sum()), int((grp & ~blk).sum()),
                           int((grp & blk & met).sum()), int((grp & ~blk & met).sum())))

    n_op = float(np.mean(nblk_list))
    print("=" * 92)
    print(f"  IRS-GROUP OVERLOAD  ·  K={cfg.K} M={cfg.M} N={cfg.N} P_S={cfg.P_S_dBm}dBm "
          f"R_LoS={cfg.R_LoS_km} D_k={Dk} lam_D={lam}")
    print(f"  oracle phase + equal power · {len(snaps)} snaps · operating point n≈#blocked={n_op:.1f}")
    print("=" * 92)
    print(f"  {'n_IRS':>5} | {'#blk':>4} {'#free':>5} | {'QoS%(all)':>9} | {'sumR':>7} | "
          f"{'reward':>8} | {'blkInGrp-met':>12} | {'freeInGrp-met':>13}")
    print("  " + "-" * 88)
    rows = {}
    for n in ns:
        nmet, sr, rew, blk_in, free_in, blk_m, free_m = np.array(agg[n], float).mean(axis=0)
        rows[n] = (nmet, sr, rew, blk_in, free_in, blk_m, free_m)
        bm = f"{100*blk_m/blk_in:.0f}%" if blk_in > 0.01 else "  -"
        fm = f"{100*free_m/free_in:.0f}%" if free_in > 0.01 else "  -"
        mark = "  <-- op (~all blocked)" if abs(n - round(n_op)) < 0.5 else ""
        print(f"  {n:>5} | {blk_in:>4.1f} {free_in:>5.1f} | {100*nmet/K:>8.1f}% | {sr:>7.3f} | "
              f"{rew:>8.3f} | {bm:>12} | {fm:>13}{mark}")
    print("=" * 92)

    rew = {n: rows[n][2] for n in ns}
    qos = {n: rows[n][0] / K for n in ns}
    peak_n = max(ns, key=lambda n: rew[n])
    qos_peak_n = max(ns, key=lambda n: qos[n])
    print(f"  Q1 BLOCKED saturation: reward PEAKS at n = {peak_n}  (reward {rew[peak_n]:.3f}), "
          f"QoS peaks at n = {qos_peak_n} ({100*qos[qos_peak_n]:.1f}%)")
    print(f"  operating point (≈#blocked) = {n_op:.1f}  →  DISTANCE op→peak = {peak_n - n_op:+.1f} users")
    print(f"     (>0 = room to add more before drop · ≈0 = knife-edge · <0 = already over-loaded)")
    if peak_n < K:
        print(f"  Q2 FREE overload: at n={K} (all free joined) reward {rew[K]:.3f} "
              f"(Δ vs peak {rew[K]-rew[peak_n]:+.3f}); blkInGrp-met "
              f"{100*rows[K][5]/max(rows[K][3],0.01):.0f}% ← dilution on blocked.")
    print("=" * 92)


if __name__ == '__main__':
    main()
