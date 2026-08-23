"""
probe_irs_group_capacity.py
---------------------------
ORACLE users-per-IRS capacity sweep — answers: "as more users share ONE IRS,
at about how many does the whole group's QoS/per-user-rate start to degrade?"

This isolates the STRUCTURAL multi-user-IRS-sharing limit from the agent's
learning quality by using ORACLE sub-actors:
  • phase  : per-element oracle (analysis/phase_oracle.oracle_phase_idx) — each
             element co-phased individually. ⚠ post per-element refactor this can
             only fully align ONE user; with n>1 sharing the IRS the oracle trades
             users off, so part of the degradation below is now PHASE CONFLICT
             (impossible under the old scalar-g_RU model, where |Σφ|=N for all).
  • routing: oracle "best-n" — the n users with the highest IRS-over-direct gain
             margin are the ones placed on the IRS (what a perfect Q-head would do).
  • power  : reference equal split (AllIRSPolicy convention) — 80% private / K,
             20% common / (G+1).  C_k = group common rate / n_members (equal).

The only thing varied is n = #users forced onto a single IRS (m=1); the rest go
direct.  The IRS-group's per-user metrics are reported vs n, so the knee where the
group degrades is visible.  CPU-only channel math; safe alongside GPU training.

Mechanism (CSI/rate.py): degradation as n grows is the RSMA COMMON stream —
  C_k = R_c_group / n        (the group common rate is split ÷ n members)
  R_c_group = log2(1 + min_k SINR_c)   (limited by the WORST user in the group)
Private rate is ~n-independent under equal split (interference-limited), so the
group "capacity" is set by how long C_k + R_private stays above D_k as n rises.

Usage:
  python analysis/probe_irs_group_capacity.py --K 10 --M 2 --P 50
  python analysis/probe_irs_group_capacity.py --K 5  --M 1 --P 50 --select random
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
from analysis.phase_oracle import oracle_phase_idx, irs_optimal_gain_mag


def _irs_margin(ch, cfg):
    """Per-user IRS(m=1)-optimal gain / direct gain, on ESTIMATED channels
    (what an oracle Q-head would route on). Returns (K,) ratio (linear)."""
    N = cfg.N
    g_dir = np.abs(ch['g_SU_hat']) ** 2                              # (K,)
    g_irs = irs_optimal_gain_mag(ch)[0] ** 2                         # (K,) best IRS_1
    return g_irs / (g_dir + 1e-30)


def _eval_group(ch, cfg, rate, n, select, rng):
    """Force `n` users onto IRS m=1 (oracle phase + ref power), rest direct.
    Returns (qos_met_frac, mean_Rtot, group_sumrate, mean_Rpriv, mean_Ck) for
    the n IRS-group users — or None if n>K."""
    K = cfg.K
    if n > K:
        return None
    margin = _irs_margin(ch, cfg)
    order = np.argsort(-margin) if select == 'best' else rng.permutation(K)
    irs_users = np.sort(order[:n])

    assignment = np.zeros(K, dtype=int)
    assignment[irs_users] = 1                       # IRS #1 (1-based)

    # oracle phase on ESTIMATED channels (self-consistent with rate computer)
    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    phase_idx = oracle_phase_idx(est, assignment, cfg)               # (M,N)
    phases = cfg.phase_levels[phase_idx]
    Phi = np.zeros((cfg.M, cfg.N, cfg.N), dtype=complex)
    di = np.arange(cfg.N)
    Phi[:, di, di] = np.exp(1j * phases)

    # reference power: 80% private / K, 20% common / (G+1), G=1 active IRS
    w_p = np.full(K, cfg.P_S * 0.8 / K)
    w_c_vec = np.full(2, cfg.P_S * 0.2 / 2)

    out = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                w_p=w_p, w_c_vec=w_c_vec,
                                active_irs_ids=[1], sigma2=cfg.sigma2)
    Rpriv = out['R_private'][irs_users]
    Ck = out['C_k'][irs_users]
    Rtot = Rpriv + Ck
    Dk = cfg.D_k_bps_hz
    return (float(np.mean(Rtot >= Dk)), float(np.mean(Rtot)),
            float(np.sum(Rtot)), float(np.mean(Rpriv)), float(np.mean(Ck)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=2)
    ap.add_argument('--N', type=int, default=None)
    ap.add_argument('--P', type=float, default=50.0, help='P_S_dBm override')
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--episodes', type=int, default=25)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--select', choices=['best', 'random'], default='best',
                    help="'best' = oracle routing (top-n by IRS margin); "
                         "'random' = random n users")
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ov = dict(K=args.K, M=args.M, P_S_dBm=args.P, R_LoS_km=args.rlos)
    if args.N is not None:
        ov['N'] = args.N
    cfg = make_config(**ov)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps,
                  reward_noise_avg=1)
    rng = np.random.default_rng(args.seed)

    # collect channel snapshots
    snaps = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            snaps.append({k: (v.copy() if hasattr(v, 'copy') else v)
                          for k, v in env.channels.items()})
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    ns = list(range(1, args.K + 1))
    agg = {n: [] for n in ns}
    for ch in snaps:
        for n in ns:
            r = _eval_group(ch, cfg, rate, n, args.select, rng)
            if r is not None:
                agg[n].append(r)

    print("=" * 80)
    print(f"  ORACLE users-per-IRS CAPACITY  ·  K={cfg.K} M={cfg.M} N={cfg.N} "
          f"P_S={cfg.P_S_dBm}dBm ({cfg.P_S:.1f} W)  R_LoS={cfg.R_LoS_km}")
    print(f"  D_k={cfg.D_k_bps_hz}  ·  select={args.select}  ·  "
          f"{len(snaps)} snapshots  ·  oracle phase + ref power")
    print("=" * 80)
    print(f"  {'n_IRS':>5} | {'QoS-met%':>9} | {'mean R/user':>12} | "
          f"{'group ΣR':>9} | {'R_priv':>7} | {'C_k':>6}")
    print("  " + "-" * 74)
    qos_by_n, peruser_by_n = {}, {}
    for n in ns:
        a = np.array(agg[n])                # (S, 5)
        qm, ru, gs, rp, ck = a.mean(axis=0)
        qos_by_n[n] = qm * 100
        peruser_by_n[n] = ru
        print(f"  {n:>5} | {qm*100:>8.1f}% | {ru:>12.4f} | {gs:>9.3f} | "
              f"{rp:>7.4f} | {ck:>6.4f}")
    print("=" * 80)

    # knee detection
    Dk = cfg.D_k_bps_hz
    knee_qos = next((n for n in ns if qos_by_n[n] < 95.0), None)
    knee_rate = next((n for n in ns if peruser_by_n[n] < Dk), None)
    base = peruser_by_n[1]
    knee_half = next((n for n in ns if peruser_by_n[n] < 0.5 * base), None)
    print(f"  KNEE (group degradation):")
    print(f"    • QoS-met drops below 95%   at n = {knee_qos if knee_qos else '> '+str(args.K)} "
          f"users on the IRS")
    print(f"    • mean per-user R_tot < D_k  at n = {knee_rate if knee_rate else '> '+str(args.K)}")
    print(f"    • per-user rate halved (vs n=1) at n = {knee_half if knee_half else '> '+str(args.K)}")
    print("=" * 80)


if __name__ == '__main__':
    main()
