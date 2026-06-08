"""
probe_blocked_direct.py
-----------------------
User question: agent routes some BLOCKED users to direct group (Blk/K > IRS/K).
Why would agent sacrifice direct-path-zero users? Two competing hypotheses:

  H-irrational : agent miss the opportunity — those blocked users could have
                 received service via IRS. Just a routing bug.
  H-rational   : multi-user IRS phase saturation — adding more users to an IRS
                 group dilutes the shared phase. Sacrificing some blocked
                 outlier users PRESERVES phase quality for remaining IRS users.
                 This is a defensible trade-off at LOW phase quality.

Method: counterfactual SWAP, all-else-equal.
  For each state, identify blocked-direct users (Blk[k]=1 ∧ phi_pol[k]=0).
  For each such user u, swap to each IRS group m ∈ {1,..,M}, picking the best.
  Keep policy's phase_idx, w_p, w_c_vec, C_k unchanged (so we ISOLATE the
  assignment effect; power allocation may not re-adapt but the test still tells
  us if the swap is a clean win at fixed downstream).

Outputs
-------
  • baseline R_user broken down by (blocked × IRS-routed) 2×2
  • per-swap: ΔR_user_u (the swapped user), ΔR_tot (everyone), QoS impact
  • aggregate: fraction of swaps that are NET POSITIVE for R_tot — H-irrational
    threshold; if most swaps positive, agent over-assigns blocked users to direct.
  • stratified by # blocked-direct per state (clustering proxy: if many blocked
    routed to direct, geography may be against IRS for them)

Usage:
  python analysis/probe_blocked_direct.py --ckpt results/result_11/checkpoints/ep_00600
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_critic_ceiling import make_checkpoint_policy


def _R_user(env, assignment, phase_idx, w_p, w_c_vec, C_k, active_irs_ids, sigmas):
    """Per-user rate (private + Ck) averaged over noise draws."""
    phases_rad = env.phase_model.index_to_phase(phase_idx)
    Phi = env.phase_model.build_phi(phases_rad)
    K = len(env.channels['g_SU'])
    R_user = np.zeros(K)
    for s2 in sigmas:
        r = env.rate_computer.compute_sum_rate(
            assignment, Phi, env.channels, w_p, w_c_vec, C_k=C_k,
            active_irs_ids=active_irs_ids, sigma2=s2)
        R_user += np.asarray(r['R_private']) + np.asarray(r['C_k'])
    return R_user / len(sigmas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_00600')
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/blocked_direct_ep600.txt')
    args = ap.parse_args()

    cfg = make_config(); K, M = cfg.K, cfg.M; D_k = cfg.D_k_bps_hz
    print("=" * 78)
    print(f"  BLOCKED→DIRECT ANALYSIS  ·  ckpt={args.ckpt}")
    print(f"  K={K} M={M} D_k={D_k:.2f} · {args.episodes}ep×{args.steps}st · noise={args.noise_avg}")
    print("=" * 78)

    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(args.ckpt, cfg)

    # 2×2 bucket sums for baseline R_user breakdown
    sum_R = np.zeros((2, 2)); cnt = np.zeros((2, 2), dtype=int)  # [blk?][IRS-routed?]
    # swap experiment storage
    swap_dR_user  = []   # per-swap, ΔR_user for the swapped user
    swap_dR_tot   = []   # per-swap, ΔR_tot for everyone
    swap_dQoS     = []   # ΔQoS pts (1 user met/unmet ⇒ ±10/K)
    swap_blocked_count = []  # n_blocked in that state (for stratification)
    swap_best_m = []
    # baseline aggregates per state
    bd_count_per_state = []  # # of (blocked + direct) users per state

    t0 = time.time()
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 17 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            action = policy(env)
            env._apply_action(action)
            phi   = action['assignment']
            blk   = env.channels['su_blocked'].astype(int)
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_avg)]

            # baseline
            R_user0 = _R_user(env, phi, action['phase_idx'], action['w_p'],
                              action['w_c_vec'], action['C_k'], env.active_irs_ids, sigmas)
            on_irs = (phi > 0).astype(int)
            for k in range(K):
                sum_R[blk[k], on_irs[k]] += R_user0[k]
                cnt[blk[k], on_irs[k]]    += 1

            R_tot0 = float(R_user0.sum())
            met0   = (R_user0 >= D_k).astype(int)
            bd_users = np.where((blk == 1) & (phi == 0))[0]
            bd_count_per_state.append(int(len(bd_users)))

            # per blocked-direct user, try swap to each ALREADY-ACTIVE IRS group, pick best.
            # Restrict to groups in env.active_irs_ids so w_c_vec layout matches.
            already_active_irs = list(env.active_irs_ids)   # 1-based
            for u in bd_users:
                if not already_active_irs:
                    continue                                 # no IRS to swap to
                best_dR_tot = -np.inf
                best_dR_u   = 0.0
                best_dQ     = 0.0
                best_m      = 0
                for m in already_active_irs:
                    phi_swap = phi.copy(); phi_swap[u] = m
                    R_user1 = _R_user(env, phi_swap, action['phase_idx'], action['w_p'],
                                      action['w_c_vec'], action['C_k'],
                                      env.active_irs_ids, sigmas)
                    dR_tot  = float(R_user1.sum() - R_tot0)
                    dR_u    = float(R_user1[u] - R_user0[u])
                    met1    = (R_user1 >= D_k).astype(int)
                    dQ      = float(met1.sum() - met0.sum()) * 100.0 / K
                    if dR_tot > best_dR_tot:
                        best_dR_tot, best_dR_u, best_dQ, best_m = dR_tot, dR_u, dQ, m
                swap_dR_tot.append(best_dR_tot)
                swap_dR_user.append(best_dR_u)
                swap_dQoS.append(best_dQ)
                swap_blocked_count.append(int(blk.sum()))
                swap_best_m.append(best_m)

            # advance env
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
        if (ep + 1) % max(1, args.episodes // 5) == 0:
            print(f"  collected {ep+1}/{args.episodes} ep  ({time.time()-t0:.0f}s)")

    swap_dR_tot  = np.asarray(swap_dR_tot)
    swap_dR_user = np.asarray(swap_dR_user)
    swap_dQoS    = np.asarray(swap_dQoS)
    swap_blk_n   = np.asarray(swap_blocked_count)
    n_swaps = len(swap_dR_tot)

    # ── report ──
    L = []
    L.append("=" * 78)
    L.append(f"  BLOCKED→DIRECT  ·  ckpt={args.ckpt}")
    L.append(f"  total states sampled: {cnt.sum()//K} · total swap experiments: {n_swaps}")
    L.append("-" * 78)

    # baseline 2×2 breakdown
    mean_R = np.where(cnt > 0, sum_R / np.maximum(cnt, 1), 0.0)
    L.append("  BASELINE per-user R_user breakdown (mean over instances):")
    L.append(f"                       on DIRECT       on IRS")
    L.append(f"    NOT-blocked     R={mean_R[0,0]:.3f} (n={cnt[0,0]:5d})   "
             f"R={mean_R[0,1]:.3f} (n={cnt[0,1]:5d})")
    L.append(f"    BLOCKED         R={mean_R[1,0]:.3f} (n={cnt[1,0]:5d})   "
             f"R={mean_R[1,1]:.3f} (n={cnt[1,1]:5d})")
    L.append(f"    D_k threshold   = {D_k:.3f}")
    L.append("")
    # interpretation guide
    if mean_R[1, 0] < 0.01:
        L.append(f"  ⚠ blocked-direct users have R≈0 ({mean_R[1,0]:.3f}) → agent EFFECTIVELY DROPS them.")
    if mean_R[1, 1] > mean_R[1, 0] + 0.02:
        L.append(f"    blocked-IRS users get R={mean_R[1,1]:.3f} (>> blocked-direct {mean_R[1,0]:.3f})"
                 f" → routing blocked → IRS DOES give them rate.")
    L.append("")
    L.append(f"  Distribution of (blocked-direct) users per state:")
    bdc = np.asarray(bd_count_per_state)
    hist, _ = np.histogram(bdc, bins=np.arange(K + 2) - 0.5)
    for i, h in enumerate(hist):
        if h > 0:
            bar = "█" * int(40 * h / max(hist.sum(), 1))
            L.append(f"    {i} BD users : {h:4d} {bar}")
    L.append(f"    mean BD/state = {bdc.mean():.2f}")
    L.append("-" * 78)

    # swap experiment
    L.append(f"  COUNTERFACTUAL SWAP — for each blocked-direct user, swap to BEST IRS group")
    L.append(f"  (keep policy phase / w_p / w_c_vec / C_k unchanged → isolates ASSIGNMENT effect)")
    L.append("")
    pos = swap_dR_tot > 0
    big = swap_dR_tot > 0.05
    L.append(f"  per-swap ΔR_user (the swapped user): mean={swap_dR_user.mean():+.4f}  "
             f"p50={np.percentile(swap_dR_user,50):+.4f}  "
             f"p90={np.percentile(swap_dR_user,90):+.4f}  max={swap_dR_user.max():+.4f}")
    L.append(f"  per-swap ΔR_tot   (all users)       : mean={swap_dR_tot.mean():+.4f}  "
             f"p50={np.percentile(swap_dR_tot,50):+.4f}  "
             f"p90={np.percentile(swap_dR_tot,90):+.4f}  max={swap_dR_tot.max():+.4f}")
    L.append(f"  per-swap ΔQoS-pts                   : mean={swap_dQoS.mean():+.2f}")
    L.append("")
    L.append(f"  fraction of swaps with ΔR_tot > 0      : {100*pos.mean():.1f}%")
    L.append(f"  fraction of swaps with ΔR_tot > +0.05  : {100*big.mean():.1f}%")
    L.append("-" * 78)

    # stratify by total blocked count
    L.append("  swap outcome stratified by # blocked users in state:")
    q1, q2 = np.percentile(swap_blk_n, [33.3, 66.7])
    lo = swap_blk_n <= q1; mid = (swap_blk_n > q1) & (swap_blk_n <= q2); hi = swap_blk_n > q2
    for nm, m in [("low_blk", lo), ("mid_blk", mid), ("hi_blk", hi)]:
        if m.sum() >= 5:
            L.append(f"    {nm} (n_swaps={m.sum():4d}, blk≈{swap_blk_n[m].mean():.1f}): "
                     f"ΔR_tot mean={swap_dR_tot[m].mean():+.4f}  "
                     f"frac_pos={100*(swap_dR_tot[m]>0).mean():.1f}%")
    L.append("-" * 78)

    # verdict
    L.append("  VERDICT:")
    if pos.mean() > 0.7 and swap_dR_tot.mean() > 0.02:
        L.append(f"   • H-irrational SUPPORTED: {100*pos.mean():.0f}% of swaps are net-positive,"
                 f" mean ΔR_tot {swap_dR_tot.mean():+.3f}.")
        L.append(f"     Agent IS losing opportunity by routing blocked users to direct."
                 f" Lever = fix assignment head bias toward IRS for blocked users.")
    elif pos.mean() < 0.4 or swap_dR_tot.mean() < 0.01:
        L.append(f"   • H-rational SUPPORTED: only {100*pos.mean():.0f}% swaps positive,"
                 f" mean ΔR_tot {swap_dR_tot.mean():+.4f}.")
        L.append(f"     Multi-user phase compromise: dropping outlier blocked users PRESERVES"
                 f" phase quality for remaining IRS group. Defensible at current phase quality.")
        L.append(f"     Expected to resolve when PhaseMLP converges to higher alignment.")
    else:
        L.append(f"   • MIXED: {100*pos.mean():.0f}% positive · mean ΔR_tot {swap_dR_tot.mean():+.4f}."
                 f" Some opportunity but not a clear winning lever.")
    L.append("=" * 78)

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
