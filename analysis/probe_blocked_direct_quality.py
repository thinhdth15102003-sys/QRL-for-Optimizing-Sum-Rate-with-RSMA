"""
probe_blocked_direct_quality.py
-------------------------------
For BLOCKED users that the agent routed to DIRECT, measure whether routing them
to IRS (using the LIVE PhaseMLP phase for the active IRS) would be physically
BETTER or WORSE than the direct link.

Distinguishes:
  IRS_live > direct  → agent UNDER-routing (the live IRS path WOULD help this
                       user, agent's choice to send to direct is sub-optimal).
  IRS_live ≤ direct  → agent RATIONAL (live IRS phase isn't good enough for this
                       user; direct is genuinely better given current PhaseMLP).

Per-user comparison hold POWER + Ck fixed, swap only the link → clean isolation
of routing vs phase quality.

Channel formula (CSI/rate.py):
  h_irs[m,k] = β[m] · conj(g_SR[m]) · (Σ_n exp(j·φ_n)) · g_RU[m,k]
  → |h_irs|² depends on coherence sum and per-user g_RU magnitude.

Usage:
  python analysis/probe_blocked_direct_quality.py \\
    --ckpt results/result_12/checkpoints/ep_00100 \\
    --episodes 15 --steps 200
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import numpy as np
import params as P
from params import make_config
from CSI.env import ISTNEnv


def _advance(env):
    env.user_pos = env._walk_users(env.user_pos)
    env.channels = env.channel_model.update_user_channels(
        env.user_pos, env.irs_pos, env.channels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_12/checkpoints/ep_00100')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260610)
    ap.add_argument('--sampled', action='store_true')
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=None,
                    help='Override R_LoS_km (default: params.py)')
    args = ap.parse_args()

    # auto-detect K/M from ckpt actor_config (mirror probe_assignment_sensitivity)
    raw_ckpt = args.ckpt
    cfgread = raw_ckpt
    if (os.path.isdir(os.path.join(raw_ckpt, 'agents'))
            and not os.path.isfile(os.path.join(raw_ckpt, 'actor_config.json'))):
        cfgread = os.path.join(raw_ckpt, 'agents')
    with open(os.path.join(cfgread, 'actor_config.json')) as f:
        ac = json.load(f)
    K_ck = int(ac.get('K', P.K)); M_ck = int(ac.get('B', P.M))
    P_S = {5: 50.0, 10: 70.0, 20: 100.0}.get(K_ck, P.P_S_dBm)
    rlos_use = args.rlos if args.rlos is not None else P.R_LoS_km
    if K_ck != P.K or M_ck != P.M or rlos_use != P.R_LoS_km:
        print(f"  auto-detect ckpt K={K_ck} M={M_ck} R_LoS={rlos_use} (params has K={P.K} M={P.M} R_LoS={P.R_LoS_km}) -> override cfg")
        cfg = make_config(K=K_ck, M=M_ck, P_S_dBm=P_S, R_LoS_km=rlos_use)
    else:
        cfg = make_config()
    K, M, N = cfg.K, cfg.M, cfg.N
    greedy = not args.sampled
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps, reward_noise_avg=1)

    ckpt = args.ckpt
    if os.path.isdir(os.path.join(ckpt, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt, 'actor_config.json')):
        ckpt = os.path.join(ckpt, 'agents')

    from RL import QuantumActor, PhaseMLP
    from train import _build_phase_state, _get_active_irs
    actor = QuantumActor.from_dir(ckpt, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt, seed=0)
    actor.n_shots = P.n_shots_train

    # Per-user gain comparisons
    # For each (step, user) where blocked && routed-to-direct: record (|h_dir|², |h_irs_live|²)
    gain_dir_list = []      # |g_SU|² for blocked-direct users
    gain_irs_list = []      # |h_irs_live|² for same users (counterfactual IRS via live PhaseMLP)
    n_blocked_total = 0
    n_blocked_direct = 0
    n_blocked_irs = 0

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            obs = env._get_obs()
            demand = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)  # (K,) {0,1}
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t, greedy=greedy)
            active_irs = _get_active_irs(phi)

            ch = env.channels
            n_blocked_total += int(blocked.sum())
            # blocked + routed to direct (phi==0)
            blocked_direct_mask = (blocked == 1) & (phi == 0)
            blocked_irs_mask = (blocked == 1) & (phi != 0)
            n_blocked_direct += int(blocked_direct_mask.sum())
            n_blocked_irs += int(blocked_irs_mask.sum())

            if blocked_direct_mask.sum() == 0 or active_irs.size == 0:
                _advance(env); continue

            # Live PhaseMLP phase for active IRS
            s_phase = _build_phase_state(env.channels, phi, cfg, info['z_t'])
            phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=greedy)

            # Compute |h_irs|² for each blocked-direct user via the BEST active IRS
            # using LIVE phase. For Case 1 (M=1) only 1 IRS choice.
            blocked_direct_idx = np.where(blocked_direct_mask)[0]
            for k in blocked_direct_idx:
                g_dir_k_sq = float(np.abs(ch['g_SU_hat'][k]) ** 2)

                # Best |h_irs|² over active IRS m
                best_irs_sq = 0.0
                for m in active_irs:
                    angles = env.phase_model.index_to_phase(phase_idx[m])  # (N,)
                    eff_phi = np.exp(1j * angles).sum()  # complex Σ exp
                    # |h_irs| = |β·g_SR·eff_phi·g_RU[k]| — magnitudes
                    h_irs_mag = (ch['beta'][m] *
                                 np.abs(ch['g_SR_hat'][m]) *
                                 np.abs(eff_phi) *
                                 np.abs(ch['g_RU_hat'][m, k]))
                    h_irs_sq = float(h_irs_mag ** 2)
                    if h_irs_sq > best_irs_sq:
                        best_irs_sq = h_irs_sq

                gain_dir_list.append(g_dir_k_sq)
                gain_irs_list.append(best_irs_sq)

            _advance(env)

    gd = np.asarray(gain_dir_list)
    gi = np.asarray(gain_irs_list)
    n_pairs = len(gd)

    print("=" * 80)
    print(f"  PROBE BLOCKED-DIRECT QUALITY  ·  K={K} M={M} N={N} R_LoS={cfg.R_LoS_km}km")
    print(f"  ckpt={args.ckpt}")
    print(f"  {args.episodes}ep × {args.steps}step  · greedy={greedy}")
    print("=" * 80)
    print(f"  Blocked users total:                    {n_blocked_total}")
    print(f"  Blocked routed to IRS by agent:         {n_blocked_irs} ({100.0*n_blocked_irs/max(1,n_blocked_total):.1f}%)")
    print(f"  Blocked routed to DIRECT by agent:      {n_blocked_direct} ({100.0*n_blocked_direct/max(1,n_blocked_total):.1f}%)  ← analyzed below")
    print("-" * 80)
    if n_pairs == 0:
        print("  No blocked-direct users with active IRS — agent routed all blocked to IRS, OR no active IRS.")
        return

    # Per-user comparison
    ratio_db = 10.0 * np.log10((gi + 1e-30) / (gd + 1e-30))
    n_irs_wins = int(np.sum(gi > gd))
    pct_irs_wins = 100.0 * n_irs_wins / n_pairs

    print(f"  Pairs analyzed (blocked-direct user-step instances): {n_pairs}")
    print()
    print(f"  IRS-live > direct in: {n_irs_wins} / {n_pairs} = {pct_irs_wins:5.1f}% of cases")
    print()
    print(f"  |h_irs_live|² / |g_SU|²  (gain ratio, dB):")
    print(f"    median = {np.median(ratio_db):+6.2f} dB")
    print(f"    p25    = {np.percentile(ratio_db, 25):+6.2f} dB")
    print(f"    p75    = {np.percentile(ratio_db, 75):+6.2f} dB")
    print(f"    mean   = {np.mean(ratio_db):+6.2f} dB")
    print("-" * 80)

    # VERDICT
    print("  VERDICT")
    print("-" * 80)
    if pct_irs_wins >= 70:
        print(f"  ❌ UNDER-ROUTING CONFIRMED ({pct_irs_wins:.0f}% blocked-direct users WOULD benefit from IRS):")
        print(f"     LIVE PhaseMLP phase already strong enough to beat direct for majority of these users.")
        print(f"     Agent's choice to route them DIRECT is reward-IRRATIONAL given current PhaseMLP.")
        print(f"     → Lever: assignment-credit (q-head), not phase (already OK).")
    elif pct_irs_wins >= 40:
        print(f"  🟡 MIXED ({pct_irs_wins:.0f}% blocked-direct WOULD benefit from IRS):")
        print(f"     Some agent calls under-routing, some rational (phase weak for those users).")
        print(f"     → Both assignment AND phase quality contribute to gap.")
    else:
        print(f"  ✓ AGENT RATIONAL ({pct_irs_wins:.0f}% blocked-direct WOULD benefit from IRS):")
        print(f"     LIVE PhaseMLP phase NOT good enough → direct genuinely better for most of these users.")
        print(f"     Agent's choice is consistent with current phase quality. Not under-routing.")
        print(f"     → Lever: improve PhaseMLP first, not assignment.")
    print("=" * 80)


if __name__ == '__main__':
    main()
