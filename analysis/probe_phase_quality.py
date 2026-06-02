"""
probe_phase_quality.py
----------------------
Does the LIVE PhaseMLP actually ALIGN the IRS channel, or just output low-entropy
but POOR phases?  (priority-#2 check: "PhaseMLP learning to optimize IRS" — phase
entropy dropping is NECESSARY but NOT SUFFICIENT; entropy can converge to a bad phase.)

Counterfactual / physical-layer probe (no ablation training). For the trained policy,
take the GREEDY PhaseMLP phase per active IRS and measure channel coherence:

    coherence  q_m = |Σ_n exp(j·φ_n^live)| / N        ∈ [~1/√N (random), 1.0 (optimal)]
    alignment  a_m = (q_m − 1/√N) / (1 − 1/√N)        ∈ [0 (random), 1 (optimal)]

(In this channel model |h_irs| ∝ |Σ_n φ_n|, max = N when all elements share one level;
random phases give ≈√N — see probe_irs_vs_direct.) Also: with the LIVE phase, does the
IRS link beat the direct link for users the agent ROUTED to that IRS?

  q → 1.0 / a → 1   ⇒ PhaseMLP IS optimizing IRS (priority-#2 solved).
  q ≈ 1/√N / a ≈ 0  ⇒ phases ≈ random → PhaseMLP IDLE regardless of entropy [F].

CPU-light (greedy forward on GPU + numpy); safe alongside training.

Usage:
  python analysis/probe_phase_quality.py --ckpt results/result_N/checkpoints/ep_00400 --episodes 15
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_8/checkpoints/ep_00400')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260602)
    ap.add_argument('--sampled', action='store_true',
                    help='use sampled phase instead of greedy (default greedy)')
    args = ap.parse_args()

    cfg = make_config(); K = cfg.K; N = cfg.N
    greedy = not args.sampled
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps, reward_noise_avg=1)

    ckpt = args.ckpt
    if os.path.isdir(os.path.join(ckpt, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt, 'actor_config.json')):
        ckpt = os.path.join(ckpt, 'agents')
    from RL import QuantumActor, PhaseMLP
    from train import _build_phase_state, _get_active_irs
    actor     = QuantumActor.from_dir(ckpt, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt, seed=0)
    actor.n_shots = P.n_shots_train

    rand_floor = 1.0 / np.sqrt(N)          # expected coherence of random phases
    q_all, irs_win, irs_tot = [], 0, 0

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            obs     = env._get_obs()
            demand  = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t, greedy=greedy)
            active_irs = _get_active_irs(phi)                       # 0-based
            if len(active_irs) == 0:
                _advance(env); continue
            s_phase = _build_phase_state(env.channels, phi, cfg, info['z_t'])
            phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=greedy)

            ch = env.channels
            for m in active_irs:                                    # 0-based IRS index
                angles = env.phase_model.index_to_phase(phase_idx[m])   # (N,) rad
                q = float(np.abs(np.exp(1j * np.asarray(angles)).sum()) / N)
                q_all.append(q)
                # live IRS gain vs direct for users routed to this IRS (gid = m+1)
                routed = np.where(phi == (m + 1))[0]
                if routed.size:
                    coeff = ch['beta'][m] * np.abs(ch['g_SR_hat'][m]) * (q * N)
                    g_irs = (coeff * np.abs(ch['g_RU_hat'][m, routed])) ** 2
                    g_dir = np.abs(ch['g_SU_hat'][routed]) ** 2
                    irs_win += int(np.sum(g_irs > g_dir)); irs_tot += routed.size
            _advance(env)

    q_all = np.asarray(q_all)
    if q_all.size == 0:
        print("No active IRS in any step — agent routed everyone to direct. PhaseMLP unused."); return
    a_all = (q_all - rand_floor) / (1.0 - rand_floor)              # alignment ratio
    qm, am = float(q_all.mean()), float(a_all.mean())

    print("=" * 74)
    print(f"  PHASE-QUALITY (live PhaseMLP) · ckpt={args.ckpt}")
    print(f"  K={K} M={cfg.M} N={N} · {args.episodes}ep×{args.steps}step · "
          f"{'greedy' if greedy else 'sampled'} · {q_all.size} active-IRS instances")
    print("=" * 74)
    print(f"  coherence q = |Σφ|/N : mean={qm:.3f}  (random≈{rand_floor:.3f}, optimal=1.000)")
    print(f"  alignment a (0=rand,1=opt): mean={am*100:.1f}%  "
          f"[p25 {np.percentile(a_all,25)*100:.0f}%  p75 {np.percentile(a_all,75)*100:.0f}%]")
    if irs_tot:
        print(f"  LIVE IRS beats direct for routed users: {100.0*irs_win/irs_tot:.1f}%  "
              f"({irs_win}/{irs_tot})")
    print("-" * 74)
    if am >= 0.6:
        print(f"  ✅ PhaseMLP OPTIMIZING IRS (alignment {am*100:.0f}% → coherent). Priority-#2 progressing.")
    elif am >= 0.25:
        print(f"  🟡 PARTIAL alignment ({am*100:.0f}%): phase learning but not converged → sharpen/more training.")
    else:
        print(f"  ❌ PhaseMLP ≈ RANDOM phase (alignment {am*100:.0f}%) → IDLE [F] regardless of entropy.")
        print(f"     Low phase entropy alone is NOT enough — confirm this before declaring #2 solved.")
    print("=" * 74)


def _advance(env):
    """Advance mobility one step without re-applying an action (mirror probe_power_qos)."""
    env.user_pos = env._walk_users(env.user_pos)
    env.channels = env.channel_model.update_user_channels(
        env.user_pos, env.irs_pos, env.channels)


if __name__ == '__main__':
    main()
