"""
probe_lambda_landscape.py
-------------------------
DISCRIMINATE the frozen-λ hypotheses (decides which fix to build) — OFFLINE, no training.

  H1  λ*(s) genuinely differs by state → conflict is fundamental → fix = λ(s)/MoE.
  H2  a good GLOBAL λ* exists but policy-gradient can't find it → fix = representation objective.
  H3  λ at init is already near-optimal / λ barely matters → frozen is fine → E-fix2 (do nothing).

Method: hold a trained checkpoint fixed (θ, AE, heads). For a batch of frozen channel
states, SCALE the encoding λ by a factor and recompute the GREEDY assignment → reward/QoS
(fixed downstream: aligned phase + equal power, so reward reflects ASSIGNMENT quality, the
only thing λ controls). λ only affects o_hat→assignment, so this isolates λ's effect cleanly.

  Test A (global sweep)  : perf(λ_factor) averaged over states. Peak location/height vs init.
  Test B (per-state gap) : mean[ max_f perf_s(f) ]  vs  max_f mean_s perf_s(f).
                           gap≈0 → a global factor suffices (H2/H3); gap≫0 → per-state needed (H1).
  Test C (structure)     : do per-state best-factors correlate with blockage? → MoE-by-blockage.

VERDICT:
  flat / peak@init             → H3  → E-fix2 (accept fixed λ, don't fix)
  peak far from init, gap≈0    → H2  → representation objective (safe, param-eff)
  gap large                    → H1  → λ(s) / MoE  (Test C: cluster → MoE-by-feature)

CPU-light (greedy forwards over a λ-grid). Works on v1 (generic+MLP) or arch-2 checkpoints
(uses only the actor's assignment path; downstream is a fixed default, so v1 PowerMLP not needed).

Usage:
  python analysis/probe_lambda_landscape.py --ckpt results/result_8/checkpoints/ep_00400 --states 60
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

FACTORS = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_8/checkpoints/ep_00400')
    ap.add_argument('--states', type=int, default=60, help='# frozen channel states to evaluate')
    ap.add_argument('--noise', type=int, default=4, help='noise draws averaged per (state,factor)')
    ap.add_argument('--seed', type=int, default=20260602)
    args = ap.parse_args()

    cfg = make_config(); K, M, N = cfg.K, cfg.M, cfg.N
    D_k = cfg.D_k_bps_hz; lamD = cfg.lambda_D; eps = getattr(cfg, 'epsilon_qp', 1e-3)
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.states + 5, reward_noise_avg=1)

    ckpt = args.ckpt
    if os.path.isdir(os.path.join(ckpt, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt, 'actor_config.json')):
        ckpt = os.path.join(ckpt, 'agents')
    from RL import QuantumActor
    actor = QuantumActor.from_dir(ckpt, seed=0); actor.n_shots = P.n_shots_train
    base_y, base_z = actor.lam_y.copy(), actor.lam_z.copy()
    rng = np.random.default_rng(args.seed)

    # ── collect frozen channel-state snapshots ──
    snaps = []
    obs = env.reset(seed=args.seed)
    for _ in range(args.states):
        demand  = np.full(K, D_k)
        blocked = env.channels['su_blocked'].astype(int)
        s_t = actor.extract_state(obs, demand, blocked)
        snaps.append({'s_t': s_t,
                      'ch': {k: (v.copy() if hasattr(v, 'copy') else v)
                             for k, v in env.channels.items()},
                      'blk': float(blocked.mean())})
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)
        obs = env._get_obs()

    zero_idx = np.zeros((M, N), dtype=int)
    Phi_aligned = env.phase_model.build_phi(env.phase_model.index_to_phase(zero_idx))

    def reward_qos(s_t, ch):
        """Greedy assignment from current actor.lam → reward & QoS on frozen state ch
        (aligned phase + equal power; averaged over noise draws)."""
        phi, _, _ = actor.forward(s_t, greedy=True)
        G = len({int(a) for a in phi if a > 0})
        env.channels = ch
        env._apply_action({'assignment': phi, 'phase_idx': zero_idx,
                            'C_k': np.zeros(K), 'w_p': np.ones(K),
                            'w_c_vec': np.ones(G + 1)})
        rw = qo = 0.0
        for _ in range(args.noise):
            s2 = env.channel_model.sample_noise_sigma2()
            r = env.rate_computer.compute_sum_rate(
                env.assignment, env.Phi, env.channels, env.w_p, env.w_c_vec,
                C_k=env.C_k, active_irs_ids=env.active_irs_ids, sigma2=s2)
            R_tot = np.asarray(r['R_private']) + np.asarray(r['C_k'])
            short = np.maximum(0.0, D_k - R_tot)
            rw += float(R_tot.sum() - lamD * np.sum((short / (D_k + eps)) ** 2))
            qo += float(np.mean(R_tot >= D_k))
        return rw / args.noise, qo / args.noise

    # ── evaluate perf[state, factor] ──
    nF = len(FACTORS)
    Rew = np.zeros((args.states, nF)); Qos = np.zeros((args.states, nF))
    for fi, f in enumerate(FACTORS):
        actor.lam_y = f * base_y; actor.lam_z = f * base_z
        for si, sn in enumerate(snaps):
            Rew[si, fi], Qos[si, fi] = reward_qos(sn['s_t'], sn['ch'])
    actor.lam_y, actor.lam_z = base_y, base_z

    init_fi = int(np.argmin(np.abs(FACTORS - 1.0)))   # factor=1.0 ≈ trained/init λ
    mean_rew = Rew.mean(axis=0)                         # (nF,) Test A curve
    best_global_fi = int(np.argmax(mean_rew))
    best_global = mean_rew[best_global_fi]
    init_perf = mean_rew[init_fi]
    # Test B: per-state oracle
    per_state_best = Rew.max(axis=1).mean()
    gap = per_state_best - best_global
    # Test C: per-state best factor vs blockage
    best_f_per_state = FACTORS[Rew.argmax(axis=1)]
    blk = np.array([s['blk'] for s in snaps])
    if best_f_per_state.std() > 1e-9 and blk.std() > 1e-9:
        corr = float(np.corrcoef(best_f_per_state, blk)[0, 1]); corr_ok = True
    else:
        corr = 0.0; corr_ok = False     # degenerate (flat landscape or constant blockage)
    spread = float(best_f_per_state.std())

    print("=" * 74)
    print(f"  λ-LANDSCAPE DISCRIMINATOR · ckpt={args.ckpt}")
    print(f"  {args.states} states × {nF} factors × {args.noise} noise · K={K} M={M} N={N}")
    print("=" * 74)
    print("  Test A — global λ-factor sweep (mean reward over states):")
    for f, r in zip(FACTORS, mean_rew):
        mark = "  ← init(×1)" if abs(f - 1.0) < 1e-6 else ("  ← best-global" if r == best_global else "")
        print(f"     ×{f:<4} : reward {r:+8.2f}{mark}")
    print(f"  init(×1) reward = {init_perf:+.2f} | best-global ×{FACTORS[best_global_fi]} = {best_global:+.2f}"
          f" | Δ(best−init) = {best_global - init_perf:+.2f}")
    print("-" * 74)
    print(f"  Test B — per-state-oracle vs best-global:")
    print(f"     mean per-state-oracle reward = {per_state_best:+.2f}")
    print(f"     best-global reward           = {best_global:+.2f}")
    print(f"     GAP (oracle − global)        = {gap:+.2f}   ⭐ (large → H1)")
    print(f"  Test C — per-state best-factor: spread(std)={spread:.2f}  "
          f"corr(best_f, blockage)={corr:+.2f}" + ("" if corr_ok else " (N/A: flat or const-blockage)"))
    print("-" * 74)
    print("  ⚠ CAVEAT: a FLAT landscape on an MLP-head checkpoint can mean the high-capacity")
    print("    post-NN ABSORBS λ-variation (so λ looks irrelevant). RE-RUN on an arch-2 (SoftmaxPQC")
    print("    LINEAR head) checkpoint — if still flat there, frozen-λ is genuinely harmless (H3).")
    rel = lambda x: x / (abs(init_perf) + 1e-6)
    improve = best_global - init_perf
    print("  VERDICT:")
    if rel(gap) > 0.15 and gap > abs(improve):
        print(f"     → H1: per-state-oracle ≫ best-global (gap {gap:+.1f}) → λ*(s) conflict is REAL.")
        print(f"       FIX = state-conditioned λ(s) / MoE." +
              (f" Test C corr={corr:+.2f} → cluster by blockage → MoE-by-feature."
               if abs(corr) > 0.3 else " Test C no clear cluster → smooth λ(s)."))
    elif rel(improve) > 0.10 and best_global_fi != init_fi:
        print(f"     → H2: a better GLOBAL λ (×{FACTORS[best_global_fi]}) exists (Δ{improve:+.1f}) but training")
        print(f"       didn't find it, and gap≈0 ({gap:+.1f}) → global suffices.")
        print(f"       FIX = representation objective (coherent signal to reach λ*). Safe, param-eff.")
    else:
        print(f"     → H3: λ at init already near-optimal (Δ{improve:+.1f}) and gap≈0 ({gap:+.1f}).")
        print(f"       λ-frozen is HARMLESS → E-fix2 (accept fixed λ). DON'T build a λ-fix.")
    print("=" * 74)


if __name__ == '__main__':
    main()
