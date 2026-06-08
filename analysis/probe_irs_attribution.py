"""
probe_irs_attribution.py
------------------------
Confirm/refute the "agent bypasses IRS" hypothesis via 3 counterfactual tests on
a fixed checkpoint. Designed to discriminate WHERE the bottleneck lives:

  Test A — IRS-disabled comparison (most decisive)
    Hold policy fixed. Evaluate same action under (i) normal channels, (ii) IRS
    reflection zeroed (g_RU_hat → 0 → h_irs = 0 → routed users get NO IRS path).
    ΔR_tot tiny (<5%) → IRS already de facto bypassed → agent doesn't benefit
                         from IRS → [J] partial confirmed.
    ΔR_tot large (>15%) → IRS contributing materially → agent IS using it, just
                          not fully optimized.

  Test B — Assignment-shape variants
    For each state, swap policy's assignment with:
      • DIRECT-ALL : phi=[0]*K (no IRS use)
      • IRS-BALANCED: phi=[k%M + 1]*K (force every user to an IRS)
    Re-run PowerMLP + CkMLP with the new assignment, KEEP policy's phase.
    Compare R_tot to policy baseline.
      IRS-BALANCED > policy → assignment head is over-avoiding IRS (bias).
      IRS-BALANCED < policy → phase quality is the real cap (not assignment).

  Test C — Oracle phase gain
    Hold policy's assignment/power/Ck. Replace phase with the best uniform-level-
    per-IRS phase (M IRS × 4 levels = 4^M combinations; this is the analytic
    optimum since the model uses Σ_n φ_n — all elements at same level maximises
    |Σ| = N, and the M angle choices optimise per-IRS rotation).
      ΔR_tot large → phase IS the bottleneck → PhaseMLP under-converged.
      ΔR_tot small → phase already near optimum → bottleneck elsewhere.

Together: Test B and C decompose the "less IRS = better" pattern into
(assignment-bias vs phase-quality) attribution.

Usage:
  python analysis/probe_irs_attribution.py --ckpt results/result_11/checkpoints/ep_00600
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import itertools
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_critic_ceiling import make_checkpoint_policy


def _eval_action(env, assignment, phase_idx, w_p, w_c_vec, C_k,
                 active_irs_ids, sigmas, D_k, channels_override=None):
    """Compute (R_tot, qp, QoS_%) for a full action, averaged over noise draws.
    channels_override: optional dict of keys to TEMPORARILY override in env.channels."""
    backup = {}
    if channels_override:
        for k, v in channels_override.items():
            backup[k] = env.channels[k]
            env.channels[k] = v
    phases_rad = env.phase_model.index_to_phase(phase_idx)
    Phi = env.phase_model.build_phi(phases_rad)
    rc = env.rate_computer
    R_user = np.zeros(len(env.channels['g_SU']))
    for s2 in sigmas:
        r = rc.compute_sum_rate(assignment, Phi, env.channels, w_p, w_c_vec,
                                C_k=C_k, active_irs_ids=active_irs_ids, sigma2=s2)
        R_user += np.asarray(r['R_private']) + np.asarray(r['C_k'])
    R_user /= len(sigmas)
    R_tot = float(R_user.sum())
    qp    = float(np.maximum(0.0, D_k - R_user).sum())
    QoS   = float(100.0 * (R_user >= D_k).mean())
    if channels_override:
        for k, v in backup.items():
            env.channels[k] = v
    return R_tot, qp, QoS, R_user


def _rerun_downstream(env, policy_internals, new_phi, cfg, z_t):
    """Re-run PhaseMLP, PowerMLP, CkMLP given a NEW assignment (and policy's z_t).
    Returns (phase_idx, w_p, w_c_vec, C_k, active_irs_ids).
    Used by Test B to evaluate "what if assignment changed; downstream adapts"."""
    from train import _build_phase_state, _build_ck_state, _get_active_irs
    phase_net, power_net, ck_net = policy_internals
    active_irs     = _get_active_irs(new_phi)               # 0-based
    active_irs_ids = [int(a) + 1 for a in active_irs]
    s_phase = _build_phase_state(env.channels, new_phi, cfg, z_t)
    phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
    phases_rad   = env.phase_model.index_to_phase(phase_idx)
    proposed_Phi = env.phase_model.build_phi(phases_rad)
    h_eff   = env.rate_computer.effective_channels_all(new_phi, proposed_Phi, env.channels)
    s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
    w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)
    partial = env.rate_computer.compute_rates_partial(
        new_phi, proposed_Phi, env.channels, w_p, w_c_vec,
        active_irs_ids=active_irs_ids)
    s_ck = _build_ck_state(np.full(cfg.K, cfg.D_k_bps_hz),
                           partial['R_private'], partial['R_c_group'], new_phi, cfg)
    C_k, _, _ = ck_net.forward(s_ck, new_phi, partial['R_c_group'])
    return phase_idx, w_p, w_c_vec, C_k, active_irs_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_00600')
    ap.add_argument('--episodes', type=int, default=10)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/irs_attribution_ep600.txt')
    args = ap.parse_args()

    cfg = make_config(); K, M, N = cfg.K, cfg.M, cfg.N; D_k = cfg.D_k_bps_hz
    print("=" * 78)
    print(f"  IRS-ATTRIBUTION  (Tests A+B+C)  ·  ckpt={args.ckpt}")
    print(f"  K={K} M={M} N={N} D_k={D_k:.2f} · {args.episodes}ep × {args.steps}st · noise={args.noise_avg}")
    print("=" * 78)

    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(args.ckpt, cfg)
    # Reach the underlying nets for downstream re-runs (Test B).
    # Re-import to grab the modules from the loaded ckpt (already cached).
    ckpt_dir = args.ckpt
    if os.path.isdir(os.path.join(ckpt_dir, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt_dir, 'actor_config.json')):
        ckpt_dir = os.path.join(ckpt_dir, 'agents')
    from RL import QuantumActor, PhaseMLP, PowerMLP, CkMLP
    actor    = QuantumActor.from_dir(ckpt_dir, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt_dir, seed=0)
    power_net = PowerMLP.from_dir(ckpt_dir, seed=0)
    ck_net    = CkMLP.from_dir(ckpt_dir, seed=0)
    actor.n_shots = getattr(P, 'n_shots_train', 1500)
    internals = (phase_net, power_net, ck_net)

    # storage
    Rpol, qppol, Qpol = [], [], []   # baseline
    RIRSoff, qIRSoff, QIRSoff = [], [], []  # Test A
    Rdir, Qdir = [], []                     # Test B direct-all
    Rirs, Qirs = [], []                     # Test B irs-balanced
    Rorc, Qorc = [], []                     # Test C oracle phase
    pol_IRS_share = []                      # baseline % users on IRS
    blocked_n = []
    phase_alignment_pol, phase_alignment_orc = [], []

    t0 = time.time()
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 13 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            # 1. policy forward (computes assignment + phase + power + Ck + caches z_t internally)
            obs = env._get_obs()
            demand = np.full(K, D_k)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi_pol, _, info = actor.forward(s_t)
            z_t = info['z_t']
            from train import _get_active_irs
            active_pol = _get_active_irs(phi_pol)
            active_pol_ids = [int(a) + 1 for a in active_pol]
            phase_pol, w_p_pol, w_c_pol, C_k_pol, _ = _rerun_downstream(
                env, internals, phi_pol, cfg, z_t)

            # Shared noise draws (fair across all variants)
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_avg)]

            # ── BASELINE (policy unchanged) ──
            R0, qp0, Q0, _ = _eval_action(env, phi_pol, phase_pol, w_p_pol, w_c_pol,
                                           C_k_pol, active_pol_ids, sigmas, D_k)
            Rpol.append(R0); qppol.append(qp0); Qpol.append(Q0)
            pol_IRS_share.append(float((phi_pol > 0).mean()))
            blocked_n.append(int(blocked.sum()))

            # ── Test A: IRS-disabled (g_RU_hat → 0) ──
            zero_g_ru = np.zeros_like(env.channels['g_RU_hat'])
            R_A, qp_A, Q_A, _ = _eval_action(env, phi_pol, phase_pol, w_p_pol, w_c_pol,
                                              C_k_pol, active_pol_ids, sigmas, D_k,
                                              channels_override={'g_RU_hat': zero_g_ru})
            RIRSoff.append(R_A); qIRSoff.append(qp_A); QIRSoff.append(Q_A)

            # ── Test B-1: DIRECT-ALL ──
            phi_d = np.zeros(K, dtype=int)
            try:
                phase_d, w_p_d, w_c_d, C_k_d, ids_d = _rerun_downstream(
                    env, internals, phi_d, cfg, z_t)
                R_d, _, Q_d, _ = _eval_action(env, phi_d, phase_d, w_p_d, w_c_d, C_k_d,
                                              ids_d, sigmas, D_k)
            except Exception:
                R_d, Q_d = R0, Q0     # if no-active-IRS path fails, fall back
            Rdir.append(R_d); Qdir.append(Q_d)

            # ── Test B-2: IRS-BALANCED (round-robin) ──
            phi_i = np.array([(k % M) + 1 for k in range(K)], dtype=int)
            try:
                phase_i, w_p_i, w_c_i, C_k_i, ids_i = _rerun_downstream(
                    env, internals, phi_i, cfg, z_t)
                R_i, _, Q_i, _ = _eval_action(env, phi_i, phase_i, w_p_i, w_c_i, C_k_i,
                                              ids_i, sigmas, D_k)
            except Exception:
                R_i, Q_i = R0, Q0
            Rirs.append(R_i); Qirs.append(Q_i)

            # ── Test C: ORACLE PHASE (uniform-level per IRS, brute-force 4^M) ──
            # Search over (level_0, ..., level_{M-1}) ∈ {0,1,2,3}^M, pick max R_tot
            # with policy's assignment/power/Ck. Σ_n φ_n is maximised at N for any uniform
            # level (just different rotation); the rotation choice matters per-user.
            n_lev = env.n_phase_levels
            best_R = -np.inf; best_phase = phase_pol; best_align = 0.0
            for combo in itertools.product(range(n_lev), repeat=M):
                ph_oracle = np.zeros((M, N), dtype=int)
                for m, lev in enumerate(combo):
                    ph_oracle[m, :] = lev
                R_c, _, Q_c, _ = _eval_action(env, phi_pol, ph_oracle, w_p_pol, w_c_pol,
                                              C_k_pol, active_pol_ids, sigmas, D_k)
                if R_c > best_R:
                    best_R, best_Q, best_phase = R_c, Q_c, ph_oracle
            Rorc.append(best_R); Qorc.append(best_Q)

            # alignment of policy vs oracle (sanity)
            if len(active_pol_ids) > 0:
                def _al(ph_idx):
                    ph_rad = env.phase_model.index_to_phase(ph_idx)
                    rbase = 1.0 / np.sqrt(N)
                    al = []
                    for mid in active_pol_ids:
                        m0 = int(mid) - 1
                        if 0 <= m0 < M:
                            q = float(np.abs(np.exp(1j * ph_rad[m0]).sum()) / N)
                            al.append((q - rbase) / (1.0 - rbase))
                    return float(np.mean(al)) if al else float('nan')
                phase_alignment_pol.append(_al(phase_pol))
                phase_alignment_orc.append(_al(best_phase))
            else:
                phase_alignment_pol.append(float('nan'))
                phase_alignment_orc.append(float('nan'))

            # advance env (mobility only — don't re-apply, save compute)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
        if (ep + 1) % max(1, args.episodes // 5) == 0:
            print(f"  collected {ep+1}/{args.episodes} ep  ({time.time()-t0:.0f}s)")

    # ── REPORT ──
    Rpol, RIRSoff, Rdir, Rirs, Rorc = map(np.asarray, [Rpol, RIRSoff, Rdir, Rirs, Rorc])
    Qpol, QIRSoff, Qdir, Qirs, Qorc = map(np.asarray, [Qpol, QIRSoff, Qdir, Qirs, Qorc])
    pol_IRS_share = np.asarray(pol_IRS_share); blocked_n = np.asarray(blocked_n)
    pal_p = np.asarray(phase_alignment_pol); pal_o = np.asarray(phase_alignment_orc)
    N_s = len(Rpol)

    L = []
    L.append("=" * 78)
    L.append(f"  IRS-ATTRIBUTION (Tests A+B+C)  ·  ckpt={args.ckpt}")
    L.append(f"  N={N_s} states · policy IRS-share mean={pol_IRS_share.mean():.2f}  (= "
             f"{100*pol_IRS_share.mean():.1f}% of users routed to IRS)")
    L.append(f"  blocked-count mean={blocked_n.mean():.1f}/{K} · "
             f"phase alignment policy={np.nanmean(pal_p):.3f}  oracle={np.nanmean(pal_o):.3f}")
    L.append("-" * 78)
    def _pct(x, ref):
        return f"{100*(x-ref)/max(abs(ref),1e-9):+5.1f}%"

    # Test A
    dR_A = Rpol - RIRSoff; rel_A = 100 * dR_A / np.maximum(Rpol, 1e-9)
    L.append(f"  TEST A — IRS disabled vs enabled (zero g_RU_hat)")
    L.append(f"    policy baseline      : R_tot={Rpol.mean():.3f}  QoS={Qpol.mean():.1f}%")
    L.append(f"    same action IRS=0    : R_tot={RIRSoff.mean():.3f}  QoS={QIRSoff.mean():.1f}%")
    L.append(f"    ΔR_tot from IRS path : mean={dR_A.mean():+.3f} ({rel_A.mean():+.1f}%)  "
             f"p25={np.percentile(dR_A,25):+.3f}  p75={np.percentile(dR_A,75):+.3f}")
    L.append(f"    ΔQoS                  : {(Qpol-QIRSoff).mean():+.1f} pts")
    if rel_A.mean() < 5:
        L.append(f"    → ⚠ IRS contribution very small ({rel_A.mean():.1f}%) → IRS being BYPASSED in practice.")
    elif rel_A.mean() < 15:
        L.append(f"    → 🟡 IRS contributes modestly ({rel_A.mean():.1f}%) → partial use, not fully exploited.")
    else:
        L.append(f"    → ✅ IRS contributes materially ({rel_A.mean():.1f}%) → IRS being used; not bypassed.")
    L.append("")

    # Test B
    L.append(f"  TEST B — assignment-shape variants (downstream heads re-react)")
    L.append(f"    policy assignment    : R_tot={Rpol.mean():.3f}  QoS={Qpol.mean():.1f}%")
    L.append(f"    DIRECT-ALL  (phi=0)  : R_tot={Rdir.mean():.3f}  QoS={Qdir.mean():.1f}%  "
             f"Δ vs policy={Rdir.mean()-Rpol.mean():+.3f} ({_pct(Rdir.mean(),Rpol.mean())})")
    L.append(f"    IRS-BALANCED (k%M+1) : R_tot={Rirs.mean():.3f}  QoS={Qirs.mean():.1f}%  "
             f"Δ vs policy={Rirs.mean()-Rpol.mean():+.3f} ({_pct(Rirs.mean(),Rpol.mean())})")
    Δb_dir = Rdir.mean() - Rpol.mean()
    Δb_irs = Rirs.mean() - Rpol.mean()
    if Δb_irs > 0.03:
        L.append(f"    → ⚠ IRS-BALANCED beats policy → assignment head OVER-AVOIDS IRS by {Δb_irs:+.3f}.")
    elif Δb_dir > 0.03:
        L.append(f"    → ⚠ DIRECT-ALL beats policy → IRS use is NET NEGATIVE under current phase quality.")
    else:
        L.append(f"    → ✅ Policy assignment ≈ best among the 3 → not a clear assignment bias.")
    L.append("")

    # Test C
    dR_C = Rorc - Rpol; rel_C = 100 * dR_C / np.maximum(Rpol, 1e-9)
    L.append(f"  TEST C — oracle uniform-phase per IRS (brute 4^{M}={4**M} per state)")
    L.append(f"    policy phase         : R_tot={Rpol.mean():.3f}  QoS={Qpol.mean():.1f}%  alignment={np.nanmean(pal_p):.3f}")
    L.append(f"    oracle phase         : R_tot={Rorc.mean():.3f}  QoS={Qorc.mean():.1f}%  alignment={np.nanmean(pal_o):.3f}")
    L.append(f"    ΔR_tot from phase    : mean={dR_C.mean():+.3f} ({rel_C.mean():+.1f}%)  "
             f"p25={np.percentile(dR_C,25):+.3f}  p75={np.percentile(dR_C,75):+.3f}")
    L.append(f"    ΔQoS                  : {(Qorc-Qpol).mean():+.1f} pts")
    if rel_C.mean() > 8:
        L.append(f"    → ⚠ PHASE IS BOTTLENECK: oracle phase lifts R_tot {rel_C.mean():.1f}% — PhaseMLP under-converged.")
    elif rel_C.mean() > 3:
        L.append(f"    → 🟡 PHASE has modest headroom ({rel_C.mean():.1f}%) — partial bottleneck.")
    else:
        L.append(f"    → ✅ Phase already near optimum ({rel_C.mean():.1f}%) — phase NOT the bottleneck.")
    L.append("-" * 78)

    # Cross-test synthesis
    L.append("  SYNTHESIS (combining A+B+C):")
    L.append(f"    IRS path real value (A) : {rel_A.mean():+.1f}% on R_tot")
    L.append(f"    Force-IRS would gain (B): {100*Δb_irs/max(Rpol.mean(),1e-9):+.1f}% (vs DIRECT: {100*Δb_dir/max(Rpol.mean(),1e-9):+.1f}%)")
    L.append(f"    Phase headroom (C)      : {rel_C.mean():+.1f}% on R_tot")
    if rel_A.mean() < 5 and rel_C.mean() > 8:
        L.append(f"    → ⚠⚠ CONFIRMED [J] CHICKEN-EGG: IRS provides little value NOW because phase is bad;")
        L.append(f"         agent rationally avoids IRS. Break loop = improve phase first (warm-up / oracle-init).")
    elif rel_A.mean() < 5 and Δb_dir > 0.03:
        L.append(f"    → ⚠ IRS GENUINELY UNHELPFUL at this ramp — direct + common stream sufficient. Not a bug;")
        L.append(f"         risk surfaces only at hard ramps where direct fails. Curriculum protection needed.")
    elif rel_A.mean() > 15 and Δb_irs > 0.03:
        L.append(f"    → 🟡 IRS already material; assignment head should route MORE. Phase OK.")
    L.append("=" * 78)

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
