"""
probe_wc_optimality.py
----------------------
Combined counterfactual probe for the wc question (Tests 1+2+4+5 in one rollout):

  Test 1 — wc-sweep            : mean[R_tot, QoS, reward](wc) → global wc*
  Test 4 — λ_D sensitivity     : recompute reward(wc) at λ_D ∈ {0.5, 1.0, 1.5, 2.0}
  Test 5 — per-state oracle    : argmax_{wc} reward(s,wc) → distribution of wc*(s)
  Test 2 — phase↔wc correlation: partial-ρ(policy_wc, phase_alignment | blocked_count)

Method
------
Hold policy fixed (trained ckpt). For each collected state s:
  1. record policy's chosen (assignment, phase_idx, w_p, w_c_vec, C_k), phase alignment,
     blocked count.
  2. for each wc_target in the sweep, RENORMALIZE: scale w_c_vec → sum=wc_target·P_S
     and w_p → sum=(1−wc_target)·P_S, **preserving relative proportions within each**.
     This isolates the wc LEVEL from the structure inside w_c_vec and w_p.
  3. evaluate R_tot and qp-penalty at that state under the renormalized power, averaging
     over a few noise draws (same draws across wc variants → fair).
Per-state output: reward(s, wc, λ_D) = R_tot(s, wc) − λ_D·qp(s, wc).

Outputs
-------
  • global curves: mean reward / R_tot / QoS vs wc, at each λ_D
  • per-state wc*: histogram + stratified by phase-quality tertile (H2 oracle test)
  • partial-ρ(policy_wc, alignment | blocked_count tertile)
  • baseline reference: equal-split wc=0.5 with equal w_p (no structure learned by RL)

Usage:
  python analysis/probe_wc_optimality.py --ckpt results/result_11/checkpoints/ep_00600
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


WC_GRID = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
LAMD_GRID = np.array([0.5, 1.0, 1.5, 2.0])


def renorm_split(w_c_vec, w_p, wc_target):
    """Scale w_c_vec & w_p so sum(w_c_vec) = wc_target·P_S and sum(w_p) = (1−wc_target)·P_S,
    preserving relative proportions within each. (Isolates wc LEVEL from structure.)"""
    P_S = float(w_c_vec.sum() + w_p.sum())
    sc, sp = float(w_c_vec.sum()), float(w_p.sum())
    n_c, n_p = len(w_c_vec), len(w_p)
    w_c_new = (w_c_vec * (wc_target * P_S / sc) if sc > 1e-9
               else np.full(n_c, wc_target * P_S / n_c))
    w_p_new = (w_p * ((1.0 - wc_target) * P_S / sp) if sp > 1e-9
               else np.full(n_p, (1.0 - wc_target) * P_S / n_p))
    return w_c_new, w_p_new


def phase_alignment(phase_idx, env, active_irs_ids):
    """Coherence |Σexp(jφ)|/N → normalised alignment (0=rand, 1=opt), averaged over
    active IRS. active_irs_ids is 1-based (env convention); convert to 0-based."""
    if len(active_irs_ids) == 0:
        return float('nan')
    phases_rad = env.phase_model.index_to_phase(phase_idx)
    M, N = phases_rad.shape
    rand_baseline = 1.0 / np.sqrt(N)
    al = []
    for mid in active_irs_ids:
        m0 = int(mid) - 1
        if 0 <= m0 < M:
            q = float(np.abs(np.exp(1j * phases_rad[m0]).sum()) / N)
            al.append((q - rand_baseline) / (1.0 - rand_baseline))
    return float(np.mean(al)) if al else float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_00600')
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=40)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/wc_optimality_ep600.txt')
    args = ap.parse_args()

    cfg = make_config(); K = cfg.K; D_k = cfg.D_k_bps_hz
    lamD_train = float(getattr(P, 'lambda_D', 1.5))
    print("=" * 78)
    print(f"  WC-OPTIMALITY (Tests 1+2+4+5)  ·  ckpt={args.ckpt}")
    print(f"  K={K} M={cfg.M} N={cfg.N} D_k={D_k:.2f} · λ_D(train)={lamD_train}")
    print(f"  collect {args.episodes}×{args.steps} (warmup {args.warmup}) · "
          f"noise_avg={args.noise_avg} · wc grid={WC_GRID.tolist()}")
    print("=" * 78)

    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(args.ckpt, cfg)
    rc = env.rate_computer

    # storage: per-state records
    n_wc = len(WC_GRID)
    Rtot_grid = []   # (n_states, n_wc)  — sum-rate
    qp_grid   = []   # (n_states, n_wc)  — total QoS deficit (Σ max(0, D_k - R_user))
    QoS_grid  = []   # (n_states, n_wc)  — % users meeting D_k
    policy_wc = []   # (n_states,)        — policy's chosen wc share
    phase_al  = []   # (n_states,)        — phase alignment (NaN if no active IRS)
    blocked_n = []   # (n_states,)        — # users blocked
    Rtot_eq, QoS_eq = [], []  # equal-split reference (wc=0.5 + uniform w_p)

    t0 = time.time()
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 7 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            action = policy(env)
            env._apply_action(action)  # sets assignment, Phi, w_p, w_c_vec, C_k

            P_S_now = float(action['w_p'].sum() + action['w_c_vec'].sum())
            policy_wc.append(float(action['w_c_vec'].sum() / max(P_S_now, 1e-12)))
            phase_al.append(phase_alignment(action['phase_idx'], env,
                                            env.active_irs_ids))
            blocked_n.append(int(env.channels['su_blocked'].sum()))

            # Sample noise draws ONCE per state, reuse across wc variants (fair)
            sigmas = [env.channel_model.sample_noise_sigma2()
                      for _ in range(args.noise_avg)]

            # ── per-wc evaluation under SAME state + SAME noise draws ──
            row_R, row_qp, row_Q = [], [], []
            for wc_t in WC_GRID:
                w_c_new, w_p_new = renorm_split(action['w_c_vec'], action['w_p'],
                                                float(wc_t))
                R_user = np.zeros(K)
                for s2 in sigmas:
                    r = rc.compute_sum_rate(env.assignment, env.Phi, env.channels,
                                            w_p_new, w_c_new, C_k=env.C_k,
                                            active_irs_ids=env.active_irs_ids,
                                            sigma2=s2)
                    R_user += np.asarray(r['R_private']) + np.asarray(r['C_k'])
                R_user /= args.noise_avg
                deficit = np.maximum(0.0, D_k - R_user)
                row_R.append(float(R_user.sum()))
                row_qp.append(float(deficit.sum()))
                row_Q.append(float(100.0 * (R_user >= D_k).mean()))
            Rtot_grid.append(row_R); qp_grid.append(row_qp); QoS_grid.append(row_Q)

            # equal-split reference: wc=0.5, w_p uniform, w_c_vec uniform across slots
            w_c_eq = np.full_like(action['w_c_vec'], 0.5 * P_S_now / len(action['w_c_vec']))
            w_p_eq = np.full(K, 0.5 * P_S_now / K)
            R_user_eq = np.zeros(K)
            for s2 in sigmas:
                r = rc.compute_sum_rate(env.assignment, env.Phi, env.channels,
                                        w_p_eq, w_c_eq, C_k=env.C_k,
                                        active_irs_ids=env.active_irs_ids,
                                        sigma2=s2)
                R_user_eq += np.asarray(r['R_private']) + np.asarray(r['C_k'])
            R_user_eq /= args.noise_avg
            Rtot_eq.append(float(R_user_eq.sum()))
            QoS_eq.append(float(100.0 * (R_user_eq >= D_k).mean()))

            # advance env without re-applying action (mobility only)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
        if (ep + 1) % max(1, args.episodes // 6) == 0:
            el = time.time() - t0
            print(f"  collected {ep+1}/{args.episodes} ep  ({el:.0f}s)")

    Rtot_grid = np.asarray(Rtot_grid)   # (N, n_wc)
    qp_grid   = np.asarray(qp_grid)
    QoS_grid  = np.asarray(QoS_grid)
    policy_wc = np.asarray(policy_wc)
    phase_al  = np.asarray(phase_al)
    blocked_n = np.asarray(blocked_n)
    N = len(policy_wc)

    # ── TESTS ─────────────────────────────────────────────────────────────────
    lines = []
    L = lines.append
    L("=" * 78)
    L(f"  WC-OPTIMALITY  ·  ckpt={args.ckpt}")
    L(f"  N={N} states · wc grid={WC_GRID.tolist()} · noise_avg={args.noise_avg}")
    L(f"  policy mean wc = {policy_wc.mean():.3f} (std {policy_wc.std():.3f})  ·  "
      f"phase alignment mean = {np.nanmean(phase_al):.3f}")
    L("-" * 78)

    # ── Test 1 + 4: global curves at each λ_D ──
    L("  TEST 1 + 4 — mean over states (R_tot, QoS) by wc_target,  reward(λ_D)")
    L("")
    L(f"  {'wc':>5s}  {'R_tot':>8s}  {'QoS%':>6s}  " +
      "  ".join([f"rwd(λ={lam})" for lam in LAMD_GRID]))
    mean_R = Rtot_grid.mean(axis=0)
    mean_Q = QoS_grid.mean(axis=0)
    rewards_by_lam = {}
    for lam in LAMD_GRID:
        rewards_by_lam[lam] = (Rtot_grid - lam * qp_grid).mean(axis=0)
    for j, wc in enumerate(WC_GRID):
        rwd_str = "  ".join([f"{rewards_by_lam[lam][j]:+9.3f}" for lam in LAMD_GRID])
        L(f"  {wc:5.2f}  {mean_R[j]:8.3f}  {mean_Q[j]:6.1f}  {rwd_str}")
    L("")
    L(f"  equal-split ref (wc=0.5, w_p uniform): R_tot={np.mean(Rtot_eq):.3f}  "
      f"QoS={np.mean(QoS_eq):.1f}%")
    L("")
    # argmax per λ_D
    best_wc = {}
    for lam in LAMD_GRID:
        j = int(np.argmax(rewards_by_lam[lam]))
        best_wc[lam] = float(WC_GRID[j])
    best_rtot = float(WC_GRID[int(np.argmax(mean_R))])
    L(f"  argmax wc:  by R_tot → {best_rtot:.2f}  · "
      + "  ".join([f"λ_D={lam}→{best_wc[lam]:.2f}" for lam in LAMD_GRID]))
    L(f"  policy mean wc = {policy_wc.mean():.2f}  → "
      f"vs sum-rate-opt = {best_rtot - policy_wc.mean():+.2f}  "
      f"vs λ_D=1.5-opt = {best_wc[1.5] - policy_wc.mean():+.2f}")
    L("-" * 78)

    # ── Test 5: per-state wc* (oracle), histogram + stratify by phase ──
    rew_train_per_state = Rtot_grid - lamD_train * qp_grid     # (N, n_wc)
    wcstar_idx = rew_train_per_state.argmax(axis=1)
    wcstar = WC_GRID[wcstar_idx]                                # (N,)
    L("  TEST 5 — per-state wc*  (oracle at λ_D=train)")
    L(f"  mean wc* = {wcstar.mean():.3f}  std = {wcstar.std():.3f}  "
      f"vs policy mean wc = {policy_wc.mean():.3f}")
    hist, edges = np.histogram(wcstar, bins=np.r_[WC_GRID - 0.05, WC_GRID[-1] + 0.05])
    L(f"  wc* histogram:")
    for j, wc in enumerate(WC_GRID):
        bar = "█" * int(40 * hist[j] / max(hist.sum(), 1))
        L(f"    {wc:4.2f} | {hist[j]:4d} {bar}")
    # stratify by phase tertile
    pa = np.where(np.isnan(phase_al), -1.0, phase_al)
    valid = pa >= 0
    if valid.sum() >= 30:
        q1, q2 = np.percentile(pa[valid], [33.3, 66.7])
        lo = valid & (pa <= q1); mid = valid & (pa > q1) & (pa <= q2); hi = valid & (pa > q2)
        L("")
        L("  wc* by phase-alignment tertile (H2 oracle test — heterogeneity?):")
        L(f"    phase LOW  (≤{q1:.2f}, n={lo.sum():4d}):  mean wc* = {wcstar[lo].mean():.3f}")
        L(f"    phase MID  ({q1:.2f}-{q2:.2f}, n={mid.sum():4d}):  mean wc* = {wcstar[mid].mean():.3f}")
        L(f"    phase HIGH (>{q2:.2f}, n={hi.sum():4d}):  mean wc* = {wcstar[hi].mean():.3f}")
        dlo_hi = wcstar[lo].mean() - wcstar[hi].mean()
        L(f"    Δ(low−high) = {dlo_hi:+.3f}  "
          f"→ H2 prediction: phase good ⇒ wc* low (Δ>0); H1: Δ≈0")
    L("-" * 78)

    # ── Test 2: partial-ρ(policy_wc, phase_alignment | blocked_count) ──
    L("  TEST 2 — correlation ρ(policy_wc, phase_alignment), partial on blocked_count")
    if valid.sum() >= 30:
        pw = policy_wc[valid]; pa_v = phase_al[valid]; bn = blocked_n[valid].astype(float)
        rho_all = float(np.corrcoef(pw, pa_v)[0, 1])
        # partial correlation via residualization on blocked_count (single covariate)
        def _resid(x, z):
            zc = (z - z.mean()) / (z.std() + 1e-9)
            beta = float(np.dot(x - x.mean(), zc) / (np.dot(zc, zc) + 1e-9))
            return x - x.mean() - beta * zc
        rho_partial = float(np.corrcoef(_resid(pw, bn), _resid(pa_v, bn))[0, 1])
        # stratified by blocked tertile
        b_q1, b_q2 = np.percentile(bn, [33.3, 66.7])
        s_lo = bn <= b_q1; s_mid = (bn > b_q1) & (bn <= b_q2); s_hi = bn > b_q2
        rho_strata = []
        for nm, mask in [("low_block", s_lo), ("mid_block", s_mid), ("hi_block", s_hi)]:
            if mask.sum() >= 10:
                r = float(np.corrcoef(pw[mask], pa_v[mask])[0, 1])
                rho_strata.append(f"{nm}(n={mask.sum()})={r:+.3f}")
            else:
                rho_strata.append(f"{nm}(n={mask.sum()})=NA")
        L(f"  ρ(policy_wc, phase_alignment) pooled  = {rho_all:+.3f}")
        L(f"  ρ partial on blocked_count            = {rho_partial:+.3f}")
        L(f"  ρ by blocked-count tertile            = " + " · ".join(rho_strata))
        L(f"  → H2 predicts ρ<0 (phase good→wc low) in all strata · H1 predicts ρ≈0")
    L("-" * 78)

    # ── PER-STATE REGRET (ΔR = R(wc*(s)) − R(policy_wc(s))) ─────────────────
    # Linearly interpolate grid-evaluated R_tot/qp at each state's continuous policy_wc
    # to get the policy's actual per-state value, then compute regret vs the per-state
    # grid argmax. This is the addressable headroom of a perfect per-state wc head.
    def _interp_row(row, target):
        return float(np.interp(target, WC_GRID, row))
    L("  PER-STATE REGRET  ΔR = R_metric(wc*(s)) − R_metric(policy_wc(s))")
    L("  (linear-interp policy_wc on the grid · UPPER BOUND on per-state-adapt headroom)")
    L("")
    metrics = {
        'R_tot only (sum-rate prio)': (Rtot_grid, np.zeros_like(qp_grid), 0.0),
        'reward λ_D=0.5'             : (Rtot_grid, qp_grid, 0.5),
        'reward λ_D=1.5 (train)'     : (Rtot_grid, qp_grid, 1.5),
    }
    L(f"  {'metric':30s}  {'mean ΔR':>9s} {'p50':>7s} {'p90':>7s} {'max':>7s}  "
      f"{'frac>0.01':>9s} {'mean R(pol)':>11s}")
    regret_by_metric = {}
    for name, (Rg, qg, lam) in metrics.items():
        rew_grid_m = Rg - lam * qg                           # (N, n_wc) — full metric
        rew_opt    = rew_grid_m.max(axis=1)                  # (N,)      — per-state opt
        rew_policy = np.array([_interp_row(rew_grid_m[i], policy_wc[i])
                               for i in range(N)])
        dR = rew_opt - rew_policy                            # ≥ 0 by construction
        regret_by_metric[name] = dR
        L(f"  {name:30s}  {dR.mean():+9.4f} {np.percentile(dR,50):7.4f} "
          f"{np.percentile(dR,90):7.4f} {dR.max():7.4f}  "
          f"{(dR > 0.01).mean()*100:8.1f}%  {rew_policy.mean():+11.4f}")
    L("")
    # context: how much of R(policy_wc) total is the regret?
    rew_pol_train = np.array([_interp_row((Rtot_grid - 1.5*qp_grid)[i], policy_wc[i])
                              for i in range(N)])
    L(f"  context: at λ_D=1.5, mean R(policy) = {rew_pol_train.mean():+.3f}; "
      f"mean ΔR = {regret_by_metric['reward λ_D=1.5 (train)'].mean():+.3f}  "
      f"(= {100*regret_by_metric['reward λ_D=1.5 (train)'].mean()/abs(rew_pol_train.mean()):.1f}% relative)")
    # stratify ΔR by phase tertile (H2-reversed: high-phase = bigger ΔR if inversion matters)
    if valid.sum() >= 30:
        L("")
        L("  ΔR (R_tot, sum-rate frame) by phase-alignment tertile:")
        dR_sr = regret_by_metric['R_tot only (sum-rate prio)']
        L(f"    phase LOW  (≤{q1:.2f}, n={lo.sum():4d}):  mean ΔR = {dR_sr[lo].mean():+.4f}  "
          f"p90 = {np.percentile(dR_sr[lo],90):.4f}")
        L(f"    phase MID  ({q1:.2f}-{q2:.2f}, n={mid.sum():4d}):  mean ΔR = {dR_sr[mid].mean():+.4f}  "
          f"p90 = {np.percentile(dR_sr[mid],90):.4f}")
        L(f"    phase HIGH (>{q2:.2f}, n={hi.sum():4d}):  mean ΔR = {dR_sr[hi].mean():+.4f}  "
          f"p90 = {np.percentile(dR_sr[hi],90):.4f}")
        L(f"    → H2-reversed prediction: HIGH-phase ΔR > LOW-phase ΔR (inversion costs most there)")
    L("-" * 78)

    # ── verdict synthesis ──
    L("  VERDICT (read with the numbers above — verdict line uses loose thresholds):")
    gap_sumrate = best_rtot - policy_wc.mean()
    gap_train   = best_wc[1.5] - policy_wc.mean()
    if abs(gap_sumrate) < 0.06 and abs(gap_train) < 0.06:
        L(f"   • POLICY ≈ OPTIMAL at current λ_D and for sum-rate: gap |wc_opt−wc_policy| < 0.06.")
        L(f"     → H4 (current wc=policy equilibrium = ~optimal) supported; H3 weak.")
    elif gap_sumrate < -0.08:
        L(f"   • SUM-RATE OPTIMAL wc ({best_rtot:.2f}) < policy ({policy_wc.mean():.2f}): "
          f"policy over-pressed toward common.")
        L(f"     → H3 supported under sum-rate priority (λ_D=1.5 pushes wc high; "
          f"sum-rate-opt wants lower).")
    if valid.sum() >= 30 and dlo_hi > 0.05:
        L(f"   • per-state wc* HETEROGENEOUS: phase-low strata want wc HIGHER than phase-high "
          f"(Δ={dlo_hi:+.3f}) → H2 supported at oracle level.")
        L(f"     → policy (which uses a ~uniform wc) leaves headroom by not adapting per-state.")
    L("=" * 78)

    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
