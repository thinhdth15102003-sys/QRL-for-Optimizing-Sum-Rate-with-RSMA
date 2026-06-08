"""
probe_assignment_decomp.py
--------------------------
Gap-decomposition probe — distinguishes ASSIGNMENT bottleneck vs DOWNSTREAM
learning as the true source of the live→oracle reward gap.

Motivation (user proposal 2026-06-05): existing probe_assignment_oracle re-runs
the POLICY's downstream for both live and oracle assignments. If the policy's
downstream is itself badly trained for some assignments, the apparent "assignment
gap" could actually be a downstream-fitting failure pretending to be an
assignment one. We need to hold downstream OPTIMAL to isolate the assignment.

Decomposition (per state):
  R_live_pol  = R(q_live,   φ_pol,    w_pol,    C_pol)        ← current "sampled"
  R_live_opt  = R(q_live,   φ_oracle, w_pol,    C_pol)        ← NEW: live q + oracle φ
  R_orw_pol   = R(q_oracle, φ_pol,    w_pol,    C_pol)        ← current "oracle-rew"
  R_orw_opt   = R(q_oracle, φ_oracle, w_pol,    C_pol)        ← NEW: oracle q + oracle φ
                  (q_oracle = coordinate-ascent maximizing reward = R_tot − λ_D · qp)

  gap_total       = R_orw_opt  − R_live_pol     [original assignment-oracle gap]
  gap_downstream  = R_live_opt − R_live_pol     [downstream's share of the gap]
  gap_assignment  = R_orw_opt  − R_live_opt     [TRUE assignment-only gap]

  Sanity:  gap_total ≈ gap_downstream + gap_assignment   (within state-variance)

VERDICT:
  gap_assignment / gap_total  →  fraction of gap that is REAL assignment bottleneck
    ≥ 0.7   ASSIGNMENT-DOMINANT — current EXP chain verdicts stand (F4 dominant
            after EXP-3 phase warmup didn't close).
    0.3-0.7 MIXED — both contribute; need to fix downstream first OR jointly.
    < 0.3   DOWNSTREAM-DOMINANT — "assignment bottleneck" was an artifact of
            mis-trained downstream. Re-focus levers on PhaseMLP / PowerMLP /
            CkMLP training, NOT assignment-side L1/L4 framing.

WHY THIS PROBE MATTERS:
  False-optimization guard. EXP-3 phase-warmup verdict (closure ≈ 0%) led us to
  "F4 dominant → L4 honest-limit framing". But if the apparent gap is mostly
  downstream-fault, that conclusion is WRONG. This probe is the cleanest cross-
  check before committing to F4 paper framing.

  Oracle-downstream choices (computational tractability):
    φ: brute over L^M uniform-per-IRS patterns (L=4, M=2 → 16 evals, optimal under
       per-IRS scalar channel model; see analysis/phase_oracle.py).
    w: keep policy power. Per L15c (probe_power_qos / wc_optimality / oracle Δ ≈ 0%),
       PowerMLP factored+bias=0.3 is at/near optimum; equal-split −0.5pt only.
       (Could brute coordinate-ascent power for full rigor; deferred — would 4×
       runtime for negligible gain on this question.)
    C: keep policy Ck. Within-group split is well-learned and oracle Δ ~ 3% max.

Usage:
  python analysis/probe_assignment_decomp.py \
         --ckpts results/result_15/checkpoints/ep_00400 \
                 results/result_11/checkpoints/ep_01000 \
         --episodes 8 --steps 20 \
         --out results/result_15/assignment_decomp_ep00400.txt
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
from probe_assignment_oracle import _load_nets, _downstream, _eval, _coord_ascent


def _eval_consistent_phase(env, nets, phi, cfg, z_t, sigmas, D_k, lamD, phase_idx):
    """
    Evaluate (R_tot, qp, reward, ...) for assignment `phi` with phase fixed to
    `phase_idx`, re-running PowerMLP + CkMLP USING the same phase (so power &
    Ck decisions are CONSISTENT with the chosen phase, not mismatched to
    PhaseMLP's policy output).

    This fixes the bug in probe_assignment_oracle._eval where power was decided
    with policy's PhaseMLP h_eff then rate computed with override phase →
    power-phase mismatch that artificially inflated/deflated R for broken
    PhaseMLP (e.g. r15 post-EXP-3 warmup).
    """
    from train import _build_ck_state, _get_active_irs
    M, K, N = cfg.M, cfg.K, cfg.N
    _, phase_net, power_net, ck_net = nets

    active_irs = _get_active_irs(phi)
    active_ids = [int(a) + 1 for a in active_irs]

    # Build Phi from the GIVEN phase_idx (not policy's PhaseMLP)
    Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(phase_idx))
    h_eff = env.rate_computer.effective_channels_all(phi, Phi, env.channels)

    # Re-run PowerMLP + CkMLP with h_eff derived from the GIVEN phase
    s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
    try:
        w_c_vec, w_p, _, _ = power_net.forward(s_power, active_ids)
        partial = env.rate_computer.compute_rates_partial(
            phi, Phi, env.channels, w_p, w_c_vec, active_irs_ids=active_ids)
        s_ck = _build_ck_state(np.full(K, D_k), partial['R_private'],
                                partial['R_c_group'], phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
    except Exception:
        return None

    # Noise-averaged rate computation
    draws = []
    for s2 in sigmas:
        r = env.rate_computer.compute_sum_rate(phi, Phi, env.channels, w_p, w_c_vec,
                                               C_k=C_k, active_irs_ids=active_ids, sigma2=s2)
        draws.append(np.asarray(r['R_private']) + np.asarray(r['C_k']))
    draws = np.array(draws)
    R_user = draws.mean(0); R_user_std = draws.std(0)
    R_tot = float(R_user.sum()); qp = float(np.maximum(0.0, D_k - R_user).sum())
    return dict(R_tot=R_tot, qp=qp, reward=R_tot - lamD * qp,
                QoS=float(100*(R_user >= D_k).mean()),
                R_user=R_user, R_user_std=R_user_std,
                IRS_share=float((phi > 0).mean()), phi=phi.copy(),
                phase_idx=phase_idx.copy(), w_p=w_p.copy(), w_c_vec=w_c_vec.copy())


def _eval_with_oracle_phase(env, nets, phi, cfg, z_t, sigmas, D_k, lamD):
    """
    For assignment `phi`, brute-search L^M uniform-per-IRS phase patterns and
    pick the one maximizing REWARD (= R_tot − λ_D · qp), with power+Ck re-run
    CONSISTENTLY for each phase candidate (no power-phase mismatch).

    Returns best metric dict over all phase candidates.
    """
    L = env.n_phase_levels
    M, N = cfg.M, cfg.N

    best = None
    for combo in itertools.product(range(L), repeat=M):
        ph = np.zeros((M, N), dtype=int)
        for mm, lev in enumerate(combo):
            ph[mm, :] = lev
        m = _eval_consistent_phase(env, nets, phi, cfg, z_t, sigmas, D_k, lamD, ph)
        if m is not None and (best is None or m['reward'] > best['reward']):
            best = m
    return best


def run_ckpt(ckpt, cfg, args, lamD, D_k):
    M, K = cfg.M, cfg.K
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(ckpt, cfg)
    actor, nets = _load_nets(ckpt)

    rows = {k: [] for k in ['live_pol', 'live_opt', 'orw_pol', 'orw_opt']}
    t0 = time.time()
    n_states = 0
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 19 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            obs = env._get_obs()
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, np.full(K, D_k), blocked)
            phi_pol, _, info = actor.forward(s_t)            # sampled live q
            z_t = info['z_t']
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_avg)]

            # 1. live q + policy downstream
            m_live_pol = _eval(env, nets, phi_pol, cfg, z_t, sigmas, D_k, lamD)
            # 2. live q + ORACLE downstream (brute phase, keep policy power/Ck)
            m_live_opt = _eval_with_oracle_phase(env, nets, phi_pol, cfg, z_t, sigmas, D_k, lamD)
            # 3. oracle q (coord-ascent on reward) + policy downstream
            m_orw_pol = _coord_ascent(env, nets, phi_pol, cfg, z_t, sigmas, D_k, lamD, M,
                                      'reward', args.passes)
            # 4. oracle q + ORACLE downstream
            if m_orw_pol is not None:
                m_orw_opt = _eval_with_oracle_phase(env, nets, m_orw_pol['phi'], cfg, z_t,
                                                    sigmas, D_k, lamD)
            else:
                m_orw_opt = None

            if any(x is None for x in [m_live_pol, m_live_opt, m_orw_pol, m_orw_opt]):
                continue
            for key, m in [('live_pol', m_live_pol), ('live_opt', m_live_opt),
                           ('orw_pol', m_orw_pol), ('orw_opt', m_orw_opt)]:
                rows[key].append((m['R_tot'], m['QoS'], m['reward'], m['IRS_share']))
            n_states += 1
        if (ep + 1) % max(1, args.episodes // 4) == 0:
            print(f"    {os.path.basename(ckpt)}: {ep+1}/{args.episodes} ep "
                  f"({n_states} states, {time.time()-t0:.0f}s)")

    out = {}
    for k, v in rows.items():
        a = np.array(v)
        out[k] = dict(R_tot=a[:, 0].mean(), QoS=a[:, 1].mean(),
                      reward=a[:, 2].mean(), IRS=a[:, 3].mean())
    out['N'] = len(rows['live_pol'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=['results/result_15/checkpoints/ep_00400',
                             'results/result_11/checkpoints/ep_01000'])
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=5)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_15/assignment_decomp.txt')
    args = ap.parse_args()

    # Auto-detect case dims from FIRST ckpt's actor_config.json (handles cross-case probing
    # when params.py is set to a different Case than the ckpts being probed).
    _first_ckpt = args.ckpts[0]
    if (os.path.isdir(os.path.join(_first_ckpt, 'agents'))
            and not os.path.isfile(os.path.join(_first_ckpt, 'actor_config.json'))):
        _first_ckpt = os.path.join(_first_ckpt, 'agents')
    import json as _json
    with open(os.path.join(_first_ckpt, 'actor_config.json')) as _f:
        _ac = _json.load(_f)
    _ckpt_K = int(_ac.get('K', P.K))
    _ckpt_M = int(_ac.get('B', P.M))   # 'B' field in actor_config is M (n_choices-1)
    if _ckpt_K != P.K or _ckpt_M != P.M:
        # Compute matching P_S (Case 1→50, Case 2→70, Case 3→100 — heuristic OR explicit)
        # Use ckpt's actor_config to derive P_S if available; else fall back to per-K map.
        _case_P_S = {5: 50.0, 10: 70.0, 20: 100.0}.get(_ckpt_K, P.P_S_dBm)
        print(f"  ⚠ Auto-detected case dims from ckpt: K={_ckpt_K} M={_ckpt_M} "
              f"(params.py has K={P.K} M={P.M}) — overriding cfg.")
        cfg = make_config(K=_ckpt_K, M=_ckpt_M, P_S_dBm=_case_P_S)
    else:
        cfg = make_config()
    D_k = cfg.D_k_bps_hz; lamD = float(getattr(P, 'lambda_D', 1.5))
    print("=" * 80)
    print(f"  GAP-DECOMP (assignment vs downstream)  ·  K={cfg.K} M={cfg.M} N={cfg.N}")
    print(f"  λ_D={lamD} D_k={D_k:.2f} · {args.episodes}ep×{args.steps}st noise={args.noise_avg}")
    print("=" * 80)

    L = []
    results = {}
    for ckpt in args.ckpts:
        print(f"  running {ckpt} ...")
        results[ckpt] = run_ckpt(ckpt, cfg, args, lamD, D_k)

    L.append("=" * 80)
    L.append("  GAP-DECOMP RESULTS  (R_oracle_opt − R_live_pol decomposed)")
    L.append("=" * 80)
    for ckpt, r in results.items():
        tag = os.path.basename(ckpt)
        L.append(f"\n  ── {tag}  (N={r['N']} states) ──")
        L.append(f"  {'level':22s} {'R_tot':>7s} {'QoS%':>6s} {'reward':>9s} {'IRS%':>6s}")
        rows = [
            ('live_pol',  'live_q + pol_down',   r['live_pol']),
            ('live_opt',  'live_q + ORACLE_φ',   r['live_opt']),
            ('orw_pol',   'oracle_q + pol_down', r['orw_pol']),
            ('orw_opt',   'oracle_q + ORACLE_φ', r['orw_opt']),
        ]
        for _, nm, d in rows:
            L.append(f"  {nm:22s} {d['R_tot']:7.3f} {d['QoS']:6.1f} {d['reward']:+9.3f} {100*d['IRS']:6.1f}")

        # Decomposition (on reward = policy's true objective)
        R_live_pol  = r['live_pol']['reward']
        R_live_opt  = r['live_opt']['reward']
        R_orw_opt   = r['orw_opt']['reward']

        gap_total      = R_orw_opt - R_live_pol
        gap_downstream = R_live_opt - R_live_pol
        gap_assignment = R_orw_opt - R_live_opt
        # Decomposition share (guard against tiny total gap)
        if abs(gap_total) > 1e-3:
            frac_down = gap_downstream / gap_total
            frac_assn = gap_assignment / gap_total
        else:
            frac_down = frac_assn = 0.0

        L.append(f"")
        L.append(f"  ── DECOMPOSITION on reward (policy's true objective) ──")
        L.append(f"     gap_total       (R_orw_opt − R_live_pol)  = {gap_total:+.3f}")
        L.append(f"     gap_downstream  (R_live_opt − R_live_pol) = {gap_downstream:+.3f}  ({100*frac_down:+.0f}% share)")
        L.append(f"     gap_assignment  (R_orw_opt  − R_live_opt) = {gap_assignment:+.3f}  ({100*frac_assn:+.0f}% share)")

        # Verdict
        if abs(gap_total) < 0.05:
            verdict = "NEGLIGIBLE GAP — policy near-optimal, no bottleneck."
        elif frac_assn >= 0.7:
            verdict = ("ASSIGNMENT-DOMINANT  — true assignment bottleneck. "
                       "EXP chain F2/F4 verdicts (L1-L4 levers) ARE the right path.")
        elif frac_assn >= 0.3:
            verdict = ("MIXED  — both assignment AND downstream contribute. "
                       "Fix downstream first (PhaseMLP/PowerMLP/CkMLP) before re-judging assignment.")
        else:
            verdict = ("DOWNSTREAM-DOMINANT  — apparent assignment gap is mostly a downstream-fitting failure. "
                       "Re-focus levers on PHASE/POWER/Ck training, NOT assignment-side L1/L4.")
        L.append(f"     → {verdict}")

    # Cross-ckpt comparison if multiple
    if len(args.ckpts) >= 2:
        L.append(f"\n  ── CROSS-CHECKPOINT (does downstream-share change over training?) ──")
        for ckpt in args.ckpts:
            r = results[ckpt]
            g_t = r['orw_opt']['reward'] - r['live_pol']['reward']
            g_d = r['live_opt']['reward'] - r['live_pol']['reward']
            f_d = (g_d / g_t * 100) if abs(g_t) > 1e-3 else 0.0
            L.append(f"    {os.path.basename(ckpt):22s}: gap_total={g_t:+.3f}  downstream={f_d:+.0f}%")

        # ⭐ ASSIGNMENT-QUALITY TRACKING (user-proposed 2026-06-05)
        # Track (live, ceiling, gap=ceiling-live) per ckpt. Different patterns mean different things:
        #   live↓ + ceiling↓ + gap STABLE       → representation drift (everything degrading proportionally)
        #   live↓ + ceiling stable + gap GROW   → execution problem (oracle has room, policy can't use)
        #   live stable + ceiling↓ + gap shrink → assignment OK but rep drift (learns "downstream-friendlier")
        #   live stable/up + ceiling stable/up  → assignment-quality preservation ⭐
        L.append(f"\n  ── ASSIGNMENT-QUALITY TRACK (live, ceiling, gap on reward) ──")
        L.append(f"    {'ckpt':22s} {'live':>7s} {'ceiling':>8s} {'gap':>7s}")
        for ckpt in args.ckpts:
            r = results[ckpt]
            live = r['live_opt']['reward']     # R(q_live,   ORACLE_φ)
            ceil = r['orw_opt']['reward']      # R(q_oracle, ORACLE_φ)
            gap  = ceil - live
            L.append(f"    {os.path.basename(ckpt):22s} {live:7.3f} {ceil:8.3f} {gap:7.3f}")
        L.append(f"    Interpretation key:")
        L.append(f"      live↓+ceil↓ gap≈stable  → REPRESENTATION DRIFT (broad degradation)")
        L.append(f"      live↓+ceil stable       → EXECUTION PROBLEM (oracle has room, policy can't use)")
        L.append(f"      live stable+ceil↓       → assignment OK but rep drifts (rare; learns downstream-friendlier)")
        L.append(f"      live≥1.475+ceil≥1.677   → ⭐ assignment-quality preservation (warmup gain locked)")
    L.append("=" * 80)

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
