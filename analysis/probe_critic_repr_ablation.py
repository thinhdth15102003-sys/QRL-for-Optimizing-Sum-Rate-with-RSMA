"""
probe_critic_repr_ablation.py
-----------------------------
STEP B (corrected): is the critic's ceiling set by its INPUT REPRESENTATION?

Holds POLICY FIXED (the trained checkpoint) and fits fresh oracle critics offline on
the SAME (state, MC-return) data, varying ONLY the input feature set:

  V0_base   : s_t  (44-dim affinity ‖ D_k ‖ |g_SU| ‖ q_m ‖ p_m) — what the live critic sees
  V1_blk    : + su_blocked flags (K)
  V2_mag    : + raw |g_SR|,|g_RU|,|g_SU| (full-resolution magnitudes)
  V3_phase  : + channel PHASES  ∠g_SR,∠g_RU,∠g_SU   (s_t is magnitude-ONLY → this is the
              prime suspect: achievable rate depends on phase alignment)
  V4_full   : base + blockage + Re/Im of all channels (everything in env.channels)

Why this design
---------------
The Tier-1 / temp-sweep ceiling (~0.82) is computed by replaying from a FULL env
snapshot → it is the predictability given the COMPLETE physical state, NOT given s_t.
The prior oracle-sweep best (0.376) was collected under a RANDOM policy, so comparing it
to the TRAINED-policy ceiling (0.81) mixed policies. This probe removes both confounds:
ONE trained policy, ONE dataset, only the input vector changes.

Reads
-----
  V0_base EV vs live critic (~0.40): V0 ≈ live → critic NOT underfitting s_t (dead-units /
       training-dynamics ruled out); V0 ≫ live → live critic underfits even s_t.
  V3/V4 EV vs full-state ceiling (~0.82): climbs toward ceiling → REPRESENTATION is the
       lever (enrich critic input — it is train-only, not deployed); stays ~V0 → richer
       features don't help → the full-state ceiling isn't reachable from these channels.

CPU fits; GPU only for the one-time trained-policy rollout collection (moderate n_shots).

Usage:
  python analysis/probe_critic_repr_ablation.py --ckpt results/result_11/checkpoints/ep_00300
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
from collections import defaultdict
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from train_oracle_critic import _state_vec, train_oracle      # sibling module
from probe_critic_temp_sweep import TempPolicy                # reuse trained-policy loader


def build_features(env, cfg) -> dict:
    """All input-variant feature vectors at the CURRENT env state."""
    base = _state_vec(env, cfg)                                  # 44, == live critic input
    blk  = env.channels['su_blocked'].astype(float).ravel()      # K
    obs  = env._get_obs()
    gsr, gru, gsu = obs['g_SR'], obs['g_RU'], obs['g_SU']
    mag = np.concatenate([np.abs(gsr).ravel(), np.abs(gru).ravel(), np.abs(gsu).ravel()])
    ph  = np.concatenate([np.angle(gsr).ravel(), np.angle(gru).ravel(), np.angle(gsu).ravel()])
    reim = np.concatenate([
        gsr.real.ravel(), gsr.imag.ravel(),
        gru.real.ravel(), gru.imag.ravel(),
        gsu.real.ravel(), gsu.imag.ravel(),
    ])
    return {
        'V0_base':  base,
        'V1_blk':   np.concatenate([base, blk]),
        'V2_mag':   np.concatenate([base, mag]),
        'V3_phase': np.concatenate([base, mag, ph]),
        'V4_full':  np.concatenate([base, blk, reim]),
    }


def collect(tp: TempPolicy, cfg, n_ep, steps, noise, gamma, seed):
    """Roll out the TRAINED policy; per step store every feature variant + reward.
    MC return (no bootstrap, truncated at episode end) per step — matches the oracle
    sweep target and the full-state MC ceiling definition."""
    env = ISTNEnv(cfg=cfg, seed=seed, n_steps_ep=steps + 2, reward_noise_avg=noise)
    variants = defaultdict(list)
    returns = []
    t0 = time.time()
    for ep in range(n_ep):
        env.reset(seed=(seed * 101 + ep) & 0xFFFF)
        ep_feats = defaultdict(list); ep_r = []
        for _ in range(steps):
            feats = build_features(env, cfg)
            a = tp.act(env, 1.0)                       # τ=1 → trained sampled policy
            _, r, _, _ = env.step(a)
            for k, v in feats.items():
                ep_feats[k].append(v)
            ep_r.append(float(r))
        ret = np.zeros(len(ep_r)); G = 0.0
        for i in reversed(range(len(ep_r))):
            G = ep_r[i] + gamma * G
            ret[i] = G
        for k in ep_feats:
            variants[k].extend(ep_feats[k])
        returns.extend(ret.tolist())
        if (ep + 1) % max(1, n_ep // 10) == 0:
            el = time.time() - t0
            print(f"  collected {ep+1}/{n_ep} ep  ({el:.0f}s, "
                  f"ETA {el*(n_ep-ep-1)/(ep+1):.0f}s)")
    return {k: np.asarray(v) for k, v in variants.items()}, np.asarray(returns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_00300')
    ap.add_argument('--n_ep', type=int, default=90)
    ap.add_argument('--steps', type=int, default=50)
    ap.add_argument('--n_shots', type=int, default=256)
    ap.add_argument('--reward_noise_avg', type=int,
                    default=getattr(P, 'reward_noise_avg', 16))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--arch', type=int, nargs='+', default=[512, 256, 128, 64])
    ap.add_argument('--ceiling', type=float, default=0.82,
                    help='full-state MC ceiling ref (temp-sweep τ=1, trained policy)')
    ap.add_argument('--live_ev', type=float, default=0.40,
                    help='live online critic explVar ref')
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/repr_ablation_ep300.txt')
    args = ap.parse_args()

    cfg = make_config()
    gamma = getattr(P, 'gamma', 0.95)
    print("=" * 78)
    print(f"  CRITIC INPUT-REPRESENTATION ABLATION (Step B)  ·  ckpt={args.ckpt}")
    print(f"  trained policy · K={cfg.K} M={cfg.M} N={cfg.N} · n_shots={args.n_shots} · "
          f"noise_avg={args.reward_noise_avg} γ={gamma}")
    print(f"  collect {args.n_ep}×{args.steps} · arch={args.arch} epochs={args.epochs}")
    print(f"  REFS: live critic ≈ {args.live_ev:+.2f} · full-state ceiling ≈ {args.ceiling:+.2f}")
    print("=" * 78)

    tp = TempPolicy(args.ckpt, cfg, n_shots=args.n_shots, seed=args.seed)
    variants, returns = collect(tp, cfg, args.n_ep, args.steps,
                                args.reward_noise_avg, gamma, args.seed)
    N = len(returns)
    print(f"\n  collected N={N} samples · return μ={returns.mean():+.2f} σ={returns.std():.2f}\n")

    order = ['V0_base', 'V1_blk', 'V2_mag', 'V3_phase', 'V4_full']
    rows = []
    for name in order:
        X = variants[name]
        ds = {'states': X, 'returns': returns, 'd_state': int(X.shape[1]),
              'gamma': gamma, 'cfg_K': cfg.K, 'cfg_M': cfg.M,
              'cfg_R_LoS_km': cfg.R_LoS_km, 'reward_noise_avg': args.reward_noise_avg,
              'n_episodes': args.n_ep, 'steps_per_ep': args.steps}
        print(f"\n── fitting {name}  (d_state={X.shape[1]}) ─────────────────────────")
        res = train_oracle(ds, hidden=list(args.arch), epochs=args.epochs,
                           normalize_targets=True, grad_clip=1000.0, seed=0)
        rows.append((name, X.shape[1], res['best_ev_raw'], res['final_ev_raw']))

    # ── report ──
    lines = []
    lines.append("=" * 78)
    lines.append("  CRITIC INPUT-REPRESENTATION ABLATION  ·  trained policy, fixed data")
    lines.append("=" * 78)
    lines.append(f"  ckpt={args.ckpt} · N={N} · arch={args.arch} · noise_avg={args.reward_noise_avg}")
    lines.append(f"  REF: live online critic ≈ {args.live_ev:+.2f} · full-state MC ceiling ≈ {args.ceiling:+.2f}")
    lines.append("-" * 78)
    lines.append(f"  {'variant':10s} {'d_in':>5s} {'best_EV':>9s} {'final_EV':>9s}  {'vs_base':>8s}")
    base_ev = rows[0][2]
    for name, d, bev, fev in rows:
        lines.append(f"  {name:10s} {d:5d} {bev:+9.3f} {fev:+9.3f}  {bev-base_ev:+8.3f}")
    lines.append("-" * 78)
    v0 = rows[0][2]; v3 = next(r[2] for r in rows if r[0] == 'V3_phase')
    v4 = rows[-1][2]; best_rich = max(r[2] for r in rows[1:])
    lines.append("  VERDICT:")
    if v0 >= args.live_ev - 0.06:
        lines.append(f"   • V0_base ({v0:+.3f}) ≈ live ({args.live_ev:+.2f}) → live critic is NOT"
                     " underfitting s_t (dead-units / training-dynamics RULED OUT).")
    else:
        lines.append(f"   • V0_base ({v0:+.3f}) ≫ live ({args.live_ev:+.2f}) → live critic UNDERFITS"
                     " even s_t → training-dynamics/dead-units back on the table.")
    if best_rich - v0 > 0.08:
        lines.append(f"   • richer input lifts EV {v0:+.3f}→{best_rich:+.3f} (Δ{best_rich-v0:+.3f})"
                     " → REPRESENTATION is a real lever. Biggest jump identifies the missing info.")
        if v3 - rows[2][2] > 0.05:
            lines.append("     PHASE (V3) adds materially beyond magnitudes → s_t being magnitude-"
                         "only is a key loss.")
    else:
        lines.append(f"   • richer input barely moves EV ({v0:+.3f}→{best_rich:+.3f}) → the full-"
                     "state ceiling is NOT reachable from these channel features → not a simple repr fix.")
    lines.append(f"   • gap to full-state ceiling after enrichment: {args.ceiling - best_rich:+.3f}")
    lines.append("=" * 78)
    report = "\n".join(lines)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
