"""
probe_uenc_ablation.py
----------------------
T4 — Prop 4 unified probe: state-dependence as a function of U_enc attenuation
AND classical-bypass attenuation. Tests Framing-A (state-dependence not reward)
via 3 modes:

  (1) ATTENUATION sweep  : o_hat → c · o_hat for c ∈ {1, 0.75, 0.5, 0.25, 0}
                           Smooth M2(KL) degradation ⇒ U_enc carries genuine
                           continuous state-info (not a cliff artefact of OOD).
  (2) BYPASS-ZERO        : z_t → 0 in head (h_t := [0 ‖ o_hat])
                           Isolates U_enc standalone contribution.
  (3) VARIANCE BAND      : run (1) across N ckpts → confirm 35% drop not
                           ckpt-specific.

NOTE on the classical bypass: head input is h_t = [z_t ‖ o_hat]. z_t (AE latent,
classical bypass [B], Δ5) varies with state even if o_hat is degenerate. So
removing U_enc only removes the QUANTUM-CHANNEL contribution; the policy stays
state-dependent via z_t. The probe quantifies BOTH channels.

Forward path: ANALYTIC (no shot noise) — critical for clean KL comparison.
Pi distribution is replicated from QuantumActor.compute_log_prob logic.

Usage:
  # Mode 1+2+3 combined (full report on one or more ckpts):
  python analysis/probe_uenc_ablation.py \
      --ckpts results/result_11/checkpoints/ep_00600/agents \
              results/result_11/checkpoints/ep_01000/agents \
              results/result_11/checkpoints/ep_01400/agents \
              results/result_11/checkpoints/ep_01700/agents \
      --c-uenc-list "1.0,0.75,0.5,0.25,0.0" \
      --include-bypass-zero \
      --episodes 12 --steps 20 --seed 0
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from RL.quantum_actor import QuantumActor
from RL.quantum_circuit import expectations_analytic
from probe_critic_ceiling import make_checkpoint_policy


# ─────────────────────────────── state sampling ───────────────────────────────

def _sample_states(env, actor, policy, args, D_k: float):
    """Run env with the trained policy; collect actor's s_t inputs per step."""
    K = actor.K
    states = []
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 19 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            obs = env._get_obs()
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, np.full(K, D_k), blocked)
            states.append(s_t.copy())
            env.step(policy(env))
    return np.array(states)


# ───────────────────────────── analytic forward ─────────────────────────────

def _pi_for_states(actor: QuantumActor, states: np.ndarray,
                   c_uenc: float = 1.0, c_bypass: float = 1.0) -> np.ndarray:
    """For each state, compute π(K, nc) using ANALYTIC quantum forward, with:
        o_hat used at head ← c_uenc · o_hat
        z_t used at head   ← c_bypass · z_t
    Returns (N, K, nc) softmax distributions."""
    pis = []
    for s in states:
        a_norm = actor._group_norm(s)
        z_t = actor._dual_encode(a_norm)
        z_t_enc, _, _ = actor._enc_norm(z_t)
        alpha = np.pi * np.tanh(actor.lam_y * z_t_enc[0::2])
        delta = np.pi * np.tanh(actor.lam_z * z_t_enc[1::2])
        o_hat = expectations_analytic(
            alpha, delta, actor.theta_y, actor.theta_z,
            n_var_layers=actor.N_VAR_LAYERS,
            data_reuploading=actor.DATA_REUPLOADING,
        )
        # Apply attenuation patches (probe-only, no retraining)
        logits, _ = actor._head_forward(c_bypass * z_t, c_uenc * o_hat)
        pi_2d = _softmax_2d(logits.reshape(actor.K, actor.n_choices))
        pis.append(pi_2d)
    return np.array(pis)


def _softmax_2d(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ──────────────────────────────── metrics ────────────────────────────────────

def _metrics(pis: np.ndarray, tag: str, verbose: bool = True) -> dict:
    """pis : (N, K, nc). Return dict of M1-M4."""
    N, K, nc = pis.shape
    log_pis = np.log(pis + 1e-12)

    # M1: mean per-user entropy across states
    H_per_user = -(pis * log_pis).sum(axis=-1)             # (N, K)
    H_mean = float(H_per_user.mean())

    # M2: mean pairwise KL per user, averaged over user and (s ≠ s')
    pi_a = pis[:, None, :, :]
    log_pi_a = log_pis[:, None, :, :]
    log_pi_b = log_pis[None, :, :, :]
    kl_pair_u = (pi_a * (log_pi_a - log_pi_b)).sum(axis=-1)  # (N, N, K)
    eye = np.eye(N, dtype=bool)
    kl_pair_u[eye] = 0.0
    n_pairs = N * (N - 1)
    kl_mean = float(kl_pair_u.sum() / (n_pairs * K))
    kl_per_user = (kl_pair_u.sum(axis=(0, 1)) / n_pairs)    # (K,)

    # M3: unique greedy assignments per user
    greedy = pis.argmax(axis=-1)
    uniq_per_user = np.array([len(np.unique(greedy[:, k])) for k in range(K)])
    uniq_mean = float(uniq_per_user.mean())

    # M4: per-state L1 deviation from marginal
    pi_bar = pis.mean(axis=0)
    l1_dev = float(np.abs(pis - pi_bar[None, :, :]).sum(axis=-1).mean())

    if verbose:
        H_max = float(np.log(nc))
        print(f"  [{tag}]")
        print(f"    M1 H̄         : {H_mean:.4f}  ({100*H_mean/H_max:.1f}% of max log {nc})")
        print(f"    M2 pairwise KL : {kl_mean:.4f}   ⭐")
        print(f"        per-user   : " + " ".join(f"{kl:.3f}" for kl in kl_per_user))
        print(f"    M3 unique-grdy : mean={uniq_mean:.2f}  per-user={uniq_per_user.tolist()}")
        print(f"    M4 ‖π−π̄‖_1    : {l1_dev:.4f}")
    return {
        "H_mean": H_mean,
        "kl_mean": kl_mean,
        "kl_per_user": kl_per_user.tolist(),
        "uniq_mean": uniq_mean,
        "l1_dev": l1_dev,
    }


# ───────────────────────────── per-ckpt driver ────────────────────────────────

def run_ckpt(ckpt_path: str, args, c_uenc_list, include_bypass_zero: bool):
    """Run all configs on one ckpt. Returns dict of tag → metrics."""
    print("=" * 76)
    print(f"  CKPT  : {ckpt_path}")
    print("=" * 76)

    cfg = make_config()
    D_k = getattr(P, "D_k_bps_hz", 0.1)
    env = ISTNEnv(cfg=cfg, seed=args.seed,
                  n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    actor = QuantumActor.from_dir(ckpt_path, seed=args.seed)
    actor.n_shots = getattr(P, "n_shots_train", 1500)
    policy = make_checkpoint_policy(ckpt_path, cfg)

    states = _sample_states(env, actor, policy, args, D_k)
    N = len(states)
    print(f"  {N} states · K={actor.K} · n_choices={actor.n_choices} · n_q={len(actor.lam_y)}")
    print(f"  λ_y |max|={float(np.max(np.abs(actor.lam_y))):.4f}  "
          f"λ_z |max|={float(np.max(np.abs(actor.lam_z))):.4f}")

    results = {}
    # Mode 1: o_hat attenuation sweep
    for c in c_uenc_list:
        print(f"\n  --- ATTENUATION c_uenc = {c:.2f} ---")
        pis = _pi_for_states(actor, states, c_uenc=c, c_bypass=1.0)
        results[f"uenc_c={c:.2f}"] = _metrics(pis, f"c_uenc={c:.2f}")

    # Mode 2: bypass-zero (full U_enc, no classical z_t)
    if include_bypass_zero:
        print(f"\n  --- BYPASS-ZERO (full U_enc, z_t := 0 at head) ---")
        pis = _pi_for_states(actor, states, c_uenc=1.0, c_bypass=0.0)
        results["bypass_zero"] = _metrics(pis, "bypass_zero")

    # Summary for this ckpt
    print(f"\n  ──── SUMMARY for {ckpt_path} ────")
    print(f"    config               H̄        M2 KL ⭐    M3 uniq   M4 L1dev")
    for tag, m in results.items():
        print(f"    {tag:<20s} {m['H_mean']:.4f}   {m['kl_mean']:.4f}   "
              f"{m['uniq_mean']:.2f}      {m['l1_dev']:.4f}")
    print()
    return results


# ───────────────────────────────── main ──────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="One or more ckpt dirs (paths to .../agents/).")
    ap.add_argument("--c-uenc-list", default="1.0,0.75,0.5,0.25,0.0",
                    help="Comma-sep c values for o_hat attenuation sweep.")
    ap.add_argument("--include-bypass-zero", action="store_true",
                    help="Also run bypass-zero (z_t := 0) config per ckpt.")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c_uenc_list = [float(c.strip()) for c in args.c_uenc_list.split(",")]

    print("=" * 76)
    print("  U_enc ABLATION (Prop 4 unified probe)")
    print(f"  ckpts={len(args.ckpts)}  c_uenc={c_uenc_list}  "
          f"bypass_zero={'yes' if args.include_bypass_zero else 'no'}")
    print("=" * 76)

    all_results = {}
    for ckpt in args.ckpts:
        all_results[ckpt] = run_ckpt(ckpt, args, c_uenc_list, args.include_bypass_zero)

    # ──────────────── cross-ckpt VARIANCE BAND summary ────────────────────────
    if len(args.ckpts) >= 2:
        print("=" * 76)
        print("  VARIANCE BAND across ckpts  (M2 KL by config)")
        print("=" * 76)
        # collect tags from first ckpt
        first = all_results[args.ckpts[0]]
        tags = list(first.keys())
        print(f"  ckpt                                      " + "   ".join(f"{t:>14s}" for t in tags))
        for ckpt, res in all_results.items():
            name = ckpt.split("/")[-2] if ckpt.endswith("agents") else ckpt.split("/")[-1]
            vals = "   ".join(f"{res[t]['kl_mean']:>14.4f}" for t in tags)
            print(f"  {name:<40s}  {vals}")
        # stats
        print("  " + "-" * 74)
        for t in tags:
            vals = [all_results[ck][t]["kl_mean"] for ck in args.ckpts]
            print(f"    {t:<20s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
                  f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")
        print()

        # ── smoothness check for attenuation sweep ──────────────────────────
        c_tags = [t for t in tags if t.startswith("uenc_c=")]
        if len(c_tags) >= 3:
            print("  SMOOTHNESS CHECK (attenuation sweep, mean across ckpts):")
            cs = [float(t.split("=")[1]) for t in c_tags]
            means = [np.mean([all_results[ck][t]["kl_mean"] for ck in args.ckpts])
                     for t in c_tags]
            for c, m in sorted(zip(cs, means)):
                bar = "█" * int(40 * m / max(means))
                print(f"    c={c:.2f}  KL={m:.4f}  |{bar}")
            # Linearity score: max gap between consecutive c's vs expected linear gap
            sorted_pairs = sorted(zip(cs, means))
            gaps = [sorted_pairs[i+1][1] - sorted_pairs[i][1] for i in range(len(sorted_pairs)-1)]
            expected = (sorted_pairs[-1][1] - sorted_pairs[0][1]) / len(gaps)
            max_gap = max(gaps)
            min_gap = min(gaps)
            ratio = max_gap / (expected + 1e-12)
            print(f"    expected linear gap (Δ_KL / n_steps) = {expected:.4f}")
            print(f"    max gap = {max_gap:.4f}  ({ratio:.2f}× expected)")
            print(f"    min gap = {min_gap:.4f}")
            if 0.5 <= ratio <= 2.0 and min_gap > 0.5 * expected:
                print(f"    → SMOOTH degradation. U_enc contribution is GENUINE continuous "
                      f"signal, not cliff/OOD artefact.")
            else:
                print(f"    → NON-SMOOTH (cliff or saturating). Inspect per-c table — "
                      f"may indicate OOD artefact at c=0 or saturation at c=1.")
        print()

    print("=" * 76)
    print("  [DONE]")
    print("=" * 76)


if __name__ == "__main__":
    main()
