"""
probe_lam_interference.py
-------------------------
Prove WHY λ (quantum encoding scales) is frozen [E]: gradient INTERFERENCE vs VANISHING.

Per-sample λ-gradient g_b = ∂L_b/∂λ over a real rollout mini-batch. Measure the mean
PAIRWISE COSINE cos(g_b, g_b') across the batch:

  mean cos ≈ 0  AND  ||g_b|| healthy  ⇒ INTERFERENCE — per-sample gradients point in
      ~random directions, so the batch-mean (Adam's 1st moment) cancels → λ stalls.
      This is NOT a barren plateau; the gradient is alive, just incoherent.
  ||g_b|| tiny (regardless of cos)     ⇒ VANISHING — depth/qubit barren plateau.

Also reports coherence ρ = ||mean_b g_b|| / mean_b||g_b||  (= the in-train λgrad 'r' but on
per-sample vectors): ρ→1 coherent, ρ→0 incoherent. And the advantage-sign vs direction split.

CPU-light (one analytic batch + critic). Run on a trained checkpoint.

Usage:
  python analysis/probe_lam_interference.py --ckpt results/result_8/checkpoints/ep_00400
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
    ap.add_argument('--episodes', type=int, default=2)
    ap.add_argument('--steps', type=int, default=120)
    ap.add_argument('--seed', type=int, default=20260602)
    ap.add_argument('--gamma', type=float, default=0.95)
    ap.add_argument('--gae', type=float, default=0.9)
    args = ap.parse_args()

    cfg = make_config(); K = cfg.K
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps, reward_noise_avg=8)

    ckpt = args.ckpt
    if os.path.isdir(os.path.join(ckpt, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt, 'actor_config.json')):
        ckpt = os.path.join(ckpt, 'agents')
    from RL import QuantumActor, ClassicalCritic
    actor  = QuantumActor.from_dir(ckpt, seed=0)        # works for v1 (generic+MLP) or arch-2
    actor.n_shots = P.n_shots_train
    critic = ClassicalCritic.from_dir(ckpt) if os.path.isfile(
        os.path.join(ckpt, 'critic_config.json')) else None

    # ── rollout: actor ASSIGNMENT (the λ-relevant action) + minimal downstream
    #    (phase=0, equal power, C_k=0) → real reward for that assignment. We only
    #    need (s_t, phi, advantage); λ affects assignment only, so default power/
    #    phase is a faithful enough reward signal for the encoding-path gradient. ──
    S, PHI, R, V = [], [], [], []
    for ep in range(args.episodes):
        obs = env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            demand  = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, _ = actor.forward(s_t)
            G = len({int(a) for a in phi if a > 0})
            action = {'assignment': phi,
                      'phase_idx': np.zeros((cfg.M, cfg.N), dtype=int),
                      'C_k': np.zeros(K),
                      'w_p': np.ones(K), 'w_c_vec': np.ones(G + 1)}
            S.append(s_t); PHI.append(phi)
            V.append(float(critic.forward(s_t)) if critic is not None else 0.0)
            obs, reward, _, _ = env.step(action)
            R.append(float(reward))

    n = len(S)
    # ── GAE advantages (same as train.py) ──
    V_next = V[1:] + [V[-1]]
    adv = np.zeros(n); gae = 0.0
    for i in reversed(range(n)):
        delta = R[i] + args.gamma * V_next[i] - V[i]
        gae = delta + args.gamma * args.gae * gae
        adv[i] = gae
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)            # per-rollout normalise

    # ── per-sample λ gradients ──
    G = actor.lam_grad_per_sample(S, PHI, adv)               # (n, 2*nq)
    norms = np.linalg.norm(G, axis=1)                        # (n,)
    keep = norms > 1e-12
    G, norms = G[keep], norms[keep]; B = len(G)
    Ghat = G / norms[:, None]
    # mean pairwise cosine = (||Σ ĝ||² − B) / (B(B−1))
    mean_cos = float((np.linalg.norm(Ghat.sum(0)) ** 2 - B) / (B * (B - 1)))
    rho = float(np.linalg.norm(G.mean(0)) / (norms.mean() + 1e-12))   # coherence (≈ in-train r)
    mag = float(norms.mean())

    # advantage-sign contribution: cos split into sign(A) flips vs direction v
    # g_b = A_b · v_b → cos(g_b,g_b') = sign(A_b A_b')·cos(v_b,v_b'). Strip |A| to isolate v.
    A = adv[keep]
    Vdir = G / (A[:, None] + np.sign(A)[:, None] * 1e-9)     # ≈ v_b (remove advantage scale)
    Vn = np.linalg.norm(Vdir, axis=1); kv = Vn > 1e-12
    Vh = Vdir[kv] / Vn[kv][:, None]; Bv = len(Vh)
    mean_cos_dir = float((np.linalg.norm(Vh.sum(0)) ** 2 - Bv) / (Bv * (Bv - 1)))

    print("=" * 72)
    print(f"  λ-GRADIENT INTERFERENCE PROBE · ckpt={args.ckpt}")
    print(f"  {B} samples · 2·nq={G.shape[1]} λ-dims")
    print("=" * 72)
    print(f"  per-sample ||g_b||      : mean={mag:.3e}  (HEALTHY O(0.01-1) → not vanishing)")
    print(f"  mean pairwise cos(g_b,g_b') : {mean_cos:+.4f}   ⭐ KEY")
    print(f"  coherence ρ=||Σg||/Σ||g||   : {rho:.4f}   (≈ in-train λgrad 'r')")
    print(f"  cos of DIRECTION only (strip advantage sign): {mean_cos_dir:+.4f}")
    print("-" * 72)
    if mag < 1e-4:
        print(f"  ❌ VANISHING: ||g_b||={mag:.1e} tiny → barren-plateau (depth/qubit). lr won't help.")
    elif abs(mean_cos) < 0.05:
        print(f"  ✅ INTERFERENCE CONFIRMED: ||g_b|| healthy ({mag:.2e}) but mean cos≈0 ({mean_cos:+.3f})")
        print(f"     → per-sample λ-grads point in ~random directions → batch-mean cancels →")
        print(f"       Adam 1st-moment≈0 → λ FROZEN.  NOT vanishing/barren-plateau.")
        if abs(mean_cos_dir) < 0.05:
            print(f"     Direction-only cos≈0 too → interference is INTRINSIC to encoding-sensitivity")
            print(f"       (data-dependent z_enc), not just advantage-sign flips. → E-fix1 (aux loss)")
            print(f"       or E-fix2 (accept fixed λ); E-fix0 (lr↑) won't fix a directionless gradient.")
        else:
            print(f"     But direction-only cos={mean_cos_dir:+.3f} ≠ 0 → much of the cancellation is")
            print(f"       ADVANTAGE-SIGN flips → a variance-reduction / baseline lever might help.")
    else:
        print(f"  🟡 PARTIALLY COHERENT: mean cos={mean_cos:+.3f}, ρ={rho:.3f} → some consistent")
        print(f"     direction → λ should move slowly; E-fix0 (lr_λ↑) may help.")
    print("=" * 72)


if __name__ == '__main__':
    main()
