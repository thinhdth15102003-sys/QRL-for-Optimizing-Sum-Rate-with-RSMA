"""
probe_representation_stability.py
---------------------------------
Localize WHICH component drifted when oracle ceiling drops across ckpts.

Motivation (2026-06-05 after EXP-3 V3 verdict): gap-decomp probe showed
oracle ceiling R(q_oracle, ORACLE_φ) dropped −0.088 between r11 ep_01000
baseline and r15 ep_00100 (immediately post-warmup, only 100ep PPO). This
drop CAN'T be the policy's q choice (we're using oracle q) nor PhaseMLP
(we brute-search oracle phase) — so it must be downstream representation
drift in either:
    1. actor's z_t  (encoder weights changed → s_power input degraded)
    2. PowerMLP weights (directly drifted)
    3. CkMLP weights  (within-group split degraded)
    4. AE decoder    (regulariser kept rep on-manifold; may be loss of)

This probe measures, for each ckpt, on a FIXED evaluation set of states:
  • z_t statistics    : mean, std, intra-sample drift vs reference ckpt
  • Layer weight norms : L2 of each weight matrix per net
  • AE reconstruction  : L_ae on fixed states (does encoder still preserve info?)
  • PhaseMLP output   : entropy + KL vs reference (only when --include-phase)
  • PowerMLP output   : split %, common %, private % distributions
  • CkMLP output      : entropy of within-group splits

REFERENCE: first ckpt in --ckpts list is the BASELINE; others measured against it.

Verdict pattern (combined with probe_assignment_decomp ceiling drop):
  z_t drifted significantly + ceiling drops    → actor encoder is the issue
  z_t stable + PowerMLP weights drifted        → PowerMLP is the issue
  All weights stable + ceiling still drops     → check rate model / channel
  z_t INTRA-sample variance dropped            → latent collapse [G]

Usage:
  python analysis/probe_representation_stability.py \
         --ckpts results/result_11/checkpoints/ep_01000 \
                 results/result_16/checkpoints/ep_00000 \
                 results/result_16/checkpoints/ep_00100 \
                 results/result_16/checkpoints/ep_00400 \
         --episodes 4 --steps 20 \
         --out results/result_16/rep_stability.txt
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
from probe_critic_ceiling import make_checkpoint_policy
from probe_assignment_oracle import _load_nets


def _weight_norms(actor, phase_net, power_net, ck_net) -> dict:
    """L2 norms of all named weight matrices across the 4 nets."""
    out = {}

    # Actor (encoder + decoder + AE branches)
    actor_named = {
        'actor.W_irs_1': actor.W_irs_1, 'actor.W_irs_2': actor.W_irs_2,
        'actor.W_u_1':   actor.W_u_1,   'actor.W_u_2':   actor.W_u_2,
        'actor.W_proj':  actor.W_proj,
        'actor.W_d_out': actor.W_d_out,
        'actor.lam_y':   actor.lam_y,   'actor.lam_z':   actor.lam_z,
        'actor.gamma_enc': actor.gamma_enc, 'actor.beta_enc': actor.beta_enc,
    }
    for i, (W, b) in enumerate(zip(actor.W_dec, actor.b_dec)):
        actor_named[f'actor.W_dec_{i}'] = W
    if actor.SOFTMAX_HEAD:
        actor_named['actor.W_sm'] = actor.W_sm
        actor_named['actor.beta_temp'] = actor.beta_temp
    else:
        actor_named['actor.W_p_out'] = actor.W_p_out
        for i, W in enumerate(actor.W_post):
            actor_named[f'actor.W_post_{i}'] = W
    for k, W in actor_named.items():
        out[k] = float(np.linalg.norm(W))

    # Sub-actors (PhaseMLP, PowerMLP, CkMLP all use the same _Adam params dict)
    for tag, net in [('phase', phase_net), ('power', power_net), ('ck', ck_net)]:
        for i, W in enumerate(net.Ws):
            out[f'{tag}.W_{i}'] = float(np.linalg.norm(W))

    return out


def _collect_zt(env, actor, n_ep, n_steps, demand, seed):
    """Gather z_t across n_ep × n_steps states. Returns array (T, n_latent)."""
    z_list = []
    for ep in range(n_ep):
        env.reset(seed=seed + ep)
        for _ in range(n_steps):
            obs = env._get_obs()
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            _, _, info = actor.forward(s_t)
            z_list.append(info['z_t'].copy())
            # Advance mobility (no policy action needed for state collection)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
    return np.array(z_list)


def _ae_recon_loss(actor, states):
    """Mean AE reconstruction L_ae on a fixed set of states (a_norm groupnormed)."""
    losses = []
    for s in states:
        a_norm = actor._group_norm(s)
        z_t = actor._dual_encode(a_norm)
        x = z_t
        for W, b in zip(actor.W_dec, actor.b_dec):
            x = np.maximum(0.0, x @ W + b)
        a_rec = x @ actor.W_d_out + actor.b_d_out
        losses.append(0.5 * float(np.sum((a_rec - a_norm) ** 2)))
    return float(np.mean(losses))


def run_ckpt(ckpt, cfg, args, demand, fixed_states):
    """Measure rep stats for one ckpt. fixed_states: list of s_t arrays."""
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + 2, reward_noise_avg=1)
    actor, (_, phase_net, power_net, ck_net) = _load_nets(ckpt)
    actor.n_shots = P.n_shots_train

    # Weight norms
    wnorms = _weight_norms(actor, phase_net, power_net, ck_net)

    # z_t collection
    z_t_b = _collect_zt(env, actor, args.episodes, args.steps, demand,
                        seed=args.seed * 13)
    zt_mean   = float(z_t_b.mean())
    zt_std    = float(z_t_b.std())
    zt_pcov   = float(np.std(z_t_b.mean(axis=1)))   # spread of per-sample means
    zt_dim_std = z_t_b.std(axis=0)                  # per-dim variance over samples
    zt_collapsed_dims = int((zt_dim_std < 0.05).sum())  # near-dead dims

    # AE recon on fixed states
    ae_loss = _ae_recon_loss(actor, fixed_states)

    return {
        'weight_norms': wnorms,
        'z_t': z_t_b,                       # raw, for cross-ckpt comparison
        'zt_mean': zt_mean, 'zt_std': zt_std,
        'zt_per_sample_mean_std': zt_pcov,
        'zt_collapsed_dims': zt_collapsed_dims,
        'zt_total_dims': len(zt_dim_std),
        'ae_loss': ae_loss,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True,
                    help='First ckpt = REFERENCE baseline; others compared against it.')
    ap.add_argument('--episodes', type=int, default=4)
    ap.add_argument('--steps',    type=int, default=20)
    ap.add_argument('--seed',     type=int, default=20260605)
    ap.add_argument('--out',      default='results/rep_stability.txt')
    args = ap.parse_args()

    cfg = make_config(); D_k = cfg.D_k_bps_hz
    demand = np.full(cfg.K, D_k)

    # Build the FIXED evaluation states from the FIRST ckpt's actor (reference).
    # We use the ref actor only for state extraction (s_t depends solely on env obs +
    # demand + blocked — not actor params). So this is just a state-collection step.
    env_ref = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + 2, reward_noise_avg=1)
    actor_ref, _ = _load_nets(args.ckpts[0])
    fixed_states = []
    for ep in range(args.episodes):
        env_ref.reset(seed=args.seed * 17 + ep)
        for _ in range(args.steps):
            obs = env_ref._get_obs()
            blocked = env_ref.channels['su_blocked'].astype(int)
            fixed_states.append(actor_ref.extract_state(obs, demand, blocked))
            env_ref.user_pos = env_ref._walk_users(env_ref.user_pos)
            env_ref.channels = env_ref.channel_model.update_user_channels(
                env_ref.user_pos, env_ref.irs_pos, env_ref.channels)

    print("=" * 80)
    print(f"  REPRESENTATION-STABILITY  ·  K={cfg.K} M={cfg.M} N={cfg.N}")
    print(f"  fixed eval set: {len(fixed_states)} states · ref ckpt: {os.path.basename(args.ckpts[0])}")
    print("=" * 80)

    results = {}
    for ckpt in args.ckpts:
        print(f"  running {ckpt} ...")
        results[ckpt] = run_ckpt(ckpt, cfg, args, demand, fixed_states)

    # Cross-ckpt comparisons (against first ckpt = reference)
    ref_key = args.ckpts[0]
    ref = results[ref_key]
    L = []
    L.append("=" * 80)
    L.append(f"  REPRESENTATION-STABILITY RESULTS  (ref = {os.path.basename(ref_key)})")
    L.append("=" * 80)

    # Summary table
    L.append(f"\n  ── SUMMARY (z_t stats + AE recon + drift vs reference) ──")
    L.append(f"  {'ckpt':22s} {'zt_mean':>9s} {'zt_std':>8s} {'zt_smplμ_std':>13s} "
             f"{'collapsed':>10s} {'AE_loss':>9s} {'Δz_t':>8s}")
    for ckpt in args.ckpts:
        r = results[ckpt]
        zt_drift_l2 = float(np.linalg.norm(r['z_t'].mean(axis=0) - ref['z_t'].mean(axis=0)))
        L.append(f"  {os.path.basename(ckpt):22s} "
                 f"{r['zt_mean']:9.4f} {r['zt_std']:8.4f} "
                 f"{r['zt_per_sample_mean_std']:13.4f} "
                 f"{r['zt_collapsed_dims']:>3d}/{r['zt_total_dims']:<3d}  "
                 f"{r['ae_loss']:9.4f} {zt_drift_l2:+8.4f}")
    L.append(f"    (Δz_t = ‖z_t_mean(ckpt) − z_t_mean(ref)‖ — large = encoder drifted)")
    L.append(f"    (collapsed = dims where per-sample std < 0.05 — latent collapse [G])")

    # Weight-norm DELTAS vs reference (% change per matrix)
    L.append(f"\n  ── WEIGHT-NORM DRIFT vs reference (largest absolute %Δ per matrix) ──")
    L.append(f"  {'ckpt':22s} " + " ".join(f"{k.split('.')[0]+'.'+k.split('.')[1][:8]:>16s}"
                                              for k in list(ref['weight_norms'].keys())[:6]))
    for ckpt in args.ckpts:
        r = results[ckpt]
        deltas = []
        for k in list(ref['weight_norms'].keys())[:6]:
            ref_v = ref['weight_norms'][k]
            cur_v = r['weight_norms'][k]
            pct = 100 * (cur_v - ref_v) / (abs(ref_v) + 1e-9)
            deltas.append(f"{pct:+15.2f}%")
        L.append(f"  {os.path.basename(ckpt):22s} " + " ".join(deltas))

    # Top-drift matrices across all
    L.append(f"\n  ── TOP 8 most-drifted weight matrices ──")
    drifts = []
    for k in ref['weight_norms']:
        max_pct = 0.0
        for ckpt in args.ckpts[1:]:
            ref_v = ref['weight_norms'][k]
            cur_v = results[ckpt]['weight_norms'][k]
            pct = abs(100 * (cur_v - ref_v) / (abs(ref_v) + 1e-9))
            max_pct = max(max_pct, pct)
        drifts.append((k, max_pct))
    drifts.sort(key=lambda x: -x[1])
    L.append(f"  {'matrix':30s} {'max_|%Δ|':>10s}  per-ckpt %Δ")
    for k, _ in drifts[:8]:
        ref_v = ref['weight_norms'][k]
        line = f"  {k:30s} "
        deltas = []
        for ckpt in args.ckpts:
            cur_v = results[ckpt]['weight_norms'][k]
            pct = 100 * (cur_v - ref_v) / (abs(ref_v) + 1e-9)
            deltas.append(f"{pct:+7.2f}%")
        max_pct = max(abs(float(d.replace('%',''))) for d in deltas)
        L.append(line + f"{max_pct:>10.2f}   " + " ".join(deltas))

    # Verdict heuristics
    L.append(f"\n  ── VERDICT HEURISTICS ──")
    last_ckpt = args.ckpts[-1]
    last = results[last_ckpt]
    zt_drift = float(np.linalg.norm(last['z_t'].mean(axis=0) - ref['z_t'].mean(axis=0)))
    ae_drift = last['ae_loss'] - ref['ae_loss']
    collapsed_jump = last['zt_collapsed_dims'] - ref['zt_collapsed_dims']

    if zt_drift > 0.5:
        L.append(f"  ⚠ z_t mean drifted ‖Δ‖={zt_drift:.3f} (>0.5) → ACTOR ENCODER drifted")
    else:
        L.append(f"  ✓ z_t mean stable ‖Δ‖={zt_drift:.3f}")

    if ae_drift > 0.1:
        L.append(f"  ⚠ AE recon loss INCREASED Δ={ae_drift:+.3f} → encoder lost reconstruction capacity")
    else:
        L.append(f"  ✓ AE recon stable Δ={ae_drift:+.3f}")

    if collapsed_jump > 2:
        L.append(f"  ⚠ z_t collapsed-dim COUNT increased +{collapsed_jump} → LATENT COLLAPSE [G]")
    else:
        L.append(f"  ✓ no collapse trend (Δcollapsed={collapsed_jump:+d})")

    top_drift_name, top_drift_pct = drifts[0]
    L.append(f"  → most-drifted weight matrix: {top_drift_name} ({top_drift_pct:+.1f}%)")
    L.append("=" * 80)

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
