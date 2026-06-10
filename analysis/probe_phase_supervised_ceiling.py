"""
probe_phase_supervised_ceiling.py
---------------------------------
SUPERVISED CEILING test for PhaseMLP — distinguishes INFO/CAPACITY ceiling
from RL CREDIT bottleneck on phase optimization.

Setup
-----
1. Load existing PhaseMLP from a ckpt (already RL-trained → baseline alignment).
2. Roll the frozen actor to collect (s_phase, oracle_phase) tuples.
3. Train PhaseMLP via SUPERVISED CE loss against oracle_phase (no RL signal,
   no entropy, no advantage) for many epochs.
4. Re-evaluate alignment after supervised convergence.

Interpretation
--------------
  supervised_alignment ≫ RL_alignment   → CREDIT bottleneck (RL can't extract
    info that's available; supervised proves info+capacity sufficient).
  supervised_alignment ≈ RL_alignment   → INFO/CAPACITY ceiling (even pure
    supervised on the same input can't go higher).

Reuses pattern from train._warmup_phase + probe_phase_quality eval.

Usage:
  python analysis/probe_phase_supervised_ceiling.py \\
    --ckpt results/result_6/checkpoints/ep_01000 \\
    --rollout-episodes 50 --eval-episodes 15 \\
    --supervised-epochs 300
"""
import os, sys
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


def eval_alignment(env, actor, phase_net, cfg, K, N, n_episodes, n_steps, seed, label):
    """Run greedy rollouts and compute mean alignment + IRS-beats-direct%."""
    from train import _build_phase_state, _get_active_irs
    rand_floor = 1.0 / np.sqrt(N)
    q_all, irs_win, irs_tot = [], 0, 0
    for ep in range(n_episodes):
        env.reset(seed=seed + ep)
        for _ in range(n_steps):
            obs = env._get_obs()
            demand = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t, greedy=True)
            active_irs = _get_active_irs(phi)
            if len(active_irs) == 0:
                _advance(env); continue
            s_phase = _build_phase_state(env.channels, phi, cfg, info['z_t'])
            phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=True)
            ch = env.channels
            for m in active_irs:
                angles = env.phase_model.index_to_phase(phase_idx[m])
                q = float(np.abs(np.exp(1j * np.asarray(angles)).sum()) / N)
                q_all.append(q)
                routed = np.where(phi == (m + 1))[0]
                if routed.size:
                    coeff = ch['beta'][m] * np.abs(ch['g_SR_hat'][m]) * (q * N)
                    g_irs = (coeff * np.abs(ch['g_RU_hat'][m, routed])) ** 2
                    g_dir = np.abs(ch['g_SU_hat'][routed]) ** 2
                    irs_win += int(np.sum(g_irs > g_dir)); irs_tot += routed.size
            _advance(env)
    q_arr = np.asarray(q_all) if q_all else np.array([rand_floor])
    a_arr = (q_arr - rand_floor) / (1.0 - rand_floor)
    qm, am = float(q_arr.mean()), float(a_arr.mean())
    beats = (100.0 * irs_win / irs_tot) if irs_tot else 0.0
    print(f"  [{label}] alignment={am*100:5.1f}%  q={qm:.3f}  IRS-beats-direct={beats:.1f}% ({irs_win}/{irs_tot})  N_samples={q_arr.size}")
    return am, qm, beats


def collect_supervised_buffer(env, actor, phase_net, cfg, n_episodes, n_steps, seed):
    """Roll frozen actor, collect (s_phase, oracle_phase_idx) tuples per active IRS."""
    from train import _build_phase_state, _get_active_irs
    from analysis.phase_oracle import oracle_phase_idx
    K = cfg.K
    buf = []
    for ep in range(n_episodes):
        env.reset(seed=seed + 100000 + ep)
        for _ in range(n_steps):
            obs = env._get_obs()
            demand = np.full(K, cfg.D_k_bps_hz)
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, demand, blocked)
            phi, _, info = actor.forward(s_t, greedy=False)
            active_irs = _get_active_irs(phi)
            if active_irs.size > 0:
                s_phase = _build_phase_state(env.channels, phi, cfg, info['z_t'])
                target_idx = oracle_phase_idx(env.channels, phi, cfg)
                buf.append({
                    's_phase':   s_phase.copy(),
                    'phi':       phi.copy(),
                    'phase_idx': target_idx.copy(),
                })
            _advance(env)
    return buf


def supervised_train(phase_net, buf, n_epochs, batch_size, seed):
    """Pure CE training, eff_adv=+1, no entropy. Mirrors train._warmup_phase Stage 2."""
    rng = np.random.default_rng(seed + 91919)
    n_buf = len(buf)
    idx_arr = np.arange(n_buf)
    eff_one = np.ones(batch_size, dtype=float)
    for epoch in range(n_epochs):
        rng.shuffle(idx_arr)
        for start in range(0, n_buf, batch_size):
            mb = idx_arr[start:start + batch_size]
            batch = [buf[i] for i in mb]
            eff_b = eff_one if len(mb) == batch_size else np.ones(len(mb), dtype=float)
            _, _, grads = phase_net.compute_grads_batch(batch, eff_b, beta_entropy=0.0)
            if grads:
                phase_net.apply_grads(grads)
        if (epoch + 1) % 50 == 0 or epoch == n_epochs - 1:
            # quick CE eval on buffer
            ce_sum = 0.0; n_pair = 0; match = 0
            from train import _get_active_irs
            for tr in buf[:min(200, n_buf)]:
                active_irs = _get_active_irs(tr['phi'])
                for m in active_irs:
                    idx_live, _, probs = phase_net._forward_irs(tr['s_phase'][m], greedy=True)
                    tgt = tr['phase_idx'][m]
                    ce_sum -= float(np.log(probs[0, tgt[0]] + 1e-10))
                    match += int(idx_live[0] == tgt[0])
                    n_pair += 1
            print(f"    epoch {epoch+1:4d}/{n_epochs}: CE={ce_sum/max(1,n_pair):.4f}  match={100.0*match/max(1,n_pair):5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_6/checkpoints/ep_01000')
    ap.add_argument('--rollout-episodes', dest='rollout_eps', type=int, default=50,
                    help='episodes to collect supervised buffer (default 50)')
    ap.add_argument('--rollout-steps', type=int, default=200)
    ap.add_argument('--eval-episodes', dest='eval_eps', type=int, default=15)
    ap.add_argument('--eval-steps', type=int, default=200)
    ap.add_argument('--supervised-epochs', dest='sup_epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=20260610)
    args = ap.parse_args()

    cfg = make_config(); K, N = cfg.K, cfg.N
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.eval_steps, reward_noise_avg=1)

    ckpt = args.ckpt
    if os.path.isdir(os.path.join(ckpt, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt, 'actor_config.json')):
        ckpt = os.path.join(ckpt, 'agents')

    from RL import QuantumActor, PhaseMLP
    actor = QuantumActor.from_dir(ckpt, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt, seed=0)
    actor.n_shots = P.n_shots_train

    print("=" * 78)
    print(f"  PHASE SUPERVISED-CEILING PROBE · K={K} M={cfg.M} N={N} R_LoS={cfg.R_LoS_km}km")
    print(f"  ckpt={args.ckpt}")
    print(f"  rollout: {args.rollout_eps}ep×{args.rollout_steps}step → buffer")
    print(f"  supervised: {args.sup_epochs} epochs × CE (eff_adv=+1, β=0)")
    print(f"  eval: {args.eval_eps}ep×{args.eval_steps}step greedy")
    print("=" * 78)

    print("\n── STAGE 1: BEFORE (RL-trained PhaseMLP, current ckpt) ──")
    a_before, q_before, beats_before = eval_alignment(
        env, actor, phase_net, cfg, K, N,
        args.eval_eps, args.eval_steps, args.seed, "BEFORE")

    print("\n── STAGE 2: collecting supervised buffer (frozen actor rollouts) ──")
    buf = collect_supervised_buffer(env, actor, phase_net, cfg,
                                     args.rollout_eps, args.rollout_steps, args.seed)
    print(f"  collected {len(buf)} active-IRS instances")

    print("\n── STAGE 3: supervised CE training on PhaseMLP ──")
    supervised_train(phase_net, buf, args.sup_epochs, P.batch_size, args.seed)

    print("\n── STAGE 4: AFTER (supervised-trained PhaseMLP) ──")
    a_after, q_after, beats_after = eval_alignment(
        env, actor, phase_net, cfg, K, N,
        args.eval_eps, args.eval_steps, args.seed, "AFTER ")

    print("\n" + "=" * 78)
    print("  VERDICT")
    print("=" * 78)
    print(f"  alignment BEFORE (RL-trained):     {a_before*100:5.1f}%")
    print(f"  alignment AFTER  (supervised):     {a_after*100:5.1f}%")
    print(f"  Δ = AFTER − BEFORE:                {(a_after-a_before)*100:+5.1f}pt")
    print(f"  IRS-beats-direct BEFORE: {beats_before:.1f}% → AFTER: {beats_after:.1f}%")
    print("-" * 78)
    delta = (a_after - a_before) * 100
    if delta >= 15.0:
        print(f"  ✅ SUPERVISED ≫ RL ({delta:+.1f}pt) → INFO+CAPACITY SUFFICIENT.")
        print(f"     RL CREDIT BOTTLENECK is the cause of phase plateau.")
        print(f"     → Δ7 (proper credit-fix) IS the right direction; design better Δ7 variant.")
    elif delta >= 5.0:
        print(f"  🟡 SUPERVISED MODEST GAIN ({delta:+.1f}pt) → MIXED.")
        print(f"     Both info-ceiling AND credit have partial contribution.")
        print(f"     → Investigate both arch (capacity/input) and credit (Δ7-full).")
    else:
        print(f"  ❌ SUPERVISED ≈ RL ({delta:+.1f}pt) → INFO/CAPACITY CEILING CONFIRMED.")
        print(f"     Even pure supervised on same input cannot break the alignment plateau.")
        print(f"     → Credit-fix levers (Δ7) WON'T help. Lever pivot: augment PhaseMLP input")
        print(f"       (add h_d / per-user power hint) OR increase PhaseMLP capacity.")
    print("=" * 78)


if __name__ == '__main__':
    main()
