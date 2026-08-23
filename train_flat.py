"""
train_flat.py
-------------
Trainer for the flat-PPO baseline (RL/flat_actor.py — method ported from
Meng et al. 2024). Deliberately a SEPARATE entry point from train.py.

Why separate: train.py exists to train the hierarchical HQC-HAC agent, and most
of its machinery — oracle warm-up, per-head warm-up schedules, the assignment /
phase / C_k auxiliary teachers, per-head PPO ratios, the AE pre-train — is
defined only for a decomposed policy. A flat agent has none of those by
construction. Branching train.py would mean disabling ten mechanisms in ten
places and leaving a reader unsure which ones were actually off; a separate
~300-line loop makes the baseline auditable at a glance.

HELD IDENTICAL to train.py, so the comparison isolates the architecture:
  · environment, cfg, D_k, λ_D  (same make_config, same params.py active case)
  · the reward — env.step returns J = ΣR − λ_D·Σ(shortfall/D_k)², untouched
  · critic class + width (P.critic_hidden), PopArt, LR and its decay schedule
  · GAE(λ) advantages with the same γ, λ and ±adv_clip normalisation
  · rollout size (P.n_rollout_episodes), minibatch size, entropy anneal
  · episode/step budget, seed handling, results/result_N layout,
    hyperparameters.json (so the existing analysis probes read these runs)

DELIBERATELY DIFFERENT — this is what is being measured:
  · ONE network, ONE forward pass, ONE joint clipped PPO ratio over the entire
    action, instead of four heads each with its own ratio
  · no auxiliary teachers, no warm-up, no oracle supervision

Usage
-----
    python3.11 train_flat.py --episodes 20000 --seed 0
K and M come from params.py's ACTIVE CASE, exactly as in train.py.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from RL.flat_actor import FlatActor, MENG_DEFAULTS
from RL.critic import ClassicalCritic
# Same per-step env feature train.py hands its actor. QuantumActor/ClassicalActor
# ignore it (their affinity features already encode link quality), but FlatActor
# reads the raw state and does consume it — so it MUST be computed the same way
# here and in infer.py, or the policy would be evaluated on a state it never saw.
from train import _compute_irs_favored


def _next_run_dir(base: str = "results") -> tuple:
    os.makedirs(base, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base, f"result_{i}")):
        i += 1
    path = os.path.join(base, f"result_{i}")
    os.makedirs(path)
    return path, i


def _save_agents(d: str, fa, critic, cfg) -> None:
    """Snapshot the policy + the topology infer.py needs to rebuild the env.

    train_flat writes the SAME training_config.json train.py does, so infer.py
    and every analysis probe evaluate a flat run through the identical path as
    an HQC-HAC run — the comparison must not differ in how it is measured.
    """
    os.makedirs(d, exist_ok=True)
    fa.save(d)
    critic.save(d)
    with open(os.path.join(d, 'training_config.json'), 'w') as f:
        json.dump({'K': cfg.K, 'M': cfg.M, 'N': cfg.N,
                   'quantization_bits': cfg.quantization_bits,
                   'P_S_dBm': cfg.P_S_dBm,
                   'D_k_bps_hz': cfg.D_k_bps_hz}, f, indent=2)


def _lr_scale(ep: int, n_episodes: int) -> float:
    """The linear decay schedule train.py uses, so the budget is comparable."""
    start = int(P.lr_decay_start * n_episodes)
    if ep < start:
        return 1.0
    prog = (ep - start) / max(1, n_episodes - start)
    return max(P.lr_min_frac, 1.0 - prog)


def _beta_entropy(ep: int, n_episodes: int) -> float:
    end = max(1, int(P.beta_entropy_anneal_end * n_episodes))
    f = min(1.0, ep / end)
    return P.beta_entropy + f * (P.beta_entropy_min - P.beta_entropy)


def rollout(env, fa, critic, demand, n_steps, seed):
    """One episode under the flat policy.

    All four sub-decisions come from ONE fa.forward. Only the C_k shares are
    drawn afterwards, and from the SAME logits — they need the per-group common
    rate that the just-executed power produces, so the draw (not the network
    evaluation) is what is deferred.
    """
    obs = env.reset(seed=seed)
    buf = []
    qos = []
    prev = None                     # Meng's (r_k, r_0) feedback; None at step 0
    for _ in range(n_steps):
        blocked = _compute_irs_favored(env)
        # Meng's combined-channel state term, evaluated at the Φ and routing the
        # env currently has applied — on ĝ, since it is an OBSERVATION.
        h_now = env.rate_computer.effective_channels_all(
            env.assignment, env.Phi, env.channels)
        s = fa.extract_state(obs, demand, blocked, h_eff=h_now, prev_rates=prev)
        act, lp, info = fa.forward(s)
        Phi = env.phase_model.build_phi(
            env.phase_model.index_to_phase(act['phase_idx']))
        part = env.rate_computer.compute_rates_partial(
            act['phi'], Phi, env.channels, info['w_p'], info['w_c_vec'],
            active_irs_ids=act['active_ids'])
        C_k, _, sel, lp_ck = fa.sample_ck(
            info['logits'], act['phi'], part['R_c_group'])
        obs, rew, _, step_info = env.step({'assignment': act['phi'],
                                           'phase_idx':  act['phase_idx'],
                                           'w_p':        info['w_p'],
                                           'w_c_vec':    info['w_c_vec'],
                                           'C_k':        C_k})
        prev = (np.asarray(step_info['R_private']), float(np.mean(C_k)))
        # s_next only feeds the critic's bootstrap, so it must NOT move the
        # running normalisation — that would count every step twice.
        s_next = fa.extract_state(
            obs, demand, _compute_irs_favored(env),
            h_eff=env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels),
            prev_rates=prev, update_norm=False)
        buf.append({'s_t': s, 's_t_next': s_next, 'act': act, 'ck_sel': sel,
                    'lp_old': lp + lp_ck, 'reward': float(rew),
                    'V_t': float(critic.forward(s))})
        R_tot = np.asarray(step_info['R_tot'])
        qos.append(float(np.mean(R_tot >= demand - 1e-12)))
    return buf, qos


def main():
    ap = argparse.ArgumentParser(
        description="Flat-PPO baseline (Meng 2024 method) on the ISTN/RSMA env")
    ap.add_argument('--episodes', type=int,   default=P.n_episodes)
    ap.add_argument('--steps',    type=int,   default=P.n_steps_per_ep)
    ap.add_argument('--gamma',    type=float, default=P.gamma)
    ap.add_argument('--seed',     type=int,   default=P.seed_default)
    # Meng's Table I actor LR is 1e-4. Measured on Case 1 (400 ep, KL-stopped):
    # 1e-4 reached J = −0.09 while 3e-4 reached +0.71 from the same start, so the
    # default here is the TUNED value, not the paper's. Using the weaker of the
    # two would understate the baseline, which is the failure mode this whole
    # comparison has to avoid. --lr 1e-4 reproduces their setting exactly.
    ap.add_argument('--lr',        type=float, default=3e-4,
                    help='actor LR (default 3e-4, tuned; Meng Table I uses 1e-4)')
    ap.add_argument('--lr-critic', type=float, default=P.lr_critic)
    ap.add_argument('--ppo-epsilon', type=float, default=MENG_DEFAULTS['ppo_epsilon'])
    ap.add_argument('--ppo-epochs',  type=int,   default=P.ppo_epochs)
    ap.add_argument('--rollout-episodes', type=int, default=P.n_rollout_episodes)
    ap.add_argument('--hidden', default='512,256',
                    help='flat trunk hidden sizes, comma-separated')
    # A joint log-density over ~250 action dimensions makes the PPO ratio very
    # sensitive: a small per-dimension policy change compounds multiplicatively.
    # Without a guard, ~98% of samples land in the clipped region and have their
    # gradient zeroed (measured on Case 1). KL early-stopping is standard PPO
    # practice, so the baseline gets it and whatever difficulty remains is
    # architectural rather than a missing safeguard. Set 0 to disable.
    ap.add_argument('--target-kl', type=float, default=0.05)
    ap.add_argument('--power-fairness', type=float, default=0.0,
                    help='OFF by default (faithful to Meng, who shape QoS purely '
                         'through the reward). Set >0 for the generous variant '
                         'giving the baseline the same prior HQC-HAC uses.')
    ap.add_argument('--power-priv-frac', type=float, default=0.8)
    ap.add_argument('--checkpoint-interval', type=int, default=1000)
    ap.add_argument('--D-k',      dest='D_k',      type=float, default=None)
    ap.add_argument('--R-LoS-km', dest='R_LoS_km', type=float, default=None)
    args = ap.parse_args()

    cfg_over = {}
    if args.D_k is not None:
        cfg_over['D_k_bps_hz'] = float(args.D_k)
    if args.R_LoS_km is not None:
        cfg_over['R_LoS_km'] = float(args.R_LoS_km)
    cfg = make_config(**cfg_over)
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps,
                  reward_noise_avg=getattr(P, 'reward_noise_avg', 1))
    demand = np.full(cfg.K, cfg.D_k_bps_hz)

    hid = tuple(int(x) for x in args.hidden.split(','))
    fa = FlatActor(cfg, hidden=hid, lr=args.lr, seed=args.seed,
                   n_levels=int(env.n_phase_levels),
                   power_fairness=args.power_fairness,
                   power_priv_frac=args.power_priv_frac)
    critic = ClassicalCritic(
        fa.d_s, d_action=0, hidden=P.critic_hidden, lr=args.lr_critic,
        gamma=args.gamma, seed=args.seed,
        popart=getattr(P, 'popart_enabled', False),
        popart_beta=getattr(P, 'popart_beta', 0.1),
        popart_sigma_floor=getattr(P, 'popart_sigma_floor', 1e-2))

    run_dir, run_id = _next_run_dir()
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    with open(os.path.join(run_dir, 'hyperparameters.json'), 'w') as f:
        json.dump({
            'actor': {'actor_mode': 'flat', 'hidden': list(hid),
                      'd_s': fa.d_s, 'd_out': fa.d_out,
                      'num_params': fa.num_params(),
                      'baseline_of': 'Meng et al. 2024 (method port, not system)'},
            'power_net': {'power_fairness': args.power_fairness,
                          'power_fairness_priv': args.power_fairness,
                          'power_priv_frac': args.power_priv_frac},
            'env': {'K': cfg.K, 'M': cfg.M, 'N': cfg.N,
                    'D_k_bps_hz': cfg.D_k_bps_hz, 'lambda_D': cfg.lambda_D,
                    'R_LoS_km': cfg.R_LoS_km, 'kappa': cfg.kappa,
                    'noise_mean_dBW': cfg.noise_mean_dBW,
                    'P_S_dBm': getattr(cfg, 'P_S_dBm', None)},
            'training': {'n_episodes': args.episodes, 'n_steps_per_ep': args.steps,
                         'gamma': args.gamma, 'gae_lambda': P.gae_lambda,
                         'lr_actor': args.lr, 'lr_critic': args.lr_critic,
                         'ppo_epsilon': args.ppo_epsilon,
                         'ppo_epochs': args.ppo_epochs,
                         'target_kl': args.target_kl,
                         'n_rollout_episodes': args.rollout_episodes,
                         'batch_size': P.batch_size},
            'seed': args.seed,
        }, f, indent=2)

    print("=" * 78)
    print(f"  FLAT-PPO BASELINE (Meng 2024 method)   →  {run_dir}")
    print("=" * 78)
    print(f"  Case      : K={cfg.K}  M={cfg.M}  N={cfg.N}  D_k={cfg.D_k_bps_hz}  "
          f"λ_D={cfg.lambda_D}")
    print(f"  Actor     : {fa.d_s} → {' → '.join(str(h) for h in hid)} → {fa.d_out}"
          f"   ({fa.num_params():,} params — ONE forward for the WHOLE action)")
    print(f"  Critic    : {critic.architecture_str}")
    print(f"  PPO       : lr_a={args.lr}  lr_c={args.lr_critic}  "
          f"eps={args.ppo_epsilon}  epochs≤{args.ppo_epochs}  "
          f"target_kl={args.target_kl}  rollout={args.rollout_episodes}ep")
    print(f"  Budget    : {args.episodes} ep × {args.steps} steps  seed={args.seed}")
    print(f"  fairness  : {args.power_fairness} "
          f"({'faithful port' if args.power_fairness == 0 else 'generous variant'})")
    print("=" * 78, flush=True)

    rng = np.random.default_rng(args.seed + 77)
    log = {'ep': [], 'J': [], 'qos': [], 'clip': [], 'kl': [], 'epochs': []}
    t0 = time.perf_counter()
    best_J = -np.inf
    gl = float(P.gae_lambda)
    aclip = float(getattr(P, 'adv_clip', 0.0) or 0.0)

    for ep0 in range(0, args.episodes, args.rollout_episodes):
        sc = _lr_scale(ep0, args.episodes)
        fa.opt.lr = args.lr * sc
        critic.opt.lr = args.lr_critic * sc
        beta = _beta_entropy(ep0, args.episodes)

        # ── collect ──
        rbuf, Js, Qs = [], [], []
        for _ in range(args.rollout_episodes):
            eb, qos = rollout(env, fa, critic, demand, args.steps,
                              int(rng.integers(1 << 30)))
            # GAE(λ) with bootstrap on the last step — identical to train.py
            gae = 0.0
            V_last = float(critic.forward(eb[-1]['s_t_next'])) if eb else 0.0
            for i in reversed(range(len(eb))):
                V_next = eb[i + 1]['V_t'] if i + 1 < len(eb) else V_last
                delta = eb[i]['reward'] + args.gamma * V_next - eb[i]['V_t']
                gae = delta + args.gamma * gl * gae
                eb[i]['gae_adv'] = float(gae)
                eb[i]['ret'] = float(gae + eb[i]['V_t'])
            rbuf += eb
            Js.append(float(np.mean([t['reward'] for t in eb])))
            Qs.append(float(np.mean(qos)))

        # ── advantage normalisation + clip (same as train.py) ──
        advs = np.array([t['gae_adv'] for t in rbuf])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        if aclip > 0:
            advs = np.clip(advs, -aclip, aclip)
        rets = np.array([t['ret'] for t in rbuf])
        lp_old = np.array([t['lp_old'] for t in rbuf])
        S = [t['s_t'] for t in rbuf]
        A = [t['act'] for t in rbuf]
        C = [t['ck_sel'] for t in rbuf]
        critic.update_popart_stats(rets)

        # ── PPO epochs, KL early-stopped ──
        n, idx = len(rbuf), np.arange(len(rbuf))
        n_done, clip_f, kl_v = 0, 0.0, 0.0
        for _epoch in range(args.ppo_epochs):
            rng.shuffle(idx)
            kls = []
            for st in range(0, n, P.batch_size):
                mb = idx[st:st + P.batch_size]
                _, _, g, clip_f, kl_v = fa.compute_logprobs_grads_batch(
                    [S[i] for i in mb], [A[i] for i in mb], [C[i] for i in mb],
                    lp_old[mb], advs[mb], args.ppo_epsilon, beta_entropy=beta)
                fa.apply_grads(g)
                _, gc = critic.compute_grads_batch([S[i] for i in mb], None, rets[mb])
                critic.apply_grads(gc)
                kls.append(kl_v)
            n_done += 1
            if args.target_kl > 0 and float(np.mean(kls)) > args.target_kl:
                break

        J, qos_m = float(np.mean(Js)), float(np.mean(Qs))
        done = ep0 + args.rollout_episodes
        log['ep'].append(done); log['J'].append(J); log['qos'].append(qos_m)
        log['clip'].append(float(clip_f)); log['kl'].append(float(kl_v))
        log['epochs'].append(n_done)

        if (done // args.rollout_episodes) % 10 == 0 or done >= args.episodes:
            el = time.perf_counter() - t0
            eta = el / max(done, 1) * (args.episodes - done)
            print(f"  ep {done:>6}/{args.episodes}  J={J:+.4f}  QoS={qos_m*100:5.1f}%  "
                  f"clip={clip_f:.3f}  kl={kl_v:.2e}  "
                  f"ep_used={n_done}/{args.ppo_epochs}  "
                  f"[{el/60:.1f}m, ETA {eta/3600:.1f}h]", flush=True)

        if done % args.checkpoint_interval == 0 or done >= args.episodes:
            _save_agents(os.path.join(run_dir, 'checkpoints',
                                      f'ep_{done:05d}', 'agents'), fa, critic, cfg)
        with open(os.path.join(run_dir, 'training_log.json'), 'w') as f:
            json.dump(log, f)
        if J > best_J:
            best_J = J
            _save_agents(os.path.join(run_dir, 'checkpoints', 'best', 'agents'),
                         fa, critic, cfg)
            _save_agents(os.path.join(run_dir, 'agents'), fa, critic, cfg)

    h = np.array(log['J'])
    q = max(1, len(h) // 10)
    print("=" * 78)
    print(f"  done in {(time.perf_counter()-t0)/3600:.2f}h   "
          f"first-10% J={h[:q].mean():+.4f}   last-10% J={h[-q:].mean():+.4f}   "
          f"best={best_J:+.4f}")
    print(f"  → {run_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
