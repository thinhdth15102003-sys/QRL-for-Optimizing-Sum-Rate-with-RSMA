"""
train_oracle_critic.py
----------------------
Tier-0 diagnostic: train an ORACLE critic offline on a heavy-averaged dataset.

Why
---
The aleatoric ceiling (Tier-1 probe) tells us the THEORETICAL maximum explVar
under a given policy.  The ORACLE critic — trained supervisedly on the same
state→return pairs the online critic sees, but with (i) extreme reward
averaging to suppress per-step n0 noise and (ii) plain MSE optimisation (no
PPO confounds) — tells us what an UNCONSTRAINED-by-training-dynamics critic
can actually reach with this architecture + this data.

Gap decomposition:
  ceiling_aleatoric     :  irreducible (env + policy stochasticity)
  oracle_explVar        :  ceiling reachable by the CURRENT architecture
  online_explVar        :  what we actually get during training

  gap(ceiling − oracle) :  architecture / capacity / data coverage limit
  gap(oracle  − online) :  training-dynamics limit (lr, PPO epochs, target choice…)

Outputs
-------
  results/oracle_dataset_K{K}M{M}_R{R}.npz       — collected (s,a,r) traces
  results/oracle_critic_K{K}M{M}_R{R}.npz        — trained oracle weights
  results/oracle_critic_K{K}M{M}_R{R}.txt        — summary report

Usage
-----
    python train_oracle_critic.py
    python train_oracle_critic.py --n_episodes 1000 --reward_noise_avg 128
    python train_oracle_critic.py --skip_collect --dataset results/oracle_dataset_K10M2_R0.20.npz
"""
from __future__ import annotations

import os
import json
import time
import argparse
import numpy as np

import params as P
from params  import make_config
from CSI.env import ISTNEnv
from RL      import ClassicalCritic


# ───────────────────────────────────────────────────────────────────────────────
# Dataset collection (very slow — heavy reward averaging)
# ───────────────────────────────────────────────────────────────────────────────

def _random_action(env, cfg, rng):
    L = env.n_phase_levels
    K, M, N = cfg.K, cfg.M, cfg.N
    assignment = rng.integers(0, M + 1, size=K)
    phase_idx  = rng.integers(0, L, size=(M, N))
    active = sorted(set(int(a) for a in assignment if a > 0))
    G_p1 = len(active) + 1
    w_c_vec = rng.dirichlet(np.ones(G_p1)) * (cfg.P_S * 0.2)
    w_p     = rng.dirichlet(np.ones(K))   * (cfg.P_S * 0.8)
    C_k     = rng.uniform(0.0, 0.5 * cfg.D_k_bps_hz, size=K)
    return {
        'assignment': assignment, 'phase_idx': phase_idx,
        'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k,
    }


def _state_vec(env, cfg):
    """Build the same state vector the live critic sees (actor.extract_state)."""
    # We avoid loading the heavy QuantumActor here — extract_state is a pure
    # function of obs/demand/blocked + cfg.  Reproduce that logic inline.
    obs = env._get_obs()
    g_sr = obs['g_SR']; g_ru = obs['g_RU']; g_su = obs['g_SU']
    M, K = cfg.M, cfg.K
    g_sr_mag = np.abs(g_sr)
    g_ru_mag = np.abs(g_ru)
    a_mat    = (g_sr_mag[:, None] * g_ru_mag).T   # (K, M)
    g_su_mag = np.abs(g_su)
    demand   = np.full(K, cfg.D_k_bps_hz)
    q_m = a_mat.mean(axis=0)
    p_m = g_sr_mag ** 2
    # Layout: [affinity (K*M), demand (K), |g_SU| (K), q_m (M), p_m (M)]
    return np.concatenate([
        a_mat.ravel(), demand, g_su_mag, q_m, p_m
    ])


def collect_oracle_dataset(cfg, n_episodes: int, steps_per_ep: int,
                           reward_noise_avg: int, seed: int,
                           gamma: float) -> dict:
    """
    Run a random policy with HEAVY reward averaging, collecting
    (state, return) pairs for supervised critic training.

    return_t = Σ γ^h r_{t+h}  truncated to episode end (bootstrap = 0).
    """
    rng_act = np.random.default_rng(seed)
    env = ISTNEnv(cfg=cfg, seed=seed, n_steps_ep=steps_per_ep,
                  reward_noise_avg=reward_noise_avg)
    d_state = len(_state_vec(env.reset(), cfg)) if False else None
    # Reset properly
    env.reset(seed=seed)
    d_state = len(_state_vec(env, cfg))

    states  = []
    returns = []
    rewards_per_ep: list = []

    t0 = time.time()
    for ep_i in range(n_episodes):
        env.reset(seed=seed + ep_i)
        ep_s = []
        ep_r = []
        for _step in range(steps_per_ep):
            s = _state_vec(env, cfg)
            a = _random_action(env, cfg, rng_act)
            _, r, _, _ = env.step(a)
            ep_s.append(s)
            ep_r.append(float(r))
        # Compute MC returns (no bootstrap)
        ret = np.zeros(len(ep_r))
        G = 0.0
        for i in reversed(range(len(ep_r))):
            G = ep_r[i] + gamma * G
            ret[i] = G
        states.extend(ep_s)
        returns.extend(ret.tolist())
        rewards_per_ep.append(ep_r)
        if (ep_i + 1) % max(1, n_episodes // 20) == 0:
            elapsed = time.time() - t0
            eta = elapsed * (n_episodes - ep_i - 1) / (ep_i + 1)
            print(f"  collected {ep_i+1}/{n_episodes} ep  "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    return {
        'states'  : np.asarray(states),
        'returns' : np.asarray(returns),
        'd_state' : int(d_state),
        'cfg_K'   : int(cfg.K),
        'cfg_M'   : int(cfg.M),
        'cfg_R_LoS_km': float(cfg.R_LoS_km),
        'reward_noise_avg': int(reward_noise_avg),
        'gamma'   : float(gamma),
        'n_episodes': int(n_episodes),
        'steps_per_ep': int(steps_per_ep),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Supervised oracle training
# ───────────────────────────────────────────────────────────────────────────────

def _clip_grad_dict(g: dict, max_norm: float) -> float:
    """In-place L2 global-norm clip on a grad dict. Returns PRE-clip norm."""
    norm = float(np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values())))
    if max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for k in g:
            g[k] *= scale
    return norm


def train_oracle(dataset: dict, hidden: list,
                 epochs: int = 200, lr: float = 1e-3,
                 batch_size: int = 256, seed: int = 0,
                 val_frac: float = 0.2, normalize_targets: bool = True,
                 grad_clip: float = 0.0) -> dict:
    """
    Train a ClassicalCritic on (states, returns) by plain MSE.  No PPO,
    no target net.  Optionally normalize targets (recommended — V predicts
    z-scored return, then we un-scale for evalulation).

    Parameters
    ----------
    grad_clip : float   L2 grad-norm clip (0 = OFF). Mirrors params.grad_clip_critic
                        so the offline sweep can test the SAME knob that binds live.

    Returns
    -------
    dict with critic, train/val explVar curves, final stats.
    """
    S = dataset['states']
    Y = dataset['returns']
    N = len(S)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    S_tr, Y_tr = S[train_idx], Y[train_idx]
    S_va, Y_va = S[val_idx],   Y[val_idx]

    if normalize_targets:
        mu_y, std_y = float(Y_tr.mean()), float(Y_tr.std() + 1e-8)
        Y_tr_n = (Y_tr - mu_y) / std_y
        Y_va_n = (Y_va - mu_y) / std_y
    else:
        mu_y, std_y = 0.0, 1.0
        Y_tr_n, Y_va_n = Y_tr, Y_va

    critic = ClassicalCritic(
        d_state=int(dataset['d_state']),
        d_action=0, hidden=hidden, lr=lr,
        gamma=float(dataset['gamma']), seed=seed,
    )

    n_tr = len(S_tr)
    history = {'train_loss': [], 'val_ev_norm': [], 'val_ev_raw': []}

    print(f"  Oracle critic: arch {critic.architecture_str}  "
          f"N_train={n_tr} N_val={n_val}  lr={lr}  epochs={epochs}")
    for ep in range(epochs):
        idx = rng.permutation(n_tr)
        losses = []
        for start in range(0, n_tr, batch_size):
            mb = idx[start:start + batch_size]
            s_b = [S_tr[i] for i in mb]
            t_b = Y_tr_n[mb]
            loss, grads = critic.compute_grads_batch(s_b, [None]*len(mb), t_b)
            _clip_grad_dict(grads, grad_clip)
            critic.apply_grads(grads)
            losses.append(loss)
        # Validation explVar — both on z-scored and raw scale
        Vva = np.array([critic.forward(s) for s in S_va])
        ev_norm = 1.0 - np.var(Y_va_n - Vva) / max(np.var(Y_va_n), 1e-12)
        Vva_raw = Vva * std_y + mu_y
        ev_raw  = 1.0 - np.var(Y_va - Vva_raw) / max(np.var(Y_va), 1e-12)
        history['train_loss'].append(float(np.mean(losses)))
        history['val_ev_norm'].append(float(ev_norm))
        history['val_ev_raw'].append(float(ev_raw))
        if (ep + 1) % max(1, epochs // 20) == 0:
            print(f"  [oracle ep {ep+1:4d}/{epochs}] "
                  f"train_loss={np.mean(losses):.4f}  "
                  f"val_EV_norm={ev_norm:+.3f}  val_EV_raw={ev_raw:+.3f}")

    return {
        'critic'   : critic,
        'history'  : history,
        'mu_y'     : mu_y,
        'std_y'    : std_y,
        'final_ev_raw'  : history['val_ev_raw'][-1],
        'final_ev_norm' : history['val_ev_norm'][-1],
        'best_ev_raw'   : float(max(history['val_ev_raw'])),
        'best_ev_norm'  : float(max(history['val_ev_norm'])),
        'd_state'  : int(dataset['d_state']),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Report
# ───────────────────────────────────────────────────────────────────────────────

def write_report(dataset: dict, result: dict, out_path: str) -> None:
    h = result['history']
    L = []
    L.append("=" * 78)
    L.append("  ORACLE CRITIC TRAINING  ·  Tier-0 diagnostic")
    L.append("=" * 78)
    L.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("")
    L.append("CONFIG")
    L.append("-" * 78)
    L.append(f"  Case: K={dataset['cfg_K']} M={dataset['cfg_M']}  "
             f"R_LoS_km={dataset['cfg_R_LoS_km']}")
    L.append(f"  Dataset: {dataset['n_episodes']} episodes × "
             f"{dataset['steps_per_ep']} steps "
             f"(reward_noise_avg={dataset['reward_noise_avg']}, "
             f"γ={dataset['gamma']})")
    L.append(f"  Critic arch: {result['critic'].architecture_str}")
    L.append("")
    L.append("RESULT")
    L.append("-" * 78)
    L.append(f"  final  val_EV (raw)  = {h['val_ev_raw'][-1]:+.4f}")
    L.append(f"  final  val_EV (norm) = {h['val_ev_norm'][-1]:+.4f}")
    L.append(f"  best   val_EV (raw)  = {result['best_ev_raw']:+.4f}")
    L.append(f"  best   val_EV (norm) = {result['best_ev_norm']:+.4f}")
    L.append("")
    L.append("INTERPRETATION GUIDE")
    L.append("-" * 78)
    L.append("  Compare with:")
    L.append("    (a) Aleatoric ceiling from probe_critic_ceiling.py for the SAME R_LoS")
    L.append("    (b) Online critic explVar from training_log.txt (rolling mean)")
    L.append("")
    L.append("  oracle_EV − online_EV  → gain achievable by fixing TRAINING DYNAMICS")
    L.append("                            (lr, PPO epochs, target choice, grad clip).")
    L.append("  ceiling   − oracle_EV  → gain achievable by changing ARCHITECTURE")
    L.append("                            or collecting more diverse data.")
    L.append("  If oracle_EV ≈ ceiling → current arch / data is near-optimal; only")
    L.append("                            training dynamics can improve.")
    L.append("  If oracle_EV << ceiling → arch / capacity is the bottleneck.")
    L.append("")
    L.append("TRAINING CURVE (val_EV_raw, last 10 logged points)")
    L.append("-" * 78)
    step = max(1, len(h['val_ev_raw']) // 10)
    for i in range(0, len(h['val_ev_raw']), step):
        L.append(f"  ep {i:4d}  loss={h['train_loss'][i]:.4f}  "
                 f"val_EV_norm={h['val_ev_norm'][i]:+.3f}  "
                 f"val_EV_raw={h['val_ev_raw'][i]:+.3f}")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier-0 diagnostic: train oracle critic offline.")
    parser.add_argument('--r_los', type=float, default=None,
                        help="R_LoS_km override; default from params.py.")
    parser.add_argument('--n_episodes', type=int, default=200,
                        help="Number of episodes to collect.")
    parser.add_argument('--steps_per_ep', type=int, default=50)
    parser.add_argument('--reward_noise_avg', type=int, default=128,
                        help="Heavy averaging to kill per-step n0 noise. "
                             "Use 256+ for very clean dataset (slower).")
    parser.add_argument('--gamma', type=float,
                        default=getattr(P, 'gamma', 0.95))
    parser.add_argument('--hidden', type=int, nargs='+',
                        default=None,
                        help="Critic hidden sizes; default = P.critic_hidden")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=20260531)
    parser.add_argument('--dataset', type=str, default=None,
                        help="Path to existing dataset npz; "
                             "skip collection if provided.")
    parser.add_argument('--skip_collect', action='store_true',
                        help="Requires --dataset; train only.")
    parser.add_argument('--out_dir', type=str, default='results')
    args = parser.parse_args()

    overrides = {}
    if args.r_los is not None:
        overrides['R_LoS_km'] = args.r_los
    cfg = make_config(**overrides)
    case_tag = f"K{cfg.K}M{cfg.M}_R{cfg.R_LoS_km:.2f}"

    os.makedirs(args.out_dir, exist_ok=True)
    ds_path = (args.dataset
               or os.path.join(args.out_dir, f"oracle_dataset_{case_tag}.npz"))

    if args.skip_collect or (args.dataset and os.path.isfile(args.dataset)):
        print(f"Loading dataset from {ds_path} ...")
        d = np.load(ds_path, allow_pickle=False)
        dataset = {k: d[k].item() if d[k].shape == () else d[k]
                   for k in d.files}
    else:
        print(f"Collecting dataset → {ds_path}")
        dataset = collect_oracle_dataset(
            cfg, n_episodes=args.n_episodes,
            steps_per_ep=args.steps_per_ep,
            reward_noise_avg=args.reward_noise_avg,
            seed=args.seed, gamma=args.gamma)
        np.savez(ds_path, **{k: np.asarray(v) for k, v in dataset.items()})
        print(f"  saved {len(dataset['states'])} samples")

    hidden = args.hidden if args.hidden is not None else list(P.critic_hidden)
    result = train_oracle(dataset, hidden=hidden,
                          epochs=args.epochs, lr=args.lr,
                          batch_size=args.batch_size, seed=args.seed)

    weights_path = os.path.join(args.out_dir, f"oracle_critic_{case_tag}.npz")
    np.savez(weights_path, **result['critic'].get_params(),
             mu_y=result['mu_y'], std_y=result['std_y'])
    print(f"\nOracle weights → {weights_path}")

    rep_path = os.path.join(args.out_dir, f"oracle_critic_{case_tag}.txt")
    write_report(dataset, result, rep_path)
    print(f"Oracle report  → {rep_path}")


if __name__ == '__main__':
    main()
