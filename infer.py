"""
infer.py
--------
Inference / evaluation script for a trained HQC-HAC agent.

Loads a trained agent from results/result_N/agents/ and benchmarks it
against four classical baselines over a fixed number of episodes.

Usage
-----
    python infer.py results/result_5
    python infer.py results/result_5 --episodes 100
    python infer.py results/result_5 --seed 42
    python infer.py results/result_5 --stochastic   # sample-based, not greedy
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure the console accepts UTF-8 box-drawing characters on Windows.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import params as P
from params     import make_config
from CSI.env    import ISTNEnv
from CSI.baselines import RandomPolicy, GreedyPolicy, DirectOnlyPolicy, AllIRSPolicy
from RL         import QuantumActor, ClassicalActor, PhaseMLP, PowerMLP, CkMLP


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline helpers  (mirror of train.py — kept local to avoid circular imports)
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase_state(channels: dict, phi: np.ndarray, cfg,
                       z_t: np.ndarray) -> np.ndarray:
    """Build (M, 2K + n_latent + K) per-IRS state for PhaseMLP (mirror of train.py).
    row m = [Re(c^SRU_m)(K), Im(c^SRU_m)(K), z_t(n_latent), phi_mask_m(K)]."""
    M, K     = cfg.M, cfg.K
    n_latent = len(z_t)
    s        = np.zeros((M, 2 * K + n_latent + K))
    for k in range(K):
        m = int(phi[k]) - 1
        if m >= 0:
            c_mk = channels['g_SR_hat'][m].conj() * channels['g_RU_hat'][m, k]
            s[m, k]                    += c_mk.real
            s[m, K + k]                += c_mk.imag
            s[m, 2 * K + n_latent + k]  = 1.0   # phi_mask: user k belongs to IRS m
    s[:, 2 * K : 2 * K + n_latent] = z_t[None, :]   # broadcast z_t to all IRS rows
    return s


def _get_active_irs(phi: np.ndarray) -> np.ndarray:
    """Sorted 0-based IRS indices with ≥1 assigned user (for PhaseMLP)."""
    return np.array(
        sorted({int(phi[k]) - 1 for k in range(len(phi)) if phi[k] > 0}),
        dtype=int,
    )


def _get_active_irs_ids(phi: np.ndarray) -> list:
    """Sorted 1-based physical IRS gids with ≥1 assigned user (for PowerMLP/rate)."""
    return sorted(set(int(phi[k]) for k in range(len(phi)) if phi[k] > 0))


def _build_ck_state(demand: np.ndarray, R_private: np.ndarray,
                    R_c_group: dict, phi: np.ndarray, cfg) -> np.ndarray:
    """Build (5·K,) [D_k, R_p_k, R_c_g_k, shortfall_k, phi_float_k] for CkMLP (mirror of train.py)."""
    K     = cfg.K
    R_c_g = np.zeros(K)
    for k in range(K):
        R_c_g[k] = float(R_c_group.get(int(phi[k]), 0.0))
    shortfall = np.maximum(0.0, demand - R_private)
    phi_float = phi[:K].astype(float)
    return np.concatenate([demand, R_private, R_c_g, shortfall, phi_float])


def _compute_blocked(env: ISTNEnv) -> np.ndarray:
    """Per-user indicator: True when best IRS path > direct link (estimated CSI)."""
    ch       = env.channels
    h_direct = np.abs(ch['g_SU_hat'])
    h_irs    = np.array([
        ch['beta'][m] * np.abs(np.sum(
            ch['g_SR_hat'][m].conj() * np.diag(env.Phi[m])[:, np.newaxis]
            * ch['g_RU_hat'][m],
            axis=0))
        for m in range(env.cfg.M)
    ])
    return h_irs.max(axis=0) > h_direct


# ══════════════════════════════════════════════════════════════════════════════
# Agent loading
# ══════════════════════════════════════════════════════════════════════════════

def load_training_cfg(run_dir: str, kappa=None, noise_var_dBW=None):
    """
    Build a SystemConfig that exactly matches the training-time topology.

    Reads agents/training_config.json (written by train.py) to recover K, M,
    N, quantization_bits, P_S_dBm, and D_k_bps_hz, then merges with remaining
    params.py defaults.  This ensures the env and agents are always consistent
    with the saved weights, regardless of what params.py currently says.

    kappa / noise_var_dBW: optional TEST-TIME overrides for the R4 noise-robustness
    sweep — evaluate a FIXED trained policy under elevated CSI error (kappa) or
    receiver noise (noise_var_dBW) WITHOUT retraining (zero-shot robustness).
    """
    topo_path = os.path.join(run_dir, 'agents', 'training_config.json')
    if not os.path.isfile(topo_path):
        raise FileNotFoundError(
            f"training_config.json not found in {run_dir}/agents/. "
            "Re-train with the current train.py to generate it.")
    with open(topo_path) as f:
        t = json.load(f)
    ov = dict(
        K=t['K'],
        M=t['M'],
        N=t['N'],
        quantization_bits=t['quantization_bits'],
        P_S_dBm=t['P_S_dBm'],
        D_k_bps_hz=t['D_k_bps_hz'],
    )
    if kappa is not None:          ov['kappa']         = float(kappa)
    if noise_var_dBW is not None:  ov['noise_var_dBW'] = float(noise_var_dBW)
    return make_config(**ov)


def load_agents(run_dir: str, seed: int = None):
    """
    Load all four trained actor networks from run_dir/agents/.

    Each network class reconstructs its own architecture from the saved
    *_config.json — no cfg argument needed.  QuantumActor.from_dir() uses
    saved B (=M) and K so it is immune to params.py changes.
    """
    agents_dir = os.path.join(run_dir, 'agents')
    if not os.path.isdir(agents_dir):
        raise FileNotFoundError(
            f"No agents/ directory found in {run_dir}. "
            f"Run training first with train.py.")
    # Detect actor type from saved config: the classical baseline
    # (--actor-mode classical) saves {"mode": "classical"} and must load via
    # ClassicalActor — QuantumActor.from_dir would KeyError on 'n_qubits'.
    _ac_path = os.path.join(agents_dir, 'actor_config.json')
    _amode = 'quantum'
    if os.path.isfile(_ac_path):
        try:
            _amode = json.load(open(_ac_path)).get('mode', 'quantum')
        except Exception:
            _amode = 'quantum'
    if _amode == 'classical':
        actor = ClassicalActor.from_dir(agents_dir, seed=seed)
    else:
        actor = QuantumActor.from_dir(agents_dir, seed=seed)
    phase_net = PhaseMLP.from_dir(agents_dir, seed=seed)
    power_net = PowerMLP.from_dir(agents_dir, seed=seed)
    ck_net    = CkMLP.from_dir(agents_dir, seed=seed)
    # power_fairness α is part of the trained policy's action mapping (blends
    # applied w_p toward equal-split) but is NOT saved in power_config.json.
    # Restore it from the run's hyperparameters.json (walk up from agents/);
    # without it the agent reverts to α=0 (concentrated power) → wrong QoS.
    _af, _d = 0.0, os.path.abspath(agents_dir)
    for _ in range(5):
        _d = os.path.dirname(_d)
        _hp = os.path.join(_d, 'hyperparameters.json')
        if os.path.isfile(_hp):
            try:
                _af = float(json.load(open(_hp)).get('power_net', {}).get('power_fairness', 0.0))
            except Exception:
                _af = 0.0
            break
    power_net.power_fairness = _af
    if _af > 0:
        print(f"  ⚙ power_fairness α={_af:.2f} restored from hyperparameters.json")
    return actor, phase_net, power_net, ck_net


# ══════════════════════════════════════════════════════════════════════════════
# Episode runners
# ══════════════════════════════════════════════════════════════════════════════

def run_hqchac_episode(env: ISTNEnv,
                       actor, phase_net, power_net, ck_net,
                       cfg, demand: np.ndarray,
                       n_steps: int,
                       greedy: bool = True) -> dict:
    """Run one episode with the HQC-HAC agent. Returns per-episode metrics."""
    obs = env.reset()

    ep_reward   = 0.0
    ep_sum_rate = []
    ep_feasible = []
    ep_qos_ok   = []
    ep_irs_frac = []

    for _ in range(n_steps):
        blocked = _compute_blocked(env)
        s_t     = actor.extract_state(obs, demand, blocked)
        phi, _, actor_info = actor.forward(s_t, greedy=greedy)
        z_t = actor_info['z_t']                       # arch-2 spatial latent (shared by phase/power)
        # [--no-ae] sub-actors consume the quantum readout o_hat instead of the AE
        # latent z_t (mirror of train.py); classical/AE actors keep z_t.
        rep = actor_info.get('o_hat', z_t) if getattr(actor, 'NO_AE', False) else z_t

        active_irs     = _get_active_irs(phi)
        active_irs_ids = _get_active_irs_ids(phi)

        s_phase   = _build_phase_state(env.channels, phi, cfg, rep)
        phase_idx, _, _ = phase_net.forward(s_phase, active_irs, greedy=greedy)
        phases_rad   = env.phase_model.index_to_phase(phase_idx)
        proposed_Phi = env.phase_model.build_phi(phases_rad)

        h_eff   = env.rate_computer.effective_channels_all(phi, proposed_Phi, env.channels)
        s_power = np.concatenate([h_eff.real, h_eff.imag, rep])
        w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)

        partial = env.rate_computer.compute_rates_partial(
            phi, proposed_Phi, env.channels, w_p, w_c_vec,
            active_irs_ids=active_irs_ids)
        s_ck = _build_ck_state(demand, partial['R_private'], partial['R_c_group'], phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])

        action = {
            'assignment': phi,
            'phase_idx':  phase_idx,
            'w_p':        w_p,
            'w_c_vec':    w_c_vec,
            'C_k':        C_k,
        }
        obs, reward, _, info = env.step(action)

        ep_reward   += reward
        ep_sum_rate.append(float(info['sum_rate']))
        ep_feasible.append(bool(info['feasible']))
        ep_qos_ok.append(int(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)))
        ep_irs_frac.append(float(np.mean(phi > 0)))

    return {
        'reward':      ep_reward,
        'sum_rate':    float(np.mean(ep_sum_rate)),
        'feasibility': float(np.mean(ep_feasible)),
        'qos_frac':    float(np.mean(ep_qos_ok)) / cfg.K,
        'irs_frac':    float(np.mean(ep_irs_frac)),
    }


def run_baseline_episode(env: ISTNEnv, policy, policy_name: str,
                         cfg, n_steps: int) -> dict:
    """Run one episode with a classical baseline policy."""
    obs = env.reset()

    ep_reward   = 0.0
    ep_sum_rate = []
    ep_feasible = []
    ep_qos_ok   = []

    for _ in range(n_steps):
        if policy_name == 'greedy':
            action = policy.act(obs, env)
        else:
            action = policy.act(obs)

        obs, reward, _, info = env.step(action)
        ep_reward   += reward
        ep_sum_rate.append(float(info['sum_rate']))
        ep_feasible.append(bool(info['feasible']))
        ep_qos_ok.append(int(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)))

    return {
        'reward':      ep_reward,
        'sum_rate':    float(np.mean(ep_sum_rate)),
        'feasibility': float(np.mean(ep_feasible)),
        'qos_frac':    float(np.mean(ep_qos_ok)) / cfg.K,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(run_dir: str, n_episodes: int, seed: int,
             greedy: bool, n_steps: int,
             kappa=None, noise_var_dBW=None) -> dict:

    # Rebuild cfg from saved topology — guarantees K/M/N/bits match the weights.
    # kappa/noise_var_dBW = optional R4 test-time noise overrides (no retrain).
    cfg    = load_training_cfg(run_dir, kappa=kappa, noise_var_dBW=noise_var_dBW)
    if kappa is not None or noise_var_dBW is not None:
        print(f"  ⚙ R4 noise override: kappa={cfg.kappa} noise_var_dBW={cfg.noise_var_dBW}")
    env    = ISTNEnv(cfg, seed=seed, n_steps_ep=n_steps)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)

    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=seed)

    rng = np.random.default_rng(seed)
    policies = {
        'HQC-HAC':    None,
        'Greedy':     GreedyPolicy(cfg),
        'AllIRS':     AllIRSPolicy(cfg),
        'DirectOnly': DirectOnlyPolicy(cfg),
        'Random':     RandomPolicy(cfg, rng=rng),
    }

    results = {name: {'reward': [], 'sum_rate': [], 'feasibility': [], 'qos_frac': []}
               for name in policies}

    W = 100
    print(f"\n{'═'*W}")
    print(f"  HQC-HAC Evaluation  ·  {run_dir}")
    print(f"  {'─'*W}")
    print(f"  Episodes : {n_episodes}  |  Steps/ep : {n_steps}  |  "
          f"Mode : {'greedy' if greedy else 'stochastic'}  |  Seed : {seed}")
    print(f"{'═'*W}")
    print(f"  {'Episode':>8}  {'HQC-HAC Reward':>16}  {'HQC-HAC Rate':>14}", flush=True)
    print(f"  {'─'*8}  {'─'*16}  {'─'*14}")

    for ep in range(n_episodes):
        r = run_hqchac_episode(env, actor, phase_net, power_net, ck_net,
                               cfg, demand, n_steps, greedy=greedy)
        for k, v in r.items():
            if k in results['HQC-HAC']:
                results['HQC-HAC'][k].append(v)

        for name, policy in policies.items():
            if name == 'HQC-HAC':
                continue
            r_bl = run_baseline_episode(env, policy, name.lower(), cfg, n_steps)
            for k, v in r_bl.items():
                results[name][k].append(v)

        if (ep + 1) % max(1, n_episodes // 10) == 0:
            hqc_r  = results['HQC-HAC']['reward']
            hqc_sr = results['HQC-HAC']['sum_rate']
            print(f"  {ep+1:>8}  {np.mean(hqc_r):>16.3f}  {np.mean(hqc_sr):>14.4f}",
                  flush=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═'*W}")
    print(f"  {'Policy':<14}  {'Reward':>10}  {'ΣRate(bps/Hz)':>14}  "
          f"{'Feasibility':>12}  {'QoS frac':>10}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*14}  {'─'*12}  {'─'*10}")

    order = ['HQC-HAC', 'Greedy', 'AllIRS', 'DirectOnly', 'Random']
    for name in order:
        r = results[name]
        rew  = float(np.mean(r['reward']))
        rate = float(np.mean(r['sum_rate']))
        feas = float(np.mean(r['feasibility'])) * 100
        qos  = float(np.mean(r['qos_frac']))  * 100
        print(f"  {name:<14}  {rew:>10.3f}  {rate:>14.4f}  {feas:>11.1f}%  {qos:>9.1f}%")

    print(f"{'═'*W}\n")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained HQC-HAC agent vs baselines.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('run_dir', type=str,
                        help='Path to results/result_N/ directory')
    parser.add_argument('--episodes', type=int, default=P.n_eval_episodes,
                        help='Number of evaluation episodes')
    parser.add_argument('--steps',    type=int, default=P.n_steps_per_ep,
                        help='Environment steps per episode')
    parser.add_argument('--seed',     type=int, default=P.seed_eval,
                        help='Evaluation seed')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic (sampled) policy instead of greedy')
    parser.add_argument('--kappa', type=float, default=None,
                        help='R4 test-time override: CSI error coefficient κ (default = training '
                             'value 0.05). Sweep up (0.1/0.2/...) for noise-robustness eval (no retrain).')
    parser.add_argument('--noise-var', dest='noise_var', type=float, default=None,
                        help='R4 test-time override: receiver noise variance noise_var_dBW (default '
                             '= training 10.0). Sweep up for SNR-robustness eval (no retrain).')
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    evaluate(
        run_dir    = args.run_dir,
        n_episodes = args.episodes,
        seed       = args.seed,
        greedy     = not args.stochastic,
        n_steps    = args.steps,
        kappa         = args.kappa,
        noise_var_dBW = args.noise_var,
    )
