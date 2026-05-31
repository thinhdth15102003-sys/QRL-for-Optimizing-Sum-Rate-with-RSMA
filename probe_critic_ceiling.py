"""
probe_critic_ceiling.py
-----------------------
Tier-1 diagnostic for the IRS-RSMA Quantum-RL critic.

Goal
----
Quantify the ALEATORIC ceiling of explained-variance (explVar) for V(s) under
the CURRENT params.py ACTIVE CASE.  Critic explVar can never exceed

    1 − E[ Var(return | s, π) ] / Var(return)

regardless of architecture or optimisation.  If observed explVar (~0.4 in
result_5) is close to this ceiling, NO critic-side fix will help — env is the
bottleneck.  If observed explVar << ceiling, training-dynamics or pipeline
choices are the bottleneck.

Outputs
-------
  results/probe_critic_ceiling_K{K}M{M}.txt   (human-readable summary)
  results/probe_critic_ceiling_K{K}M{M}.json  (machine-readable raw numbers)

Sections of the report
----------------------
  A. Aleatoric vs total variance summary  → headline ceiling estimate
  B. Per-source ablation                  → which env noise contributes most
  C. Cross-R_LoS comparison               → does ceiling shift with curriculum?
  D. Policy comparison                    → random vs AE-pretrained vs ckpt
  E. Action-conditional variance          → would Q(s,a) buy anything over V(s)?

Usage
-----
    python probe_critic_ceiling.py
    python probe_critic_ceiling.py --ckpt results/result_5/checkpoints/ep_01000
    python probe_critic_ceiling.py --r_los_list 0.2 0.3 0.5
    python probe_critic_ceiling.py --quick    # smaller sample sizes for sanity
"""
from __future__ import annotations

import os
import json
import time
import argparse
import copy
from collections import OrderedDict
from typing import Callable, Optional

import numpy as np

import params as P
from params  import make_config
from CSI.env import ISTNEnv
from CSI.env_probe import (
    snapshot_env_state, restore_env_state,
    step_with_source_control,
    reward_std_intra_state, return_std_intra_state,
)

# Sub-actors imported lazily only when --ckpt is used (avoid heavy GPU init for
# the random-policy probe).


# ───────────────────────────────────────────────────────────────────────────────
# Policy adapters — every policy is callable(env) → action_dict
# ───────────────────────────────────────────────────────────────────────────────

def make_random_policy(cfg, rng_seed: int = 0) -> Callable:
    """Uniform random policy: sample each action component independently."""
    rng = np.random.default_rng(rng_seed)
    N_LEVELS = 4  # phase quantisation (matches IRSPhaseModel default for n_bits=2)

    def policy(env) -> dict:
        K, M, N = cfg.K, cfg.M, cfg.N
        # assignment: K integers in {0,...,M}
        assignment = rng.integers(0, M + 1, size=K)
        # phase_idx: (M, N) in {0,..,L-1}.  Use env.n_phase_levels to be exact.
        L = env.n_phase_levels
        phase_idx = rng.integers(0, L, size=(M, N))
        # Power: Dirichlet split → sums approx to 1, scaled by P_S
        active = sorted(set(int(a) for a in assignment if a > 0))
        G_p1 = len(active) + 1
        w_c_vec = rng.dirichlet(np.ones(G_p1)) * (cfg.P_S * 0.2)
        w_p     = rng.dirichlet(np.ones(K))   * (cfg.P_S * 0.8)
        # C_k uniform in [0, 0.5*D_k_bps_hz]
        C_k = rng.uniform(0.0, 0.5 * cfg.D_k_bps_hz, size=K)
        return {
            'assignment': assignment, 'phase_idx': phase_idx,
            'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k,
        }
    return policy


def make_checkpoint_policy(ckpt_dir: str, cfg) -> Callable:
    """Load a trained checkpoint and return its forward-pass policy.

    Auto-detect whether the user passed the parent dir (containing agents/)
    or the agents/ subdir directly — both should work.

    Uses train.py's helpers (_build_phase_state / _build_ck_state) directly
    so that state construction matches the live training pipeline byte-for-byte.
    """
    if os.path.isdir(os.path.join(ckpt_dir, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt_dir, 'actor_config.json')):
        ckpt_dir = os.path.join(ckpt_dir, 'agents')

    from RL import QuantumActor, PhaseMLP, PowerMLP, CkMLP
    from train import (_build_phase_state, _build_ck_state,
                       _get_active_irs)

    actor     = QuantumActor.from_dir(ckpt_dir, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt_dir, seed=0)
    power_net = PowerMLP.from_dir(ckpt_dir, seed=0)
    ck_net    = CkMLP.from_dir(ckpt_dir, seed=0)
    actor.n_shots = P.n_shots_train

    def policy(env) -> dict:
        K = cfg.K
        obs     = env._get_obs()
        demand  = np.full(K, cfg.D_k_bps_hz)
        blocked = env.channels['su_blocked'].astype(int)
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, info = actor.forward(s_t)
        z_t = info['z_t']

        active_irs     = _get_active_irs(phi)            # 0-based
        active_irs_ids = [int(a) + 1 for a in active_irs]

        s_phase = _build_phase_state(env.channels, phi, cfg, z_t)
        phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
        phases_rad   = env.phase_model.index_to_phase(phase_idx)
        proposed_Phi = env.phase_model.build_phi(phases_rad)

        h_eff   = env.rate_computer.effective_channels_all(
                      phi, proposed_Phi, env.channels)
        s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
        w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)

        partial = env.rate_computer.compute_rates_partial(
            phi, proposed_Phi, env.channels, w_p, w_c_vec,
            active_irs_ids=active_irs_ids)
        s_ck = _build_ck_state(demand, partial['R_private'],
                               partial['R_c_group'], phi, cfg)
        C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
        return {
            'assignment': phi, 'phase_idx': phase_idx,
            'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k,
        }
    return policy


# ───────────────────────────────────────────────────────────────────────────────
# Dataset collection — run policy under live env to harvest (state, action) pairs
# ───────────────────────────────────────────────────────────────────────────────

def collect_states(env, policy: Callable,
                   n_episodes: int = 20,
                   steps_per_ep: int = 50,
                   warmup_steps: int = 5) -> list:
    """
    Run policy for n_episodes × steps_per_ep steps, recording at every step the
    env state SNAPSHOT and the action that was applied.

    Returns
    -------
    list of dicts: { 'snap', 'action', 'reward', 'R_tot', 'sigma2',
                     'episode', 'step' }
    """
    records = []
    for ep_idx in range(n_episodes):
        env.reset(seed=int(time.time_ns()) & 0xFFFF + ep_idx)
        # discard the first few steps so distributions are mixed
        for _ in range(warmup_steps):
            a = policy(env)
            env.step(a)
        for step_idx in range(steps_per_ep):
            snap = snapshot_env_state(env)
            a    = policy(env)
            _, r, _, info = env.step(a)
            records.append({
                'snap'   : snap,
                'action' : copy.deepcopy(a),
                'reward' : float(r),
                'R_tot'  : float(np.sum(info['R_tot'])),
                'sigma2' : float(info['sigma2']),
                'episode': ep_idx,
                'step'   : step_idx,
            })
    return records


# ───────────────────────────────────────────────────────────────────────────────
# Variance probes
# ───────────────────────────────────────────────────────────────────────────────

def aleatoric_per_step(env, records: list,
                       n_samples: int = 100,
                       R_intra: int = 100,
                       disable: tuple = ()) -> dict:
    """
    For n_samples random records, replay env.step under noise resampling to
    estimate σ(reward | s, a).

    Returns
    -------
    dict with per-sample stats + aggregated mean / quantiles.
    """
    rng    = np.random.default_rng(20260531)
    picks  = rng.choice(len(records), size=min(n_samples, len(records)),
                        replace=False)
    sigmas = np.empty(len(picks))
    means  = np.empty(len(picks))
    for i, idx in enumerate(picks):
        rec = records[idx]
        restore_env_state(env, rec['snap'])
        rewards, info = reward_std_intra_state(
            env, rec['action'], R=R_intra, disable=disable,
            reseed_each=True, base_seed=int(idx))
        sigmas[i] = info['std']
        means[i]  = info['mean']
    return {
        'sigma_per_sample' : sigmas.tolist(),
        'mean_per_sample'  : means.tolist(),
        'sigma_mean'   : float(sigmas.mean()),
        'sigma_median' : float(np.median(sigmas)),
        'sigma_p25'    : float(np.percentile(sigmas, 25)),
        'sigma_p75'    : float(np.percentile(sigmas, 75)),
        'sigma_max'    : float(sigmas.max()),
        'n_samples'    : len(picks),
        'R_intra'      : R_intra,
        'disable'      : list(disable),
    }


def aleatoric_per_return(env, records: list, policy: Callable,
                         n_samples: int = 40,
                         H: int = 10,
                         R_intra: int = 30,
                         gamma: float = 0.95,
                         disable: tuple = ()) -> dict:
    """
    For n_samples random records, restore env state then roll out for H steps
    R_intra times.  Measures σ(H-step return | s_t, π).

    This is the variance that matters for V(s) since V predicts the return,
    not a single reward.
    """
    rng    = np.random.default_rng(20260601)
    picks  = rng.choice(len(records), size=min(n_samples, len(records)),
                        replace=False)
    sigmas = np.empty(len(picks))
    means  = np.empty(len(picks))
    for i, idx in enumerate(picks):
        rec = records[idx]
        restore_env_state(env, rec['snap'])
        rets, info = return_std_intra_state(
            env, policy, H=H, R=R_intra, gamma=gamma,
            disable=disable, base_seed=int(idx) * 7919)
        sigmas[i] = info['std']
        means[i]  = info['mean']
    return {
        'sigma_per_sample' : sigmas.tolist(),
        'mean_per_sample'  : means.tolist(),
        'sigma_mean'   : float(sigmas.mean()),
        'sigma_median' : float(np.median(sigmas)),
        'sigma_p25'    : float(np.percentile(sigmas, 25)),
        'sigma_p75'    : float(np.percentile(sigmas, 75)),
        'sigma_max'    : float(sigmas.max()),
        'n_samples'    : len(picks),
        'R_intra'      : R_intra,
        'H'            : H,
        'gamma'        : gamma,
        'disable'      : list(disable),
    }


def cross_state_stats(records: list, gamma: float = 0.95,
                      H: int = 10) -> dict:
    """
    σ(reward) and σ(H-step return) computed ACROSS all recorded transitions.

    For σ(return) we take the on-policy single rollout return starting at each
    record's step (using the rewards already collected sequentially).
    Records are grouped by episode so the H-window does not cross boundaries.
    """
    rewards = np.array([r['reward'] for r in records])
    # H-step return per record using the on-policy sequential rewards within
    # the same episode (truncate at episode end)
    returns = []
    for i, rec in enumerate(records):
        ep = rec['episode']
        G  = 0.0
        disc = 1.0
        for h in range(H):
            j = i + h
            if j >= len(records) or records[j]['episode'] != ep:
                break
            G += disc * records[j]['reward']
            disc *= gamma
        returns.append(G)
    returns = np.array(returns)
    return {
        'reward_mean': float(rewards.mean()),
        'reward_std' : float(rewards.std()),
        'return_mean': float(returns.mean()),
        'return_std' : float(returns.std()),
        'n_records'  : len(records),
    }


def aleatoric_fraction_and_ceiling(sigma_intra_mean: float,
                                   sigma_total: float) -> tuple:
    """
    aleatoric_fraction = E[Var(X|s)] / Var(X)  ≈ σ²_intra_mean / σ²_total
    ceiling_explVar     = 1 − aleatoric_fraction
    """
    if sigma_total < 1e-12:
        return float('nan'), float('nan')
    frac = (sigma_intra_mean ** 2) / (sigma_total ** 2)
    frac = float(np.clip(frac, 0.0, 1.0))
    ceiling = 1.0 - frac
    return frac, ceiling


# ───────────────────────────────────────────────────────────────────────────────
# Action-conditional variance — σ(reward | s, varying a)
# ───────────────────────────────────────────────────────────────────────────────

def action_conditional_variance(env, records: list, policy_fn: Callable,
                                n_states: int = 40,
                                n_actions: int = 20) -> dict:
    """
    For each of n_states sampled records, fix the env state and apply
    n_actions DIFFERENT action samples from policy_fn (typically random
    policy) → measure σ(reward | s, varying a).

    Compares to σ(reward | s, fixed a) from aleatoric probe:
      ratio = σ_action / σ_aleatoric.
      If ratio >> 1 → action choice matters a lot → Q(s,a) could help.
      If ratio ≈ 1 → reward is dominated by env noise → V(s) is fine.
    """
    rng   = np.random.default_rng(20260602)
    picks = rng.choice(len(records), size=min(n_states, len(records)),
                       replace=False)
    sigmas_action = np.empty(len(picks))
    for i, idx in enumerate(picks):
        rec = records[idx]
        snap = rec['snap']
        rewards = np.empty(n_actions)
        for j in range(n_actions):
            restore_env_state(env, snap)
            env.rng = np.random.default_rng(int(idx) * 1009 + j)
            env.channel_model.rng = env.rng
            a = policy_fn(env)
            _, r, _, _ = env.step(a)
            rewards[j] = float(r)
        sigmas_action[i] = float(rewards.std())
    return {
        'sigma_action_mean'   : float(sigmas_action.mean()),
        'sigma_action_median' : float(np.median(sigmas_action)),
        'sigma_action_p25'    : float(np.percentile(sigmas_action, 25)),
        'sigma_action_p75'    : float(np.percentile(sigmas_action, 75)),
        'n_states'  : len(picks),
        'n_actions' : n_actions,
    }


# ───────────────────────────────────────────────────────────────────────────────
# Full probe driver
# ───────────────────────────────────────────────────────────────────────────────

ABLATION_SOURCES = (
    ('all_off',           ('n0', 'csi_kappa', 'rain',
                           'mobility', 'channel_resample')),
    ('only_n0',           ('csi_kappa', 'rain',
                           'mobility', 'channel_resample')),
    ('only_csi_kappa',    ('n0', 'rain',
                           'mobility', 'channel_resample')),
    ('only_rain',         ('n0', 'csi_kappa',
                           'mobility', 'channel_resample')),
    ('only_mobility',     ('n0', 'csi_kappa', 'rain',
                           'channel_resample')),
    ('only_channel_resample', ('n0', 'csi_kappa', 'rain', 'mobility')),
    ('full_noise',        ()),
)


def run_one_scenario(scenario_name: str,
                     policy: Callable,
                     cfg,
                     args) -> dict:
    """
    Run the full probe under one (policy, R_LoS) configuration.

    Steps:
      1. Build env, collect records via policy
      2. Cross-state stats (reward + H-step return)
      3. Aleatoric per-step (full noise) → ceiling for V(reward)
      4. Aleatoric per H-return (full noise) → ceiling for V(s)
      5. Source ablation (per-step variance under each subset)
      6. Action-conditional variance (random actions over fixed states)
    """
    print(f"\n══════════════════════════════════════════════════")
    print(f"  SCENARIO: {scenario_name}  · K={cfg.K} M={cfg.M} "
          f"R_LoS_km={cfg.R_LoS_km:.2f}")
    print(f"══════════════════════════════════════════════════")
    out = OrderedDict()
    out['scenario'] = scenario_name
    out['K']        = cfg.K
    out['M']        = cfg.M
    out['R_LoS_km'] = cfg.R_LoS_km
    out['reward_noise_avg'] = args.reward_noise_avg
    out['kappa']    = cfg.kappa
    out['args']     = {
        'n_episodes_collect': args.n_episodes_collect,
        'steps_per_ep'      : args.steps_per_ep,
        'n_samples_aleatoric': args.n_samples_aleatoric,
        'R_intra_step'      : args.R_intra_step,
        'n_samples_return'  : args.n_samples_return,
        'R_intra_return'    : args.R_intra_return,
        'H'                 : args.H,
        'n_states_action'   : args.n_states_action,
        'n_actions_action'  : args.n_actions_action,
    }

    # 1) Env + collect
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps_per_ep,
                  reward_noise_avg=args.reward_noise_avg)
    t0 = time.time()
    print(f"\n[1/6] Collecting {args.n_episodes_collect} episodes × "
          f"{args.steps_per_ep} steps under '{scenario_name}' policy ...")
    records = collect_states(env, policy,
                             n_episodes=args.n_episodes_collect,
                             steps_per_ep=args.steps_per_ep,
                             warmup_steps=5)
    t1 = time.time()
    print(f"  collected {len(records)} records in {t1-t0:.1f}s")

    # 2) Cross-state
    print(f"[2/6] Cross-state reward / return stats ...")
    cs = cross_state_stats(records, gamma=args.gamma, H=args.H)
    out['cross_state'] = cs
    print(f"  reward μ={cs['reward_mean']:+.3f} σ={cs['reward_std']:.3f}  ·  "
          f"H={args.H} return μ={cs['return_mean']:+.3f} σ={cs['return_std']:.3f}")

    # 3) Per-step aleatoric (full noise)
    print(f"[3/6] Aleatoric σ(reward|s,a)  full noise  "
          f"n={args.n_samples_aleatoric} R={args.R_intra_step} ...")
    t0 = time.time()
    al_step = aleatoric_per_step(env, records,
                                 n_samples=args.n_samples_aleatoric,
                                 R_intra=args.R_intra_step,
                                 disable=())
    t1 = time.time()
    out['aleatoric_step_full'] = al_step
    frac_step, ceil_step = aleatoric_fraction_and_ceiling(
        al_step['sigma_mean'], cs['reward_std'])
    out['aleatoric_step_full']['fraction']         = frac_step
    out['aleatoric_step_full']['ceiling_explVar']  = ceil_step
    print(f"  σ_intra mean={al_step['sigma_mean']:.3f} "
          f"med={al_step['sigma_median']:.3f}  "
          f"→ aleatoric_fraction={frac_step:.3f}  "
          f"ceiling explVar(reward) ≈ {ceil_step:.3f}  "
          f"({t1-t0:.1f}s)")

    # 4) Per-return aleatoric (full noise) — KEY metric for V(s)
    print(f"[4/6] Aleatoric σ(return_{args.H}|s,π) full noise  "
          f"n={args.n_samples_return} R={args.R_intra_return} ...")
    t0 = time.time()
    al_ret = aleatoric_per_return(env, records, policy,
                                  n_samples=args.n_samples_return,
                                  H=args.H, R_intra=args.R_intra_return,
                                  gamma=args.gamma, disable=())
    t1 = time.time()
    out['aleatoric_return_full'] = al_ret
    frac_ret, ceil_ret = aleatoric_fraction_and_ceiling(
        al_ret['sigma_mean'], cs['return_std'])
    out['aleatoric_return_full']['fraction']        = frac_ret
    out['aleatoric_return_full']['ceiling_explVar'] = ceil_ret
    print(f"  σ_intra return mean={al_ret['sigma_mean']:.3f}  "
          f"→ aleatoric_fraction={frac_ret:.3f}  "
          f"⭐ ceiling explVar(V(s)) ≈ {ceil_ret:.3f}  "
          f"({t1-t0:.1f}s)")

    # 5a) Per-step ablation (only n0 acts per-step; other sources are constant
    #     within one step → mostly 0 for non-n0).  Kept for completeness.
    print(f"[5a/6] Per-step source ablation (n=20 R={args.R_intra_step//2}) ...")
    ablation = OrderedDict()
    for name, disable in ABLATION_SOURCES:
        t0 = time.time()
        a = aleatoric_per_step(env, records,
                               n_samples=20,
                               R_intra=max(20, args.R_intra_step // 2),
                               disable=disable)
        t1 = time.time()
        a['fraction'], a['ceiling_explVar'] = aleatoric_fraction_and_ceiling(
            a['sigma_mean'], cs['reward_std'])
        ablation[name] = a
        print(f"  {name:25s}  σ_intra={a['sigma_mean']:8.3f}  "
              f"frac={a['fraction']:.3f}  ceil={a['ceiling_explVar']:+.3f}  "
              f"({t1-t0:.1f}s)")
    out['ablation_per_step'] = ablation

    # 5b) H-step RETURN ablation — the meaningful decomposition for V(s) because
    #     mobility / channel_resample only affect FUTURE steps, not the current
    #     reward.  This shows which env source contributes the most to return
    #     variance across the H-step rollout horizon.
    print(f"[5b/6] H-step return source ablation "
          f"(n={max(10, args.n_samples_return // 2)} "
          f"R={max(8, args.R_intra_return // 2)} H={args.H}) ...")
    ablation_ret = OrderedDict()
    n_ret = max(10, args.n_samples_return // 2)
    R_ret = max(8, args.R_intra_return // 2)
    for name, disable in ABLATION_SOURCES:
        t0 = time.time()
        a = aleatoric_per_return(env, records, policy,
                                 n_samples=n_ret, H=args.H,
                                 R_intra=R_ret, gamma=args.gamma,
                                 disable=disable)
        t1 = time.time()
        a['fraction'], a['ceiling_explVar'] = aleatoric_fraction_and_ceiling(
            a['sigma_mean'], cs['return_std'])
        ablation_ret[name] = a
        print(f"  {name:25s}  σ_intra_ret={a['sigma_mean']:8.3f}  "
              f"frac={a['fraction']:.3f}  ceil={a['ceiling_explVar']:+.3f}  "
              f"({t1-t0:.1f}s)")
    out['ablation_per_return'] = ablation_ret

    # 6) Action-conditional variance (uses random policy actions)
    print(f"[6/6] Action-conditional variance "
          f"n_states={args.n_states_action} n_actions={args.n_actions_action} ...")
    random_pol_for_actions = make_random_policy(cfg, rng_seed=20260603)
    t0 = time.time()
    ac = action_conditional_variance(env, records, random_pol_for_actions,
                                     n_states=args.n_states_action,
                                     n_actions=args.n_actions_action)
    t1 = time.time()
    out['action_conditional'] = ac
    ratio = ac['sigma_action_mean'] / max(al_step['sigma_mean'], 1e-12)
    out['action_conditional']['sigma_ratio_action_over_aleatoric'] = float(ratio)
    print(f"  σ(reward|s,varying a) mean={ac['sigma_action_mean']:.3f}  "
          f"ratio σ_action/σ_aleatoric={ratio:.2f}  "
          f"({t1-t0:.1f}s)")
    if ratio > 3:
        print(f"  → action choice swings reward {ratio:.1f}× more than env noise "
              f"→ Q(s,a) could help over V(s).")
    else:
        print(f"  → action choice swings reward similar to env noise "
              f"→ V(s) is appropriate.")

    return out


# ───────────────────────────────────────────────────────────────────────────────
# Report writers
# ───────────────────────────────────────────────────────────────────────────────

def write_text_report(all_results: list, out_path: str) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("  CRITIC ALEATORIC-CEILING PROBE  ·  Tier-1 diagnostic")
    lines.append("=" * 78)
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("HOW TO READ")
    lines.append("-" * 78)
    lines.append("  ceiling_explVar(V(s)) = 1 − E[Var(return | s, π)] / Var(return)")
    lines.append("    No critic — however well-tuned — can exceed this.")
    lines.append("  ⚠ CAVEAT: variance is over BOTH env stochasticity AND policy")
    lines.append("    stochasticity (action sampling).  For a random policy the")
    lines.append("    ceiling is dominated by ACTION variance; for a trained policy")
    lines.append("    (lower entropy) the ceiling should be HIGHER.")
    lines.append("    → Always run BOTH a random and a checkpoint scenario.")
    lines.append("")
    lines.append("  Compare to observed explVar in training_log.txt:")
    lines.append("    if observed ≈ ceiling → env+policy-bound, accept noisy critic.")
    lines.append("    if observed << ceiling → pipeline/theory fixes can still help.")
    lines.append("")
    lines.append("  Per-source ablation interpretation:")
    lines.append("    Per-STEP: only n0 acts within a step (others affect future steps).")
    lines.append("    H-step RETURN: 'only_X' = ONLY source X enabled, others disabled.")
    lines.append("      → σ_intra(only_X) − σ_intra(all_off) ≈ contribution of source X.")
    lines.append("      'all_off' is NOT zero because policy stochasticity remains.")
    lines.append("")
    for res in all_results:
        lines.append("=" * 78)
        lines.append(f"  SCENARIO: {res['scenario']}  ·  "
                     f"K={res['K']} M={res['M']} R_LoS={res['R_LoS_km']:.2f}  ·  "
                     f"reward_noise_avg={res['reward_noise_avg']}  κ={res['kappa']}")
        lines.append("=" * 78)
        cs = res['cross_state']
        lines.append(f"  cross-state  reward:  μ={cs['reward_mean']:+.3f}  "
                     f"σ={cs['reward_std']:.3f}  (n={cs['n_records']})")
        lines.append(f"  cross-state  return:  μ={cs['return_mean']:+.3f}  "
                     f"σ={cs['return_std']:.3f}  H={res['args']['H']}")
        lines.append("")
        a = res['aleatoric_step_full']
        lines.append("  ── per-STEP aleatoric (reward|s,a, full noise) ──")
        lines.append(f"    σ_intra: mean={a['sigma_mean']:.3f}  "
                     f"med={a['sigma_median']:.3f}  "
                     f"p25={a['sigma_p25']:.3f}  p75={a['sigma_p75']:.3f}  "
                     f"max={a['sigma_max']:.3f}")
        lines.append(f"    aleatoric_fraction = {a['fraction']:.3f}")
        lines.append(f"    ceiling explVar(reward) ≈ {a['ceiling_explVar']:+.3f}")
        lines.append("")
        ar = res['aleatoric_return_full']
        lines.append("  ── per-RETURN aleatoric (return_H|s,π, full noise) ⭐ ──")
        lines.append(f"    σ_intra: mean={ar['sigma_mean']:.3f}  "
                     f"med={ar['sigma_median']:.3f}  "
                     f"p25={ar['sigma_p25']:.3f}  p75={ar['sigma_p75']:.3f}")
        lines.append(f"    aleatoric_fraction = {ar['fraction']:.3f}")
        lines.append(f"    ⭐ CEILING explVar(V(s)) ≈ {ar['ceiling_explVar']:+.3f}")
        lines.append("")
        lines.append("  ── per-source ablation (per-STEP σ — only n0 acts per-step) ──")
        lines.append("    name                       σ_intra   fraction   ceiling")
        for name, ab in res['ablation_per_step'].items():
            lines.append(f"    {name:25s}  {ab['sigma_mean']:7.3f}    "
                         f"{ab['fraction']:.3f}    {ab['ceiling_explVar']:+.3f}")
        lines.append("")
        if 'ablation_per_return' in res:
            lines.append("  ── per-source ablation (H-step RETURN σ ⭐ — meaningful for V(s)) ──")
            lines.append("    name                       σ_intra_ret  fraction   ceiling")
            for name, ab in res['ablation_per_return'].items():
                lines.append(f"    {name:25s}  {ab['sigma_mean']:9.3f}    "
                             f"{ab['fraction']:.3f}    {ab['ceiling_explVar']:+.3f}")
            lines.append("")
        ac = res['action_conditional']
        lines.append("  ── action-conditional variance ──")
        lines.append(f"    σ(reward|s, varying a) mean={ac['sigma_action_mean']:.3f}")
        lines.append(f"    ratio σ_action / σ_aleatoric  = "
                     f"{ac['sigma_ratio_action_over_aleatoric']:.2f}")
        if ac['sigma_ratio_action_over_aleatoric'] > 3:
            lines.append(f"    → Q(s,a) MAY help over V(s) "
                         f"(actions swing reward {ac['sigma_ratio_action_over_aleatoric']:.1f}× env noise)")
        else:
            lines.append(f"    → V(s) is appropriate (action ≲ env noise)")
        lines.append("")
    # ── Cross-scenario summary ──────────────────────────────────────────────
    lines.append("=" * 78)
    lines.append("  CROSS-SCENARIO SUMMARY  ·  ⭐ ceiling explVar(V(s))")
    lines.append("=" * 78)
    lines.append("  scenario                  R_LoS   ceiling(V)   σ_return")
    for res in all_results:
        ar = res['aleatoric_return_full']
        lines.append(f"  {res['scenario']:25s} {res['R_LoS_km']:.2f}   "
                     f"{ar['ceiling_explVar']:+.3f}      "
                     f"{res['cross_state']['return_std']:.3f}")
    lines.append("")
    lines.append("  Compare to observed mean explVar from training_log.txt for result_5:")
    lines.append("  ≈ 0.40 (wobble 0.02-0.69).  Interpretation:")
    lines.append("    ceiling 0.6+ → critic NOT at ceiling; pipeline/theory still has room.")
    lines.append("    ceiling 0.3-0.5 → critic near ceiling; env-bound, accept noisy V.")
    lines.append("    ceiling <0.3 → env aleatoric dominates; switch tactics entirely.")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\nText report written to {out_path}")


def write_json_report(all_results: list, out_path: str) -> None:
    # Strip per-sample lists if huge to keep file small
    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items()
                    if k not in ('sigma_per_sample', 'mean_per_sample')}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(_strip(all_results), f, indent=2)
    print(f"JSON report written to {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tier-1 diagnostic: critic aleatoric ceiling probe.")
    parser.add_argument('--ckpt', type=str, default=None,
                        help="Optional checkpoint dir to add as a scenario "
                             "(e.g. results/result_5/checkpoints/ep_01000)")
    parser.add_argument('--r_los_list', type=float, nargs='+',
                        default=[0.2, 0.3, 0.5],
                        help="R_LoS_km values to sweep (one scenario per value)")
    parser.add_argument('--seed', type=int, default=20260531)
    parser.add_argument('--reward_noise_avg', type=int,
                        default=getattr(P, 'reward_noise_avg', 16))
    parser.add_argument('--gamma', type=float, default=getattr(P, 'gamma', 0.95))
    # Sample sizes
    parser.add_argument('--n_episodes_collect', type=int, default=20)
    parser.add_argument('--steps_per_ep',       type=int, default=50)
    parser.add_argument('--n_samples_aleatoric', type=int, default=100)
    parser.add_argument('--R_intra_step',        type=int, default=100)
    parser.add_argument('--n_samples_return',    type=int, default=40)
    parser.add_argument('--R_intra_return',      type=int, default=30)
    parser.add_argument('--H', type=int, default=10)
    parser.add_argument('--n_states_action',  type=int, default=40)
    parser.add_argument('--n_actions_action', type=int, default=20)
    parser.add_argument('--quick', action='store_true',
                        help="Reduce sample sizes ~3× for fast sanity check")
    parser.add_argument('--out_dir', type=str, default='results',
                        help="Where to write probe report files")
    args = parser.parse_args()

    if args.quick:
        args.n_episodes_collect  = 8
        args.steps_per_ep        = 25
        args.n_samples_aleatoric = 30
        args.R_intra_step        = 40
        args.n_samples_return    = 15
        args.R_intra_return      = 12
        args.n_states_action     = 15
        args.n_actions_action    = 10

    os.makedirs(args.out_dir, exist_ok=True)
    cfg_base = make_config()
    print(f"Active case: K={cfg_base.K} M={cfg_base.M} "
          f"P_S_dBm={getattr(P, 'P_S_dBm', '?')} "
          f"n_qubits={getattr(P, 'n_qubits', '?')} κ={cfg_base.kappa}")
    print(f"reward_noise_avg={args.reward_noise_avg}  γ={args.gamma}")
    print(f"Sample sizes: collect {args.n_episodes_collect}×{args.steps_per_ep}, "
          f"aleatoric_step n={args.n_samples_aleatoric}×R={args.R_intra_step}, "
          f"aleatoric_return n={args.n_samples_return}×R={args.R_intra_return} H={args.H}, "
          f"action n={args.n_states_action}×{args.n_actions_action}")

    all_results = []

    # Build the policy ONCE per scenario type (random / ckpt).  R_LoS sweep is
    # per scenario type via cfg override.
    for r_los in args.r_los_list:
        cfg = make_config(R_LoS_km=r_los)
        # --- Scenario 1: random policy
        rand_policy = make_random_policy(cfg, rng_seed=args.seed)
        res = run_one_scenario(f"random_R{r_los:.2f}", rand_policy, cfg, args)
        all_results.append(res)
        # --- Scenario 2: checkpoint policy (only if path provided AND R_LoS
        #                  matches what the checkpoint was trained at — we run
        #                  it for every r_los anyway so user can see behaviour)
        if args.ckpt is not None:
            try:
                ckpt_policy = make_checkpoint_policy(args.ckpt, cfg)
                res2 = run_one_scenario(
                    f"ckpt_{os.path.basename(args.ckpt)}_R{r_los:.2f}",
                    ckpt_policy, cfg, args)
                all_results.append(res2)
            except Exception as exc:  # noqa
                print(f"  [warn] checkpoint policy load failed: {exc}")

    case_tag = f"K{cfg_base.K}M{cfg_base.M}"
    txt_path  = os.path.join(args.out_dir, f"probe_critic_ceiling_{case_tag}.txt")
    json_path = os.path.join(args.out_dir, f"probe_critic_ceiling_{case_tag}.json")
    write_text_report(all_results, txt_path)
    write_json_report(all_results, json_path)


if __name__ == '__main__':
    main()
