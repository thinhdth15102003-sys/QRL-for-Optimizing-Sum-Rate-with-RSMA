"""
env_probe.py
------------
Utility helpers for diagnostic probes over ISTNEnv.  Provides:

  * snapshot_env_state(env)  / restore_env_state(env, snap)
      Capture and restore the FULL stochastic state of an env so that the
      same step() call can be replayed under different noise realisations.

  * step_with_source_control(env, action, disable=())
      Run env.step(action) with specific aleatoric sources temporarily
      disabled (or held fixed).  Useful for source-attribution probes.

  * rollout_with_source_control(env, actor_fn, H, disable=())
      Multi-step rollout variant — needed when the critic target is the
      H-step return (most return variance comes from mobility + channel
      resample BETWEEN steps, not from n0 within step).

These wrappers do NOT modify env.py.  They monkey-patch attributes of the
provided env instance for the duration of one call and restore them after.

Disable sources currently understood
------------------------------------
  'n0'              : per-step receiver noise variance.  Fixed to mean.
  'csi_kappa'       : multiplicative CSI estimation error (kappa).  Set to 0.
  'channel_resample': suppress update_user_channels mid-step.  CSI of next
                      step = CSI of current (drift only via mobility coords).
  'mobility'        : suppress _walk_users.  user_pos frozen.
  'rain'            : zero rain variance (rain locked to mean).

Combine via tuple, e.g. disable=('n0', 'csi_kappa').
"""

import copy
import numpy as np
from typing import Iterable, Callable, Optional


# ───────────────────────────────────────────────────────────────────────────────
# State snapshot / restore
# ───────────────────────────────────────────────────────────────────────────────

_ENV_SCALARS  = ('_step_count', 'reward_noise_avg')
_ENV_OBJECTS  = ('assignment', 'w_p', 'w_c_vec', 'C_k',
                 'user_pos', 'irs_pos', 'Phi',
                 '_confined_users', '_user_building')


def snapshot_env_state(env) -> dict:
    """Deep-copy all stateful attributes of an ISTNEnv into a plain dict."""
    snap = {
        'channels'        : copy.deepcopy(env.channels),
        'active_irs_ids'  : list(env.active_irs_ids),
        'rng_state'       : env.rng.bit_generator.state,
        'channel_rng_state': env.channel_model.rng.bit_generator.state,
    }
    for name in _ENV_SCALARS:
        snap[name] = getattr(env, name)
    for name in _ENV_OBJECTS:
        v = getattr(env, name)
        snap[name] = (None if v is None else
                      (v.copy() if hasattr(v, 'copy') else copy.deepcopy(v)))
    return snap


def restore_env_state(env, snap: dict) -> None:
    """Restore every attribute captured by snapshot_env_state."""
    env.channels       = copy.deepcopy(snap['channels'])
    env.active_irs_ids = list(snap['active_irs_ids'])
    env.rng.bit_generator.state          = snap['rng_state']
    env.channel_model.rng.bit_generator.state = snap['channel_rng_state']
    for name in _ENV_SCALARS:
        setattr(env, name, snap[name])
    for name in _ENV_OBJECTS:
        v = snap[name]
        setattr(env, name,
                None if v is None else (v.copy() if hasattr(v, 'copy') else
                                        copy.deepcopy(v)))


# ───────────────────────────────────────────────────────────────────────────────
# Per-source disabling via monkey-patch
# ───────────────────────────────────────────────────────────────────────────────

class _SourceControl:
    """Context manager that monkey-patches env to disable specified noise
    sources for the duration of a single env.step() call."""

    def __init__(self, env, disable: Iterable[str]):
        self.env      = env
        self.disable  = set(disable)
        self._saved: dict = {}

    def __enter__(self):
        env = self.env
        ch_model = env.channel_model

        # ── 'n0': pin sigma2 to its expected value (linear of mean dBW) ──
        if 'n0' in self.disable:
            self._saved['sample_noise_sigma2'] = ch_model.sample_noise_sigma2
            fixed_sigma2 = 10.0 ** (ch_model.cfg.noise_mean_dBW / 10.0)
            ch_model.sample_noise_sigma2 = lambda: fixed_sigma2  # type: ignore

        # ── 'csi_kappa': zero the CSI estimation error coefficient ──
        if 'csi_kappa' in self.disable:
            self._saved['kappa'] = ch_model.cfg.kappa
            ch_model.cfg.kappa = 0.0

        # ── 'rain': pin rain to mean ──
        if 'rain' in self.disable:
            self._saved['rain_var_SR'] = ch_model.cfg.rain_var_SR
            self._saved['rain_var_SU'] = ch_model.cfg.rain_var_SU
            self._saved['rain_var_RU'] = ch_model.cfg.rain_var_RU
            ch_model.cfg.rain_var_SR = 0.0
            ch_model.cfg.rain_var_SU = 0.0
            ch_model.cfg.rain_var_RU = 0.0

        # ── 'mobility': freeze _walk_users → returns input unchanged ──
        if 'mobility' in self.disable:
            self._saved['_walk_users'] = env._walk_users
            env._walk_users = lambda pos: pos  # type: ignore

        # ── 'channel_resample': skip update_user_channels (CSI frozen) ──
        if 'channel_resample' in self.disable:
            self._saved['update_user_channels'] = ch_model.update_user_channels
            ch_model.update_user_channels = lambda up, ip, ch: ch  # type: ignore

        return self

    def __exit__(self, *exc):
        env = self.env
        ch_model = env.channel_model
        if 'sample_noise_sigma2' in self._saved:
            ch_model.sample_noise_sigma2 = self._saved['sample_noise_sigma2']
        if 'kappa' in self._saved:
            ch_model.cfg.kappa = self._saved['kappa']
        if 'rain_var_SR' in self._saved:
            ch_model.cfg.rain_var_SR = self._saved['rain_var_SR']
            ch_model.cfg.rain_var_SU = self._saved['rain_var_SU']
            ch_model.cfg.rain_var_RU = self._saved['rain_var_RU']
        if '_walk_users' in self._saved:
            env._walk_users = self._saved['_walk_users']
        if 'update_user_channels' in self._saved:
            ch_model.update_user_channels = self._saved['update_user_channels']


def step_with_source_control(env, action: dict,
                             disable: Iterable[str] = ()) -> tuple:
    """env.step(action) with specified noise sources disabled."""
    with _SourceControl(env, disable):
        return env.step(action)


# ───────────────────────────────────────────────────────────────────────────────
# Aleatoric variance probe helpers
# ───────────────────────────────────────────────────────────────────────────────

def reward_std_intra_state(env, action: dict,
                           R: int = 100,
                           disable: Iterable[str] = (),
                           reseed_each: bool = True,
                           base_seed: int = 0) -> tuple:
    """
    Compute σ(reward | fixed state, fixed action) by replaying env.step R times
    with the env state SNAPSHOTTED-AND-RESTORED between repeats.

    Parameters
    ----------
    env         : ISTNEnv-like instance, ALREADY positioned at the target state
                  (typically after env.reset() + N env.step() warm-up calls)
    action      : action dict to apply each repeat
    R           : number of repeats
    disable     : noise sources to suppress
    reseed_each : if True, advance the RNG with a different seed each repeat
                  (so that re-enabled sources actually re-draw differently)
    base_seed   : base RNG seed; per-repeat seed = base_seed + i

    Returns
    -------
    rewards : (R,) np.ndarray of per-repeat rewards
    info    : dict with 'mean', 'std', 'p25', 'p75', 'min', 'max'
    """
    snap = snapshot_env_state(env)
    rewards = np.empty(R)
    for i in range(R):
        restore_env_state(env, snap)
        if reseed_each:
            env.rng = np.random.default_rng(base_seed + i)
            env.channel_model.rng = env.rng
        _, r, _, _ = step_with_source_control(env, action, disable=disable)
        rewards[i] = float(r)
    # Final restore so caller env is back at the original state
    restore_env_state(env, snap)
    info = {
        'mean': float(rewards.mean()),
        'std' : float(rewards.std()),
        'p25' : float(np.percentile(rewards, 25)),
        'p75' : float(np.percentile(rewards, 75)),
        'min' : float(rewards.min()),
        'max' : float(rewards.max()),
    }
    return rewards, info


def return_std_intra_state(env, policy_fn: Callable,
                           H: int = 10,
                           R: int = 50,
                           gamma: float = 0.95,
                           disable: Iterable[str] = (),
                           base_seed: int = 0) -> tuple:
    """
    Compute σ(H-step discounted return | fixed start state) by replaying H
    steps of policy R times.

    The policy is queried fresh each repeat; if policy is deterministic the
    only variation is from env stochasticity (target of this probe).

    Parameters
    ----------
    policy_fn : callable(env) → action_dict
    H         : rollout horizon
    R         : number of repeats
    gamma     : discount

    Returns
    -------
    returns : (R,) np.ndarray
    info    : dict with mean / std / quantiles
    """
    snap = snapshot_env_state(env)
    returns = np.empty(R)
    for i in range(R):
        restore_env_state(env, snap)
        env.rng = np.random.default_rng(base_seed + i)
        env.channel_model.rng = env.rng
        G, disc = 0.0, 1.0
        for _t in range(H):
            a = policy_fn(env)
            _, r, _, _ = step_with_source_control(env, a, disable=disable)
            G   += disc * float(r)
            disc *= gamma
        returns[i] = G
    restore_env_state(env, snap)
    info = {
        'mean': float(returns.mean()),
        'std' : float(returns.std()),
        'p25' : float(np.percentile(returns, 25)),
        'p75' : float(np.percentile(returns, 75)),
        'min' : float(returns.min()),
        'max' : float(returns.max()),
    }
    return returns, info
