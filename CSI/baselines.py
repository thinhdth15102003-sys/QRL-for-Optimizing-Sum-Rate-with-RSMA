"""
baselines.py
------------
Simple baseline policies used to benchmark the environment.

Baselines
---------
  RandomPolicy     — random action every step
  GreedyPolicy     — assign each user to the link with highest |h_k|
  DirectOnlyPolicy — all users use direct satellite link (no IRS)
  AllIRSPolicy     — all users routed through IRS (round-robin per IRS)

Channel convention: all policies observe TRUE channels (g_SR, g_RU, g_SU).
Rate-relevant decisions (h_eff in GreedyPolicy) use ESTIMATED channels (g_hat).
w_c_vec is always (G+1,) where G = number of active IRS in the chosen assignment.
"""

import numpy as np
from typing import Optional

from istn.config import SystemConfig
from .env    import ISTNEnv


def _make_wc_vec(assignment: np.ndarray, cfg: SystemConfig) -> np.ndarray:
    """
    Build (G+1,) common-power vector for the given assignment.
    20% of P_S split equally among G+1 active groups (direct + G IRS).
    """
    active_irs_ids = sorted(set(int(a) for a in assignment if a > 0))
    G = len(active_irs_ids)
    n = G + 1
    return np.full(n, cfg.P_S * 0.2 / n)


# ------------------------------------------------------------------ #
# Random policy
# ------------------------------------------------------------------ #

class RandomPolicy:
    """Uniformly random assignment + phase indices each step."""

    def __init__(self, cfg: SystemConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    def act(self, obs: dict) -> dict:
        cfg        = self.cfg
        assignment = self.rng.integers(0, cfg.M + 1, size=cfg.K)
        return {
            'assignment': assignment,
            'phase_idx':  self.rng.integers(0, 2 ** cfg.quantization_bits,
                                            size=(cfg.M, cfg.N)),
            'w_p':     np.abs(self.rng.normal(size=cfg.K)),
            'w_c_vec': _make_wc_vec(assignment, cfg),
        }


# ------------------------------------------------------------------ #
# Greedy channel-gain policy
# ------------------------------------------------------------------ #

class GreedyPolicy:
    """
    Assign each user to the link (direct or IRS_m) that maximises
    effective channel magnitude |h_k|, using ESTIMATED channels (g_hat).
    """

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def act(self, obs: dict, env: ISTNEnv) -> dict:
        cfg = self.cfg
        ch  = env.channels
        Phi = env.Phi

        assignment = np.zeros(cfg.K, dtype=int)
        for k in range(cfg.K):
            # Direct link gain (estimated channel)
            best_gain = np.abs(ch['g_SU_hat'][k])
            best_a    = 0

            # IRS reflected link gain for each IRS (using estimated CSI)
            for m in range(cfg.M):
                beta_m = ch['beta'][m]
                h_m    = beta_m * np.sum(
                    ch['g_SR_hat'][m].conj() * np.diag(Phi[m]) * ch['g_RU_hat'][m, k]
                )
                gain = np.abs(h_m)
                if gain > best_gain:
                    best_gain = gain
                    best_a    = m + 1   # 1-based

            assignment[k] = best_a

        return {
            'assignment': assignment,
            'w_p':     np.full(cfg.K, cfg.P_S * 0.8 / cfg.K),
            'w_c_vec': _make_wc_vec(assignment, cfg),
        }


# ------------------------------------------------------------------ #
# Direct-only policy
# ------------------------------------------------------------------ #

class DirectOnlyPolicy:
    """All users communicate via direct satellite link (assignment = 0)."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def act(self, obs: dict) -> dict:
        cfg        = self.cfg
        assignment = np.zeros(cfg.K, dtype=int)
        return {
            'assignment': assignment,
            'w_p':     np.full(cfg.K, cfg.P_S * 0.8 / cfg.K),
            'w_c_vec': _make_wc_vec(assignment, cfg),   # (1,) — direct group only
        }


# ------------------------------------------------------------------ #
# All-IRS policy  (round-robin IRS assignment)
# ------------------------------------------------------------------ #

class AllIRSPolicy:
    """Every user is assigned to an IRS (round-robin across M panels)."""

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    def act(self, obs: dict) -> dict:
        cfg        = self.cfg
        assignment = np.array([(k % cfg.M) + 1 for k in range(cfg.K)])
        return {
            'assignment': assignment,
            'w_p':     np.full(cfg.K, cfg.P_S * 0.8 / cfg.K),
            'w_c_vec': _make_wc_vec(assignment, cfg),
        }
