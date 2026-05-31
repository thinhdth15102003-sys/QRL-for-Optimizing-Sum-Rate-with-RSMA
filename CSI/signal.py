"""
signal.py
---------
Baseband transmit-signal model for the multi-group RSMA system.

RSMA signal decomposition  (M+1 common streams, one per group)
--------------------------------------------------------------
  Stage 1 — each user k decodes its own GROUP's common message first:
      x_c_g = √w_c_vec[g] · s_c_g      s_c_g ~ CN(0,1), independent across groups

  Stage 2 — subtract x_c_g (SIC), then decode the private message:
      x_p_k = √w_p[k] · s_p_k          s_p_k ~ CN(0,1)

Transmitted signal
------------------
  x = Σ_g √w_c_vec[g] · s_c_g  +  Σ_k √w_p[k] · s_p_k

Power budget:
  E[|x|²] = Σ_g w_c_vec[g] + Σ_k w_p[k]  ≤  P_S

Received signal at user k  (group gid = assignment[k])
--------------------------
  y_k = h_k · x + n_k
      = h_k · √w_c_vec[gid] · s_c_gid          ← desired common signal
        + h_k · Σ_{g≠gid} √w_c_vec[g] · s_c_g  ← other-group common interference
        + h_k · √w_p[k] · s_p_k                 ← desired private signal
        + h_k · Σ_{j≠k} √w_p[j] · s_p_j        ← inter-user private interference
        + n_k,   n_k ~ CN(0, σ²)

SINR expressions (consistent with CSI/rate.py):
  SINR_common_k  = |h_k|² · w_c[gid] / (|h_k|² · (Σ_j w_p[j] + Σ_g w_c[g] − w_c[gid]) + σ²)
  SINR_private_k = |h_k|² · w_p[k]   / (|h_k|² · (Σ_{j≠k} w_p[j] + Σ_{g≠gid} w_c[g]) + σ²)
"""

import numpy as np
from typing import Optional

from istn.config import SystemConfig


class BasebandSignal:
    """Generate and verify RSMA baseband transmit samples (multi-group)."""

    def __init__(self, cfg: SystemConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Symbol generation
    # ------------------------------------------------------------------ #

    def sample_symbols(self) -> tuple:
        """
        Draw one block of i.i.d. RSMA symbols for all M+1 groups and K users.

        Returns
        -------
        s_c : (M+1,) complex   CN(0,1) — one independent common symbol per group
        s_p : (K,)   complex   CN(0,1) — private symbols, one per user
        """
        M, K = self.cfg.M, self.cfg.K
        s_c = ((self.rng.normal(size=M + 1) + 1j * self.rng.normal(size=M + 1))
               / np.sqrt(2))
        s_p = ((self.rng.normal(size=K) + 1j * self.rng.normal(size=K))
               / np.sqrt(2))
        return s_c, s_p

    # ------------------------------------------------------------------ #
    # Transmit sample
    # ------------------------------------------------------------------ #

    def transmit(self, w_c_vec: np.ndarray, w_p: np.ndarray) -> complex:
        """
        Generate one scalar transmitted sample.

          x = Σ_g √w_c_vec[g] · s_c_g  +  Σ_k √w_p[k] · s_p_k

        Parameters
        ----------
        w_c_vec : (M+1,)  common-stream powers per group (Watts)
        w_p     : (K,)    private powers per user (Watts)

        Returns
        -------
        x : complex  scalar transmit sample
        """
        s_c, s_p = self.sample_symbols()
        x = np.sum(np.sqrt(w_c_vec) * s_c) + np.sum(np.sqrt(w_p) * s_p)
        return x

    # ------------------------------------------------------------------ #
    # Power verification
    # ------------------------------------------------------------------ #

    def verify_power(self, w_c_vec: np.ndarray, w_p: np.ndarray,
                     n_samples: int = 10_000) -> dict:
        """
        Monte Carlo verification that E[|x|²] = Σ w_c_vec[g] + Σ w_p[k].

        Returns
        -------
        dict:
          P_theoretical : float  Σ w_c_vec + Σ w_p
          P_empirical   : float  sample mean of |x|²
          relative_error: float  |theory − empirical| / theory
        """
        P_theory    = float(np.sum(w_c_vec)) + float(np.sum(w_p))
        samples     = np.array([self.transmit(w_c_vec, w_p) for _ in range(n_samples)])
        P_empirical = float(np.mean(np.abs(samples) ** 2))
        return {
            'P_theoretical': P_theory,
            'P_empirical':   P_empirical,
            'relative_error': abs(P_theory - P_empirical) / max(P_theory, 1e-12),
        }

    # ------------------------------------------------------------------ #
    # Convenience: received signal at one user (scalar beamforming)
    # ------------------------------------------------------------------ #

    def received(self, h_k: complex,
                 w_c_vec: np.ndarray, w_p: np.ndarray,
                 sigma2: Optional[float] = None) -> complex:
        """
        Scalar received signal at user k under MRC-aligned beamforming.

          y_k = h_k · x + n_k,   n_k ~ CN(0, sigma2)

        Parameters
        ----------
        h_k     : complex   effective channel scalar for user k
        w_c_vec : (M+1,)    common-stream powers per group
        w_p     : (K,)      private powers per user
        sigma2  : float     noise variance; defaults to cfg.sigma2 if None
        """
        s2  = sigma2 if sigma2 is not None else self.cfg.sigma2
        x   = self.transmit(w_c_vec, w_p)
        n_k = (np.sqrt(s2 / 2)
               * (self.rng.normal() + 1j * self.rng.normal()))
        return h_k * x + n_k
