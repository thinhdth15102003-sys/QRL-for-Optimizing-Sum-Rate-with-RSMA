"""
irs.py
------
Manages discrete phase-shifts for M IRS, each with N elements.

2-bit quantisation (extension vs. paper)
-----------------------------------------
  Phase levels : {0, π/2, π, 3π/2}
  Index map    : 0→0, 1→π/2, 2→π, 3→3π/2

The RL agent outputs integer indices in {0,1,2,3} per element;
this module converts them to complex diagonal entries of Φ_m.
"""

import numpy as np
from typing import Optional

from istn.config import SystemConfig


class IRSPhaseModel:

    def __init__(self, cfg: SystemConfig):
        self.cfg     = cfg
        self.levels   = cfg.phase_levels          # (2^bits,) rad
        self.n_levels = len(self.levels)         # 2^bits

    # ------------------------------------------------------------------ #
    # Conversion utilities
    # ------------------------------------------------------------------ #

    def quantize(self, continuous_phases: np.ndarray) -> np.ndarray:
        """
        Map continuous phases (any shape) in [0, 2π) to the nearest
        discrete level.

        Parameters
        ----------
        continuous_phases : ndarray  (arbitrary shape), values in radians

        Returns
        -------
        ndarray  same shape, values in {0, π/2, π, 3π/2}
        """
        phases_mod = continuous_phases % (2 * np.pi)
        diff = np.abs(phases_mod[..., np.newaxis] - self.levels)  # (..., 4)
        diff = np.minimum(diff, 2 * np.pi - diff)                 # circular wrap
        idx  = np.argmin(diff, axis=-1)
        return self.levels[idx]

    def index_to_phase(self, idx: np.ndarray) -> np.ndarray:
        """
        Convert integer index array to phase values.

        Parameters
        ----------
        idx : ndarray  (M, N)  integers in {0, 1, 2, 3}

        Returns
        -------
        ndarray  (M, N)  phase values in {0, π/2, π, 3π/2}
        """
        idx_clipped = np.clip(idx, 0, self.n_levels - 1).astype(int)
        return self.levels[idx_clipped]

    def build_phi(self, phases: np.ndarray) -> np.ndarray:
        """
        Build diagonal phase-shift matrices Φ_m = diag(e^{jφ_1}, …, e^{jφ_N}).

        Parameters
        ----------
        phases : ndarray (M, N)  quantized phase values in radians

        Returns
        -------
        Phi : ndarray (M, N, N) complex
              Phi[m] = diag(e^{jφ_{m,1}}, …, e^{jφ_{m,N}}), |diag entries| = 1
        """
        M, N = phases.shape
        diag_entries = np.exp(1j * phases)          # (M, N), unit modulus
        Phi = np.zeros((M, N, N), dtype=complex)
        idx = np.arange(N)
        Phi[:, idx, idx] = diag_entries
        return Phi

    # ------------------------------------------------------------------ #
    # Sampling helpers
    # ------------------------------------------------------------------ #

    def random_phases(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample uniformly random discrete phases.

        Returns
        -------
        phases : ndarray (M, N)  values in {0, π/2, π, 3π/2}
        """
        rng = rng or np.random.default_rng()
        idx = rng.integers(0, self.n_levels, size=(self.cfg.M, self.cfg.N))
        return self.levels[idx]

    def random_phi(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Convenience: random phases already converted to complex Phi.

        Returns
        -------
        Phi : ndarray (M, N, N) complex  —  Phi[m] = diag(e^{jθ_1},…,e^{jθ_N})
        """
        return self.build_phi(self.random_phases(rng))

    def all_zero_phi(self) -> np.ndarray:
        """Return Phi with all phases set to 0 (all diagonal entries = e^{j·0} = 1)."""
        M, N = self.cfg.M, self.cfg.N
        Phi = np.zeros((M, N, N), dtype=complex)
        idx = np.arange(N)
        Phi[:, idx, idx] = 1.0
        return Phi