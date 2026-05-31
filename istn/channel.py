"""
channel.py
----------
Generates all channel coefficients for one episode/step snapshot.

Channel convention
------------------
  g_SR[m]     satellite → IRS_m     TRUE physical channel
  g_RU[m, k]  IRS_m    → user k     TRUE physical channel
  g_SU[k]     satellite → user k    TRUE physical channel

  g_SR_hat[m]    estimated/imperfect CSI: g_hat = g + Δg,  Δg = κ|g|ε, ε ~ CN(0,1)
  g_RU_hat[m, k] ditto for IRS→user link
  g_SU_hat[k]    ditto for direct link

  The RL agent observes the TRUE channels |g| (fed from env observation).
  Rate/h computation (rate.py) uses the ESTIMATED channels g_hat — reflecting
  that the real system designs precoding based on imperfect CSI.

Building blocking  (3-D)
------------------------
  Each IRS sits on a rectangular building of footprint
  (2·d_block_km × 2·d_block_km) and height h_IRS_km.  The satellite is at
  (0, 0, h_SR_km); a user's DIRECT link is blocked when the 3-D LoS ray
  from the satellite crosses any building's 3-D box (su_blocked).
  IRS→user paths are assumed always unobstructed (no blk_mat).

Mobility update
---------------
  update_user_channels() recomputes only g_SU/g_SU_hat and g_RU/g_RU_hat.
  g_SR / g_SR_hat are unchanged within an episode (IRS position is fixed).
"""

import numpy as np
from typing import Optional

from .config import SystemConfig


class ChannelModel:

    def __init__(self, cfg: SystemConfig,
                 rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def sample_noise_sigma2(self) -> float:
        """Per-step noise variance from N(noise_mean_dBW, sqrt(noise_var_dBW)) dBW."""
        noise_dBW = self.rng.normal(self.cfg.noise_mean_dBW,
                                    np.sqrt(self.cfg.noise_var_dBW))
        return 10.0 ** (noise_dBW / 10.0)

    # ------------------------------------------------------------------ #
    # Distance geometry
    # ------------------------------------------------------------------ #

    def _slant_ranges(self, xy_pos: np.ndarray) -> np.ndarray:
        h = self.cfg.h_SR_km
        return np.sqrt(xy_pos[:, 0]**2 + xy_pos[:, 1]**2 + h**2)

    def _irs_user_distances(self, user_pos: np.ndarray,
                            irs_pos: np.ndarray) -> np.ndarray:
        dx = user_pos[np.newaxis, :, 0] - irs_pos[:, np.newaxis, 0]
        dy = user_pos[np.newaxis, :, 1] - irs_pos[:, np.newaxis, 1]
        return np.sqrt(dx**2 + dy**2 + self.cfg.h_IRS_km**2)

    def _irs_user_path_loss(self, d_IRS_U: np.ndarray) -> np.ndarray:
        return (self.cfg.d0_ref_km / d_IRS_U) ** self.cfg.path_loss_exp

    # ------------------------------------------------------------------ #
    # Building blocking geometry  (3-D ray–box intersection, vectorised)
    # ------------------------------------------------------------------ #

    def _building_blocks_matrix(self, user_pos: np.ndarray,
                                irs_pos: np.ndarray) -> np.ndarray:
        """
        Fully vectorised (M, K) ray–box intersection test.
        Replaces the O(M×K) Python double-for-loop.
        """
        a    = self.cfg.d_block_km
        H    = self.cfg.h_IRS_km
        h_SR = self.cfg.h_SR_km
        tol  = 1e-12

        # r_min_z: parametric value where ray enters the vertical slab [0, H]
        r_min_z = max(0.0, 1.0 - H / h_SR)

        # Broadcast (M, K) from (M, 1) and (1, K)
        x_k = user_pos[np.newaxis, :, 0]   # (1, K)
        y_k = user_pos[np.newaxis, :, 1]
        x_m = irs_pos[:, np.newaxis, 0]    # (M, 1)
        y_m = irs_pos[:, np.newaxis, 1]

        # ── X slab ──────────────────────────────────────────────────────
        xk_near0 = np.abs(x_k) < tol                       # (M, K) bool
        xm_in    = np.abs(x_m) <= a
        xk_safe  = np.where(xk_near0, 1.0, x_k)            # avoid /0
        lo_x = (x_m - a) / xk_safe
        hi_x = (x_m + a) / xk_safe
        r_xlo = np.where(xk_near0, 0.0,              np.minimum(lo_x, hi_x))
        r_xhi = np.where(xk_near0,
                         np.where(xm_in, 1.0, -1.0), np.maximum(lo_x, hi_x))

        # ── Y slab ──────────────────────────────────────────────────────
        yk_near0 = np.abs(y_k) < tol
        ym_in    = np.abs(y_m) <= a
        yk_safe  = np.where(yk_near0, 1.0, y_k)
        lo_y = (y_m - a) / yk_safe
        hi_y = (y_m + a) / yk_safe
        r_ylo = np.where(yk_near0, 0.0,              np.minimum(lo_y, hi_y))
        r_yhi = np.where(yk_near0,
                         np.where(ym_in, 1.0, -1.0), np.maximum(lo_y, hi_y))

        # ── Slab intersection ────────────────────────────────────────────
        r_lo = np.maximum(np.maximum(r_xlo, r_ylo), r_min_z)
        r_hi = np.minimum(np.minimum(r_xhi, r_yhi), 1.0)
        return r_lo <= r_hi   # (M, K) bool

    def _direct_link_blocked(self, user_pos: np.ndarray,
                             irs_pos: np.ndarray) -> np.ndarray:
        return self._building_blocks_matrix(user_pos, irs_pos).any(axis=0)

    # ------------------------------------------------------------------ #
    # Channel samplers — vectorised, no Python loops over K or M
    # ------------------------------------------------------------------ #

    def _sample_g_SR(self, d_SR: np.ndarray):
        """Sample sat→IRS channels. Returns (g (M,) true, g_hat (M,) estimated)."""
        cfg = self.cfg
        M   = cfg.M
        G_S = 10 ** (cfg.G_S_dBi / 10)

        rain_dB = self.rng.normal(cfg.rain_mean, np.sqrt(cfg.rain_var_SR), size=M)
        eps     = 10.0 ** (-rain_dB / 10.0)
        f       = cfg.f_GHz * 1e9
        d       = d_SR * 1e3
        mag     = (cfg.c * np.sqrt(G_S * eps)) / (4 * np.pi * f * d)
        phi     = self.rng.uniform(0, 2 * np.pi, size=M)
        g       = mag * np.exp(1j * phi)

        delta_n = (self.rng.standard_normal(M) + 1j * self.rng.standard_normal(M)) / np.sqrt(2)
        g_hat   = g + cfg.kappa * np.abs(g) * delta_n
        return g, g_hat

    def _sample_g_SU(self, d_SU: np.ndarray):
        """Sample sat→user channels. Returns (g (K,) true, g_hat (K,) estimated)."""
        cfg = self.cfg
        K   = cfg.K
        G_S = 10 ** (cfg.G_S_dBi / 10)
        G_U = 10 ** (cfg.G_U_dBi / 10)

        rain_dB = self.rng.normal(cfg.rain_mean, np.sqrt(cfg.rain_var_SU), size=K)
        eps     = 10.0 ** (-rain_dB / 10.0)
        f       = cfg.f_GHz * 1e9
        d       = d_SU * 1e3
        mag     = (cfg.c * np.sqrt(G_S * G_U * eps)) / (4 * np.pi * f * d)
        phi     = self.rng.uniform(0, 2 * np.pi, size=K)
        g       = mag * np.exp(1j * phi)

        delta_n = (self.rng.standard_normal(K) + 1j * self.rng.standard_normal(K)) / np.sqrt(2)
        g_hat   = g + cfg.kappa * np.abs(g) * delta_n
        return g, g_hat

    def _sample_g_RU(self, d_IRS_U: np.ndarray):
        """Sample IRS→user channels. Returns (g (M,K) true, g_hat (M,K) estimated)."""
        cfg      = self.cfg
        M, K     = cfg.M, cfg.K
        G_U      = 10 ** (cfg.G_U_dBi / 10)
        g_sf_lin = 10 ** (cfg.g_sf_dB / 10)
        pl       = self._irs_user_path_loss(d_IRS_U)   # (M, K)

        rain_dB = self.rng.normal(cfg.rain_mean, np.sqrt(cfg.rain_var_RU), size=(M, K))
        eps_RU  = 10.0 ** (-rain_dB / 10.0)
        g_sf    = (self.rng.standard_normal((M, K))
                   + 1j * self.rng.standard_normal((M, K))) / np.sqrt(2)
        g       = np.sqrt(G_U * g_sf_lin * pl * eps_RU) * g_sf

        delta_n = (self.rng.standard_normal((M, K))
                   + 1j * self.rng.standard_normal((M, K))) / np.sqrt(2)
        g_hat   = g + cfg.kappa * np.abs(g) * delta_n
        return g, g_hat

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def generate(self, user_pos: np.ndarray, irs_pos: np.ndarray) -> dict:
        """
        Sample a full channel realisation for one episode.

        Returns
        -------
        dict with keys:
          g_SR      (M,)  complex   TRUE satellite→IRS channels
          g_RU      (M,K) complex   TRUE IRS→user channels
          g_SU      (K,)  complex   TRUE satellite→user direct channels
          g_SR_hat  (M,)  complex   estimated (imperfect) counterparts
          g_RU_hat  (M,K) complex
          g_SU_hat  (K,)  complex
          d_SU      (K,)  float
          d_IRS_U   (M,K) float
          beta      (M,)  float
          su_blocked (K,) bool      direct link blocked by a building
        """
        d_SU    = self._slant_ranges(user_pos)
        d_SR    = self._slant_ranges(irs_pos)
        d_IRS_U = self._irs_user_distances(user_pos, irs_pos)

        beta = np.full(self.cfg.M, self.cfg.beta_IRS)

        g_SR, g_SR_hat = self._sample_g_SR(d_SR)
        g_SU, g_SU_hat = self._sample_g_SU(d_SU)
        g_RU, g_RU_hat = self._sample_g_RU(d_IRS_U)

        su_blocked = self._building_blocks_matrix(user_pos, irs_pos).any(axis=0)
        if np.any(su_blocked):
            g_SU[su_blocked]     *= self.cfg.beta_blocking
            g_SU_hat[su_blocked] *= self.cfg.beta_blocking

        return {
            'g_SR':       g_SR,
            'g_RU':       g_RU,
            'g_SU':       g_SU,
            'g_SR_hat':   g_SR_hat,
            'g_RU_hat':   g_RU_hat,
            'g_SU_hat':   g_SU_hat,
            'd_SU':       d_SU,
            'd_IRS_U':    d_IRS_U,
            'beta':       beta,
            'su_blocked': su_blocked,
        }

    def update_user_channels(self, user_pos: np.ndarray,
                             irs_pos: np.ndarray,
                             channels: dict) -> dict:
        """
        Recompute user-dependent channels after a mobility step.
        g_SR / g_SR_hat are kept unchanged (IRS position fixed per episode).
        """
        d_SU    = self._slant_ranges(user_pos)
        d_IRS_U = self._irs_user_distances(user_pos, irs_pos)

        g_SU, g_SU_hat = self._sample_g_SU(d_SU)
        g_RU, g_RU_hat = self._sample_g_RU(d_IRS_U)

        su_blocked = self._building_blocks_matrix(user_pos, irs_pos).any(axis=0)
        if np.any(su_blocked):
            g_SU[su_blocked]     *= self.cfg.beta_blocking
            g_SU_hat[su_blocked] *= self.cfg.beta_blocking

        return {
            **channels,
            'g_SU':       g_SU,
            'g_SU_hat':   g_SU_hat,
            'g_RU':       g_RU,
            'g_RU_hat':   g_RU_hat,
            'd_SU':       d_SU,
            'd_IRS_U':    d_IRS_U,
            'su_blocked': su_blocked,
        }
