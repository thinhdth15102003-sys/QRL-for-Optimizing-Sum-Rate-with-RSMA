"""
rate.py
-------
SINR and achievable rate computation for the Multi-IRS RSMA system.

Channel convention  (see effective_channels_all / compute_sum_rate `use_true`)
------------------
  DESIGN view  (use_true=False, default): effective channel h_k from ESTIMATED
    ĝ (g_*_hat) — precoding (assignment/phase/power/Ck) is designed on the
    imperfect CSI estimate the system actually has.
  ACHIEVED view (use_true=True): h_k from the TRUE channel g (g_*) — the
    reward / achieved-rate / QoS score the ĝ-designed precoding against reality
    (env.step reward, counterfactual reward).  Imperfect CSI costs performance
    precisely through this design-vs-reality gap.

RSMA grouping  (G+1 common streams, G = active IRS count)
----------------------------------------------------------
  Direct-link users (assignment=0) form group 0 with their own common stream.
  Each ACTIVE IRS m (assignment=m, m ∈ {1..M}, with at least one user) forms
  one group.  Only G ≤ M IRS are active per step → G+1 common streams total.

  active_irs_ids : sorted list of physical IRS gids (1-based) with ≥1 user.
  gid_to_wc_idx  : {0: 0, active_irs_ids[0]: 1, active_irs_ids[1]: 2, ...}
  w_c_vec        : (G+1,) — w_c_vec[gid_to_wc_idx[gid]] = power for group gid

Effective channel
-----------------
  IRS user (assignment=m, m ≥ 1):
      h_k = β_m · Σ_n [ conj(g_SR_hat[m]) · φ_n · g_RU_hat[m,n,k] ]
      (g_SR scalar under far-field sat→IRS assumption; φ_n = diag(Φ_m)[n];
       g_RU is PER-ELEMENT, so the φ_n are a real degree of freedom — aligned
       phases give Σ_n |g_RU[m,n,k]|, i.e. coherent combining)
  Direct user (assignment=0):
      h_k = g_SU_hat[k]

SINR formulas  (per-group SIC, gid = 0..G)
------------------------------------------
  Let i = gid_to_wc_idx[gid].

  SINR_c[k] = |h_k|² · w_c_vec[i]
              ───────────────────────────────────────────────────
              |h_k|² · (Σ w_p + Σ w_c_vec − w_c_vec[i]) + σ²

  SINR_p[k] = |h_k|² · w_p[k]
              ─────────────────────────────────────────────────────
              |h_k|² · (Σ w_p − w_p[k] + Σ w_c_vec − w_c_vec[i]) + σ²
"""

import numpy as np
from typing import Optional, List

from istn.config import SystemConfig


class RateComputer:

    def __init__(self, cfg: SystemConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    # gid → w_c_vec index mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _make_wc_map(active_irs_ids: List[int]) -> dict:
        """Build gid → w_c_vec index mapping.  gid 0 → 0, active_irs_ids[i] → i+1"""
        wc_map = {0: 0}
        for i, gid in enumerate(active_irs_ids):
            wc_map[gid] = i + 1
        return wc_map

    # ------------------------------------------------------------------ #
    # Effective channel  (fully vectorised — no Python loop over K)
    # ------------------------------------------------------------------ #

    def effective_channels_all(self, assignment: np.ndarray,
                               Phi: np.ndarray,
                               channels: dict,
                               use_true: bool = False) -> np.ndarray:
        """
        Compute h[k] for all K users simultaneously.  Returns (K,) complex.

        For IRS user k with assignment[k]=m (1-based):
            h[k] = beta[m-1] * conj(g_SR[m-1]) * sum_n Phi[m-1,n,n] * g_RU[m-1, n, k]
        For direct user k with assignment[k]=0:
            h[k] = g_SU[k]

        use_true : which channel realisation to evaluate the effective channel on.
            False (default) → ESTIMATED ĝ (g_*_hat): the imperfect CSI the precoding
                              is DESIGNED on (assignment/phase/power decisions).
            True            → TRUE g (g_*): the actual physical channel the signal
                              propagates through — used for the ACHIEVED rate/reward,
                              so the ĝ-designed precoding is scored on reality.
        """
        K = self.cfg.K
        N = Phi.shape[1]
        sfx = '' if use_true else '_hat'                 # '' = true g, '_hat' = ĝ
        g_SR = channels['g_SR' + sfx]
        g_RU = channels['g_RU' + sfx]
        g_SU = channels['g_SU' + sfx]

        # per-element phase of each IRS: (M, N)
        phi_diag = Phi[:, np.arange(N), np.arange(N)]

        # PER-ELEMENT combining: the n-th element reflects with its own phase onto
        # its own channel, so the phases are a real degree of freedom:
        #   h_irs[m,k] = beta[m] * conj(g_SR[m]) * sum_n phi_diag[m,n] * g_RU[m,n,k]
        # (aligned phases give sum_n |g_RU[m,n,k]|, i.e. coherent combining).
        irs_coeff = channels['beta'] * g_SR.conj()                 # (M,)
        h_irs = irs_coeff[:, np.newaxis] * np.einsum('mn,mnk->mk',
                                                     phi_diag, g_RU)

        # Select per user: IRS path or direct path
        mask  = assignment > 0                           # (K,) bool
        m_idx = np.clip(assignment - 1, 0, None)         # (K,) 0-based, clamped
        h_from_irs = h_irs[m_idx, np.arange(K)]          # (K,)
        return np.where(mask, h_from_irs, g_SU)

    # ------------------------------------------------------------------ #
    # Vectorised SINR helper
    # ------------------------------------------------------------------ #

    def _sinr_all(self, h: np.ndarray, assignment: np.ndarray,
                  w_p: np.ndarray, w_c_vec: np.ndarray,
                  wc_map: dict, s2):
        """
        Compute private and common SINR for all K users in one pass.

        s2 : float | (K,) ndarray — per-user noise variance, broadcasts against h2.

        Returns
        -------
        sinr_p : (K,) float
        sinr_c : (K,) float
        own_wc : (K,) float   — w_c_vec[wc_idx[k]] per user
        """
        M  = self.cfg.M
        h2 = np.abs(h) ** 2               # (K,)
        total_wp = float(np.sum(w_p))
        total_wc = float(np.sum(w_c_vec))

        # Vectorised wc_idx lookup: build (M+1,) lookup table
        wc_idx_arr       = np.zeros(M + 1, dtype=int)
        for gid, idx in wc_map.items():
            wc_idx_arr[gid] = idx
        wc_idx_k = wc_idx_arr[assignment]  # (K,)
        own_wc   = w_c_vec[wc_idx_k]      # (K,)

        sinr_p = h2 * w_p / (h2 * (total_wp - w_p + total_wc - own_wc) + s2)
        sinr_c = h2 * own_wc / (h2 * (total_wp + total_wc - own_wc) + s2)
        return sinr_p, sinr_c, own_wc

    # ------------------------------------------------------------------ #
    # Group building (vectorised)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_groups(assignment: np.ndarray) -> dict:
        """Returns {gid: array_of_user_indices} using np.where instead of a K-loop."""
        groups = {}
        for gid in np.unique(assignment):
            groups[int(gid)] = list(np.where(assignment == int(gid))[0])
        return groups

    # ------------------------------------------------------------------ #
    # Partial-rate helper  (needed by CkMLP before step)
    # ------------------------------------------------------------------ #

    def compute_rates_partial(self,
                              assignment: np.ndarray,
                              Phi: np.ndarray,
                              channels: dict,
                              w_p: np.ndarray,
                              w_c_vec: np.ndarray,
                              active_irs_ids: Optional[List[int]] = None,
                              sigma2=None) -> dict:
        """
        Compute effective channels, private rates, and per-group common rates
        without making any C_k allocation decision.

        sigma2 : float | (K,) ndarray | None — per-user noise variance;
                 falls back to the nominal scalar cfg.sigma2 when None.

        Returns dict with: R_private, R_c_group, h_eff, SINR_p, SINR_c, groups
        """
        cfg = self.cfg
        K   = cfg.K
        s2  = sigma2 if sigma2 is not None else cfg.sigma2

        if active_irs_ids is None:
            active_irs_ids = sorted(set(int(a) for a in assignment[:K] if a > 0))

        wc_map = self._make_wc_map(active_irs_ids)
        h      = self.effective_channels_all(assignment, Phi, channels)
        wp     = w_p[:K]

        groups = self._build_groups(assignment)
        sinr_p, sinr_c, _ = self._sinr_all(h, assignment, wp, w_c_vec, wc_map, s2)

        R_c_group: dict = {}
        for gid, members in groups.items():
            R_c_group[gid] = float(np.log2(1.0 + sinr_c[members].min()))

        R_private = np.log2(1.0 + sinr_p)

        return {
            'R_private':  R_private,
            'R_c_group':  R_c_group,
            'h_eff':      h,
            'SINR_p':     sinr_p,
            'SINR_c':     sinr_c,
            'groups':     groups,
        }

    # ------------------------------------------------------------------ #
    # Main rate computation
    # ------------------------------------------------------------------ #

    def compute_sum_rate(self,
                         assignment: np.ndarray,
                         Phi: np.ndarray,
                         channels: dict,
                         w_p: np.ndarray,
                         w_c_vec: np.ndarray,
                         C_k: Optional[np.ndarray] = None,
                         active_irs_ids: Optional[List[int]] = None,
                         sigma2=None,
                         use_true: bool = False) -> dict:
        """
        Compute achievable sum-rate for all users under multi-group RSMA.

        use_true : evaluate the achieved rate on the TRUE channel g (reward /
            achieved-rate / QoS) when True; on the ESTIMATED ĝ (design view) when
            False.  The precoding (Phi, w_p, w_c) is unchanged — only the channel
            the signal is scored against differs (see effective_channels_all).

        Parameters
        ----------
        assignment     : (K,) int      0=direct, 1..M=IRS (1-based)
        Phi            : (M, N, N) complex
        channels       : dict from ChannelModel.generate()
        w_p            : (K,) float    private power per user
        w_c_vec        : (G+1,) float  common-stream power; G = |active_irs_ids|
        C_k            : (K,) float or None  explicit common-rate share per user
        active_irs_ids : sorted list of physical IRS gids with ≥1 user
        sigma2         : float | (K,) ndarray | None  per-step noise variance;
                         PER-USER since the per-user-noise change, None → cfg.sigma2

        Returns
        -------
        dict: sum_rate, R_private, R_common_group, C_k, SINR_p, SINR_c,
              h_eff, feasible, power_ok, groups
        """
        cfg = self.cfg
        K   = cfg.K
        s2  = sigma2 if sigma2 is not None else cfg.sigma2

        if active_irs_ids is None:
            active_irs_ids = sorted(set(int(a) for a in assignment[:K] if a > 0))

        pre = self._sigma_invariants(assignment, Phi, channels, active_irs_ids, use_true)
        return self._finish_sum_rate(pre, assignment, w_p, w_c_vec, C_k, s2)

    # ------------------------------------------------------------------ #
    # σ²-invariant / σ²-dependent split
    # ------------------------------------------------------------------ #
    #
    # ⚡ 2026-07-21. env.step averages the reward over `reward_noise_avg` (=16)
    # σ² draws that share ONE channel realisation. h_eff, the user groups and the
    # common-power map do not depend on σ² at all, yet they were recomputed on
    # every draw — 36% of compute_sum_rate wasted 15 times over (h_eff alone is
    # 26%, it runs the einsum over all N reflecting elements).
    #
    # Splitting them out is EXACTLY equivalent: same function, same inputs, just
    # evaluated once. Verified bit-identical against the old path — this is a
    # pure book-keeping change, no numerical trade whatsoever.

    def _sigma_invariants(self, assignment, Phi, channels, active_irs_ids, use_true):
        """Everything in compute_sum_rate that does NOT depend on σ²."""
        return {
            'wc_map': self._make_wc_map(active_irs_ids),
            'h':      self.effective_channels_all(assignment, Phi, channels,
                                                  use_true=use_true),
            'groups': self._build_groups(assignment),
            'active_irs_ids': active_irs_ids,
        }

    def _finish_sum_rate(self, pre, assignment, w_p, w_c_vec, C_k, s2) -> dict:
        """The σ²-dependent remainder, given precomputed invariants."""
        cfg = self.cfg
        K   = cfg.K
        wc_map, h, groups = pre['wc_map'], pre['h'], pre['groups']
        wp = w_p[:K]
        sinr_p, sinr_c, _ = self._sinr_all(h, assignment, wp, w_c_vec, wc_map, s2)

        R_private = np.log2(1.0 + sinr_p)

        R_common_group: dict = {}
        for gid, members in groups.items():
            R_common_group[gid] = float(np.log2(1.0 + sinr_c[members].min()))

        # Common-rate allocation per user
        if C_k is None:
            C_k_out = np.zeros(K)
            for gid, members in groups.items():
                share = R_common_group.get(gid, 0.0) / max(len(members), 1)
                C_k_out[np.array(members)] = share
        else:
            # Enforce per-group constraints: Σ_{k∈g} C_k = R_c_g  (rescaling)
            C_k_out = np.array(C_k[:K], dtype=float)
            for gid, members in groups.items():
                R_c_g = float(R_common_group.get(gid, 0.0))
                idx = np.array(members)
                if R_c_g < 1e-12:
                    C_k_out[idx] = 0.0
                else:
                    C_k_out[idx] = np.maximum(C_k_out[idx], 1e-10)
                    C_k_out[idx] *= R_c_g / float(C_k_out[idx].sum())

        R_total  = float(np.sum(C_k_out + R_private))
        rate_ok  = bool(np.all(R_private + C_k_out >= cfg.D_k_bps_hz))
        power_ok = (float(np.sum(w_c_vec)) + float(np.sum(w_p[:K]))) <= cfg.P_S * (1 + 1e-9)
        feasible = rate_ok and power_ok

        return {
            'sum_rate':       R_total,
            'R_private':      R_private,
            'R_common_group': R_common_group,
            'C_k':            C_k_out,
            'SINR_p':         sinr_p,
            'SINR_c':         sinr_c,
            'h_eff':          h,
            'feasible':       feasible,
            'power_ok':       power_ok,
            'groups':         groups,
        }
