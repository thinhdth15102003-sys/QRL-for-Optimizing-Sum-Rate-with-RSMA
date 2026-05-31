"""
env.py
------
Main environment class for the Multi-IRS ISTN system.

Key design decisions
--------------------
  Fixed K: all K users are active every step (no K variation).

  G active IRS: at each step the agent assigns users to G ≤ M IRS panels.
    - assignment[k] ∈ {0, …, M}: 0 = direct link, 1..M = physical IRS index.
    - active_irs_ids: sorted list of non-zero assignment values that appear.
    - G = len(active_irs_ids).
    - w_c_vec: (G+1,)  — index 0 = direct group, index i = active_irs_ids[i-1].

  Channel convention:
    - Env feeds TRUE complex channels g to the agent via _get_obs().
    - Rate computation (rate.py) uses estimated channels g_hat.

Action dict keys accepted by step()
-------------------------------------
  'assignment'  (K,) int    in {0, …, M}
  'phase_idx'   (M, N) int  in {0,..,L-1}
  'w_p'         (K,)   float                private power per user
  'w_c_vec'     (G+1,) float                common-stream power per active group
                                            index 0 → direct, i → active_irs_ids[i-1]
  'C_k'         (K,)   float                common-rate share per user

Any subset of keys may be provided.  When 'assignment' and 'w_c_vec' are both
provided they must be consistent (w_c_vec.shape[0] == G+1 for the NEW assignment).

Observation dict (from _get_obs)
----------------------------------
  'g_SR'        (M,)   complex  TRUE satellite→IRS channels
  'g_RU'        (M, K) complex  TRUE IRS→user channels
  'g_SU'        (K,)   complex  TRUE satellite→user channels
  'beta'        (M,)   float    per-IRS reflection efficiency
  'Phi_angle'   (M, N) float    current IRS phase angles (radians)
  'assignment'  (K,)   int      current user → link assignment
  'K_active'    int             = cfg.K (kept for API compatibility)
"""

import numpy as np
from typing import Optional, List

from istn.config  import SystemConfig
from istn.channel import ChannelModel
from CSI.irs      import IRSPhaseModel
from CSI.rate     import RateComputer


class ISTNEnv:

    def __init__(self,
                 cfg:        Optional[SystemConfig] = None,
                 seed:       Optional[int]          = None,
                 n_steps_ep: int                    = 150,
                 reward_noise_avg: int              = 1):
        self.cfg           = cfg or SystemConfig()
        self.rng           = np.random.default_rng(seed)
        self._n_confined_steps = n_steps_ep // 2
        # Average the reward over this many per-step noise realisations to cut
        # reward variance (lets the critic fit); preserves E[reward]. 1 = off.
        self.reward_noise_avg = max(1, int(reward_noise_avg))

        self.channel_model = ChannelModel(self.cfg, self.rng)
        self.phase_model   = IRSPhaseModel(self.cfg)
        self.rate_computer = RateComputer(self.cfg)

        # Episode state (initialised by reset)
        self.channels:        Optional[dict]       = None
        self.Phi:             Optional[np.ndarray] = None   # (M, N, N) complex
        self.assignment:      Optional[np.ndarray] = None   # (K,) int
        self.w_p:             Optional[np.ndarray] = None   # (K,) float
        self.w_c_vec:         Optional[np.ndarray] = None   # (G+1,) float
        self.C_k:             Optional[np.ndarray] = None   # (K,) float or None
        self.active_irs_ids:  List[int]            = []     # sorted physical IRS gids
        self.user_pos:        Optional[np.ndarray] = None   # (K, 2) km
        self.irs_pos:         Optional[np.ndarray] = None   # (M, 2) km
        self._step_count:     int = 0

        # Confined-user tracking (populated in reset)
        self._confined_users = np.zeros(self.cfg.K, dtype=bool)
        self._user_building  = np.full(self.cfg.K, -1, dtype=int)

    @property
    def K_active(self) -> int:
        """Always equal to cfg.K — retained for API compatibility."""
        return self.cfg.K

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #

    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Start a new episode.
        K is fixed (= cfg.K) across all episodes and steps.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.channel_model.rng = self.rng

        self.irs_pos = self._sample_irs_positions()

        # Spawn 2/3 of users inside building footprints (confined), rest free.
        K = self.cfg.K
        n_blocked = int(round(K * 2 / 3))
        n_free    = K - n_blocked

        self._confined_users[:] = False
        self._user_building[:]  = -1

        blocked_bld = (self.rng.integers(0, self.cfg.M, size=n_blocked)
                       if n_blocked > 0 else np.empty(0, dtype=int))
        self._confined_users[:n_blocked] = True
        self._user_building[:n_blocked]  = blocked_bld

        user_pos = np.zeros((K, 2))
        if n_blocked > 0:
            user_pos[:n_blocked] = self._sample_in_footprints(blocked_bld)
        user_pos[n_blocked:]     = self._sample_free_positions(n_free)
        self.user_pos = user_pos

        self.channels   = self.channel_model.generate(self.user_pos, self.irs_pos)
        self.Phi        = self.phase_model.random_phi(self.rng)
        self.assignment = self.rng.integers(0, self.cfg.M + 1, size=K)

        # Derive active IRS from initial assignment
        self.active_irs_ids = sorted(set(int(a) for a in self.assignment if a > 0))
        G = len(self.active_irs_ids)

        # Equal power split: 80% private, 20% common (G+1 groups)
        P = self.cfg.P_S
        n_common = max(G + 1, 1)
        self.w_c_vec = np.full(n_common, P * 0.2 / n_common)
        self.w_p     = np.full(K, P * 0.8 / K)
        self.C_k     = None

        self._step_count = 0
        return self._get_obs()

    def step(self, action: dict) -> tuple:
        """
        Apply action, compute reward, then advance user positions.

        Returns
        -------
        obs    : dict
        reward : float  sum-rate minus QoS penalty
        done   : bool   always False
        info   : dict
        """
        self._apply_action(action)

        K   = self.cfg.K
        D_k = np.full(K, self.cfg.D_k_bps_hz)

        # Monte-Carlo average the reward over R independent noise realisations.
        # compute_sum_rate is a pure function of sigma2 here (CSI g_hat is fixed
        # within a step), so this reduces reward variance ~1/√R without biasing
        # the expected reward — the critic can then actually fit V(s).
        R_avg       = self.reward_noise_avg
        reward_acc  = 0.0
        sum_rate_acc = 0.0
        qp_acc      = 0.0
        R_tot_acc   = np.zeros(K)
        sigma2_acc  = 0.0
        result      = None
        for _ in range(R_avg):
            sigma2_step = self.channel_model.sample_noise_sigma2()
            res = self.rate_computer.compute_sum_rate(
                self.assignment, self.Phi, self.channels,
                self.w_p, self.w_c_vec, C_k=self.C_k,
                active_irs_ids=self.active_irs_ids,
                sigma2=sigma2_step,
            )
            R_tot_r    = res['R_private'] + res['C_k']             # (K,)
            shortfall  = np.maximum(0.0, D_k - R_tot_r)
            qp_terms   = (shortfall / (D_k + self.cfg.epsilon_qp)) ** 2
            qp_r       = self.cfg.lambda_D * float(np.sum(qp_terms))

            reward_acc   += res['sum_rate'] - qp_r
            sum_rate_acc += res['sum_rate']
            qp_acc       += qp_r
            R_tot_acc    += R_tot_r
            sigma2_acc   += sigma2_step
            result        = res                                   # keep last (groups, SINR, feasible)

        reward      = reward_acc / R_avg
        sigma2_step = sigma2_acc / R_avg
        # Overwrite noise-dependent fields with their averages so logged metrics
        # (sum_rate, QoS count from R_tot) match the averaged reward.
        result      = dict(result)
        result['sum_rate'] = sum_rate_acc / R_avg
        R_tot       = R_tot_acc / R_avg
        qp_penalty  = qp_acc / R_avg

        self._step_count += 1
        self.user_pos = self._walk_users(self.user_pos)
        self.channels = self.channel_model.update_user_channels(
            self.user_pos, self.irs_pos, self.channels
        )

        info = {
            **result,
            'qp_penalty':      qp_penalty,
            'R_tot':           R_tot,
            'D_k':             D_k,
            'sigma2':          sigma2_step,
            'K_active':        K,
            'active_irs_ids':  list(self.active_irs_ids),
            'beta':            self.channels['beta'],
            'assignment':      self.assignment.copy(),
            'step':            self._step_count,
            'user_pos':        self.user_pos.copy(),
            'irs_pos':         self.irs_pos.copy(),
        }
        return self._get_obs(), reward, False, info

    # ------------------------------------------------------------------ #
    # Position helpers
    # ------------------------------------------------------------------ #

    def _sample_irs_positions(self) -> np.ndarray:
        """IRS within irs_spawn_radius_frac·R_LoS, min-separated (no clumping)."""
        cfg   = self.cfg
        M     = cfg.M
        r_max = cfg.R_LoS_km * getattr(cfg, 'irs_spawn_radius_frac', 1.0)
        # Target minimum mutual separation so M IRS spread over the inner disk.
        min_sep = (r_max / np.sqrt(M)) if M > 1 else 0.0
        pos = np.zeros((M, 2))
        for m in range(M):
            cand = pos[m]
            for _attempt in range(100):                 # rejection sampling
                r  = r_max * np.sqrt(self.rng.uniform(0, 1))
                th = self.rng.uniform(0, 2 * np.pi)
                cand = np.array([r * np.cos(th), r * np.sin(th)])
                if m == 0 or np.all(
                        np.linalg.norm(pos[:m] - cand, axis=1) >= min_sep):
                    break                               # accept (else retry / last)
            pos[m] = cand
        return pos

    def _sample_free_positions(self, n: int) -> np.ndarray:
        """Free (non-confined) users within user_free_radius_frac·R_LoS."""
        if n == 0:
            return np.zeros((0, 2))
        r_max = self.cfg.R_LoS_km * getattr(self.cfg, 'user_free_radius_frac', 1.0)
        r     = r_max * np.sqrt(self.rng.uniform(0, 1, size=n))
        theta = self.rng.uniform(0, 2 * np.pi, size=n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def _sample_in_footprints(self, building_indices: np.ndarray) -> np.ndarray:
        a  = self.cfg.d_block_km
        n  = len(building_indices)
        cx = self.irs_pos[building_indices, 0]
        cy = self.irs_pos[building_indices, 1]
        x  = self.rng.uniform(-a, a, size=n) + cx
        y  = self.rng.uniform(-a, a, size=n) + cy
        return np.column_stack([x, y])

    def _walk_users(self, user_pos: np.ndarray) -> np.ndarray:
        cfg   = self.cfg
        angle = self.rng.uniform(0, 2 * np.pi, size=cfg.K)
        delta = cfg.user_step_km * np.column_stack([np.cos(angle), np.sin(angle)])
        new_pos = user_pos + delta

        # First half of episode: confined users cannot leave building footprint
        if self._step_count <= self._n_confined_steps:
            conf_idx = np.where(self._confined_users)[0]
            if len(conf_idx) > 0:
                a  = cfg.d_block_km
                mi = self._user_building[conf_idx]          # building idx per user
                cx = self.irs_pos[mi, 0]
                cy = self.irs_pos[mi, 1]
                new_pos[conf_idx, 0] = np.clip(new_pos[conf_idx, 0], cx - a, cx + a)
                new_pos[conf_idx, 1] = np.clip(new_pos[conf_idx, 1], cy - a, cy + a)

        # Keep all users within LoS zone
        dist    = np.sqrt(new_pos[:, 0]**2 + new_pos[:, 1]**2)
        outside = dist > cfg.R_LoS_km
        if np.any(outside):
            scale            = cfg.R_LoS_km / dist[outside]
            new_pos[outside] *= scale[:, np.newaxis]

        return new_pos

    # ------------------------------------------------------------------ #
    # RL helper properties
    # ------------------------------------------------------------------ #

    def get_state_vector(self) -> np.ndarray:
        """
        Flatten observation to a 1-D float array for MLP input.

        Layout (total = state_dim):
          Re/Im g_SR    : 2M        TRUE sat→IRS channel Re and Im parts
          Re/Im g_RU    : 2×M×K    TRUE IRS→user channel Re and Im parts
          Re/Im g_SU    : 2K        TRUE sat→user channel Re and Im parts
          beta          : M         IRS reflection efficiency
          cos(Phi_angle): M×N  ┐   circular encoding of IRS phase angles
          sin(Phi_angle): M×N  ┘
          assignment    : K
        """
        obs = self._get_obs()
        g_sr = obs['g_SR']                       # (M,) complex
        g_ru = obs['g_RU']                       # (M, K) complex
        g_su = obs['g_SU']                       # (K,) complex
        phi  = obs['Phi_angle']                  # (M, N)
        return np.concatenate([
            g_sr.real, g_sr.imag,
            g_ru.real.flatten(), g_ru.imag.flatten(),
            g_su.real, g_su.imag,
            obs['beta'],
            np.cos(phi).flatten(),
            np.sin(phi).flatten(),
            obs['assignment'].astype(float),
        ])

    @property
    def state_dim(self) -> int:
        cfg = self.cfg
        # 2M (g_SR Re+Im) + 2MK (g_RU Re+Im) + 2K (g_SU Re+Im)
        # + M (beta) + 2MN (cos/sin Phi) + K (assignment)
        return (2 * cfg.M
              + 2 * cfg.M * cfg.K
              + 2 * cfg.K
              + cfg.M
              + 2 * cfg.M * cfg.N
              + cfg.K)

    @property
    def n_phase_levels(self) -> int:
        return self.phase_model.n_levels

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _apply_action(self, action: dict) -> None:
        """
        Apply action dict, enforcing:
          assignment  ∈ {0, …, M}
          phase_idx   ∈ {0, …, n_levels-1}  → Phi (unit-modulus diagonal)
          w_p, w_c_vec ≥ 0,  Σ w_p + Σ w_c_vec = P_S
          C_k ≥ 0  (group-sum equality enforced by rate.py)
        """
        cfg = self.cfg

        # ── IRS assignment ────────────────────────────────────────────────
        if 'assignment' in action:
            self.assignment = np.clip(
                np.array(action['assignment'], dtype=int), 0, cfg.M)

        # Recompute active_irs_ids from (possibly updated) assignment
        self.active_irs_ids = sorted(
            set(int(a) for a in self.assignment if a > 0))
        G = len(self.active_irs_ids)

        # ── Phase shifts ──────────────────────────────────────────────────
        if 'phase_idx' in action:
            idx      = np.clip(np.array(action['phase_idx'], dtype=int),
                               0, self.phase_model.n_levels - 1)
            phases   = self.phase_model.index_to_phase(idx)
            self.Phi = self.phase_model.build_phi(phases)

        # ── Common-rate shares ────────────────────────────────────────────
        if 'C_k' in action:
            self.C_k = np.maximum(np.array(action['C_k'], dtype=float), 0.0)

        # ── Power allocation: Σ w_p + Σ w_c_vec = P_S ───────────────────
        if 'w_p' in action or 'w_c_vec' in action:
            w_p_raw = (np.abs(np.array(action['w_p'], dtype=float))
                       if 'w_p' in action else self.w_p.copy())

            if 'w_c_vec' in action:
                # Caller provides (G+1,) matching the current active_irs_ids
                w_c_raw = np.abs(np.array(action['w_c_vec'], dtype=float))
            elif len(self.active_irs_ids) + 1 == len(self.w_c_vec):
                # G unchanged — keep existing w_c_vec
                w_c_raw = self.w_c_vec.copy()
            else:
                # G changed but no w_c_vec provided — equal split default
                w_c_raw = np.ones(G + 1) * cfg.P_S * 0.2 / max(G + 1, 1)

            total = float(np.sum(w_c_raw)) + float(np.sum(w_p_raw))
            scale = cfg.P_S / max(total, 1e-12)
            self.w_p     = w_p_raw * scale
            self.w_c_vec = w_c_raw * scale

        elif len(self.active_irs_ids) + 1 != len(self.w_c_vec):
            # Only assignment changed; resize w_c_vec to new G+1
            total_wc     = float(np.sum(self.w_c_vec))
            self.w_c_vec = np.full(G + 1, total_wc / max(G + 1, 1))

    def _get_obs(self) -> dict:
        ch = self.channels
        return {
            # TRUE channels (complex) — split into Re/Im by each sub-actor's state builder
            'g_SR':       ch['g_SR'].copy(),
            'g_RU':       ch['g_RU'].copy(),
            'g_SU':       ch['g_SU'].copy(),
            'beta':       ch['beta'].copy(),
            'Phi_angle':  np.angle(
                self.Phi[:, np.arange(self.cfg.N), np.arange(self.cfg.N)]
            ) % (2 * np.pi),
            'assignment': self.assignment.copy(),
            'K_active':   self.cfg.K,
        }
