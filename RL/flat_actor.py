"""
flat_actor.py
-------------
FlatActor — the flat-PPO baseline, porting the METHOD of

    Y. Meng et al., "Sum-Rate Maximization in STAR-RIS-Assisted RSMA Networks:
    A PPO-Based Algorithm", IEEE IoT-J, 2024.

into this project's ISTN/grouped-RSMA environment.

WHAT IS BEING PORTED
--------------------
Meng's agent is a SINGLE PPO actor that maps the raw channel state to the ENTIRE
action vector in one forward pass — beamforming, amplitude/phase coefficients and
the common-rate split all come out of one network head, are scored by one joint
log-density, and are updated by one PPO objective. There is no decomposition of
the action, no sub-policy conditioned on another sub-policy's choice, and no
auxiliary supervision.

That single architectural property — flat vs. hierarchical — is what this class
reproduces. Their SYSTEM (terrestrial 4-antenna BS, STAR-RIS with a transmission/
reflection split, K=2 users, Rician terrestrial path loss, perfect CSI) is NOT
reproduced: it is a different physical model from the satellite/IRS/grouped-RSMA
setting here, so their absolute sum-rate numbers are not comparable to ours and
are not claimed to be. Their Table I hyper-parameters that DO transfer are used
verbatim (see `MENG_DEFAULTS` below).

AUDIT AGAINST THE ORIGINAL — every element of their agent, and its status here.
Cite as: methodology from Meng et al. 2024, adapted to this environment.

  KEPT AS-IS (their design, reproduced)
    · one actor network → the ENTIRE action in one forward pass       [core idea]
    · clipped surrogate J^CLIP, ε = 0.2                               [their (22)]
    · separate critic network, actor/critic updated in separate loops [their Alg 1]
    · state = raw per-link channels (Re/Im) + the COMBINED channel
      under the currently-applied phase config + last-step rate feedback
    · per-feature Z-score input normalisation over the episode        [their §III-B]
    · a constraint-satisfaction step guaranteeing the power budget    [their CSP,
      (26)-(27)] — here the Dirichlet is budget-feasible by construction, so the
      projection is unnecessary rather than omitted
    · critic LR 3e-4, γ, clip ε from their Table I

  CHANGED — FORCED by the different environment
    · Gaussian policy → categorical (routing, 2-bit phase) + Dirichlet (power,
      C_k). Their action is fully continuous; ours has a discrete IRS assignment
      and a discrete phase codebook, where a Gaussian is not defined. Dirichlet
      rather than softmax-of-Gaussian on the simplex is also what HQC-HAC uses,
      and for a measured reason: a policy that samples one object and executes
      another has an identically-zero expected gradient (memory
      `dirichlet-phantom-action-fix`).
    · reward = this env's J = ΣR − λ_D·Σ(shortfall/D_k)² instead of their
      (12a)/(12b). The baseline MUST optimise the same objective as the agent it
      is compared against, or the comparison means nothing.
    · state gains `demand` and `blocked`, which their system has no counterpart
      for. Added, not substituted.

  CHANGED — to MATCH HQC-HAC so the comparison isolates the architecture.
  Each of these gives the baseline the BETTER of the two options, never the worse:
    · advantage: GAE(λ) + normalisation + ±adv_clip, instead of their plain
      Monte-Carlo return minus V (their (19)-(21)). GAE has strictly lower
      variance; using their higher-variance estimator would handicap the baseline.
    · entropy bonus (annealed), which they do not use.
    · PPO epochs: P.ppo_epochs with KL early-stopping, instead of their fixed
      J = 40. Without a KL guard ~98% of samples clip and their gradient is
      zeroed — measured, see `--target-kl`.
    · γ, rollout size, minibatch size, LR decay: this project's values.
    · plain value regression, not their clipped value loss (25) — inherited from
      sharing HQC-HAC's critic class.
    · actor and critic are updated in the same minibatch loop; their Alg 1 runs
      the two loops separately (lines 7-9 then 10-12). Same data, same number of
      gradient steps each.

  RE-TUNED for this environment
    · actor LR 3e-4, not their 1e-4. Their §IV-B-2 sweeps six orders of magnitude
      and selects 1e-4 for THEIR system, i.e. they establish that this constant is
      setting-dependent. Re-tuned here on Case 1: 1e-4 → J −0.09, 3e-4 → +0.71.
      `--lr 1e-4` restores their value exactly.

HOW IT DIFFERS FROM HQC-HAC (this is the comparison the paper makes)
--------------------------------------------------------------------
HQC-HAC decomposes the action into a chain and each stage SEES the previous
stage's realised choice:

    assignment ──▶ phase  (input: cascade channel of the CHOSEN groups)
               ──▶ power  (input: h_eff AFTER the chosen phases)
               ──▶ C_k    (input: the realised per-group common rate)

FlatActor gets none of that conditioning. Every logit — routing, phase, power,
C_k — is produced by ONE evaluation of ONE trunk on the raw state, before any of
the four sub-decisions is known. Whatever coupling exists between them has to be
learned inside the trunk from the reward alone.

The only structure that remains is mechanical and unavoidable: the common-power
simplex is masked to the IRS groups that routing actually populated, and the C_k
shares are grouped by routing. Both re-use the logits from the SAME forward pass
— no second network evaluation — so "one network evaluation per state" is exact.

STATE
-----
Meng's state (their Sec. III-A-1) is, term for term:

    Re/Im{H_br}, Re/Im{h_bu,k}, Re/Im{h_ru,k}      raw per-link channels
    Re/Im{h_ru,k^H Θ_p H_br + h_bu,k^H}            the COMBINED channel under the
                                                   currently-applied Θ_p
    r_k, r_0                                       last step's private and common
                                                   rate feedback

Each term is reproduced here on its ISTN analogue, using the ESTIMATED channels
because the agent decides on ĝ = g + Δg, identically to every other actor here:

    [ ĝ_SR (2M) | ĝ_SU (2K) | ĝ_RU (2·M·N·K)   ← their three raw-channel terms
    | h_eff (2K)                                ← their combined-channel term,
    |                                             evaluated at the Φ and routing
    |                                             the env currently has applied
    | r_k (K) | r_0 (1)                         ← their rate feedback
    | demand (K) | blocked (K) ]                ← env-specific extras (see below)

demand and blocked have no counterpart in Meng because their system has neither
per-user QoS variation nor blockage. They are ADDED, not substituted: withholding
env information the hierarchical agent receives would weaken the baseline for a
reason unrelated to its architecture.

NORMALISATION — Meng use "Z-score normalization ... aggregating the data over the
course of the entire episode", i.e. PER-FEATURE running statistics accumulated
across timesteps. That is what is implemented (`_normalize`), and the distinction
from a per-sample LayerNorm matters a great deal here: normalising each feature by
its own history preserves the relative magnitudes BETWEEN features within one
sample, whereas a LayerNorm across the row destroys both the cascade-channel phase
and the user magnitude ordering (both measured in this project — see memory
`phase-state-layernorm-fix` and `power-head-layernorm-ranking-bug`). Running
statistics are frozen and saved with the checkpoint so inference sees training's
normalisation.

ACTION
------
    routing   K × (M+1)      categorical per user   (0 = direct, 1..M = IRS)
    phase     M × N × L      categorical per element
    power     2 + (M+1) + K  three Dirichlets (split / common / private)
    C_k       K              one Dirichlet per routing group

Log-density is the sum of the four blocks; they are conditionally independent
GIVEN the state precisely because they share one forward pass. That is the flat
assumption, and it is the thing under test.
"""
import os
import json
import numpy as np

from RL import dirichlet as _dir
from RL.sub_actors import _Adam, _he, _relu, _softmax_1d


# Meng 2024, Table I — the hyper-parameters that carry over to any PPO agent
# regardless of the physical system. Learning rates, clip range, discount and
# epochs-per-batch are theirs; anything system-specific (N antennas, K=2 users,
# P_max) is NOT here because it belongs to their model, not ours.
MENG_DEFAULTS = {
    'lr_actor':    1e-4,
    'lr_critic':   3e-4,
    'ppo_epsilon': 0.2,
    'gamma':       0.99,
    'ppo_epochs':  40,
    'buffer':      1000,
}


class FlatActor:
    """Single-trunk PPO actor emitting the complete action (Meng 2024 style)."""

    def __init__(self, cfg,
                 hidden=(512, 256),
                 lr: float = MENG_DEFAULTS['lr_actor'],
                 seed: int = None,
                 n_levels: int = 4,
                 power_fairness: float = 0.0,
                 power_priv_frac: float = 0.8,
                 power_fairness_priv: float = None):
        self.K = int(cfg.K)
        self.M = int(cfg.M)
        self.N = int(cfg.N)
        self.L = int(n_levels)
        self.P_S = float(getattr(cfg, 'P_S', 1.0))
        self.HIDDEN = list(hidden)
        self.lr = float(lr)
        self.rng = np.random.default_rng(seed)

        K, M, N, L = self.K, self.M, self.N, self.L

        # ── state layout (block offsets kept so the probes can slice it) ──
        o = 0
        self._sl_sr = slice(o, o + 2 * M);          o += 2 * M
        self._sl_su = slice(o, o + 2 * K);          o += 2 * K
        self._sl_ru = slice(o, o + 2 * M * N * K);  o += 2 * M * N * K
        self._sl_heff = slice(o, o + 2 * K);        o += 2 * K   # Meng: combined ch.
        self._sl_rk = slice(o, o + K);              o += K       # Meng: r_k
        self._sl_r0 = slice(o, o + 1);              o += 1       # Meng: r_0
        self._sl_dem = slice(o, o + K);             o += K
        self._sl_blk = slice(o, o + K);             o += K
        self.d_s = o

        # ── per-feature running Z-score (Meng's input normalisation) ──
        # Welford accumulators over every state the agent has seen. Updated at
        # collection time only, so the states stored in the PPO buffer keep the
        # exact normalisation their log-probs were computed under.
        # TRUE Welford (M2 accumulator, no prior). A "prior of var=1" was tried and
        # is wrong here: it only decays as 1/n, and channel features have variance
        # ~1e-6, so after 600 steps the normalised state still had std 0.046
        # instead of 1 — the network would spend early training on a silently
        # shrunken input. Accumulating M2 from nothing gives a real estimate after
        # two samples; the ±10 clip covers the transient.
        self.rn_mean = np.zeros(self.d_s)
        self.rn_m2 = np.zeros(self.d_s)
        self.rn_count = 0.0

        # ── action layout ──
        n_route = K * (M + 1)
        n_phase = M * N * L
        n_pw    = 2 + (M + 1) + K
        n_ck    = K
        self._sl_route = slice(0, n_route)
        self._sl_phase = slice(n_route, n_route + n_phase)
        o = n_route + n_phase
        self._sl_pw_split   = slice(o, o + 2)
        self._sl_pw_common  = slice(o + 2, o + 2 + (M + 1))
        self._sl_pw_private = slice(o + 2 + (M + 1), o + n_pw)
        self._sl_ck = slice(o + n_pw, o + n_pw + n_ck)
        self.d_out = n_route + n_phase + n_pw + n_ck

        # ── power-fairness prior: OFF by default ──
        # This is *our* device, not Meng's — they shape QoS purely through the
        # reward (their eq. 12b), exactly as this env does through
        # J = ΣR − λ_D·Σ(shortfall/D)². Defaulting it off keeps the port
        # faithful; the flag exists so the baseline can also be run with the
        # same prior HQC-HAC gets, which is the generous variant.
        self.power_fairness = float(power_fairness)
        self.power_priv_frac = float(power_priv_frac)
        self.power_fairness_priv = (float(power_fairness_priv)
                                    if power_fairness_priv is not None
                                    else float(power_fairness))

        self._init_params(seed)
        self.opt = _Adam(lr=self.lr)

    # ── parameters ──────────────────────────────────────────────────────────
    def _init_params(self, seed: int = None) -> None:
        rng = np.random.default_rng(seed)
        dims = [self.d_s] + self.HIDDEN + [self.d_out]
        self._n_layers = len(dims) - 1
        self.params = {}
        for i in range(self._n_layers):
            self.params[f'W{i}'] = _he(dims[i], dims[i + 1], rng)
            self.params[f'b{i}'] = np.zeros(dims[i + 1])
        # The last layer starts at zero so the initial policy is exactly uniform
        # over every block (categorical uniform, Dirichlet α = softplus(0)+α_min).
        # Without this a He-scaled output layer on a ~1000-dim input produces
        # logits of order 10, i.e. a near-deterministic and arbitrary initial
        # policy — the baseline would spend its first thousands of episodes
        # escaping its own initialisation rather than learning.
        self.params[f'W{self._n_layers - 1}'] *= 0.0

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ── state ───────────────────────────────────────────────────────────────
    def _normalize(self, x: np.ndarray, update: bool) -> np.ndarray:
        """Per-feature Z-score against running statistics (Meng's normalisation)."""
        if update:
            self.rn_count += 1.0
            d = x - self.rn_mean
            self.rn_mean = self.rn_mean + d / self.rn_count
            self.rn_m2 = self.rn_m2 + d * (x - self.rn_mean)
        if self.rn_count < 2.0:
            return np.zeros_like(x)          # no scale known yet
        # The floor is RELATIVE to each feature's own magnitude, not absolute: an
        # absolute epsilon would be enormous next to a 1e-6-variance channel
        # feature and would flatten it, which is the failure this replaced.
        sd = np.sqrt(self.rn_m2 / self.rn_count)
        sd = np.maximum(sd, 1e-6 * (np.abs(self.rn_mean) + 1e-12))
        return np.clip((x - self.rn_mean) / np.maximum(sd, 1e-30), -10.0, 10.0)

    def extract_state(self, obs: dict,
                      demand: np.ndarray,
                      blocked: np.ndarray,
                      h_eff: np.ndarray = None,
                      prev_rates: tuple = None,
                      update_norm: bool = True) -> np.ndarray:
        """Meng's state on its ISTN analogue, Z-scored per feature.

        Uses ĝ (estimated CSI) for the same reason every other actor here does:
        the agent must decide on what it can measure. Falls back to the true keys
        only for hand-built mock obs dicts.

        h_eff      : (K,) complex — the combined channel under the Φ and routing
                     the env currently has applied. Meng's
                     `h_ru,k^H Θ_p H_br + h_bu,k^H` term. None → zeros (step 0).
        prev_rates : (R_private (K,), R_common scalar) from the previous step —
                     Meng's r_k and r_0. None → zeros (step 0).
        update_norm: fold this sample into the running statistics. True while
                     collecting, False at inference so evaluation cannot drift
                     the normalisation the policy was trained under.
        """
        K = self.K
        g_sr = np.asarray(obs.get('g_SR_hat', obs['g_SR'])).ravel()
        g_su = np.asarray(obs.get('g_SU_hat', obs['g_SU'])).ravel()
        g_ru = np.asarray(obs.get('g_RU_hat', obs['g_RU'])).ravel()
        he = (np.zeros(K, dtype=complex) if h_eff is None
              else np.asarray(h_eff).ravel())
        if prev_rates is None:
            r_k, r_0 = np.zeros(K), 0.0
        else:
            r_k = np.asarray(prev_rates[0], dtype=float).ravel()
            r_0 = float(prev_rates[1])

        raw = np.concatenate([
            g_sr.real, g_sr.imag,
            g_su.real, g_su.imag,
            g_ru.real, g_ru.imag,
            he.real, he.imag,
            r_k, [r_0],
            np.asarray(demand, dtype=float).ravel(),
            np.asarray(blocked, dtype=float).ravel(),
        ])
        return self._normalize(raw, update_norm)

    # ── trunk ───────────────────────────────────────────────────────────────
    def _forward_logits(self, s_t: np.ndarray):
        acts, pres = [], []
        x = np.asarray(s_t, dtype=float)
        for i in range(self._n_layers):
            acts.append(x)
            z = x @ self.params[f'W{i}'] + self.params[f'b{i}']
            if i < self._n_layers - 1:
                pres.append(z)
                x = _relu(z)
            else:
                return z, {'acts': acts, 'pres': pres}

    def _forward_logits_batch(self, S: np.ndarray):
        acts, pres = [], []
        x = np.asarray(S, dtype=float)
        for i in range(self._n_layers):
            acts.append(x)
            z = x @ self.params[f'W{i}'] + self.params[f'b{i}']
            if i < self._n_layers - 1:
                pres.append(z)
                x = _relu(z)
            else:
                return z, {'acts': acts, 'pres': pres}

    # ── mechanical masks (no extra network evaluation) ──────────────────────
    def _common_mask(self, active_irs_ids) -> np.ndarray:
        m = np.zeros(self.M + 1, dtype=bool)
        m[0] = True
        for gid in active_irs_ids:
            if 1 <= int(gid) <= self.M:
                m[int(gid)] = True
        return m

    @staticmethod
    def _groups(phi: np.ndarray, K_used: int) -> dict:
        g: dict = {}
        for k in range(K_used):
            g.setdefault(int(phi[k]), []).append(k)
        return g

    def _conc_power(self, logits, mask):
        """Dirichlet concentrations for the three power axes, fairness blend applied.

        Identical algebra to PowerMLP._conc so that, when the fairness flag IS
        enabled, the flat baseline and HQC-HAC share exactly the same prior and
        the comparison isolates the architecture.
        """
        a_s = _dir.logits_to_conc(logits[self._sl_pw_split])
        a_c = _dir.logits_to_conc(logits[self._sl_pw_common][mask])
        a_p = _dir.logits_to_conc(logits[self._sl_pw_private])
        af, ap = self.power_fairness, self.power_fairness_priv
        if af > 0.0:
            f = self.power_priv_frac
            a_s = (1.0 - af) * a_s + af * a_s.sum() * np.array([1.0 - f, f])
            a_c = (1.0 - af) * a_c + af * a_c.sum() / a_c.shape[0]
        if ap > 0.0:
            a_p = (1.0 - ap) * a_p + ap * a_p.sum() / self.K
        return a_s, a_c, a_p

    def _dconc_blend(self, g_s, g_c, g_p):
        af, ap = self.power_fairness, self.power_fairness_priv
        if af > 0.0:
            f = self.power_priv_frac
            t_s = np.array([1.0 - f, f])
            g_s = (1.0 - af) * g_s + af * float(g_s @ t_s)
            g_c = (1.0 - af) * g_c + af * float(g_c.sum()) / g_c.shape[0]
        if ap > 0.0:
            g_p = (1.0 - ap) * g_p + ap * float(g_p.sum()) / self.K
        return g_s, g_c, g_p

    # ── act ─────────────────────────────────────────────────────────────────
    def forward(self, s_t: np.ndarray, greedy: bool = False) -> tuple:
        """One trunk evaluation → routing, phase and power together.

        Returns (action, log_prob, info). `action` carries everything PPO needs
        to re-score this transition later; `info['logits']` is retained so the
        C_k shares can be drawn from the SAME forward pass once the group common
        rates are known (see `sample_ck`).
        """
        K, M, N, L = self.K, self.M, self.N, self.L
        logits, _ = self._forward_logits(s_t)

        # ① routing — categorical per user
        lr_2d = logits[self._sl_route].reshape(K, M + 1)
        pi_r = np.stack([_softmax_1d(lr_2d[k]) for k in range(K)])
        if greedy:
            phi = pi_r.argmax(axis=1).astype(int)
        else:
            phi = np.array([self.rng.choice(M + 1, p=pi_r[k]) for k in range(K)], dtype=int)
        lp = float(np.log(pi_r[np.arange(K), phi] + 1e-10).sum())

        active_ids = sorted({int(v) for v in phi if v > 0})
        active_irs = np.array([g - 1 for g in active_ids], dtype=int)

        # ② phase — categorical per element; only populated IRS rows are scored,
        #    matching PhaseMLP (an unused IRS reflects nothing, so its phase
        #    neither affects the reward nor should receive gradient).
        lp_3d = logits[self._sl_phase].reshape(M, N, L)
        e = np.exp(lp_3d - lp_3d.max(axis=2, keepdims=True))
        pi_p = e / e.sum(axis=2, keepdims=True)
        phase_idx = np.zeros((M, N), dtype=int)
        for m in active_irs:
            if greedy:
                phase_idx[m] = pi_p[m].argmax(axis=1)
            else:
                cdf = pi_p[m].cumsum(axis=1)
                u = self.rng.random((N, 1))
                phase_idx[m] = (u > cdf).sum(axis=1).clip(0, L - 1)
            lp += float(np.log(pi_p[m, np.arange(N), phase_idx[m]] + 1e-10).sum())

        # ③ power — three Dirichlets, common axis masked to populated groups
        mask = self._common_mask(active_ids)
        a_s, a_c, a_p = self._conc_power(logits, mask)
        if greedy:
            x_s, x_c_act, x_p = a_s / a_s.sum(), a_c / a_c.sum(), a_p / a_p.sum()
        else:
            x_s = _dir.sample(self.rng, a_s)
            x_c_act = _dir.sample(self.rng, a_c)
            x_p = _dir.sample(self.rng, a_p)
        lp += float(_dir.log_prob(x_s, a_s) + _dir.log_prob(x_c_act, a_c)
                    + _dir.log_prob(x_p, a_p))

        x_c = np.zeros(M + 1)
        x_c[mask] = x_c_act
        w_c_full = x_c * (float(x_s[0]) * self.P_S)
        w_p = x_p * (float(x_s[1]) * self.P_S)
        w_c_vec = w_c_full[[0] + list(active_ids)]

        action = {
            'phi':       phi,
            'phase_idx': phase_idx,
            'power':     {'split': x_s, 'common': x_c, 'private': x_p},
            'active_ids': list(active_ids),
        }
        info = {'logits': logits, 'w_c_vec': w_c_vec, 'w_p': w_p,
                'active_irs': active_irs, 'pi_route': pi_r}
        return action, lp, info

    def sample_ck(self, logits: np.ndarray, phi: np.ndarray,
                  R_c_group: dict, K_active: int = None,
                  greedy: bool = False) -> tuple:
        """Draw the per-group common-rate shares from the ALREADY-COMPUTED logits.

        Deferred only because the group common rates R_c_group are a function of
        the power that was just executed — the network is not re-evaluated, so
        the "one forward pass per state" property is preserved.
        """
        K_used = K_active if K_active is not None else self.K
        z = logits[self._sl_ck]
        C_k = np.zeros(self.K)
        alpha_k = np.ones(self.K) / max(self.K, 1)
        sel: dict = {}
        lp = 0.0
        for gid, mem in self._groups(np.asarray(phi), K_used).items():
            R_g = float(R_c_group.get(gid, 0.0))
            if R_g < 1e-12 or not mem:
                alpha_k[mem] = 1.0 / max(len(mem), 1)
                sel[gid] = None                     # deterministic: not scored
                continue
            a_g = _dir.logits_to_conc(z[mem])
            x_g = (a_g / a_g.sum()) if greedy else _dir.sample(self.rng, a_g)
            alpha_k[mem] = x_g
            C_k[mem] = x_g * R_g
            sel[gid] = x_g
            lp += float(_dir.log_prob(x_g, a_g))
        return C_k, alpha_k, sel, lp

    # ── re-scoring (PPO ratio) ──────────────────────────────────────────────
    def _logp_from_logits(self, logits, action, ck_sel, K_used) -> float:
        K, M, N, L = self.K, self.M, self.N, self.L
        phi = np.asarray(action['phi'], dtype=int)

        lr_2d = logits[self._sl_route].reshape(K, M + 1)
        e = np.exp(lr_2d - lr_2d.max(axis=1, keepdims=True))
        pi_r = e / e.sum(axis=1, keepdims=True)
        lp = float(np.log(pi_r[np.arange(K), phi] + 1e-10)[:K_used].sum())

        lp_3d = logits[self._sl_phase].reshape(M, N, L)
        ep = np.exp(lp_3d - lp_3d.max(axis=2, keepdims=True))
        pi_p = ep / ep.sum(axis=2, keepdims=True)
        pidx = np.asarray(action['phase_idx'], dtype=int)
        for gid in action['active_ids']:
            m = int(gid) - 1
            lp += float(np.log(pi_p[m, np.arange(N), pidx[m]] + 1e-10).sum())

        mask = self._common_mask(action['active_ids'])
        a_s, a_c, a_p = self._conc_power(logits, mask)
        pw = action['power']
        lp += float(_dir.log_prob(pw['split'], a_s)
                    + _dir.log_prob(np.asarray(pw['common'])[mask], a_c)
                    + _dir.log_prob(pw['private'], a_p))

        if ck_sel:
            z = logits[self._sl_ck]
            for gid, x_g in ck_sel.items():
                if x_g is None:
                    continue
                mem = [k for k in range(K_used) if int(phi[k]) == int(gid)]
                if len(mem) != len(x_g):
                    continue
                lp += float(_dir.log_prob(x_g, _dir.logits_to_conc(z[mem])))
        return lp

    def compute_log_prob(self, s_t: np.ndarray, action: dict,
                         ck_sel: dict = None, K_active: int = None) -> float:
        logits, _ = self._forward_logits(s_t)
        K_used = K_active if K_active is not None else self.K
        return self._logp_from_logits(logits, action, ck_sel, K_used)

    # ── PPO update ──────────────────────────────────────────────────────────
    def compute_logprobs_grads_batch(self, s_t_list, act_list, ck_sel_list,
                                     lp_old_arr, adv_arr,
                                     ppo_epsilon: float,
                                     beta_entropy: float = 0.0,
                                     K_active: int = None):
        """Fused PPO log-prob + gradient over a mini-batch.

        ONE clipped objective over the JOINT action — this is the defining
        property of the flat baseline. HQC-HAC instead forms a separate clipped
        ratio per head; here routing, phase, power and C_k share a single ratio,
        so a bad phase choice clips the routing gradient too.

        Returns (loss_arr, L_ent_avg, grads, clip_frac, kl).
        """
        B = len(s_t_list)
        K, M, N, L = self.K, self.M, self.N, self.L
        K_used = K_active if K_active is not None else K

        S = np.asarray(s_t_list, dtype=float)
        logits_b, cache = self._forward_logits_batch(S)

        lp_b = np.array([self._logp_from_logits(logits_b[i], act_list[i],
                                                ck_sel_list[i], K_used)
                         for i in range(B)])
        ratio = np.exp(np.clip(lp_b - np.asarray(lp_old_arr), -10.0, 10.0))
        surr1 = ratio * adv_arr
        surr2 = np.clip(ratio, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv_arr
        # eff = the advantage that actually flows; zero inside the clipped region
        eff = np.where(surr1 <= surr2, ratio * adv_arr, 0.0)
        loss_arr = -np.minimum(surr1, surr2)
        clip_frac = float(np.mean(surr1 > surr2))
        kl = float(np.mean((ratio - 1.0) - np.log(ratio + 1e-10)))

        dL = np.zeros_like(logits_b)
        L_ent_tot = 0.0
        for i in range(B):
            g, H = self._dlogits(logits_b[i], act_list[i], ck_sel_list[i],
                                 eff[i], beta_entropy, K_used)
            dL[i] = g
            L_ent_tot += -beta_entropy * H
        L_ent_avg = L_ent_tot / max(B, 1)

        # ── backprop through the trunk ──
        grads = {}
        acts, pres = cache['acts'], cache['pres']
        dx = dL
        for i in reversed(range(self._n_layers)):
            if i < self._n_layers - 1:
                dx = dx * (pres[i] > 0)
            grads[f'W{i}'] = acts[i].T @ dx / B
            grads[f'b{i}'] = dx.mean(axis=0)
            dx = dx @ self.params[f'W{i}'].T

        return loss_arr, L_ent_avg, grads, clip_frac, kl

    def _dlogits(self, logits, action, ck_sel, eff, beta_entropy, K_used):
        """dL/dlogits for one transition, plus the joint entropy H(π(·|s))."""
        K, M, N, L = self.K, self.M, self.N, self.L
        phi = np.asarray(action['phi'], dtype=int)
        dL = np.zeros(self.d_out)
        H_tot = 0.0

        # ① routing — categorical PG + entropy
        lr_2d = logits[self._sl_route].reshape(K, M + 1)
        e = np.exp(lr_2d - lr_2d.max(axis=1, keepdims=True))
        pi_r = e / e.sum(axis=1, keepdims=True)
        log_pi = np.log(pi_r + 1e-10)
        H_r = -(pi_r * log_pi).sum(axis=1)
        one_hot = np.zeros_like(pi_r)
        one_hot[np.arange(K), phi] = 1.0
        d_r = -eff * (one_hot - pi_r) + beta_entropy * pi_r * (log_pi + H_r[:, None])
        d_r[K_used:] = 0.0
        dL[self._sl_route] = d_r.ravel()
        H_tot += float(H_r[:K_used].sum())

        # ② phase — only populated IRS rows carry gradient
        lp_3d = logits[self._sl_phase].reshape(M, N, L)
        ep = np.exp(lp_3d - lp_3d.max(axis=2, keepdims=True))
        pi_p = ep / ep.sum(axis=2, keepdims=True)
        log_pp = np.log(pi_p + 1e-10)
        H_p = -(pi_p * log_pp).sum(axis=2)
        pidx = np.asarray(action['phase_idx'], dtype=int)
        d_p = np.zeros_like(pi_p)
        for gid in action['active_ids']:
            m = int(gid) - 1
            oh = np.zeros((N, L))
            oh[np.arange(N), pidx[m]] = 1.0
            d_p[m] = (-eff * (oh - pi_p[m])
                      + beta_entropy * pi_p[m] * (log_pp[m] + H_p[m][:, None]))
            H_tot += float(H_p[m].sum())
        dL[self._sl_phase] = d_p.ravel()

        # ③ power — three Dirichlets (same chain rule as PowerMLP._dlogits)
        mask = self._common_mask(action['active_ids'])
        a_s, a_c, a_p = self._conc_power(logits, mask)
        pw = action['power']
        x_s = np.asarray(pw['split'], dtype=float)
        x_c = np.asarray(pw['common'], dtype=float)[mask]
        x_p = np.asarray(pw['private'], dtype=float)
        g_s = -eff * _dir.dlogp_dconc(x_s, a_s) - beta_entropy * _dir.dentropy_dconc(a_s)
        g_c = -eff * _dir.dlogp_dconc(x_c, a_c) - beta_entropy * _dir.dentropy_dconc(a_c)
        g_p = -eff * _dir.dlogp_dconc(x_p, a_p) - beta_entropy * _dir.dentropy_dconc(a_p)
        H_tot += float(_dir.entropy(a_s) + _dir.entropy(a_c) + _dir.entropy(a_p))
        g_s, g_c, g_p = self._dconc_blend(g_s, g_c, g_p)
        dL[self._sl_pw_split] = g_s * _dir.dconc_dlogits(logits[self._sl_pw_split])
        dc = np.zeros(M + 1)
        dc[mask] = g_c * _dir.dconc_dlogits(logits[self._sl_pw_common][mask])
        dL[self._sl_pw_common] = dc
        dL[self._sl_pw_private] = g_p * _dir.dconc_dlogits(logits[self._sl_pw_private])

        # ④ C_k — one Dirichlet per populated group
        if ck_sel:
            z = logits[self._sl_ck]
            d_ck = np.zeros(K)
            for gid, x_g in ck_sel.items():
                if x_g is None:
                    continue
                mem = [k for k in range(K_used) if int(phi[k]) == int(gid)]
                if len(mem) != len(x_g):
                    continue
                a_g = _dir.logits_to_conc(z[mem])
                g_g = (-eff * _dir.dlogp_dconc(np.asarray(x_g), a_g)
                       - beta_entropy * _dir.dentropy_dconc(a_g))
                d_ck[mem] = g_g * _dir.dconc_dlogits(z[mem])
                H_tot += float(_dir.entropy(a_g))
            dL[self._sl_ck] = d_ck

        return dL, H_tot

    def apply_grads(self, grads: dict) -> None:
        self.opt.step(self.params, {k: v for k, v in grads.items() if k in self.params})

    # ── persistence ─────────────────────────────────────────────────────────
    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self.params.items()}

    def set_params(self, snap: dict) -> None:
        for k, v in snap.items():
            if k in self.params:
                self.params[k] = v.copy()

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        # The running Z-score statistics are PART OF THE POLICY: the same weights
        # against different normalisation are a different function. They travel
        # with the weights so inference reproduces training's input scaling.
        np.savez(os.path.join(path, 'flat_params.npz'),
                 _rn_mean=self.rn_mean, _rn_m2=self.rn_m2,
                 _rn_count=np.array([self.rn_count]), **self.params)
        with open(os.path.join(path, 'flat_config.json'), 'w') as f:
            json.dump({'mode': 'flat', 'K': self.K, 'M': self.M, 'N': self.N,
                       'n_levels': self.L, 'P_S': self.P_S,
                       'hidden': self.HIDDEN, 'lr': self.lr,
                       'power_fairness': self.power_fairness,
                       'power_priv_frac': self.power_priv_frac,
                       'power_fairness_priv': self.power_fairness_priv}, f, indent=2)

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        from types import SimpleNamespace
        with open(os.path.join(path, 'flat_config.json')) as f:
            c = json.load(f)
        mini = SimpleNamespace(K=int(c['K']), M=int(c['M']), N=int(c['N']),
                               P_S=float(c.get('P_S', 1.0)))
        a = cls(mini, hidden=tuple(c.get('hidden', (512, 256))),
                lr=float(c.get('lr', MENG_DEFAULTS['lr_actor'])), seed=seed,
                n_levels=int(c.get('n_levels', 4)),
                power_fairness=float(c.get('power_fairness', 0.0)),
                power_priv_frac=float(c.get('power_priv_frac', 0.8)),
                power_fairness_priv=c.get('power_fairness_priv'))
        d = np.load(os.path.join(path, 'flat_params.npz'))
        for k in a.params:
            a.params[k] = d[k]
        if '_rn_mean' in d:
            a.rn_mean = d['_rn_mean']
            a.rn_m2 = d['_rn_m2']
            a.rn_count = float(d['_rn_count'][0])
        return a
