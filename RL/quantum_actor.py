"""
quantum_actor.py
----------------
Quantum Actor for IRS-assisted user selection.

Architecture  (sizes for Case 3: n_qubits=16, n_latent=32, M=4, K=20)
----------------------------------------------------------------------
  a_t  →  GroupNorm  →  Dual-Branch Encoder (B2)  →  z_t ∈ R^{n_latent}
       IRS branch:  [q_m(M), p_m(M)] → FC(2M,16,ReLU) → FC(16,2M) → z_IRS ∈ R^{2M}
       User branch: shared φ: [a_{k,0..M-1},D_k] → FC(M+1,8,ReLU) → FC(8,2) → z_k ∈ R^2
                    stack K embeddings → linear projection P → z_user ∈ R^{2(nq-M)}
                    z_t = [z_user | z_IRS] ∈ R^{2(nq-M)+2M} = R^{n_latent}
       →  LN(z_t)  →  Encoding angles (λ)
       →  n_qubits-qubit quantum circuit  →  o_hat ∈ R^{N_QUANTUM}
       →  h_t = [z_t, o_hat] ∈ R^{n_latent + N_QUANTUM}
       →  Post-NN (ξ)  →  logits ∈ R^{K(B+1)}  →  per-user softmax  →  φ ∈ {0,…,B}^K

Observables: ⟨Z_i⟩ (nq) + configurable ZZ pairs via configure_topology()
             B4: N_QUANTUM = 2·nq − 1 + len(extra_zz_pairs)
             B1: N_QUANTUM = nq + len(full_zz_pairs)  [full cross-block set]

Affinity feature vector  (B2 — replaces raw Re/Im channels)
------------------------------------------------------------
  a_t = [a_{k,m} affinities (K×M), D_k demands (K), |g_SU_k| direct (K),
         q_m means (M), p_m powers (M)]
  d_aff = K*(M+2) + 2*M   (e.g. 17 for K=5, M=1)

Parameters  Ω = {ω, λ, θ, ξ}
  ω : AE weights  (encoder + decoder)
  λ : quantum encoding scales  (λ^y, λ^z) ∈ R^{8×2}
  θ : quantum variational angles  (θ^y, θ^z) ∈ R^{8×2}
  ξ : post-processing NN weights

Gradient computation
--------------------
  Classical layers (ω, ξ) : standard backprop
  Quantum parameters (λ, θ): parameter-shift rule (see quantum_circuit.py)

All architecture hyperparameters are defined in params.py and must be
passed explicitly at construction — there are no fallback defaults.
"""

import os
import json
import numpy as np
from RL.quantum_circuit import (
    expectations_analytic, expectations_analytic_batch,
    expectations_shots, param_shift_jacobian,
    param_shift_jacobian_all_batch, spsa_jacobian_all_batch,
    configure_topology,
)

# ── Adam optimizer ─────────────────────────────────────────────────────────────

class _Adam:
    """Minimal Adam with per-parameter moment tracking."""

    def __init__(self, lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self._m:   dict = {}
        self._v:   dict = {}
        self._t:   int  = 0

    def step(self, params: dict, grads: dict) -> None:
        """In-place Adam update for a dict of parameter arrays."""
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for k, g in grads.items():
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * g * g
            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for a 2-D array."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _he(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)


# ── Quantum Actor ──────────────────────────────────────────────────────────────

class QuantumActor:
    """
    Quantum actor for IRS-user assignment  φ ∈ {0, 1, …, B}^K.

    All architecture and training hyperparameters must be supplied explicitly
    from params.py — see train.py for the canonical call site.

    Parameters
    ----------
    cfg           : SystemConfig  (needs .M, .K)
    n_qubits      : quantum register size  (params.n_qubits)
    n_latent      : AE latent dimension    (params.n_latent; must equal 2×n_qubits)
    n_hidden_ae   : list of encoder hidden-layer sizes (params.n_hidden_ae);
                    decoder mirrors in reverse. A plain int is wrapped to [int].
    n_hidden_post : list of post-NN hidden-layer sizes (params.n_hidden_post)
    n_var_layers  : variational circuit depth L (params.n_var_layers)
    n_shots       : measurement shots for inference forward pass (params.n_shots_train)
    lr_ae/qc/xi   : Adam learning rates for ω / λθ / ξ (params.lr_actor_*)
    data_reuploading : whether to re-apply U_E before each U_var (params.data_reuploading)
    ae_pretrain_lr : Adam lr for hot-start AE pre-training (params.ae_pretrain_lr)
    seed          : numpy RNG seed
    """

    def __init__(self, cfg,
                 n_qubits:         int,
                 n_latent:         int,
                 n_hidden_ae,               # list[int] | int — from params.n_hidden_ae
                 n_hidden_post,             # list[int] | int — from params.n_hidden_post
                 n_var_layers:     int,
                 n_shots:          int,
                 lr_ae:            float,
                 lr_qc:            float,
                 lr_xi:            float,
                 data_reuploading: bool,
                 ae_pretrain_lr:   float,
                 spsa_n_reps:      int   = 0,
                 spsa_epsilon:     float = 0.1,
                 extra_cz_pairs:   tuple = (),   # B3: cross-block CZ pairs
                 extra_zz_pairs:   tuple = (),   # B4: cross-block ZZ observable pairs
                 full_zz_pairs:    tuple = (),   # B1: full ZZ set (replaces NN ZZ + extra)
                 readout_mode:     str   = 'generic',  # Δ2: 'generic' | 'r1'
                 softmax_head:     bool  = False,       # Δ4: SoftmaxPQC linear+β head
                 softmax_beta_init: float = 1.0,        # Δ4: initial inverse-temperature β
                 seed:             int   = None):

        assert n_latent == 2 * n_qubits, (
            f"n_latent ({n_latent}) must equal 2 × n_qubits ({n_qubits}). "
            f"Each qubit needs one α (even z index) and one δ (odd z index)."
        )

        if isinstance(n_hidden_ae, int):
            n_hidden_ae = [n_hidden_ae]
        n_hidden_ae = list(n_hidden_ae)

        if isinstance(n_hidden_post, int):
            n_hidden_post = [n_hidden_post]
        n_hidden_post = list(n_hidden_post)

        # Architecture dimensions (instance attrs, readable from outside)
        self.N_QUBITS      = n_qubits
        self.N_LATENT      = n_latent
        # Δ2 R1 readout: N_QUANTUM = (nq-M)(M+2)+M  (structured per-action)
        # B1 active   : N_QUANTUM = nq + len(full_zz_pairs)
        # B4 active   : N_QUANTUM = 2*nq-1 + len(extra_zz_pairs)
        # baseline    : N_QUANTUM = 2*nq-1
        self.READOUT_MODE = str(readout_mode)
        self.SOFTMAX_HEAD = bool(softmax_head)
        if self.READOUT_MODE == 'r1':
            _nu = n_qubits - cfg.M
            self.N_QUANTUM = _nu * (cfg.M + 2) + cfg.M
        elif full_zz_pairs:
            self.N_QUANTUM = n_qubits + len(full_zz_pairs)
        else:
            self.N_QUANTUM = 2 * n_qubits - 1 + len(extra_zz_pairs)
        self.N_HIDDEN_AE   = n_hidden_ae   # list[int]
        self.N_HIDDEN_POST = n_hidden_post # list[int]
        self.N_VAR_LAYERS     = n_var_layers
        self.DATA_REUPLOADING = data_reuploading

        self.B         = cfg.M
        self.K         = cfg.K
        self.n_choices = self.B + 1        # 0=direct, 1..B = IRS index
        # B2: affinity feature dimension (replaces raw Re/Im channels)
        #     d_aff = K*(M+2) + 2*M
        #           = [K*M affinities + K demands + K direct |g_SU|] + [M means + M powers]
        self.d_s       = self.K * (self.B + 2) + 2 * self.B
        self.IRS_HIDDEN      = 16  # B2: IRS branch hidden size
        self.USER_ENC_HIDDEN = 8   # B2: user micro-encoder hidden size
        self.n_shots      = n_shots
        self.spsa_n_reps  = spsa_n_reps    # 0 = exact param-shift; >0 = SPSA
        self.spsa_epsilon = spsa_epsilon   # SPSA finite-difference step size
        self.EXTRA_CZ_PAIRS = tuple(extra_cz_pairs)  # B3: stored for save/load
        self.EXTRA_ZZ_PAIRS = tuple(extra_zz_pairs)  # B4: stored for save/load
        self.FULL_ZZ_PAIRS  = tuple(full_zz_pairs)   # B1: stored for save/load
        self.LAM_MAX        = 1.5          # B6: λ clipping bound (|λ|>1.5 → tanh grad<0.01)
        self.SOFTMAX_BETA_INIT = float(softmax_beta_init)
        self.rng            = np.random.default_rng(seed)

        # B3/B4/B1 + Δ2: set circuit topology BEFORE parameter init (head size uses N_QUANTUM)
        configure_topology(extra_cz_pairs, extra_zz_pairs, full_zz_pairs,
                           readout_mode=self.READOUT_MODE, r1_m=cfg.M)

        self._init_params()

        self.opt_ae     = _Adam(lr=lr_ae)
        self.opt_qc     = _Adam(lr=lr_qc)
        self.opt_xi     = _Adam(lr=lr_xi)
        self._opt_ae_pre = _Adam(lr=ae_pretrain_lr)

    # ── Initialisation ─────────────────────────────────────────────────────────

    def _init_params(self) -> None:
        nL   = self.N_LATENT    # n_latent  (e.g. 32 for n_qubits=16, M=4)
        nNq  = self.N_QUBITS    # n_qubits
        nObs = self.N_QUANTUM   # observable count
        nOut = self.K * self.n_choices
        _K, _M = self.K, self.B
        _nqM   = nNq - _M                    # user-block qubits (e.g. 12 for nq=16, M=4)
        _IH    = self.IRS_HIDDEN             # 16
        _UH    = self.USER_ENC_HIDDEN        # 8

        # ── B2 Dual-Branch Encoder ────────────────────────────────────────────────
        # IRS branch:  [q_m (M), p_m (M)] → FC(2M→IH,ReLU) → FC(IH→2M) → z_IRS ∈ R^(2M)
        self.W_irs_1 = _he(2*_M, _IH, self.rng);   self.b_irs_1 = np.zeros(_IH)
        self.W_irs_2 = _he(_IH, 2*_M, self.rng);   self.b_irs_2 = np.zeros(2*_M)

        # User micro-encoder φ (shared across K users):
        #   [a_{k,0..M-1}, D_k, |g_SU_k|] ∈ R^(M+2) → FC(M+2→UH,ReLU) → FC(UH→2) → z_k ∈ R^2
        self.W_u_1 = _he(_M+2, _UH, self.rng);     self.b_u_1 = np.zeros(_UH)
        self.W_u_2 = _he(_UH, 2, self.rng);         self.b_u_2 = np.zeros(2)

        # Projection P:  R^(2K) → R^(2*(nq-M))  — linear, no bias
        self.W_proj = _he(2*_nqM, 2*_K, self.rng)  # (2*(nq-M), 2K)

        # ── Decoder (single MLP, mirrors N_HIDDEN_AE depth, output → d_aff) ──────
        # z_t → h[-1] → ... → h[0] (all ReLU) → d_aff (linear output)
        dec_dims = [nL] + list(reversed(self.N_HIDDEN_AE))
        self.W_dec   = [_he(dec_dims[i], dec_dims[i+1], self.rng) for i in range(len(dec_dims)-1)]
        self.b_dec   = [np.zeros(dec_dims[i+1])                    for i in range(len(dec_dims)-1)]
        self.W_d_out = _he(dec_dims[-1], self.d_s, self.rng);  self.b_d_out = np.zeros(self.d_s)

        # ── λ — Quantum encoding scales ───────────────────────────────────────────
        self.lam_y = self.rng.uniform(-0.5, 0.5, nNq)
        self.lam_z = self.rng.uniform(-0.5, 0.5, nNq)

        # ── Δ1 (2026-06-01): LayerNorm learnable affine on z_t_enc (unfreeze λ) ────
        # Encoding norm: z_t_enc = LayerNorm(z_t) * gamma_enc + beta_enc.
        # The per-sample LayerNorm (mean-subtract + std-divide) is KEPT for drift
        # stability. The learnable affine adds a DC offset (beta) that breaks the
        # zero-mean structure of LN(z_t) — the cause of the frozen-λ gradient
        # cancellation: dL/dλ_i = γ_i·mean_b[g·LN] + β_i·mean_b[g], so a non-zero
        # β recovers the DC gradient component that pure LN (β≡0) annihilates.
        # init γ=1 (identity scale), β=tiny noise (immediate gradient path, ~no
        # change to the encoding at init). Trained via opt_qc alongside λ, θ.
        self.gamma_enc = np.ones(nL)
        self.beta_enc  = self.rng.normal(0.0, 0.02, nL)

        # ── θ — Quantum variational angles ────────────────────────────────────────
        self.theta_y = self.rng.uniform(-np.pi, np.pi, (self.N_VAR_LAYERS, nNq))
        self.theta_z = self.rng.uniform(-np.pi, np.pi, (self.N_VAR_LAYERS, nNq))

        # ── ξ — Assignment head ───────────────────────────────────────────────────
        # Input = [z_t (nL) ‖ o_hat (nObs)]  — z_t is the classical bypass [B] (Δ5).
        if self.SOFTMAX_HEAD:
            # Δ4 SoftmaxPQC: single linear map + trainable inverse-temperature β.
            #   logits = β · (h_t @ W_sm + b_sm);  π = softmax(logits per user).
            # ~ (nL+nObs)·K·nc params (Case2 66·30 ≈ 2K) vs MLP ~50K. β scalar.
            self.W_sm = _he(nL + nObs, nOut, self.rng)
            self.b_sm = np.zeros(nOut)
            self.beta_temp = np.array([self.SOFTMAX_BETA_INIT], dtype=float)  # shape (1,)
            self.W_post = [];  self.b_post = []   # unused
        else:
            post_dims = [nL + nObs] + self.N_HIDDEN_POST
            self.W_post  = [_he(post_dims[i], post_dims[i+1], self.rng) for i in range(len(post_dims)-1)]
            self.b_post  = [np.zeros(post_dims[i+1])                     for i in range(len(post_dims)-1)]
            self.W_p_out = _he(post_dims[-1], nOut, self.rng);  self.b_p_out = np.zeros(nOut)

        # ── Parameter dicts for Adam (keyed by name, point to live arrays) ────────
        self._p_ae = {
            'W_irs_1': self.W_irs_1, 'b_irs_1': self.b_irs_1,
            'W_irs_2': self.W_irs_2, 'b_irs_2': self.b_irs_2,
            'W_u_1':   self.W_u_1,   'b_u_1':   self.b_u_1,
            'W_u_2':   self.W_u_2,   'b_u_2':   self.b_u_2,
            'W_proj':  self.W_proj,
        }
        for i, (W, b) in enumerate(zip(self.W_dec, self.b_dec)):
            self._p_ae[f'W_dec_{i}'] = W;  self._p_ae[f'b_dec_{i}'] = b
        self._p_ae['W_d_out'] = self.W_d_out;  self._p_ae['b_d_out'] = self.b_d_out

        self._p_qc = dict(lam_y=self.lam_y, lam_z=self.lam_z,
                          theta_y=self.theta_y, theta_z=self.theta_z,
                          gamma_enc=self.gamma_enc, beta_enc=self.beta_enc)
        self._p_xi = {}
        if self.SOFTMAX_HEAD:
            self._p_xi['W_sm'] = self.W_sm;  self._p_xi['b_sm'] = self.b_sm
            self._p_xi['beta_temp'] = self.beta_temp
        else:
            for i, (W, b) in enumerate(zip(self.W_post, self.b_post)):
                self._p_xi[f'W_post_{i}'] = W;  self._p_xi[f'b_post_{i}'] = b
            self._p_xi['W_p_out'] = self.W_p_out;  self._p_xi['b_p_out'] = self.b_p_out

    # ── State extraction ───────────────────────────────────────────────────────

    def extract_state(self, obs: dict,
                      demand: np.ndarray,
                      blocked: np.ndarray) -> np.ndarray:
        """
        Build affinity feature vector a_t from env obs.  (B2 dual-branch AE)

        Parameters
        ----------
        obs     : dict from ISTNEnv._get_obs()
        demand  : (K,) D_k demand per user (bps/Hz)
        blocked : (K,) boolean blocking indicator — NOT used in affinity features
                        (blocking is implicitly encoded: blocked link → a_{k,m}≈0)

        Returns
        -------
        a_t : (d_aff,) float  with d_aff = K*(M+2) + 2*M
              BLOCK layout (contiguous per feature type):
                [ affinity (K*M) | demand (K) | |g_SU| (K) | q_m (M) | p_m (M) ]
              affinity block is user-major: [a_{0,0..M-1}, a_{1,0..M-1}, …].
        """
        g_sr = obs['g_SR']                          # (M,) complex
        g_ru = obs['g_RU']                          # (M, K) complex
        g_su = obs['g_SU']                          # (K,) complex — direct sat→user
        _M, _K = self.B, self.K

        # Affinity a_{k,m} = |g_SR[m]| × |g_RU[m,k]|  (IRS-path quality)
        g_sr_mag = np.abs(g_sr)                     # (M,)
        g_ru_mag = np.abs(g_ru)                     # (M, K)
        # a_mat[k, m] = g_sr_mag[m] * g_ru_mag[m, k]
        a_mat    = (g_sr_mag[:, None] * g_ru_mag).T  # (K, M)

        d_t      = demand.astype(np.float64)         # (K,)
        g_su_mag = np.abs(g_su)                      # (K,)  direct-path quality

        # IRS features: q_m = mean affinity per IRS, p_m = |g_SR[m]|²
        q_m = a_mat.mean(axis=0)                    # (M,)  mean over K users
        p_m = g_sr_mag ** 2                          # (M,)

        return np.concatenate([
            a_mat.flatten(),                         # K*M   affinity (user-major)
            d_t,                                     # K     demand
            g_su_mag,                                # K     direct |g_SU|
            q_m, p_m,                                # 2M    irs_feat
        ])

    # ── Layer norm ─────────────────────────────────────────────────────────────

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu  = x.mean()
        std = x.std() + eps
        return (x - mu) / std

    # ── Group-wise layer norm (PRIMARY normalisation for all forward paths) ────

    def _group_norm(self, s_t: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Per-group layer normalisation of the affinity feature vector.  (B2)

        Groups (d_aff = K*(M+2) + 2*M):
          affinity  [0 : K*M]              a_{k,m} — normalised        (K*M dims)
          demand    [K*M : K*(M+1)]        D_k — kept raw (constant)   (K   dims)
          gsu       [K*(M+1) : K*(M+2)]    |g_SU_k| — normalised       (K   dims)
          irs_feat  [K*(M+2) : d_aff]      [q_m, p_m] — normalised     (2M  dims)

        Demand is kept raw because it is a fixed constant (D_k_bps_hz ≈ 0.1
        for all k); std ≈ 0 → layer-norm would yield all-zeros, discarding the
        only absolute scale information the network receives.
        """
        _M, _K = self.B, self.K
        out = s_t.copy()
        for sl in [
            slice(0,              _K * _M),            # affinities  (K*M)
            slice(_K * (_M + 1),   _K * (_M + 2)),      # |g_SU|      (K)
            slice(_K * (_M + 2),   None),               # irs_feat    (2M)
        ]:
            x = s_t[sl];  mu = x.mean();  std = x.std() + eps
            out[sl] = (x - mu) / std
        # demand slice kept raw (std ≈ 0 in practice — constant across users)
        return out

    def _group_norm_batch(self, s_t_b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Batch version: s_t_b is (B_s, d_aff)."""
        _M, _K = self.B, self.K
        out = s_t_b.copy()
        for sl in [
            slice(0,              _K * _M),            # affinities
            slice(_K * (_M + 1),   _K * (_M + 2)),      # |g_SU|
            slice(_K * (_M + 2),   None),               # irs_feat
        ]:
            x   = s_t_b[:, sl]
            mu  = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1,  keepdims=True) + eps
            out[:, sl] = (x - mu) / std
        return out

    # ── Δ1: encoding normalisation (LayerNorm + learnable affine) ──────────────

    def _enc_norm(self, z_t: np.ndarray):
        """z_t_enc = LayerNorm(z_t)·γ + β.  Returns (z_enc, ln, sig) — ln & sig
        cached for backward.  Per-sample LN kept for drift stability; affine adds
        the DC offset that unfreezes λ (see _init_params Δ1 note)."""
        mu  = z_t.mean()
        sig = z_t.std() + 1e-6
        ln  = (z_t - mu) / sig
        z_enc = ln * self.gamma_enc + self.beta_enc
        return z_enc, ln, sig

    def _enc_norm_batch(self, z_t_b: np.ndarray):
        """Batch version. z_t_b (B_s, nL). Returns (z_enc_b, ln_b, sig_b)."""
        mu  = z_t_b.mean(axis=1, keepdims=True)
        sig = z_t_b.std(axis=1, keepdims=True) + 1e-6
        ln  = (z_t_b - mu) / sig
        z_enc = ln * self.gamma_enc + self.beta_enc
        return z_enc, ln, sig

    # ── Δ4: assignment head (SoftmaxPQC linear+β  OR  classical MLP) ────────────

    def _head_forward(self, z_t: np.ndarray, o_hat: np.ndarray):
        """Single-sample head. Returns (logits_flat (K*nc,), cache).
        h_t = [z_t ‖ o_hat] (z_t = classical bypass [B], Δ5)."""
        h_t = np.concatenate([z_t, o_hat])
        if self.SOFTMAX_HEAD:
            pre    = h_t @ self.W_sm + self.b_sm        # (K*nc,)
            logits = self.beta_temp[0] * pre
            return logits, ('sm', h_t, pre)
        acts = [h_t];  pres = [];  x = h_t
        for W, b in zip(self.W_post, self.b_post):
            p = x @ W + b;  pres.append(p);  x = _relu(p);  acts.append(x)
        logits = x @ self.W_p_out + self.b_p_out
        return logits, ('mlp', acts, pres)

    def _head_backward(self, dL_dlogits: np.ndarray, cache):
        """Single-sample head backward. dL_dlogits (K*nc,).
        Returns (grads_xi dict, dL_dz (nL,), dL_do_hat (nObs,))."""
        kind = cache[0]
        if kind == 'sm':
            _, h_t, pre = cache
            dL_dbeta = np.array([float(np.sum(dL_dlogits * pre))])   # (1,)
            dL_dpre  = self.beta_temp[0] * dL_dlogits
            grads = {'W_sm': np.outer(h_t, dL_dpre),
                     'b_sm': dL_dpre.copy(),
                     'beta_temp': dL_dbeta}
            dL_dh = dL_dpre @ self.W_sm.T
        else:
            _, acts, pres = cache
            dL_dW_p_out = np.outer(acts[-1], dL_dlogits)
            dL_db_p_out = dL_dlogits.copy()
            dx = dL_dlogits @ self.W_p_out.T
            dW = [None] * len(self.W_post);  db = [None] * len(self.b_post)
            for i in reversed(range(len(self.W_post))):
                dx = dx * (pres[i] > 0)
                dW[i] = np.outer(acts[i], dx);  db[i] = dx.copy()
                dx = dx @ self.W_post[i].T
            grads = {'W_p_out': dL_dW_p_out, 'b_p_out': dL_db_p_out}
            for i in range(len(self.W_post)):
                grads[f'W_post_{i}'] = dW[i];  grads[f'b_post_{i}'] = db[i]
            dL_dh = dx
        return grads, dL_dh[:self.N_LATENT], dL_dh[self.N_LATENT:]

    def _head_forward_batch(self, z_t_b: np.ndarray, o_hat_b: np.ndarray):
        """Batch head. Returns (logits_b (B_s, K*nc), cache)."""
        h_t_b = np.concatenate([z_t_b, o_hat_b], axis=1)
        if self.SOFTMAX_HEAD:
            pre_b    = h_t_b @ self.W_sm + self.b_sm
            logits_b = self.beta_temp[0] * pre_b
            return logits_b, ('sm', h_t_b, pre_b)
        acts = [h_t_b];  pres = [];  x = h_t_b
        for W, b in zip(self.W_post, self.b_post):
            p = x @ W + b;  pres.append(p);  x = _relu(p);  acts.append(x)
        logits_b = x @ self.W_p_out + self.b_p_out
        return logits_b, ('mlp', acts, pres)

    def _head_backward_batch(self, dL_dlogits_b: np.ndarray, cache, B_s: int):
        """Batch head backward (B_s-averaged grads). dL_dlogits_b (B_s, K*nc).
        Returns (grads_xi dict, dL_dz_b (B_s,nL), dL_do_hat_b (B_s,nObs))."""
        kind = cache[0]
        if kind == 'sm':
            _, h_t_b, pre_b = cache
            dL_dbeta = np.array([float((dL_dlogits_b * pre_b).sum() / B_s)])  # (1,)
            dL_dpre_b = self.beta_temp[0] * dL_dlogits_b
            grads = {'W_sm': h_t_b.T @ dL_dpre_b / B_s,
                     'b_sm': dL_dpre_b.mean(axis=0),
                     'beta_temp': dL_dbeta}
            dL_dh_b = dL_dpre_b @ self.W_sm.T
        else:
            _, acts, pres = cache
            dL_dW_p_out = acts[-1].T @ dL_dlogits_b / B_s
            dL_db_p_out = dL_dlogits_b.mean(axis=0)
            dx = dL_dlogits_b @ self.W_p_out.T
            dW = [None] * len(self.W_post);  db = [None] * len(self.b_post)
            for i in reversed(range(len(self.W_post))):
                dx = dx * (pres[i] > 0)
                dW[i] = acts[i].T @ dx / B_s;  db[i] = dx.mean(axis=0)
                dx = dx @ self.W_post[i].T
            grads = {'W_p_out': dL_dW_p_out, 'b_p_out': dL_db_p_out}
            for i in range(len(self.W_post)):
                grads[f'W_post_{i}'] = dW[i];  grads[f'b_post_{i}'] = db[i]
            dL_dh_b = dx
        return grads, dL_dh_b[:, :self.N_LATENT], dL_dh_b[:, self.N_LATENT:]

    # ── B2 dual-branch encoder helpers ────────────────────────────────────────

    def _dual_encode(self, a_norm: np.ndarray) -> np.ndarray:
        """
        Inference-only dual-branch forward.  No activations stored.
        a_norm : (d_aff,) group-normed affinity vector.
        Returns z_t : (n_latent,).
        """
        _K, _M, _nq = self.K, self.B, self.N_QUBITS
        a_flat   = a_norm[:_K * _M].reshape(_K, _M)
        d_flat   = a_norm[_K * _M: _K * (_M + 1)]
        gsu_flat = a_norm[_K * (_M + 1): _K * (_M + 2)]             # (K,)
        irs_in   = a_norm[_K * (_M + 2):]                          # (2M,)
        # IRS branch
        z_IRS  = (_relu(irs_in @ self.W_irs_1 + self.b_irs_1)
                  @ self.W_irs_2 + self.b_irs_2)                    # (2M,)
        # User branch
        x_uk    = np.concatenate(
            [a_flat, d_flat[:, None], gsu_flat[:, None]], axis=1)   # (K, M+2)
        z_k     = (_relu(x_uk @ self.W_u_1 + self.b_u_1)
                   @ self.W_u_2 + self.b_u_2)                       # (K, 2)
        z_user  = z_k.reshape(-1) @ self.W_proj.T                   # (2*(nq-M),)
        return np.concatenate([z_user, z_IRS])                      # (n_latent,)

    def _dual_encode_batch(self, a_norm_b: np.ndarray):
        """
        Inference-only dual-branch forward — batch.  No activations stored.
        a_norm_b : (B_s, d_aff).
        Returns z_t_b : (B_s, n_latent).
        """
        B_s = a_norm_b.shape[0]
        _K, _M, _nq = self.K, self.B, self.N_QUBITS
        a_flat_b   = a_norm_b[:, :_K*_M].reshape(B_s, _K, _M)
        d_flat_b   = a_norm_b[:, _K*_M: _K*(_M+1)]                  # (B_s, K)
        gsu_flat_b = a_norm_b[:, _K*(_M+1): _K*(_M+2)]              # (B_s, K)
        irs_in_b   = a_norm_b[:, _K*(_M+2):]                        # (B_s, 2M)
        # IRS branch
        z_IRS_b  = (_relu(irs_in_b @ self.W_irs_1 + self.b_irs_1)
                    @ self.W_irs_2 + self.b_irs_2)                  # (B_s, 2M)
        # User branch
        x_uk_b    = np.concatenate([a_flat_b,
                                    d_flat_b[:, :, None],
                                    gsu_flat_b[:, :, None]], axis=2)  # (B_s, K, M+2)
        x_uk_flat = x_uk_b.reshape(B_s * _K, _M + 2)
        z_k_flat  = (_relu(x_uk_flat @ self.W_u_1 + self.b_u_1)
                     @ self.W_u_2 + self.b_u_2)                     # (B_s*K, 2)
        z_user_b  = z_k_flat.reshape(B_s, _K * 2) @ self.W_proj.T  # (B_s, 2*(nq-M))
        return np.concatenate([z_user_b, z_IRS_b], axis=1)          # (B_s, n_latent)

    # ── Forward (inference) ────────────────────────────────────────────────────

    def forward(self, s_t: np.ndarray, greedy: bool = False) -> tuple:
        """
        Forward pass with shot-based quantum measurement.

        Parameters
        ----------
        greedy : bool  If True, select argmax assignment; if False, sample.

        Returns
        -------
        phi      : (K,) int    IRS assignment per user
        log_prob : float       Σ_k log π(φ_k | s_t)
        info     : dict        z_t, o_hat, pi (for debugging / logging)
        """
        a_norm = self._group_norm(s_t)
        z_t    = self._dual_encode(a_norm)             # (n_latent,)

        z_t_enc, _, _ = self._enc_norm(z_t)            # Δ1: LN + affine
        alpha = np.pi * np.tanh(self.lam_y * z_t_enc[0::2])
        delta = np.pi * np.tanh(self.lam_z * z_t_enc[1::2])

        # Shot-based quantum measurement
        o_hat = expectations_shots(alpha, delta, self.theta_y, self.theta_z,
                                   self.n_shots, self.rng,
                                   n_var_layers=self.N_VAR_LAYERS,
                                   data_reuploading=self.DATA_REUPLOADING)  # (Nq,)

        # Assignment head (Δ4: SoftmaxPQC linear+β or classical MLP)
        logits, _ = self._head_forward(z_t, o_hat)     # (K*(B+1),)

        logits_2d = logits.reshape(self.K, self.n_choices)
        pi_2d     = _softmax(logits_2d)

        if greedy:
            phi = pi_2d.argmax(axis=1).astype(int)
        else:
            phi = np.array([
                self.rng.choice(self.n_choices, p=pi_2d[k])
                for k in range(self.K)
            ])
        log_prob = float(sum(
            np.log(pi_2d[k, phi[k]] + 1e-10) for k in range(self.K)
        ))

        return phi, log_prob, {'z_t': z_t, 'o_hat': o_hat, 'pi': pi_2d}

    # ── PPO: analytic log-prob for ratio computation ───────────────────────────

    def compute_log_prob(self, s_t: np.ndarray,
                         phi: np.ndarray,
                         K_active: int = None) -> float:
        """
        Analytic log π(phi|s_t) under the CURRENT parameters.
        Used at training time to compute the PPO importance-sampling ratio
        r_t = exp(log_π_current − log_π_old).

        Uses expectations_analytic (no shot noise) so the ratio is consistent
        with the analytic forward pass inside compute_grads.
        """
        K_used = K_active if K_active is not None else self.K
        a_norm = self._group_norm(s_t)
        z_t    = self._dual_encode(a_norm)

        z_t_enc, _, _ = self._enc_norm(z_t)            # Δ1: LN + affine
        alpha = np.pi * np.tanh(self.lam_y * z_t_enc[0::2])
        delta = np.pi * np.tanh(self.lam_z * z_t_enc[1::2])
        o_hat = expectations_analytic(alpha, delta, self.theta_y, self.theta_z,
                                      n_var_layers=self.N_VAR_LAYERS,
                                      data_reuploading=self.DATA_REUPLOADING)

        logits, _ = self._head_forward(z_t, o_hat)
        pi_2d    = _softmax(logits.reshape(self.K, self.n_choices))
        return float(sum(np.log(pi_2d[k, phi[k]] + 1e-10) for k in range(K_used)))

    def compute_logprobs_batch(self, s_t_list, phi_list,
                               K_active: int = None) -> np.ndarray:
        """
        Batch analytic log π(phi|s_t) for all B_s samples — one GPU forward pass.
        Replaces B_s sequential compute_log_prob calls in the PPO per-sample loop.

        Returns
        -------
        log_probs : (B_s,) float  sum of log π_k over K_used users, per sample
        """
        B_s    = len(s_t_list)
        K_used = K_active if K_active is not None else self.K
        L      = self.N_VAR_LAYERS
        DR     = self.DATA_REUPLOADING
        nc     = self.n_choices

        a_norm_b  = self._group_norm_batch(np.array(s_t_list))
        z_t_b     = self._dual_encode_batch(a_norm_b)      # (B_s, N_LATENT)

        z_t_enc_b, _, _ = self._enc_norm_batch(z_t_b)      # Δ1: LN + affine
        alpha_b = np.pi * np.tanh(self.lam_y * z_t_enc_b[:, 0::2])
        delta_b = np.pi * np.tanh(self.lam_z * z_t_enc_b[:, 1::2])

        o_hat_b = expectations_analytic_batch(
            alpha_b, delta_b, self.theta_y, self.theta_z, L, DR)

        logits_b, _ = self._head_forward_batch(z_t_b, o_hat_b)  # (B_s, K*nc)
        logits_2d_b = logits_b.reshape(B_s, self.K, nc)
        e_b  = np.exp(logits_2d_b - logits_2d_b.max(axis=2, keepdims=True))
        pi_b = e_b / e_b.sum(axis=2, keepdims=True)        # (B_s, K, nc)

        phi_np   = np.array(phi_list)                       # (B_s, K)
        log_pi_b = np.log(pi_b + 1e-10)
        lp_sel   = log_pi_b[np.arange(B_s)[:, None],
                              np.arange(self.K)[None, :],
                              phi_np]                       # (B_s, K)
        return lp_sel[:, :K_used].sum(axis=1)              # (B_s,)

    # ── Training update ────────────────────────────────────────────────────────

    def compute_grads(self, s_t: np.ndarray,
                      phi: np.ndarray,
                      advantage: float,
                      ae_weight: float = 0.1,
                      beta_entropy: float = 0.0,
                      K_active: int = None) -> tuple:
        """
        Compute losses and parameter gradients without applying updates.

        Loss form (per call):
          L = -A · log π(φ|s) + α_AE · L_AE - β · H(π)

        where H(π) = -Σ_k π_k log π_k is the categorical entropy per user
        (summed across the K user-wise distributions).

        Returns
        -------
        L_pg     : float   policy gradient loss  (-A · log π)
        L_ae     : float   AE reconstruction loss (unscaled)
        L_ent    : float   entropy bonus contribution  (-β · ΣH), <=0
        grads_ae : dict    gradients for AE parameters (ω)
        grads_qc : dict    gradients for quantum parameters (λ, θ)
        grads_xi : dict    gradients for post-NN parameters (ξ)
        """
        # ──────────────────────────────────────────────────────────────────────
        # Forward pass (analytic quantum for stable gradients)
        # ──────────────────────────────────────────────────────────────────────
        a_norm = self._group_norm(s_t)
        _K, _M, _nq = self.K, self.B, self.N_QUBITS

        # B2 dual-branch encoder — store activations for backprop
        a_flat   = a_norm[:_K * _M].reshape(_K, _M)
        d_flat   = a_norm[_K * _M: _K * (_M + 1)]
        gsu_flat = a_norm[_K * (_M + 1): _K * (_M + 2)]            # (K,)
        irs_in   = a_norm[_K * (_M + 2):]                          # (2M,)
        irs_pre1 = irs_in @ self.W_irs_1 + self.b_irs_1            # (IH,)
        irs_act1 = _relu(irs_pre1)
        z_IRS    = irs_act1 @ self.W_irs_2 + self.b_irs_2          # (2M,)
        x_uk     = np.concatenate(
            [a_flat, d_flat[:, None], gsu_flat[:, None]], axis=1)   # (K, M+2)
        u_pre1   = x_uk @ self.W_u_1 + self.b_u_1                  # (K, UH)
        u_act1   = _relu(u_pre1)
        z_k      = u_act1 @ self.W_u_2 + self.b_u_2                # (K, 2)
        z_stack  = z_k.reshape(-1)                                  # (2K,)
        z_user   = z_stack @ self.W_proj.T                          # (2*(nq-M),)
        z_t      = np.concatenate([z_user, z_IRS])                  # (n_latent,)

        # Decoder (AE loss — reconstructs affinity features)
        dec_acts = [z_t]
        dec_pres = []
        x = z_t
        for W, b in zip(self.W_dec, self.b_dec):
            pre = x @ W + b
            dec_pres.append(pre)
            x = _relu(pre)
            dec_acts.append(x)
        a_rec  = dec_acts[-1] @ self.W_d_out + self.b_d_out        # (d_aff,)
        ae_res = a_rec - a_norm
        L_ae   = float(0.5 * np.sum(ae_res ** 2))

        # Layer-norm z_t before VQC encoding; kept separate from raw z_t
        # (decoder and post-NN use raw z_t — only quantum angles use z_t_enc)
        # Δ1: LN + learnable affine (γ,β). _ln, _z_sig cached for backward.
        z_t_enc, _ln, _z_sig = self._enc_norm(z_t)        # (n_latent,)

        # Encoding angles
        u_y   = self.lam_y * z_t_enc[0::2]                # (nq,)
        u_z   = self.lam_z * z_t_enc[1::2]                # (nq,)
        alpha = np.pi * np.tanh(u_y)                       # (nq,)
        delta = np.pi * np.tanh(u_z)                       # (nq,)

        # Analytic quantum measurement
        o_hat = expectations_analytic(alpha, delta, self.theta_y, self.theta_z,
                                      n_var_layers=self.N_VAR_LAYERS,
                                      data_reuploading=self.DATA_REUPLOADING)

        # Assignment head (Δ4: SoftmaxPQC linear+β or classical MLP) — cache for backward
        logits, _head_cache = self._head_forward(z_t, o_hat)  # (K*(B+1),)

        logits_2d = logits.reshape(self.K, self.n_choices)
        pi_2d     = _softmax(logits_2d)                    # (K, B+1)
        K_used    = K_active if K_active is not None else self.K
        log_prob  = float(sum(
            np.log(pi_2d[k, phi[k]] + 1e-10) for k in range(K_used)
        ))
        L_pg = -advantage * log_prob

        # ──────────────────────────────────────────────────────────────────────
        # Backward: policy gradient through post-NN
        # ──────────────────────────────────────────────────────────────────────
        # ∂(-log_prob)/∂logits_2d[k] = π_k - e_{φ_k}  (only active users)
        one_hot = np.zeros_like(pi_2d)
        for k in range(K_used):
            one_hot[k, phi[k]] = 1.0
        dL_pg_dlogits = (-advantage * (one_hot - pi_2d))  # (K, B+1)

        # ── Entropy regularization: L_ent = -β · Σ_k H(π_k) ─────────────────
        log_pi      = np.log(pi_2d + 1e-10)                    # (K, B+1)
        H_per_user  = -np.sum(pi_2d * log_pi, axis=1, keepdims=True)  # (K, 1)
        H_total     = float(np.sum(H_per_user[:K_used]))       # active users only
        L_ent       = -beta_entropy * H_total
        dL_ent_dlogits = beta_entropy * pi_2d * (log_pi + H_per_user)  # (K, B+1)

        dL_dlogits = (dL_pg_dlogits + dL_ent_dlogits).flatten()  # (K*(B+1),)

        # Assignment head backward (Δ4) → head grads + dL/dz_post + dL/do_hat
        _grads_head, dL_dz_post, dL_do_hat = self._head_backward(dL_dlogits, _head_cache)

        # ──────────────────────────────────────────────────────────────────────
        # Backward: AE reconstruction through decoder (reversed layer order)
        # ──────────────────────────────────────────────────────────────────────
        dL_ds_rec   = ae_weight * ae_res                          # (d_s,)
        dL_dW_d_out = np.outer(dec_acts[-1], dL_ds_rec)
        dL_db_d_out = dL_ds_rec.copy()
        dx = dL_ds_rec @ self.W_d_out.T

        dL_dW_dec = [None] * len(self.W_dec)
        dL_db_dec = [None] * len(self.b_dec)
        for i in reversed(range(len(self.W_dec))):
            dx = dx * (dec_pres[i] > 0)
            dL_dW_dec[i] = np.outer(dec_acts[i], dx)
            dL_db_dec[i] = dx.copy()
            dx = dx @ self.W_dec[i].T
        dL_dz_ae = dx                                             # (n_latent,)

        # ──────────────────────────────────────────────────────────────────────
        # Backward: quantum parameters via parameter-shift rule
        # ──────────────────────────────────────────────────────────────────────
        # θ variational parameters — J shape (Nq, L*Nq) for 2-D theta
        L   = self.N_VAR_LAYERS
        DR  = self.DATA_REUPLOADING
        J_ty = param_shift_jacobian(alpha, delta, self.theta_y, self.theta_z,
                                    'theta_y', n_var_layers=L, data_reuploading=DR)
        J_tz = param_shift_jacobian(alpha, delta, self.theta_y, self.theta_z,
                                    'theta_z', n_var_layers=L, data_reuploading=DR)
        dL_dtheta_y = (J_ty.T @ dL_do_hat).reshape(L, self.N_QUBITS)
        dL_dtheta_z = (J_tz.T @ dL_do_hat).reshape(L, self.N_QUBITS)

        # Encoding angles α, δ  →  chain-rule into λ and z
        J_a = param_shift_jacobian(alpha, delta, self.theta_y, self.theta_z,
                                   'alpha', n_var_layers=L, data_reuploading=DR)
        J_d = param_shift_jacobian(alpha, delta, self.theta_y, self.theta_z,
                                   'delta', n_var_layers=L, data_reuploading=DR)
        dL_dalpha = J_a.T @ dL_do_hat                           # (8,)
        dL_ddelta = J_d.T @ dL_do_hat                           # (8,)

        # sech²(u) = 1 - tanh²(u)
        sech2_y = 1.0 - np.tanh(u_y) ** 2                      # (8,)
        sech2_z = 1.0 - np.tanh(u_z) ** 2                      # (8,)

        # ∂α_i/∂λ^y_i  =  π · sech²(u_y_i) · z_enc_{2i}   (use normed z)
        dL_dlam_y = dL_dalpha * (np.pi * sech2_y * z_t_enc[0::2])  # (nq,)
        dL_dlam_z = dL_ddelta * (np.pi * sech2_z * z_t_enc[1::2])  # (nq,)

        # ∂α_i/∂z_enc_{2i}  =  π · sech²(u_y_i) · λ^y_i  (z_enc = affine output)
        dL_dz_enc = np.zeros(self.N_LATENT)
        dL_dz_enc[0::2] = dL_dalpha * (np.pi * sech2_y * self.lam_y)
        dL_dz_enc[1::2] = dL_ddelta * (np.pi * sech2_z * self.lam_z)

        # Δ1 affine backward: z_enc = ln·γ + β
        dL_dbeta_enc  = dL_dz_enc.copy()
        dL_dgamma_enc = dL_dz_enc * _ln
        dL_dln        = dL_dz_enc * self.gamma_enc

        # LayerNorm backward: dL/dln → dL/dz_qc  (ln = (z − μ)/σ)
        _n   = self.N_LATENT
        _sg  = dL_dln.sum()
        _sgz = (dL_dln * _ln).sum()
        dL_dz_qc = (dL_dln - _sg / _n - _ln * _sgz / _n) / _z_sig

        # ──────────────────────────────────────────────────────────────────────
        # Total gradient on z  →  backprop through encoder (reversed layer order)
        # ──────────────────────────────────────────────────────────────────────
        # B5: stop_gradient — post-NN does NOT flow into encoder
        dL_dz = dL_dz_qc + dL_dz_ae                              # (n_latent,)

        # Split gradient on z_t by block
        _n_user    = 2 * (_nq - _M)
        dL_dz_user = dL_dz[:_n_user]                             # (2*(nq-M),)
        dL_dz_IRS  = dL_dz[_n_user:]                             # (2M,)

        # IRS branch backward
        dL_dW_irs_2 = np.outer(irs_act1, dL_dz_IRS)
        dL_db_irs_2 = dL_dz_IRS.copy()
        dL_irs1     = (dL_dz_IRS @ self.W_irs_2.T) * (irs_pre1 > 0)
        dL_dW_irs_1 = np.outer(irs_in, dL_irs1)
        dL_db_irs_1 = dL_irs1.copy()

        # User projection backward
        dL_dW_proj  = np.outer(dL_dz_user, z_stack)
        dL_dz_stack = dL_dz_user @ self.W_proj                   # (2K,)

        # User micro-encoder backward (sum over K users)
        dL_dz_k   = dL_dz_stack.reshape(_K, 2)
        dL_dW_u_2 = u_act1.T @ dL_dz_k                          # (UH, 2)
        dL_db_u_2 = dL_dz_k.sum(axis=0)                          # (2,)
        dL_du1    = (dL_dz_k @ self.W_u_2.T) * (u_pre1 > 0)    # (K, UH)
        dL_dW_u_1 = x_uk.T @ dL_du1                              # (M+1, UH)
        dL_db_u_1 = dL_du1.sum(axis=0)                           # (UH,)

        grads_ae = {
            'W_irs_1': dL_dW_irs_1, 'b_irs_1': dL_db_irs_1,
            'W_irs_2': dL_dW_irs_2, 'b_irs_2': dL_db_irs_2,
            'W_u_1':   dL_dW_u_1,   'b_u_1':   dL_db_u_1,
            'W_u_2':   dL_dW_u_2,   'b_u_2':   dL_db_u_2,
            'W_proj':  dL_dW_proj,
        }
        for i in range(len(self.W_dec)):
            grads_ae[f'W_dec_{i}'] = dL_dW_dec[i]
            grads_ae[f'b_dec_{i}'] = dL_db_dec[i]
        grads_ae['W_d_out'] = dL_dW_d_out
        grads_ae['b_d_out'] = dL_db_d_out
        grads_qc = dict(lam_y=dL_dlam_y, lam_z=dL_dlam_z,
                        theta_y=dL_dtheta_y, theta_z=dL_dtheta_z,
                        gamma_enc=dL_dgamma_enc, beta_enc=dL_dbeta_enc)
        grads_xi = _grads_head
        return L_pg, L_ae, L_ent, grads_ae, grads_qc, grads_xi

    # ── Fused Jacobian dispatcher (param-shift or SPSA) ──────────────────────

    def _jacobian_all_batch(self, alpha_b: np.ndarray, delta_b: np.ndarray):
        """
        Compute all four Jacobians (J_ty, J_tz, J_a, J_d) in ONE GPU pass.

        Dispatches to:
          spsa_n_reps == 0  →  param_shift_jacobian_all_batch  (exact, 16,384 circuits at nq=16)
          spsa_n_reps >  0  →  spsa_jacobian_all_batch         (estimate,   512 circuits at nq=16, n_reps=4)

        Returns
        -------
        J_ty_b : (B_s, n_obs, L*nq)
        J_tz_b : (B_s, n_obs, L*nq)
        J_a_b  : (B_s, n_obs, nq)
        J_d_b  : (B_s, n_obs, nq)
        """
        L  = self.N_VAR_LAYERS
        DR = self.DATA_REUPLOADING
        if self.spsa_n_reps > 0:
            return spsa_jacobian_all_batch(
                alpha_b, delta_b, self.theta_y, self.theta_z,
                n_var_layers=L, data_reuploading=DR,
                spsa_epsilon=self.spsa_epsilon,
                n_reps=self.spsa_n_reps,
                rng=self.rng)
        else:
            return param_shift_jacobian_all_batch(
                alpha_b, delta_b, self.theta_y, self.theta_z,
                n_var_layers=L, data_reuploading=DR)

    # ── Batched training update (replaces B_s sequential compute_grads calls) ─

    def compute_grads_batch(self,
                            s_t_list,          # list of B_s (d_s,) arrays
                            phi_list,          # list of B_s (K,) int arrays
                            advantages: np.ndarray,   # (B_s,) effective PPO advantages
                            ae_weight:  float,
                            beta_entropy: float,
                            K_active: int = None) -> tuple:
        """
        Compute averaged gradients for the whole mini-batch in 4 GPU passes.

        Instead of calling compute_grads B_s times (each triggering 4 sequential
        param_shift_jacobian calls), this method:
          1. Runs all MLP forward/backward passes per-sample (fast NumPy)
          2. Issues one fused batched Jacobian pass (param_shift_jacobian_all_batch,
             all four wrt at once) processing all B_s samples on the GPU.

        Returns the same tuple as compute_grads but with values already averaged
        over the mini-batch (do NOT divide by B again in the caller).

        Returns
        -------
        L_pg_avg    : float   mean policy gradient loss
        L_ae_avg    : float   mean AE reconstruction loss (unscaled)
        L_ent_avg   : float   mean entropy bonus contribution
        grads_ae    : dict    averaged AE gradients
        grads_qc    : dict    averaged quantum parameter gradients
        grads_xi    : dict    averaged post-NN gradients
        """
        B_s  = len(s_t_list)
        L    = self.N_VAR_LAYERS
        DR   = self.DATA_REUPLOADING
        nq   = self.N_QUBITS
        n_obs = self.N_QUANTUM          # B1: nq+|full_zz|; B4: 2*nq-1+|extra_zz|; base: 2*nq-1
        K    = self.K
        K_used = K_active if K_active is not None else K
        nc   = self.n_choices           # B+1

        # ── Group-wise layer normalisation ────────────────────────────────────
        a_norm_b = self._group_norm_batch(np.array(s_t_list))   # (B_s, d_aff)
        _K, _M, _nq = self.K, self.B, self.N_QUBITS

        # ── B2 dual-branch encoder forward ────────────────────────────────────
        a_flat_b   = a_norm_b[:, :_K*_M].reshape(B_s, _K, _M)
        d_flat_b   = a_norm_b[:, _K*_M: _K*(_M+1)]             # (B_s, K)
        gsu_flat_b = a_norm_b[:, _K*(_M+1): _K*(_M+2)]         # (B_s, K)
        irs_in_b   = a_norm_b[:, _K*(_M+2):]                   # (B_s, 2M)
        irs_pre1_b = irs_in_b @ self.W_irs_1 + self.b_irs_1     # (B_s, IH)
        irs_act1_b = _relu(irs_pre1_b)
        z_IRS_b    = irs_act1_b @ self.W_irs_2 + self.b_irs_2   # (B_s, 2M)
        x_uk_b     = np.concatenate([a_flat_b,
                                     d_flat_b[:, :, None],
                                     gsu_flat_b[:, :, None]], axis=2)  # (B_s, K, M+2)
        x_uk_flat  = x_uk_b.reshape(B_s * _K, _M + 2)
        u_pre1_flat = x_uk_flat @ self.W_u_1 + self.b_u_1       # (B_s*K, UH)
        u_act1_flat = _relu(u_pre1_flat)
        z_k_flat    = u_act1_flat @ self.W_u_2 + self.b_u_2     # (B_s*K, 2)
        z_stack_b   = z_k_flat.reshape(B_s, _K * 2)             # (B_s, 2K)
        z_user_b    = z_stack_b @ self.W_proj.T                  # (B_s, 2*(nq-M))
        z_t_b       = np.concatenate([z_user_b, z_IRS_b], axis=1)  # (B_s, N_LATENT)

        # ── Decoder forward (AE loss) ──────────────────────────────────────────
        dec_acts = [z_t_b]
        dec_pres = []
        x = z_t_b
        for W, b in zip(self.W_dec, self.b_dec):
            pre = x @ W + b
            dec_pres.append(pre)
            x = _relu(pre)
            dec_acts.append(x)
        a_rec_b  = x @ self.W_d_out + self.b_d_out              # (B_s, d_aff)
        ae_res_b = a_rec_b - a_norm_b

        # ── Layer-norm z_t before VQC encoding (Δ1: LN + affine) ──────────────
        z_t_enc_b, _ln_b, _z_sig_b = self._enc_norm_batch(z_t_b)  # (B_s, N_LATENT)

        # ── Encoding angles ────────────────────────────────────────────────────
        u_y_b   = self.lam_y * z_t_enc_b[:, 0::2]           # (B_s, nq)
        u_z_b   = self.lam_z * z_t_enc_b[:, 1::2]
        alpha_b = np.pi * np.tanh(u_y_b)
        delta_b = np.pi * np.tanh(u_z_b)

        # ── Batched quantum forward ────────────────────────────────────────────
        o_hat_b = expectations_analytic_batch(
            alpha_b, delta_b, self.theta_y, self.theta_z, L, DR)  # (B_s, n_obs)

        # ── Assignment head forward (Δ4) — cache for backward ─────────────────
        logits_b, _head_cache = self._head_forward_batch(z_t_b, o_hat_b)  # (B_s, K*nc)
        logits_2d_b = logits_b.reshape(B_s, K, nc)
        # row-wise softmax over last axis
        e_b  = np.exp(logits_2d_b - logits_2d_b.max(axis=2, keepdims=True))
        pi_b = e_b / e_b.sum(axis=2, keepdims=True)     # (B_s, K, nc)

        # ── Losses ────────────────────────────────────────────────────────────
        phi_np = np.array(phi_list)                      # (B_s, K) int
        adv_b  = advantages                              # (B_s,)

        log_pi_b  = np.log(pi_b + 1e-10)                # (B_s, K, nc)
        lp_sel    = log_pi_b[np.arange(B_s)[:, None],
                              np.arange(K)[None, :],
                              phi_np]                    # (B_s, K)
        log_prob_b = lp_sel.sum(axis=1)                  # (B_s,)

        L_pg_avg  = float((-adv_b * log_prob_b).mean())
        L_ae_avg  = float((0.5 * np.sum(ae_res_b**2, axis=1)).mean())

        H_b       = -(pi_b * log_pi_b).sum(axis=2)      # (B_s, K)
        L_ent_avg = float((-beta_entropy * H_b[:, :K_used].sum(axis=1)).mean())

        # ── Post-NN backward ──────────────────────────────────────────────────
        one_hot_b  = np.zeros_like(pi_b)
        one_hot_b[np.arange(B_s)[:, None], np.arange(K)[None, :], phi_np] = 1.0

        dL_dlogits_b = (
            -adv_b[:, None, None] * (one_hot_b - pi_b)          # policy gradient
            + beta_entropy * pi_b * (log_pi_b + H_b[:, :, None]) # entropy
        ).reshape(B_s, K * nc)                                   # (B_s, K*nc)

        # Assignment head backward (Δ4) → head grads + dL/dz_post + dL/do_hat
        _grads_head, dL_dz_post_b, dL_do_hat_b = self._head_backward_batch(
            dL_dlogits_b, _head_cache, B_s)

        # ── Decoder backward (AE) ─────────────────────────────────────────────
        dL_ds_rec_b  = ae_weight * ae_res_b                      # (B_s, d_s)
        dL_dW_d_out  = dec_acts[-1].T @ dL_ds_rec_b / B_s
        dL_db_d_out  = dL_ds_rec_b.mean(axis=0)
        dx_b         = dL_ds_rec_b @ self.W_d_out.T

        dL_dW_dec_list = [None] * len(self.W_dec)
        dL_db_dec_list = [None] * len(self.b_dec)
        for i in reversed(range(len(self.W_dec))):
            dx_b = dx_b * (dec_pres[i] > 0)
            dL_dW_dec_list[i] = dec_acts[i].T @ dx_b / B_s
            dL_db_dec_list[i] = dx_b.mean(axis=0)
            dx_b = dx_b @ self.W_dec[i].T
        dL_dz_ae_b = dx_b                                        # (B_s, N_LATENT)

        # ── Fused quantum Jacobians (1 GPU pass — param-shift or SPSA) ──────────
        J_ty_b, J_tz_b, J_a_b, J_d_b = self._jacobian_all_batch(alpha_b, delta_b)

        # ── Quantum parameter gradients (vectorised over batch) ───────────────
        # J_*_b : (B_s, n_obs, n_params);  dL_do_hat_b : (B_s, n_obs)
        # mean over s of  J[s].T @ dL_do_hat[s]  =  einsum('sop,so->p') / B_s
        dL_dtheta_y = np.einsum('sop,so->p', J_ty_b, dL_do_hat_b) / B_s
        dL_dtheta_z = np.einsum('sop,so->p', J_tz_b, dL_do_hat_b) / B_s
        dL_dtheta_y = dL_dtheta_y.reshape(L, nq)
        dL_dtheta_z = dL_dtheta_z.reshape(L, nq)

        # per-sample alpha/delta gradients → needed for lam/enc grads
        dL_dalpha_b = np.einsum('sop,so->sp', J_a_b, dL_do_hat_b)  # (B_s, nq)
        dL_ddelta_b = np.einsum('sop,so->sp', J_d_b, dL_do_hat_b)

        sech2_y_b = 1.0 - np.tanh(u_y_b) ** 2                  # (B_s, nq)
        sech2_z_b = 1.0 - np.tanh(u_z_b) ** 2

        dL_dlam_y = (dL_dalpha_b * (np.pi * sech2_y_b * z_t_enc_b[:, 0::2])).mean(axis=0)
        dL_dlam_z = (dL_ddelta_b * (np.pi * sech2_z_b * z_t_enc_b[:, 1::2])).mean(axis=0)

        dL_dz_enc_b = np.zeros((B_s, self.N_LATENT))
        dL_dz_enc_b[:, 0::2] = dL_dalpha_b * (np.pi * sech2_y_b * self.lam_y)
        dL_dz_enc_b[:, 1::2] = dL_ddelta_b * (np.pi * sech2_z_b * self.lam_z)

        # Δ1 affine backward: z_enc = ln·γ + β  (γ,β shared → B_s-mean like lam)
        dL_dbeta_enc  = dL_dz_enc_b.mean(axis=0)                      # (N_LATENT,)
        dL_dgamma_enc = (dL_dz_enc_b * _ln_b).mean(axis=0)           # (N_LATENT,)
        dL_dln_b      = dL_dz_enc_b * self.gamma_enc                  # (B_s, N_LATENT)

        # LayerNorm backward: dL/dln_b → dL/dz_qc_b  (ln = (z − μ)/σ)
        _n     = self.N_LATENT
        _sg_b  = dL_dln_b.sum(axis=1, keepdims=True)                  # (B_s, 1)
        _sgz_b = (dL_dln_b * _ln_b).sum(axis=1, keepdims=True)        # (B_s, 1)
        dL_dz_qc_b = (dL_dln_b - _sg_b / _n
                       - _ln_b * _sgz_b / _n) / _z_sig_b              # (B_s, N_LATENT)

        # ── B2 dual-branch encoder backward (B5: no post-NN gradient) ──────────
        dL_dz_b    = dL_dz_qc_b + dL_dz_ae_b                    # (B_s, N_LATENT)

        _n_user        = 2 * (_nq - _M)
        dL_dz_user_b   = dL_dz_b[:, :_n_user]                   # (B_s, 2*(nq-M))
        dL_dz_IRS_b    = dL_dz_b[:, _n_user:]                   # (B_s, 2M)

        # IRS branch backward
        dL_dW_irs_2    = irs_act1_b.T @ dL_dz_IRS_b / B_s       # (IH, 2M)
        dL_db_irs_2    = dL_dz_IRS_b.mean(axis=0)
        dL_irs1_b      = dL_dz_IRS_b @ self.W_irs_2.T
        dL_irs1_b     *= (irs_pre1_b > 0)                        # ReLU mask
        dL_dW_irs_1    = irs_in_b.T @ dL_irs1_b / B_s           # (2M, IH)
        dL_db_irs_1    = dL_irs1_b.mean(axis=0)

        # User projection backward
        dL_dW_proj     = dL_dz_user_b.T @ z_stack_b / B_s       # (2*(nq-M), 2K)
        dL_dz_stack_b  = dL_dz_user_b @ self.W_proj              # (B_s, 2K)

        # User micro-encoder backward (batch over B_s*K)
        dL_dz_k_flat   = dL_dz_stack_b.reshape(B_s * _K, 2)
        dL_dW_u_2      = u_act1_flat.T @ dL_dz_k_flat / B_s     # (UH, 2)
        dL_db_u_2      = dL_dz_k_flat.sum(axis=0) / B_s          # (2,)
        dL_du1_flat    = dL_dz_k_flat @ self.W_u_2.T
        dL_du1_flat   *= (u_pre1_flat > 0)                       # ReLU mask
        dL_dW_u_1      = x_uk_flat.T @ dL_du1_flat / B_s        # (M+1, UH)
        dL_db_u_1      = dL_du1_flat.sum(axis=0) / B_s           # (UH,)

        # ── Assemble gradient dicts ────────────────────────────────────────────
        grads_ae = {
            'W_irs_1': dL_dW_irs_1, 'b_irs_1': dL_db_irs_1,
            'W_irs_2': dL_dW_irs_2, 'b_irs_2': dL_db_irs_2,
            'W_u_1':   dL_dW_u_1,   'b_u_1':   dL_db_u_1,
            'W_u_2':   dL_dW_u_2,   'b_u_2':   dL_db_u_2,
            'W_proj':  dL_dW_proj,
        }
        for i in range(len(self.W_dec)):
            grads_ae[f'W_dec_{i}'] = dL_dW_dec_list[i]
            grads_ae[f'b_dec_{i}'] = dL_db_dec_list[i]
        grads_ae['W_d_out'] = dL_dW_d_out
        grads_ae['b_d_out'] = dL_db_d_out

        grads_qc = dict(lam_y=dL_dlam_y, lam_z=dL_dlam_z,
                        theta_y=dL_dtheta_y, theta_z=dL_dtheta_z,
                        gamma_enc=dL_dgamma_enc, beta_enc=dL_dbeta_enc)

        grads_xi = _grads_head

        return L_pg_avg, L_ae_avg, L_ent_avg, grads_ae, grads_qc, grads_xi

    # ── Diagnostic: per-sample λ gradient (frozen-λ interference vs vanishing) ──

    def lam_grad_per_sample(self, s_t_list, phi_list, advantages,
                            beta_entropy: float = 0.0, K_active: int = None) -> np.ndarray:
        """Per-sample λ-gradient matrix G (B_s, 2·nq) — the terms BEFORE the batch-mean
        in dL_dlam (cols [0:nq]=λ_y, [nq:2nq]=λ_z, one row per sample). Diagnoses
        frozen-λ [E]: compute mean pairwise cos(G_b, G_b') across the batch.
          mean cos ≈ 0 + healthy ||G_b|| → INTERFERENCE (per-sample dirs random → batch-mean
                                            cancels → Adam stalls). NOT a barren-plateau.
          small ||G_b||                  → VANISHING (depth/qubit barren plateau).
        Mirrors the first half of compute_logprobs_grads_batch (analytic, no PPO clip)."""
        B_s = len(s_t_list); L = self.N_VAR_LAYERS; DR = self.DATA_REUPLOADING
        nq = self.N_QUBITS; K = self.K; nc = self.n_choices
        K_used = K_active if K_active is not None else K
        adv_b = np.asarray(advantages, dtype=float)

        a_norm_b  = self._group_norm_batch(np.array(s_t_list))
        z_t_b     = self._dual_encode_batch(a_norm_b)
        z_t_enc_b, _ln_b, _sig_b = self._enc_norm_batch(z_t_b)
        u_y_b = self.lam_y * z_t_enc_b[:, 0::2]
        u_z_b = self.lam_z * z_t_enc_b[:, 1::2]
        alpha_b = np.pi * np.tanh(u_y_b)
        delta_b = np.pi * np.tanh(u_z_b)
        o_hat_b = expectations_analytic_batch(alpha_b, delta_b, self.theta_y, self.theta_z, L, DR)

        logits_b, cache = self._head_forward_batch(z_t_b, o_hat_b)
        pi_b = _softmax(logits_b.reshape(B_s, K, nc).reshape(B_s * K, nc)).reshape(B_s, K, nc)
        log_pi_b = np.log(pi_b + 1e-10)
        phi_np = np.array(phi_list)
        one_hot_b = np.zeros_like(pi_b)
        one_hot_b[np.arange(B_s)[:, None], np.arange(K)[None, :], phi_np] = 1.0
        H_b = -(pi_b * log_pi_b).sum(axis=2)
        dL_dlogits_b = (-adv_b[:, None, None] * (one_hot_b - pi_b)
                        + beta_entropy * pi_b * (log_pi_b + H_b[:, :, None])).reshape(B_s, K * nc)
        # per-sample dL/do_hat (head backward gives it un-averaged)
        _, _, dL_do_hat_b = self._head_backward_batch(dL_dlogits_b, cache, B_s)

        J_ty, J_tz, J_a, J_d = self._jacobian_all_batch(alpha_b, delta_b)
        dL_dalpha_b = np.einsum('sop,so->sp', J_a, dL_do_hat_b)   # (B_s, nq)
        dL_ddelta_b = np.einsum('sop,so->sp', J_d, dL_do_hat_b)
        sech2_y_b = 1.0 - np.tanh(u_y_b) ** 2
        sech2_z_b = 1.0 - np.tanh(u_z_b) ** 2
        G_y = dL_dalpha_b * (np.pi * sech2_y_b * z_t_enc_b[:, 0::2])   # (B_s, nq)
        G_z = dL_ddelta_b * (np.pi * sech2_z_b * z_t_enc_b[:, 1::2])
        return np.concatenate([G_y, G_z], axis=1)                     # (B_s, 2nq)

    # ── Fused log-prob + gradient (single quantum forward pass) ───────────────

    def compute_logprobs_grads_batch(self,
                                     s_t_list,
                                     phi_list,
                                     lp_old_arr:   np.ndarray,
                                     adv_arr:      np.ndarray,
                                     ppo_epsilon:  float,
                                     ae_weight:    float,
                                     beta_entropy: float,
                                     K_active: int = None) -> tuple:
        """
        Fused log-prob + gradient in ONE quantum forward pass.

        Eliminates the redundant expectations_analytic_batch call that otherwise
        occurs separately in compute_logprobs_batch and compute_grads_batch.
        PPO clipping is handled internally using lp_old_arr and adv_arr.

        Returns
        -------
        l_q_arr   : (B_s,)  PPO surrogate loss per sample  (for L_pg logging)
        L_ae_avg  : float   mean AE reconstruction loss
        L_ent_avg : float   mean entropy bonus
        grads_ae, grads_qc, grads_xi : averaged gradient dicts
        clip_frac_q : float  fraction of samples whose PPO gradient was zeroed
        kl_q        : float  approx KL(old‖new) of the quantum policy (PPO health)
        """
        B_s    = len(s_t_list)
        K_used = K_active if K_active is not None else self.K
        L      = self.N_VAR_LAYERS
        DR     = self.DATA_REUPLOADING
        nq     = self.N_QUBITS
        n_obs  = self.N_QUANTUM
        K      = self.K
        nc     = self.n_choices

        # ── Group-wise layer normalisation ────────────────────────────────────
        a_norm_b = self._group_norm_batch(np.array(s_t_list))
        _K, _M, _nq = self.K, self.B, self.N_QUBITS

        # ── B2 dual-branch encoder forward ────────────────────────────────────
        a_flat_b   = a_norm_b[:, :_K*_M].reshape(B_s, _K, _M)
        d_flat_b   = a_norm_b[:, _K*_M: _K*(_M+1)]
        gsu_flat_b = a_norm_b[:, _K*(_M+1): _K*(_M+2)]
        irs_in_b   = a_norm_b[:, _K*(_M+2):]
        irs_pre1_b = irs_in_b @ self.W_irs_1 + self.b_irs_1
        irs_act1_b = _relu(irs_pre1_b)
        z_IRS_b    = irs_act1_b @ self.W_irs_2 + self.b_irs_2
        x_uk_b     = np.concatenate([a_flat_b,
                                     d_flat_b[:, :, None],
                                     gsu_flat_b[:, :, None]], axis=2)
        x_uk_flat  = x_uk_b.reshape(B_s * _K, _M + 2)
        u_pre1_flat = x_uk_flat @ self.W_u_1 + self.b_u_1
        u_act1_flat = _relu(u_pre1_flat)
        z_k_flat    = u_act1_flat @ self.W_u_2 + self.b_u_2
        z_stack_b   = z_k_flat.reshape(B_s, _K * 2)
        z_user_b    = z_stack_b @ self.W_proj.T
        z_t_b       = np.concatenate([z_user_b, z_IRS_b], axis=1)

        # ── Decoder forward (AE loss) ──────────────────────────────────────────
        dec_acts = [z_t_b]
        dec_pres = []
        x = z_t_b
        for W, b in zip(self.W_dec, self.b_dec):
            pre = x @ W + b
            dec_pres.append(pre)
            x = _relu(pre)
            dec_acts.append(x)
        a_rec_b  = x @ self.W_d_out + self.b_d_out
        ae_res_b = a_rec_b - a_norm_b

        # ── Layer-norm z_t before VQC encoding (Δ1: LN + affine) ──────────────
        z_t_enc_b, _ln_b, _z_sig_b = self._enc_norm_batch(z_t_b)

        # ── Encoding angles ────────────────────────────────────────────────────
        u_y_b   = self.lam_y * z_t_enc_b[:, 0::2]
        u_z_b   = self.lam_z * z_t_enc_b[:, 1::2]
        alpha_b = np.pi * np.tanh(u_y_b)
        delta_b = np.pi * np.tanh(u_z_b)

        # ── Batched quantum forward (ONE GPU call) ─────────────────────────────
        o_hat_b = expectations_analytic_batch(
            alpha_b, delta_b, self.theta_y, self.theta_z, L, DR)

        # ── Assignment head forward (Δ4) — cache for backward ─────────────────
        logits_b, _head_cache = self._head_forward_batch(z_t_b, o_hat_b)
        logits_2d_b = logits_b.reshape(B_s, K, nc)
        e_b  = np.exp(logits_2d_b - logits_2d_b.max(axis=2, keepdims=True))
        pi_b = e_b / e_b.sum(axis=2, keepdims=True)

        # ── PPO clipping (uses current log-probs from this forward pass) ───────
        phi_np     = np.array(phi_list)
        log_pi_b   = np.log(pi_b + 1e-10)
        lp_sel     = log_pi_b[np.arange(B_s)[:, None],
                               np.arange(K)[None, :],
                               phi_np]
        log_prob_b = lp_sel[:, :K_used].sum(axis=1)          # (B_s,) current lp

        ratio_b = np.exp(np.clip(log_prob_b - lp_old_arr, -10.0, 10.0))
        surr1_b = ratio_b * adv_arr
        surr2_b = np.clip(ratio_b, 1.0 - ppo_epsilon,
                                   1.0 + ppo_epsilon) * adv_arr
        eff_q_b = np.where(surr1_b <= surr2_b, ratio_b * adv_arr, 0.0)
        l_q_arr = -np.minimum(surr1_b, surr2_b)              # (B_s,) for logging
        clip_frac_q = float(np.mean(surr1_b > surr2_b))      # frac of zeroed-grad samples
        kl_q = float(np.mean((ratio_b - 1.0) - np.log(ratio_b + 1e-10)))  # approx KL(old‖new)

        # ── Losses ────────────────────────────────────────────────────────────
        L_ae_avg  = float((0.5 * np.sum(ae_res_b**2, axis=1)).mean())
        H_b       = -(pi_b * log_pi_b).sum(axis=2)
        L_ent_avg = float((-beta_entropy * H_b[:, :K_used].sum(axis=1)).mean())

        # ── Post-NN backward ──────────────────────────────────────────────────
        one_hot_b = np.zeros_like(pi_b)
        one_hot_b[np.arange(B_s)[:, None], np.arange(K)[None, :], phi_np] = 1.0

        dL_dlogits_b = (
            -eff_q_b[:, None, None] * (one_hot_b - pi_b)
            + beta_entropy * pi_b * (log_pi_b + H_b[:, :, None])
        ).reshape(B_s, K * nc)

        # Assignment head backward (Δ4) → head grads + dL/dz_post + dL/do_hat
        _grads_head, dL_dz_post_b, dL_do_hat_b = self._head_backward_batch(
            dL_dlogits_b, _head_cache, B_s)

        # ── Decoder backward ──────────────────────────────────────────────────
        dL_ds_rec_b = ae_weight * ae_res_b
        dL_dW_d_out = dec_acts[-1].T @ dL_ds_rec_b / B_s
        dL_db_d_out = dL_ds_rec_b.mean(axis=0)
        dx_b        = dL_ds_rec_b @ self.W_d_out.T

        dL_dW_dec_list = [None] * len(self.W_dec)
        dL_db_dec_list = [None] * len(self.b_dec)
        for i in reversed(range(len(self.W_dec))):
            dx_b = dx_b * (dec_pres[i] > 0)
            dL_dW_dec_list[i] = dec_acts[i].T @ dx_b / B_s
            dL_db_dec_list[i] = dx_b.mean(axis=0)
            dx_b = dx_b @ self.W_dec[i].T
        dL_dz_ae_b = dx_b

        # ── Fused quantum Jacobians (1 GPU pass — param-shift or SPSA) ──────────
        J_ty_b, J_tz_b, J_a_b, J_d_b = self._jacobian_all_batch(alpha_b, delta_b)

        # ── Quantum parameter gradients ───────────────────────────────────────
        dL_dtheta_y = np.einsum('sop,so->p', J_ty_b, dL_do_hat_b) / B_s
        dL_dtheta_z = np.einsum('sop,so->p', J_tz_b, dL_do_hat_b) / B_s
        dL_dtheta_y = dL_dtheta_y.reshape(L, nq)
        dL_dtheta_z = dL_dtheta_z.reshape(L, nq)

        dL_dalpha_b = np.einsum('sop,so->sp', J_a_b, dL_do_hat_b)
        dL_ddelta_b = np.einsum('sop,so->sp', J_d_b, dL_do_hat_b)

        sech2_y_b = 1.0 - np.tanh(u_y_b) ** 2
        sech2_z_b = 1.0 - np.tanh(u_z_b) ** 2

        dL_dlam_y = (dL_dalpha_b * (np.pi * sech2_y_b * z_t_enc_b[:, 0::2])).mean(axis=0)
        dL_dlam_z = (dL_ddelta_b * (np.pi * sech2_z_b * z_t_enc_b[:, 1::2])).mean(axis=0)

        dL_dz_enc_b = np.zeros((B_s, self.N_LATENT))
        dL_dz_enc_b[:, 0::2] = dL_dalpha_b * (np.pi * sech2_y_b * self.lam_y)
        dL_dz_enc_b[:, 1::2] = dL_ddelta_b * (np.pi * sech2_z_b * self.lam_z)

        # Δ1 affine backward: z_enc = ln·γ + β  (γ,β shared → B_s-mean like lam)
        dL_dbeta_enc  = dL_dz_enc_b.mean(axis=0)
        dL_dgamma_enc = (dL_dz_enc_b * _ln_b).mean(axis=0)
        dL_dln_b      = dL_dz_enc_b * self.gamma_enc

        # LayerNorm backward: dL/dln_b → dL/dz_qc_b  (ln = (z − μ)/σ)
        _n     = self.N_LATENT
        _sg_b  = dL_dln_b.sum(axis=1, keepdims=True)
        _sgz_b = (dL_dln_b * _ln_b).sum(axis=1, keepdims=True)
        dL_dz_qc_b = (dL_dln_b - _sg_b / _n
                       - _ln_b * _sgz_b / _n) / _z_sig_b

        # ── B2 dual-branch encoder backward (B5: no post-NN gradient) ──────────
        dL_dz_b    = dL_dz_qc_b + dL_dz_ae_b                    # (B_s, N_LATENT)

        _n_user        = 2 * (_nq - _M)
        dL_dz_user_b   = dL_dz_b[:, :_n_user]
        dL_dz_IRS_b    = dL_dz_b[:, _n_user:]

        dL_dW_irs_2    = irs_act1_b.T @ dL_dz_IRS_b / B_s
        dL_db_irs_2    = dL_dz_IRS_b.mean(axis=0)
        dL_irs1_b      = dL_dz_IRS_b @ self.W_irs_2.T
        dL_irs1_b     *= (irs_pre1_b > 0)
        dL_dW_irs_1    = irs_in_b.T @ dL_irs1_b / B_s
        dL_db_irs_1    = dL_irs1_b.mean(axis=0)

        dL_dW_proj     = dL_dz_user_b.T @ z_stack_b / B_s
        dL_dz_stack_b  = dL_dz_user_b @ self.W_proj

        dL_dz_k_flat   = dL_dz_stack_b.reshape(B_s * _K, 2)
        dL_dW_u_2      = u_act1_flat.T @ dL_dz_k_flat / B_s
        dL_db_u_2      = dL_dz_k_flat.sum(axis=0) / B_s
        dL_du1_flat    = dL_dz_k_flat @ self.W_u_2.T
        dL_du1_flat   *= (u_pre1_flat > 0)
        dL_dW_u_1      = x_uk_flat.T @ dL_du1_flat / B_s
        dL_db_u_1      = dL_du1_flat.sum(axis=0) / B_s

        # ── Assemble gradient dicts ────────────────────────────────────────────
        grads_ae = {
            'W_irs_1': dL_dW_irs_1, 'b_irs_1': dL_db_irs_1,
            'W_irs_2': dL_dW_irs_2, 'b_irs_2': dL_db_irs_2,
            'W_u_1':   dL_dW_u_1,   'b_u_1':   dL_db_u_1,
            'W_u_2':   dL_dW_u_2,   'b_u_2':   dL_db_u_2,
            'W_proj':  dL_dW_proj,
        }
        for i in range(len(self.W_dec)):
            grads_ae[f'W_dec_{i}'] = dL_dW_dec_list[i]
            grads_ae[f'b_dec_{i}'] = dL_db_dec_list[i]
        grads_ae['W_d_out'] = dL_dW_d_out
        grads_ae['b_d_out'] = dL_db_d_out

        grads_qc = dict(lam_y=dL_dlam_y, lam_z=dL_dlam_z,
                        theta_y=dL_dtheta_y, theta_z=dL_dtheta_z,
                        gamma_enc=dL_dgamma_enc, beta_enc=dL_dbeta_enc)

        grads_xi = _grads_head

        return l_q_arr, L_ae_avg, L_ent_avg, grads_ae, grads_qc, grads_xi, clip_frac_q, kl_q

    # ── Parameter I/O ─────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save config + parameters to directory `path`."""
        os.makedirs(path, exist_ok=True)
        cfg_dict = {
            'n_qubits':         self.N_QUBITS,
            'n_latent':         self.N_LATENT,
            'n_hidden_ae':      self.N_HIDDEN_AE,
            'n_hidden_post':    self.N_HIDDEN_POST,
            'n_var_layers':     self.N_VAR_LAYERS,
            'data_reuploading': self.DATA_REUPLOADING,
            'B':                self.B,
            'K':                self.K,
            'd_s':              self.d_s,
            'n_shots':          self.n_shots,
            'lr_ae':            self.opt_ae.lr,
            'lr_qc':            self.opt_qc.lr,
            'lr_xi':            self.opt_xi.lr,
            'ae_pretrain_lr':   self._opt_ae_pre.lr,
            'spsa_n_reps':      self.spsa_n_reps,
            'spsa_epsilon':     self.spsa_epsilon,
            'extra_cz_pairs':   [list(p) for p in self.EXTRA_CZ_PAIRS],
            'extra_zz_pairs':   [list(p) for p in self.EXTRA_ZZ_PAIRS],
            'full_zz_pairs':    [list(p) for p in self.FULL_ZZ_PAIRS],
            'readout_mode':     self.READOUT_MODE,
            'softmax_head':     self.SOFTMAX_HEAD,
            'softmax_beta_init': self.SOFTMAX_BETA_INIT,
        }
        with open(os.path.join(path, 'actor_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'actor_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, cfg=None, seed: int = None):
        """
        Reconstruct a QuantumActor from a saved directory.

        Architecture dimensions (B=M, K) are taken from the saved
        actor_config.json — the `cfg` argument is ignored for structural
        parameters so the actor always matches its training-time layout
        regardless of what params.py currently contains.
        """
        with open(os.path.join(path, 'actor_config.json')) as f:
            c = json.load(f)

        # Minimal stand-in so __init__ can read .M and .K from saved values
        class _Dims:
            M = c['B']
            K = c['K']

        actor = cls(
            cfg=_Dims(),
            n_qubits=c['n_qubits'],
            n_latent=c['n_latent'],
            n_hidden_ae=c['n_hidden_ae'],
            n_hidden_post=c['n_hidden_post'],
            n_var_layers=c['n_var_layers'],
            n_shots=c['n_shots'],
            lr_ae=c['lr_ae'],
            lr_qc=c['lr_qc'],
            lr_xi=c['lr_xi'],
            data_reuploading=c['data_reuploading'],
            ae_pretrain_lr=c['ae_pretrain_lr'],
            spsa_n_reps=c.get('spsa_n_reps', 0),
            spsa_epsilon=c.get('spsa_epsilon', 0.1),
            extra_cz_pairs=[tuple(p) for p in c.get('extra_cz_pairs', [])],
            extra_zz_pairs=[tuple(p) for p in c.get('extra_zz_pairs', [])],
            full_zz_pairs= [tuple(p) for p in c.get('full_zz_pairs',  [])],
            readout_mode=c.get('readout_mode', 'generic'),
            softmax_head=c.get('softmax_head', False),
            softmax_beta_init=c.get('softmax_beta_init', 1.0),
            seed=seed,
        )
        actor.set_params(dict(np.load(os.path.join(path, 'actor_params.npz'))))
        return actor

    # ── AE hot-start ──────────────────────────────────────────────────────────

    def pretrain_ae_step(self, s_t: np.ndarray) -> float:
        """
        Single AE reconstruction step for hot-start pre-training.
        Updates only AE parameters (ω) using a dedicated Adam optimizer
        so pre-training momentum does not bleed into joint training.

        Returns
        -------
        L_ae : float   MSE reconstruction loss (unscaled)
        """
        a_norm = self._group_norm(s_t)
        _K, _M, _nq = self.K, self.B, self.N_QUBITS

        # B2 dual-branch encoder forward
        a_flat   = a_norm[:_K * _M].reshape(_K, _M)
        d_flat   = a_norm[_K * _M: _K * (_M + 1)]
        gsu_flat = a_norm[_K * (_M + 1): _K * (_M + 2)]
        irs_in   = a_norm[_K * (_M + 2):]
        irs_pre1 = irs_in @ self.W_irs_1 + self.b_irs_1
        irs_act1 = _relu(irs_pre1)
        z_IRS    = irs_act1 @ self.W_irs_2 + self.b_irs_2
        x_uk     = np.concatenate(
            [a_flat, d_flat[:, None], gsu_flat[:, None]], axis=1)
        u_pre1   = x_uk @ self.W_u_1 + self.b_u_1
        u_act1   = _relu(u_pre1)
        z_k      = u_act1 @ self.W_u_2 + self.b_u_2
        z_stack  = z_k.reshape(-1)
        z_user   = z_stack @ self.W_proj.T
        z_t      = np.concatenate([z_user, z_IRS])

        # Decoder forward
        dec_acts = [z_t]
        dec_pres = []
        x = z_t
        for W, b in zip(self.W_dec, self.b_dec):
            pre = x @ W + b
            dec_pres.append(pre)
            x = _relu(pre)
            dec_acts.append(x)
        a_rec  = dec_acts[-1] @ self.W_d_out + self.b_d_out
        ae_res = a_rec - a_norm
        L_ae   = float(0.5 * np.sum(ae_res ** 2))

        # Decoder backward
        dL_da_rec   = ae_res
        dL_dW_d_out = np.outer(dec_acts[-1], dL_da_rec)
        dL_db_d_out = dL_da_rec.copy()
        dx = dL_da_rec @ self.W_d_out.T

        dL_dW_dec = [None] * len(self.W_dec)
        dL_db_dec = [None] * len(self.b_dec)
        for i in reversed(range(len(self.W_dec))):
            dx = dx * (dec_pres[i] > 0)
            dL_dW_dec[i] = np.outer(dec_acts[i], dx)
            dL_db_dec[i] = dx.copy()
            dx = dx @ self.W_dec[i].T
        dL_dz = dx   # (n_latent,)

        # B2 dual-branch encoder backward
        _n_user    = 2 * (_nq - _M)
        dL_dz_user = dL_dz[:_n_user]
        dL_dz_IRS  = dL_dz[_n_user:]

        dL_dW_irs_2 = np.outer(irs_act1, dL_dz_IRS)
        dL_db_irs_2 = dL_dz_IRS.copy()
        dL_irs1     = (dL_dz_IRS @ self.W_irs_2.T) * (irs_pre1 > 0)
        dL_dW_irs_1 = np.outer(irs_in, dL_irs1)
        dL_db_irs_1 = dL_irs1.copy()

        dL_dW_proj  = np.outer(dL_dz_user, z_stack)
        dL_dz_stack = dL_dz_user @ self.W_proj

        dL_dz_k   = dL_dz_stack.reshape(_K, 2)
        dL_dW_u_2 = u_act1.T @ dL_dz_k
        dL_db_u_2 = dL_dz_k.sum(axis=0)
        dL_du1    = (dL_dz_k @ self.W_u_2.T) * (u_pre1 > 0)
        dL_dW_u_1 = x_uk.T @ dL_du1
        dL_db_u_1 = dL_du1.sum(axis=0)

        grads_ae = {
            'W_irs_1': dL_dW_irs_1, 'b_irs_1': dL_db_irs_1,
            'W_irs_2': dL_dW_irs_2, 'b_irs_2': dL_db_irs_2,
            'W_u_1':   dL_dW_u_1,   'b_u_1':   dL_db_u_1,
            'W_u_2':   dL_dW_u_2,   'b_u_2':   dL_db_u_2,
            'W_proj':  dL_dW_proj,
        }
        for i in range(len(self.W_dec)):
            grads_ae[f'W_dec_{i}'] = dL_dW_dec[i]
            grads_ae[f'b_dec_{i}'] = dL_db_dec[i]
        grads_ae['W_d_out'] = dL_dW_d_out
        grads_ae['b_d_out'] = dL_db_d_out

        self._opt_ae_pre.step(self._p_ae, grads_ae)
        return L_ae

    def apply_grads(self, grads_ae: dict, grads_qc: dict, grads_xi: dict) -> None:
        """Apply pre-computed gradients to all parameter groups."""
        self.opt_ae.step(self._p_ae, grads_ae)
        self.opt_qc.step(self._p_qc, grads_qc)
        # B6: clip encoding scales λ — prevents tanh saturation.
        # |λ| > 1.5 → sech²(λz) < 0.01 → near-zero gradient → λ frozen.
        # In-place clip on live arrays (they're in _p_qc by reference).
        np.clip(self.lam_y, -self.LAM_MAX, self.LAM_MAX, out=self.lam_y)
        np.clip(self.lam_z, -self.LAM_MAX, self.LAM_MAX, out=self.lam_z)
        self.opt_xi.step(self._p_xi, grads_xi)

    def update(self, s_t: np.ndarray,
               phi: np.ndarray,
               advantage: float,
               ae_weight: float = 0.1,
               beta_entropy: float = 0.0) -> tuple:
        """Compute gradients and apply in one call. Returns (L_pg, L_ae, L_ent)."""
        L_pg, L_ae, L_ent, g_ae, g_qc, g_xi = self.compute_grads(
            s_t, phi, advantage, ae_weight, beta_entropy)
        self.apply_grads(g_ae, g_qc, g_xi)
        return L_pg, L_ae, L_ent

    # ── Parameter I/O ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Return a snapshot of all parameters (for checkpointing)."""
        return {k: v.copy() for d in (self._p_ae, self._p_qc, self._p_xi)
                for k, v in d.items()}

    def set_params(self, snapshot: dict) -> None:
        """Restore parameters from a snapshot."""
        for d in (self._p_ae, self._p_qc, self._p_xi):
            for k in d:
                if k in snapshot:
                    d[k][:] = snapshot[k]
