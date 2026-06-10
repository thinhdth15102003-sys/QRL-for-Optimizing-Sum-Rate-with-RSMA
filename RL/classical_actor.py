"""
classical_actor.py
------------------
ClassicalActor — a pure-MLP drop-in replacement for QuantumActor, used as the
DNN baseline for the "VQC vs DNN" ablation (paper A2 / result table R5) and the
param-efficiency claim (C2/C3: SoftmaxPQC ~2K params vs MLP ~tens-of-K).

Design (decided 2026-06-10 with user): replace the WHOLE high-level actor
(AE + VQC + assignment head) with a single classical MLP that maps the state
feature vector s_t to the per-user assignment logits:

    s_t  ──group_norm──▶  ENCODER MLP  ──▶  z_t (N_LATENT)  ──▶  POLICY MLP  ──▶  logits
                                              │
                                              └──▶ fed to PhaseMLP/PowerMLP/CkMLP (unchanged)

The shared latent z_t is preserved (same dim N_LATENT) so the downstream
sub-actors (phase/power/Ck — already classical) consume identical features and
the comparison isolates exactly the quantum-vs-classical actor block.

Interface parity: implements the same public surface train.py / probes call on
QuantumActor — extract_state (inherited, identical features), forward,
compute_log_prob, compute_logprobs_batch, compute_logprobs_grads_batch,
apply_grads, save, from_dir, pretrain_ae_step — plus the diagnostic attributes
(lam_y/lam_z stubs = 0, N_QUBITS/N_QUANTUM stubs) so the training diag panel and
hyperparameters dump don't need special-casing.

extract_state / _group_norm{,_batch} are INHERITED from QuantumActor unchanged
(they only read self.K, self.B) → guarantees the DNN sees the SAME input
features as the VQC actor (fair comparison).
"""
import os
import json
import numpy as np

from RL.quantum_actor import QuantumActor, _Adam, _relu, _softmax, _he


class ClassicalActor(QuantumActor):
    """Pure-MLP actor. Subclasses QuantumActor ONLY to inherit the identical
    state-feature extraction (extract_state, _group_norm, _group_norm_batch);
    every quantum code path is overridden below and none is reachable."""

    # NOTE: deliberately does NOT call super().__init__ (skips VQC/circuit init).
    def __init__(self, cfg,
                 n_latent:   int,
                 enc_hidden=(128,),
                 pol_hidden=(256, 128),
                 lr:         float = 1e-3,
                 seed:       int   = None):
        # ── dims shared with QuantumActor (extract_state / _group_norm use these) ──
        self.B         = cfg.M
        self.K         = cfg.K
        self.n_choices = self.B + 1                     # 0=direct, 1..M = IRS id
        self.d_s       = self.K * (self.B + 2) + 2 * self.B
        self.N_LATENT  = int(n_latent)
        self.ENC_HIDDEN = list(enc_hidden)
        self.POL_HIDDEN = list(pol_hidden)

        # ── diag/dump stubs so train.py never special-cases classical mode ──
        self.N_QUBITS        = 0
        self.N_QUANTUM       = 0
        self.N_VAR_LAYERS    = 0
        self.DATA_REUPLOADING = False
        self.N_HIDDEN_AE     = self.ENC_HIDDEN
        self.N_HIDDEN_POST   = self.POL_HIDDEN
        self.FULL_ZZ_PAIRS   = ()
        self.EXTRA_ZZ_PAIRS  = ()
        self.EXTRA_CZ_PAIRS  = ()
        self.spsa_n_reps     = 0
        self.spsa_epsilon    = 0.0
        self.n_shots         = 0
        self.lam_y = np.zeros(1)        # frozen-λ diag reads max|·| → 0.00 (no quantum scales)
        self.lam_z = np.zeros(1)
        self.READOUT_MODE = 'classical'
        self.SOFTMAX_HEAD = False

        self.lr  = float(lr)
        self.rng = np.random.default_rng(seed)
        self._init_params(seed)
        # Single optimizer over all MLP params. train.py's per-episode LR schedule
        # sets actor.opt_{ae,qc,xi}.lr — we alias opt_qc to the real optimizer so the
        # schedule drives it; opt_ae/opt_xi are inert (no AE/ξ params in classical mode).
        self.opt    = _Adam(lr=self.lr)
        self.opt_qc = self.opt
        self.opt_ae = _Adam(lr=self.lr)
        self.opt_xi = _Adam(lr=self.lr)

    # ── Parameter init ─────────────────────────────────────────────────────────
    def _init_params(self, seed: int = None) -> None:
        rng = np.random.default_rng(seed)
        self.params = {}
        # Encoder: d_s → enc_hidden... → N_LATENT
        dims = [self.d_s] + self.ENC_HIDDEN + [self.N_LATENT]
        self._enc_layers = len(dims) - 1
        for i in range(self._enc_layers):
            self.params[f'W_enc_{i}'] = _he(dims[i], dims[i + 1], rng)
            self.params[f'b_enc_{i}'] = np.zeros(dims[i + 1])
        # Policy: N_LATENT → pol_hidden... → K*n_choices
        out = self.K * self.n_choices
        pdims = [self.N_LATENT] + self.POL_HIDDEN + [out]
        self._pol_layers = len(pdims) - 1
        for i in range(self._pol_layers):
            self.params[f'W_pol_{i}'] = _he(pdims[i], pdims[i + 1], rng)
            self.params[f'b_pol_{i}'] = np.zeros(pdims[i + 1])

    def num_params(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # ── Forward helpers ─────────────────────────────────────────────────────────
    def _encode(self, a_norm: np.ndarray) -> np.ndarray:
        """s_norm → z_t (single sample). ReLU hidden, linear last → z_t (N_LATENT)."""
        x = a_norm
        for i in range(self._enc_layers):
            x = x @ self.params[f'W_enc_{i}'] + self.params[f'b_enc_{i}']
            if i < self._enc_layers - 1:
                x = _relu(x)
        return x                                       # z_t (N_LATENT,) linear output

    def _policy_logits(self, z_t: np.ndarray) -> np.ndarray:
        x = z_t
        for i in range(self._pol_layers):
            x = x @ self.params[f'W_pol_{i}'] + self.params[f'b_pol_{i}']
            if i < self._pol_layers - 1:
                x = _relu(x)
        return x                                       # logits (K*n_choices,)

    def _forward_logits_batch(self, a_norm_b: np.ndarray):
        """Batch forward with cached pre-activations for backprop.
        Returns logits_b (B_s, K*nc), z_t_b (B_s, N_LATENT), cache."""
        enc_acts = [a_norm_b]
        enc_pres = []
        x = a_norm_b
        for i in range(self._enc_layers):
            pre = x @ self.params[f'W_enc_{i}'] + self.params[f'b_enc_{i}']
            enc_pres.append(pre)
            x = _relu(pre) if i < self._enc_layers - 1 else pre
            enc_acts.append(x)
        z_t_b = x                                       # (B_s, N_LATENT)

        pol_acts = [z_t_b]
        pol_pres = []
        x = z_t_b
        for i in range(self._pol_layers):
            pre = x @ self.params[f'W_pol_{i}'] + self.params[f'b_pol_{i}']
            pol_pres.append(pre)
            x = _relu(pre) if i < self._pol_layers - 1 else pre
            pol_acts.append(x)
        logits_b = x                                    # (B_s, K*nc)
        cache = dict(enc_acts=enc_acts, enc_pres=enc_pres,
                     pol_acts=pol_acts, pol_pres=pol_pres)
        return logits_b, z_t_b, cache

    # ── Public forward (sampling) ────────────────────────────────────────────────
    def forward(self, s_t: np.ndarray, greedy: bool = False) -> tuple:
        a_norm = self._group_norm(s_t)
        z_t    = self._encode(a_norm)
        logits = self._policy_logits(z_t)
        pi_2d  = _softmax(logits.reshape(self.K, self.n_choices))
        if greedy:
            phi = pi_2d.argmax(axis=1).astype(int)
        else:
            phi = np.array([self.rng.choice(self.n_choices, p=pi_2d[k])
                            for k in range(self.K)])
        log_prob = float(sum(np.log(pi_2d[k, phi[k]] + 1e-10) for k in range(self.K)))
        # o_hat kept for info-dict parity (unused downstream); z_t is the real shared latent
        return phi, log_prob, {'z_t': z_t, 'o_hat': np.zeros(0), 'pi': pi_2d}

    # ── Analytic log-prob (PPO ratio) ────────────────────────────────────────────
    def compute_log_prob(self, s_t: np.ndarray, phi: np.ndarray,
                         K_active: int = None) -> float:
        K_used = K_active if K_active is not None else self.K
        a_norm = self._group_norm(s_t)
        logits = self._policy_logits(self._encode(a_norm))
        pi_2d  = _softmax(logits.reshape(self.K, self.n_choices))
        return float(sum(np.log(pi_2d[k, phi[k]] + 1e-10) for k in range(K_used)))

    def compute_logprobs_batch(self, s_t_list, phi_list,
                               K_active: int = None) -> np.ndarray:
        B_s    = len(s_t_list)
        K_used = K_active if K_active is not None else self.K
        a_norm_b = self._group_norm_batch(np.array(s_t_list))
        logits_b, _, _ = self._forward_logits_batch(a_norm_b)
        logits_2d = logits_b.reshape(B_s, self.K, self.n_choices)
        e = np.exp(logits_2d - logits_2d.max(axis=2, keepdims=True))
        pi = e / e.sum(axis=2, keepdims=True)
        phi_np = np.array(phi_list)
        lp = np.log(pi + 1e-10)[np.arange(B_s)[:, None],
                                np.arange(self.K)[None, :], phi_np]
        return lp[:, :K_used].sum(axis=1)

    # ── Fused PPO log-prob + gradient (main training path) ───────────────────────
    def compute_logprobs_grads_batch(self, s_t_list, phi_list,
                                     lp_old_arr, adv_arr,
                                     ppo_epsilon, ae_weight, beta_entropy,
                                     K_active: int = None,
                                     routing_target_b=None,
                                     shape_mask_b=None,
                                     shape_coef: float = 0.0):
        """Returns the SAME 8-tuple as QuantumActor:
        (l_q_arr, L_ae_avg, L_ent_avg, grads_ae, grads_qc, grads_xi, clip_frac_q, kl_q).
        L_ae_avg is 0 (no AE). All MLP grads go in grads_qc (with dummy lam_y/lam_z=0
        so the frozen-λ diag never KeyErrors); grads_ae/grads_xi are empty."""
        B_s    = len(s_t_list)
        K_used = K_active if K_active is not None else self.K
        K, nc  = self.K, self.n_choices

        a_norm_b = self._group_norm_batch(np.array(s_t_list))
        logits_b, z_t_b, cache = self._forward_logits_batch(a_norm_b)
        logits_2d_b = logits_b.reshape(B_s, K, nc)
        e_b  = np.exp(logits_2d_b - logits_2d_b.max(axis=2, keepdims=True))
        pi_b = e_b / e_b.sum(axis=2, keepdims=True)

        # ── PPO clipping (identical to QuantumActor) ──────────────────────────
        phi_np     = np.array(phi_list)
        log_pi_b   = np.log(pi_b + 1e-10)
        lp_sel     = log_pi_b[np.arange(B_s)[:, None], np.arange(K)[None, :], phi_np]
        log_prob_b = lp_sel[:, :K_used].sum(axis=1)
        ratio_b = np.exp(np.clip(log_prob_b - lp_old_arr, -10.0, 10.0))
        surr1_b = ratio_b * adv_arr
        surr2_b = np.clip(ratio_b, 1.0 - ppo_epsilon, 1.0 + ppo_epsilon) * adv_arr
        eff_q_b = np.where(surr1_b <= surr2_b, ratio_b * adv_arr, 0.0)
        l_q_arr = -np.minimum(surr1_b, surr2_b)
        clip_frac_q = float(np.mean(surr1_b > surr2_b))
        kl_q = float(np.mean((ratio_b - 1.0) - np.log(ratio_b + 1e-10)))

        H_b       = -(pi_b * log_pi_b).sum(axis=2)
        L_ent_avg = float((-beta_entropy * H_b[:, :K_used].sum(axis=1)).mean())

        # ── dL/dlogits — SAME formula as QuantumActor (PPO PG + entropy) ──────
        one_hot_b = np.zeros_like(pi_b)
        one_hot_b[np.arange(B_s)[:, None], np.arange(K)[None, :], phi_np] = 1.0
        dL_dlogits_3d = (
            -eff_q_b[:, None, None] * (one_hot_b - pi_b)
            + beta_entropy * pi_b * (log_pi_b + H_b[:, :, None])
        )
        # Routing shaping (Option 1) — same directed CE pull as QuantumActor.
        if shape_coef > 0.0 and routing_target_b is not None:
            tgt_b = np.asarray(routing_target_b)
            msk_b = np.asarray(shape_mask_b, dtype=float)
            onehot_tgt = np.zeros_like(pi_b)
            onehot_tgt[np.arange(B_s)[:, None], np.arange(K)[None, :], tgt_b] = 1.0
            dL_dlogits_3d = dL_dlogits_3d + shape_coef * msk_b[:, :, None] * (pi_b - onehot_tgt)
        dL_dlogits_b = dL_dlogits_3d.reshape(B_s, K * nc)

        # ── Backprop through policy MLP → encoder MLP ─────────────────────────
        grads = {}
        pol_acts, pol_pres = cache['pol_acts'], cache['pol_pres']
        dx = dL_dlogits_b
        for i in reversed(range(self._pol_layers)):
            if i < self._pol_layers - 1:
                dx = dx * (pol_pres[i] > 0)
            grads[f'W_pol_{i}'] = pol_acts[i].T @ dx / B_s
            grads[f'b_pol_{i}'] = dx.mean(axis=0)
            dx = dx @ self.params[f'W_pol_{i}'].T
        dL_dz_t = dx                                    # (B_s, N_LATENT)

        enc_acts, enc_pres = cache['enc_acts'], cache['enc_pres']
        dx = dL_dz_t
        for i in reversed(range(self._enc_layers)):
            if i < self._enc_layers - 1:
                dx = dx * (enc_pres[i] > 0)
            grads[f'W_enc_{i}'] = enc_acts[i].T @ dx / B_s
            grads[f'b_enc_{i}'] = dx.mean(axis=0)
            dx = dx @ self.params[f'W_enc_{i}'].T

        # dummy quantum-scale grads so the frozen-λ diag (g_qc_acc['lam_y']) is safe
        grads['lam_y'] = np.zeros(1)
        grads['lam_z'] = np.zeros(1)

        return l_q_arr, 0.0, L_ent_avg, {}, grads, {}, clip_frac_q, kl_q

    # ── Apply update ─────────────────────────────────────────────────────────────
    def apply_grads(self, grads_ae: dict, grads_qc: dict, grads_xi: dict) -> None:
        """All MLP grads arrive in grads_qc (lam_y/lam_z dummies skipped). One Adam."""
        upd = {k: v for k, v in grads_qc.items() if k in self.params}
        if upd:
            self.opt.step(self.params, upd)

    # ── No-op AE pretrain (classical actor has no autoencoder) ───────────────────
    def pretrain_ae_step(self, s_t: np.ndarray) -> float:
        return 0.0

    # ── Persistence ──────────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        np.savez(os.path.join(path, 'actor_params.npz'), **self.params)
        cfg = {
            'mode':       'classical',
            'K':          self.K,
            'B':          self.B,
            'n_latent':   self.N_LATENT,
            'enc_hidden': self.ENC_HIDDEN,
            'pol_hidden': self.POL_HIDDEN,
            'lr':         self.lr,
        }
        with open(os.path.join(path, 'actor_config.json'), 'w') as f:
            json.dump(cfg, f, indent=2)

    @classmethod
    def from_dir(cls, path: str, cfg=None, seed: int = None):
        from types import SimpleNamespace
        with open(os.path.join(path, 'actor_config.json')) as f:
            ac = json.load(f)
        mini_cfg = SimpleNamespace(K=int(ac['K']), M=int(ac['B']))
        actor = cls(mini_cfg,
                    n_latent=int(ac['n_latent']),
                    enc_hidden=tuple(ac.get('enc_hidden', (128,))),
                    pol_hidden=tuple(ac.get('pol_hidden', (256, 128))),
                    lr=float(ac.get('lr', 1e-3)),
                    seed=seed)
        data = np.load(os.path.join(path, 'actor_params.npz'))
        for k in actor.params:
            actor.params[k] = data[k]
        return actor
