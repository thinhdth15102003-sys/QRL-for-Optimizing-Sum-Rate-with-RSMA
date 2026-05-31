"""
critic.py
---------
Action-conditioned MLP critic  Q(s_t, a_t; ψ)  (CTDE style).

Architecture
------------
  [s_t ‖ a_t]  →  LayerNorm  →  [hidden[0], ReLU]  →  …  →  scalar Q

The joint action vector a_t is a fixed-size float encoding of all four
sub-actor outputs (assignment, phase, power, C_k).  During inference each
actor runs with only its local state; the critic is never deployed.

Hidden layer sizes are configured in params.py via  critic_hidden = [512,256,128,64].

Gradient
--------
  TD loss  L = ½ (Q(s_t,a_t) - target)²,
  target   = r + γ · Q(s_{t+1}, a_{t+1})  (frozen before PPO epochs)
  Standard backprop through all layers; Adam optimiser.
"""

import os
import json

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _he(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)


class _Adam:
    def __init__(self, lr=3e-4, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self._m: dict = {}; self._v: dict = {}; self._t = 0

    def step(self, params: dict, grads: dict) -> None:
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t
        for k, g in grads.items():
            if k not in self._m:
                self._m[k] = np.zeros_like(g)
                self._v[k] = np.zeros_like(g)
            self._m[k] = self.beta1 * self._m[k] + (1 - self.beta1) * g
            self._v[k] = self.beta2 * self._v[k] + (1 - self.beta2) * g * g
            params[k] -= self.lr * (self._m[k] / bc1) / (
                np.sqrt(self._v[k] / bc2) + self.eps)


class ClassicalCritic:
    """
    Dynamic-depth MLP action-conditioned critic  Q(s_t, a_t; ψ).

    Parameters
    ----------
    d_state  : int         state dimension (= actor.d_s)
    d_action : int         action encoding dimension (= K + M*N + M+1 + 2*K)
    hidden   : list[int]   hidden layer sizes, e.g. [512, 256, 128, 64]
    lr       : float       Adam learning rate
    gamma    : float       TD discount factor (used by caller to compute target)
    seed     : int
    """

    def __init__(self, d_state: int,
                 d_action: int  = 0,
                 hidden:   list = None,
                 lr:       float = 3e-4,
                 gamma:    float = 0.99,
                 seed:     int   = None,
                 popart:       bool  = False,
                 popart_beta:  float = 0.1,
                 popart_sigma_floor: float = 1e-2):

        self.d_state  = d_state
        self.d_action = d_action
        self.d_in     = d_state + d_action
        self.gamma    = gamma
        self.hidden   = list(hidden) if hidden is not None else [256, 128, 64]

        # ── PopArt (van Hasselt 2016) — value-target normalisation + output preservation ──
        # The network's last layer outputs a NORMALISED value ṽ; the real value is
        #   V = σ·ṽ + μ,  with (μ,σ) running stats of the returns.
        # On every stats update we RESCALE the last layer (W,b) so V is unchanged for
        # the same input (the "POP" = Preserve Outputs Precisely part) — this kills the
        # discontinuity a naive target-normalisation would cause.  Regressing ṽ to the
        # normalised target keeps gradients O(1) regardless of return scale/drift.
        self.popart        = bool(popart)
        self.pa_beta       = float(popart_beta)        # EMA rate per stats update (per rollout)
        self.pa_sig_floor  = float(popart_sigma_floor) # σ lower bound (avoid div-by-0 early)
        self.pa_mu         = 0.0                        # running mean of returns
        self.pa_nu         = 1.0                        # running 2nd moment E[G²]
        self.pa_sigma      = 1.0                        # running std = sqrt(nu − mu²)
        self.pa_initialized = False

        rng = np.random.default_rng(seed)

        # Build layer sizes:  [d_in, h0, h1, …, 1]
        sizes = [self.d_in] + self.hidden + [1]
        self.n_layers = len(sizes) - 1

        self.Ws = [_he(sizes[i], sizes[i + 1], rng) for i in range(self.n_layers)]
        self.bs = [np.zeros(sizes[i + 1])            for i in range(self.n_layers)]

        self._params: dict = {}
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            self._params[f'W{i}'] = W
            self._params[f'b{i}'] = b

        self.opt = _Adam(lr=lr)

        # Frozen target network for stable TD targets (Polyak soft update)
        self._target_params: dict = {k: v.copy() for k, v in self._params.items()}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return (x - x.mean()) / (x.std() + eps)

    def _sa(self, s_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        """Concatenate state and action; layer-normalise the joint vector."""
        sa = np.concatenate([s_t, a_t]) if self.d_action > 0 else s_t
        return self._layer_norm(sa)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, s_t: np.ndarray, a_t: np.ndarray = None) -> float:
        """
        Parameters
        ----------
        s_t : (d_state,) state vector
        a_t : (d_action,) action encoding, or None when d_action == 0

        Returns
        -------
        Q : float  Q(s_t, a_t)
        """
        x = self._sa(s_t, a_t if a_t is not None else np.empty(0))
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            x = x @ W + b
            if i < self.n_layers - 1:
                x = _relu(x)
        return self._denorm(float(x[0]))      # ṽ → real V = σ·ṽ+μ (PopArt)

    def forward_target(self, s_t: np.ndarray, a_t: np.ndarray = None) -> float:
        """Q-value from the frozen target network (used to compute TD targets)."""
        x  = self._sa(s_t, a_t if a_t is not None else np.empty(0))
        Ws = [self._target_params[f'W{i}'] for i in range(self.n_layers)]
        bs = [self._target_params[f'b{i}'] for i in range(self.n_layers)]
        for i, (W, b) in enumerate(zip(Ws, bs)):
            x = x @ W + b
            if i < self.n_layers - 1:
                x = _relu(x)
        return self._denorm(float(x[0]))      # PopArt un-normalise

    def polyak_update(self, tau: float) -> None:
        """Soft update: target ← τ·current + (1−τ)·target."""
        for k in self._params:
            self._target_params[k] *= (1.0 - tau)
            self._target_params[k] += tau * self._params[k]

    # ── PopArt ─────────────────────────────────────────────────────────────────

    def update_popart_stats(self, returns: np.ndarray) -> None:
        """Update running (μ,σ) from a batch of returns and PRESERVE OUTPUTS by
        rescaling the last layer (W,b).  Call ONCE per rollout BEFORE the PPO
        epochs (not per minibatch).  No-op when popart is disabled.

        Stats: EMA of mean μ and 2nd-moment ν=E[G²]; σ=sqrt(ν−μ²) with a floor.
        POP rescale (last layer L), old (μ,σ) → new (μ',σ'):
            W_L ← (σ/σ')·W_L
            b_L ← (σ·b_L + μ − μ')/σ'
        so that V = σ'·(W_L·h+b_L')+μ' equals σ·(W_L_old·h+b_L_old)+μ for all h.
        """
        if not self.popart:
            return
        G = np.asarray(returns, dtype=float).ravel()
        if G.size == 0:
            return
        b_mean = float(G.mean())
        b_nu   = float(np.mean(G * G))

        if not self.pa_initialized:
            # First call: snap stats to the batch (last layer still ~fresh → no rescale).
            self.pa_mu    = b_mean
            self.pa_nu    = max(b_nu, b_mean * b_mean + self.pa_sig_floor ** 2)
            self.pa_sigma = max(np.sqrt(max(self.pa_nu - self.pa_mu ** 2, 0.0)),
                                self.pa_sig_floor)
            self.pa_initialized = True
            return

        mu_old, sig_old = self.pa_mu, self.pa_sigma
        b = self.pa_beta
        mu_new = (1.0 - b) * self.pa_mu + b * b_mean
        nu_new = (1.0 - b) * self.pa_nu + b * b_nu
        sig_new = max(np.sqrt(max(nu_new - mu_new * mu_new, 0.0)), self.pa_sig_floor)

        # POP: rescale last layer so un-normalised outputs are preserved.
        L  = self.n_layers - 1
        WL = self._params[f'W{L}']
        bL = self._params[f'b{L}']
        WL *= (sig_old / sig_new)
        bL[:] = (sig_old * bL + mu_old - mu_new) / sig_new
        # keep target net consistent (it un-normalises with the same stats)
        self._target_params[f'W{L}'] *= (sig_old / sig_new)
        self._target_params[f'b{L}'][:] = (
            sig_old * self._target_params[f'b{L}'] + mu_old - mu_new) / sig_new

        self.pa_mu, self.pa_nu, self.pa_sigma = mu_new, nu_new, sig_new

    def _denorm(self, v_norm: float) -> float:
        """ṽ → real value V = σ·ṽ + μ (identity when popart off, μ=0 σ=1)."""
        return self.pa_sigma * v_norm + self.pa_mu if self.popart else v_norm

    def normalize_target(self, target):
        """G → (G−μ)/σ for the regression head (identity when popart off)."""
        if not self.popart:
            return target
        return (np.asarray(target, dtype=float) - self.pa_mu) / self.pa_sigma

    # ── Update ────────────────────────────────────────────────────────────────

    def compute_grads(self, s_t: np.ndarray,
                      a_t:   np.ndarray,
                      target: float) -> tuple:
        """
        TD(0) gradient computation.

        Parameters
        ----------
        s_t    : (d_state,)  current state
        a_t    : (d_action,) action encoding (pass np.empty(0) if d_action==0)
        target : float       r_t + γ · Q(s_{t+1}, a_{t+1})

        Returns
        -------
        td_loss : float
        grads   : dict  {'W0','b0',...}  gradients w.r.t. ψ
        """
        sa_norm = self._sa(s_t, a_t)

        inputs:   list = [sa_norm]
        pre_acts: list = []

        x = sa_norm
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            pre = x @ W + b
            pre_acts.append(pre)
            x = _relu(pre) if i < self.n_layers - 1 else pre
            inputs.append(x)

        # PopArt: the network outputs the NORMALISED value ṽ; regress it to the
        # normalised target (G−μ)/σ.  (Identity when popart is off.)
        Q       = float(inputs[-1][0])                 # ṽ (normalised head output)
        td_err  = Q - float(self.normalize_target(target))
        td_loss = 0.5 * td_err ** 2

        grads: dict = {}
        d_out = np.array([td_err])

        for i in reversed(range(self.n_layers)):
            d_pre = d_out * (pre_acts[i] > 0) if i < self.n_layers - 1 else d_out
            grads[f'W{i}'] = np.outer(inputs[i], d_pre)
            grads[f'b{i}'] = d_pre.copy()
            d_out = d_pre @ self.Ws[i].T

        return td_loss, grads

    def apply_grads(self, grads: dict) -> None:
        self.opt.step(self._params, grads)

    def update(self, s_t: np.ndarray, a_t: np.ndarray, target: float) -> float:
        """Compute gradients and apply in one call. Returns td_loss."""
        td_loss, grads = self.compute_grads(s_t, a_t, target)
        self.apply_grads(grads)
        return td_loss

    def compute_grads_batch(self, s_t_list: list,
                             a_t_list:  list,
                             targets:   np.ndarray) -> tuple:
        """
        Vectorised TD gradient over B transitions in one matrix pass.

        Replaces B sequential compute_grads calls with batched matrix
        multiplications, eliminating Python-loop overhead.

        Returns
        -------
        td_loss : float   mean TD loss over the batch
        grads   : dict    averaged gradients {'W0','b0',...}
        """
        B = len(targets)
        # Build (B, d_in) batch — layer-norm each row independently
        if self.d_action > 0:
            sa_b = np.stack([
                np.concatenate([s, a]) for s, a in zip(s_t_list, a_t_list)
            ])                                             # (B, d_in)
        else:
            sa_b = np.stack(s_t_list)                     # (B, d_in)
        # Per-row LayerNorm (vectorised)
        mu  = sa_b.mean(axis=1, keepdims=True)
        std = sa_b.std(axis=1,  keepdims=True) + 1e-6
        x   = (sa_b - mu) / std                           # (B, d_in)

        # Forward pass
        inputs: list = [x]
        pre_acts: list = []
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            pre = x @ W + b                               # (B, out)
            pre_acts.append(pre)
            x = _relu(pre) if i < self.n_layers - 1 else pre
            inputs.append(x)

        # PopArt: network outputs normalised value ṽ; regress to normalised target.
        Q_b     = inputs[-1][:, 0]                        # (B,) ṽ (normalised)
        td_errs = Q_b - self.normalize_target(targets)    # (B,)
        td_loss = float(0.5 * np.mean(td_errs ** 2))

        # Backward pass — d_out starts as (B, 1), averaged over B
        grads: dict = {}
        d_out = (td_errs / B)[:, None]                    # (B, 1)
        for i in reversed(range(self.n_layers)):
            d_pre = d_out * (pre_acts[i] > 0) if i < self.n_layers - 1 else d_out
            grads[f'W{i}'] = inputs[i].T @ d_pre          # (in, out) — already /B
            grads[f'b{i}'] = d_pre.sum(axis=0)
            d_out = d_pre @ self.Ws[i].T

        return td_loss, grads

    # ── Summary ───────────────────────────────────────────────────────────────

    @property
    def architecture_str(self) -> str:
        sizes = [self.d_in] + self.hidden + [1]
        tag   = f'[{self.d_state}s‖{self.d_action}a]→' if self.d_action > 0 else ''
        return tag + ' → '.join(str(s) for s in sizes)

    # ── Parameter I/O ─────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self._params.items()}

    def set_params(self, snapshot: dict) -> None:
        for k in self._params:
            if k in snapshot:
                self._params[k][:] = snapshot[k]

    def save(self, path: str) -> None:
        """Save config + parameters to `path` (critic_config.json + critic_params.npz)."""
        os.makedirs(path, exist_ok=True)
        cfg_dict = {
            'd_state':  self.d_state,
            'd_action': self.d_action,
            'hidden':   list(self.hidden),
            'lr':       self.opt.lr,
            'gamma':    self.gamma,
            # PopArt config + running stats (so resume keeps the value scale)
            'popart':       self.popart,
            'popart_beta':  self.pa_beta,
            'popart_sigma_floor': self.pa_sig_floor,
            'pa_mu':        self.pa_mu,
            'pa_nu':        self.pa_nu,
            'pa_sigma':     self.pa_sigma,
            'pa_initialized': self.pa_initialized,
        }
        with open(os.path.join(path, 'critic_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'critic_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        """
        Reconstruct a ClassicalCritic from a saved directory.

        Structural dimensions (d_state, d_action, hidden) are taken from the
        saved critic_config.json so the network always matches its training-time
        layout regardless of params.py. The target network is synced to the
        loaded weights, so bootstrapped TD targets are calibrated from step one
        instead of cold-starting from a random init.
        """
        with open(os.path.join(path, 'critic_config.json')) as f:
            c = json.load(f)
        critic = cls(
            d_state  = c['d_state'],
            d_action = c.get('d_action', 0),
            hidden   = c['hidden'],
            lr       = c['lr'],
            gamma    = c['gamma'],
            seed     = seed,
            popart       = c.get('popart', False),
            popart_beta  = c.get('popart_beta', 0.1),
            popart_sigma_floor = c.get('popart_sigma_floor', 1e-2),
        )
        # Restore PopArt running stats (value scale) if present.
        critic.pa_mu          = c.get('pa_mu', 0.0)
        critic.pa_nu          = c.get('pa_nu', 1.0)
        critic.pa_sigma       = c.get('pa_sigma', 1.0)
        critic.pa_initialized = c.get('pa_initialized', False)
        critic.set_params(dict(np.load(os.path.join(path, 'critic_params.npz'))))
        critic._target_params = {k: v.copy() for k, v in critic._params.items()}
        return critic
