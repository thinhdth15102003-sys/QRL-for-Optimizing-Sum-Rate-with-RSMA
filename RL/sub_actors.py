"""
sub_actors.py
-------------
Classical MLP sub-policies for the three remaining action components.

Pipeline position
-----------------
  After the quantum actor produces the IRS assignment φ:

  1. PhaseMLP  — IRS phase shifts φ_{m,n} ∈ {0,..,n_levels-1}^{M×N}
                 State: c^SRU_{m,k} = g_SR[m] · g_RU[m,k] for assigned pairs,
                        zero otherwise.  Input dim = 2·M·K (Re + Im stacked).

  2. PowerMLP  — power allocation [w_c_vec, w_p] summing to P_S
                 State: |h_eff[k]| for all K users (effective channel magnitudes
                        after applying the phase shifts from step 1).
                        Input dim = K.  Output dim = M + K.
                 Inactive IRS (no assigned users) are masked to -∞ before softmax.

  3. CkMLP     — common-rate split C_k per user with Σ_{k∈g} C_k = R_c_g
                 State: [D_k, R_p_k, R_c_g_k] per user.  Input dim = 3·K.
                 Within-group softmax ensures the group budget constraint.

REINFORCE gradient
------------------
  PhaseMLP : per-element categorical  →  -A · (one_hot - softmax) per element
  PowerMLP : categorical over M+K slots with active mask  →  same gradient form
  CkMLP    : per-group categorical (one member sampled per group)
              →  -A · (one_hot_g - alpha_g) for each group g

Hidden layers
-------------
  All three classes accept `hidden` as either an int (single hidden layer) or a
  list of ints (arbitrary depth), e.g. hidden=[128, 256, 128].  The network is
  always: input → hidden[0] → ReLU → … → hidden[-1] → ReLU → output (no ReLU).
"""

import os
import json
import numpy as np


# ── Shared helpers ─────────────────────────────────────────────────────────────

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


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _he(in_dim: int, out_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((in_dim, out_dim)) * np.sqrt(2.0 / in_dim)


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.nanmax(x))
    return e / e.sum()


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax for 2-D input."""
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _parse_hidden(hidden) -> list:
    """Normalise hidden spec to a list of ints."""
    if isinstance(hidden, int):
        return [hidden]
    return list(int(h) for h in hidden)


def _build_layers(d_in: int, d_out: int,
                  hidden_sizes: list,
                  rng: np.random.Generator):
    """
    Allocate weight matrices and bias vectors for a fully-connected MLP.

    Returns
    -------
    Ws : list of (d_prev, d_next) arrays  — length = len(hidden_sizes) + 1
    bs : list of (d_next,) arrays
    """
    dims = [d_in] + hidden_sizes + [d_out]
    Ws = [_he(dims[i], dims[i + 1], rng) for i in range(len(dims) - 1)]
    bs = [np.zeros(dims[i + 1])          for i in range(len(dims) - 1)]
    return Ws, bs


def _make_params(Ws: list, bs: list) -> dict:
    """Return a {f'W{i}': array, f'b{i}': array} dict of references."""
    params = {}
    for i, (W, b) in enumerate(zip(Ws, bs)):
        params[f'W{i}'] = W
        params[f'b{i}'] = b
    return params


def _mlp_forward(s_t: np.ndarray, Ws: list, bs: list):
    """
    Full MLP forward pass (all hidden layers use ReLU; output is linear).

    Returns
    -------
    pres   : list of pre-ReLU activations for each hidden layer
    hs     : list of post-ReLU activations for each hidden layer
    logits : final linear output
    """
    pres, hs = [], []
    x = s_t
    for W, b in zip(Ws[:-1], bs[:-1]):
        pre = x @ W + b
        h   = _relu(pre)
        pres.append(pre)
        hs.append(h)
        x = h
    logits = x @ Ws[-1] + bs[-1]
    return pres, hs, logits


def _mlp_backward(s_t: np.ndarray,
                  pres: list, hs: list,
                  Ws: list,
                  dL_dlogits: np.ndarray) -> dict:
    """
    Backprop through the MLP produced by _mlp_forward.

    Returns grad dict {f'W{i}': grad, f'b{i}': grad}.
    """
    grads = {}
    n = len(Ws)

    # Gradient for the output (last) layer
    x_in = hs[-1] if hs else s_t
    grads[f'W{n - 1}'] = np.outer(x_in, dL_dlogits)
    grads[f'b{n - 1}'] = dL_dlogits.copy()

    delta = dL_dlogits @ Ws[-1].T

    for i in range(n - 2, -1, -1):
        dpre  = delta * (pres[i] > 0)               # ReLU mask
        x_in  = hs[i - 1] if i > 0 else s_t
        grads[f'W{i}'] = np.outer(x_in, dpre)
        grads[f'b{i}'] = dpre.copy()
        if i > 0:
            delta = dpre @ Ws[i].T

    return grads


def _arch_str(d_in: int, hidden_sizes: list, d_out: int) -> str:
    """Human-readable architecture string, e.g. '120 → 64 → 128 → 1440'."""
    parts = [str(d_in)] + [str(h) for h in hidden_sizes] + [str(d_out)]
    return ' → '.join(parts)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-sample layer normalisation: zero-mean, unit-std across all elements."""
    mu = x.mean()
    return (x - mu) / (x.std() + eps)


def _active_irs_from_phi(phi: np.ndarray) -> np.ndarray:
    """Sorted 0-based IRS indices with ≥1 assigned user.
    phi: (K,) int — 0=direct, 1..M=IRS (1-based gid).
    Mirrors train._get_active_irs without importing train.
    """
    return np.array(
        sorted({int(phi[k]) - 1 for k in range(len(phi)) if phi[k] > 0}),
        dtype=int,
    )


def _layer_norm_batch(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Row-wise layer normalisation for a 2-D batch (B, d)."""
    mu  = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + eps
    return (X - mu) / std


def _mlp_forward_batch(X: np.ndarray, Ws: list, bs: list):
    """
    Batched MLP forward pass (B, d_in) → pre_acts, acts, logits (B, d_out).
    Hidden layers use ReLU; output layer is linear (no activation).
    """
    pre_acts, acts = [], []
    x = X
    for W, b in zip(Ws[:-1], bs[:-1]):
        pre = x @ W + b          # (B, h)
        h   = _relu(pre)
        pre_acts.append(pre)
        acts.append(h)
        x = h
    logits = x @ Ws[-1] + bs[-1]  # (B, d_out)
    return pre_acts, acts, logits


def _mlp_backward_batch(X: np.ndarray,
                        pre_acts: list, acts: list,
                        Ws: list,
                        dL_dlogits: np.ndarray) -> dict:
    """
    Batched backprop through the MLP produced by _mlp_forward_batch.
    Divides the accumulated gradient by X.shape[0] (= batch size B or T).

    PhaseMLP caller note: pass  dL_dlogits * (T / B)  so that the division
    by T here yields the correct B-normalised gradient
    (T = total IRS panels across the mini-batch, B = number of transitions).
    """
    B = X.shape[0]
    n = len(Ws)
    grads: dict = {}

    d_out = dL_dlogits / B           # (B, d_out) — normalise
    x_in  = acts[-1] if acts else X
    grads[f'W{n - 1}'] = x_in.T @ d_out       # (h, d_out)
    grads[f'b{n - 1}'] = d_out.sum(axis=0)    # (d_out,)

    delta = d_out @ Ws[-1].T        # (B, h_prev)

    for i in range(n - 2, -1, -1):
        dpre = delta * (pre_acts[i] > 0)       # ReLU mask, (B, h)
        x_in = acts[i - 1] if i > 0 else X
        grads[f'W{i}'] = x_in.T @ dpre
        grads[f'b{i}'] = dpre.sum(axis=0)
        if i > 0:
            delta = dpre @ Ws[i].T

    return grads


# ── Phase-shift MLP ────────────────────────────────────────────────────────────

class PhaseMLP:
    """
    Per-IRS discrete phase-shift policy with shared weights.

    The same MLP processes each ACTIVE IRS independently.  Inactive IRS
    (no users assigned after IRS-selection) are skipped — their phase_idx
    defaults to 0 and receive no gradient.

    Input per IRS
    -------------
    c^SRU_{m,k} = g_SR[m] · g_RU[m,k]  for IRS m, all K users.
    Stacked as [Re(c_m), Im(c_m)], shape (2·K,).

    Parameters
    ----------
    d_s      : int             input dim per IRS = 2·K
    M        : int             total IRS panels (for output shape only)
    N        : int             elements per IRS
    n_levels : int             discrete phase levels = 2^bits
    hidden   : int | list[int]
    lr       : float
    seed     : int | None
    """

    def __init__(self, d_s: int, M: int, N: int, n_levels: int,
                 hidden=64, lr: float = 3e-4, seed: int = None):
        self.M        = M
        self.N        = N
        self.n_levels = n_levels
        self.d_s      = d_s           # 2·K per IRS
        d_out         = N * n_levels  # logits per IRS

        rng = np.random.default_rng(seed)
        self.rng = rng

        hidden_sizes      = _parse_hidden(hidden)
        self.hidden_sizes = hidden_sizes
        self.Ws, self.bs  = _build_layers(d_s, d_out, hidden_sizes, rng)
        self._params      = _make_params(self.Ws, self.bs)
        self.opt          = _Adam(lr=lr)
        self.architecture = _arch_str(d_s, hidden_sizes, d_out)

    # ── Internal: process one IRS ─────────────────────────────────────────

    def _forward_irs(self, s_m: np.ndarray, greedy: bool):
        """Forward for one IRS panel. Returns (idx (N,), log_prob, probs (N, n_levels))."""
        s_m = _layer_norm(s_m)
        _, _, logits = _mlp_forward(s_m, self.Ws, self.bs)
        probs = _softmax_rows(logits.reshape(self.N, self.n_levels))  # (N, n_levels)
        if greedy:
            idx = probs.argmax(axis=1)
        else:
            idx = np.array([self.rng.choice(self.n_levels, p=probs[n])
                            for n in range(self.N)])
        log_prob = float(sum(np.log(probs[n, idx[n]] + 1e-10) for n in range(self.N)))
        return idx, log_prob, probs

    # ── Public forward ────────────────────────────────────────────────────

    def forward(self, s_phase_mat: np.ndarray, active_irs: np.ndarray,
                greedy: bool = False) -> tuple:
        """
        Parameters
        ----------
        s_phase_mat : (M, 2K) float  per-IRS cascade channel states;
                      row m = [Re(c^SRU_m), Im(c^SRU_m)]; inactive rows = 0
        active_irs  : (G,) int       0-based indices of IRS with ≥1 assigned user

        Returns
        -------
        phase_idx     : (M, N) int              inactive IRS rows default to 0
        log_prob      : float                   Σ over active IRS and elements
        probs_per_irs : list[(N, n_levels)]     one entry per active IRS, length G
        """
        phase_idx     = np.zeros((self.M, self.N), dtype=int)
        log_prob      = 0.0
        probs_per_irs = []

        for m in active_irs:
            idx, lp, probs = self._forward_irs(s_phase_mat[m], greedy)
            phase_idx[m]   = idx
            log_prob      += lp
            probs_per_irs.append(probs)

        return phase_idx, log_prob, probs_per_irs

    # ── Gradient computation ──────────────────────────────────────────────

    def compute_grads(self, s_phase_mat: np.ndarray, active_irs: np.ndarray,
                      phase_idx: np.ndarray, advantage: float,
                      beta_entropy: float = 0.0) -> tuple:
        """
        REINFORCE gradients accumulated over all active IRS (shared weights).

        Parameters
        ----------
        s_phase_mat : (M, 2K) float
        active_irs  : (G,) int        0-based active IRS indices
        phase_idx   : (M, N) int      only active IRS rows are used
        advantage   : float
        beta_entropy: float

        Returns (L_pg, L_ent, grads).  grads is {} when G=0 (no active IRS).
        """
        L_pg:  float = 0.0
        L_ent: float = 0.0
        grads: dict  = {}

        for m in active_irs:
            s_m  = _layer_norm(s_phase_mat[m])
            pres, hs, logits = _mlp_forward(s_m, self.Ws, self.bs)
            probs = _softmax_rows(logits.reshape(self.N, self.n_levels))  # (N, n_levels)

            idx_n   = phase_idx[m]                    # (N,) int
            one_hot = np.zeros_like(probs)
            for n, a in enumerate(idx_n):
                one_hot[n, a] = 1.0

            L_pg += float(-advantage * sum(
                np.log(probs[n, idx_n[n]] + 1e-10) for n in range(self.N)
            ))
            dL_pg = -advantage * (one_hot - probs)    # (N, n_levels)

            log_probs  = np.log(probs + 1e-10)
            H_per_elem = -np.sum(probs * log_probs, axis=1, keepdims=True)  # (N,1)
            L_ent     += -beta_entropy * float(np.sum(H_per_elem))
            dL_ent     = beta_entropy * probs * (log_probs + H_per_elem)

            dL_dlogits = (dL_pg + dL_ent).flatten()
            g_m = _mlp_backward(s_m, pres, hs, self.Ws, dL_dlogits)

            # Accumulate into shared weight gradients
            if not grads:
                grads = {k: v.copy() for k, v in g_m.items()}
            else:
                for k in grads:
                    grads[k] += g_m[k]

        return L_pg, L_ent, grads

    def compute_log_prob(self, s_phase_mat: np.ndarray, active_irs: np.ndarray,
                         phase_idx: np.ndarray) -> float:
        """Log π(phase_idx|s) under current policy. Used for PPO ratio."""
        log_prob = 0.0
        for m in active_irs:
            _, _, logits = _mlp_forward(_layer_norm(s_phase_mat[m]), self.Ws, self.bs)
            probs = _softmax_rows(logits.reshape(self.N, self.n_levels))
            log_prob += float(sum(
                np.log(probs[n, phase_idx[m, n]] + 1e-10) for n in range(self.N)
            ))
        return log_prob

    def compute_log_prob_batch(self, trans_list: list) -> np.ndarray:
        """
        Vectorised log π(phase_idx|s) for a mini-batch.

        Each transition dict must contain 's_phase' (M, 2K), 'phase_idx' (M, N),
        and 'phi' (K,) for deriving active IRS.

        Returns
        -------
        lp_b : (B,) float  per-transition total log-probability
        """
        B    = len(trans_list)
        lp_b = np.zeros(B)

        panels_s   = []   # layer-normed inputs, each (d_s,)
        panels_idx = []   # phase_idx per panel, each (N,) int
        sample_ids = []   # which transition each panel belongs to

        for b, trans in enumerate(trans_list):
            for m in _active_irs_from_phi(trans['phi']):
                panels_s.append(_layer_norm(trans['s_phase'][m]))
                panels_idx.append(trans['phase_idx'][m])
                sample_ids.append(b)

        if not panels_s:
            return lp_b   # no active IRS in any transition

        T        = len(panels_s)
        X        = np.stack(panels_s)    # (T, d_s)
        idx_all  = np.stack(panels_idx)  # (T, N) int

        _, _, logits = _mlp_forward_batch(X, self.Ws, self.bs)   # (T, N*L)
        probs_3d = _softmax_rows(
            logits.reshape(T * self.N, self.n_levels)
        ).reshape(T, self.N, self.n_levels)                       # (T, N, L)

        log_p_elem   = np.log(
            probs_3d[np.arange(T)[:, None],
                     np.arange(self.N)[None, :],
                     idx_all] + 1e-10
        )                                                         # (T, N)
        log_p_panels = log_p_elem.sum(axis=1)                     # (T,)
        np.add.at(lp_b, sample_ids, log_p_panels)
        return lp_b

    def compute_grads_batch(self, trans_list: list,
                            eff_adv_b: np.ndarray,
                            beta_entropy: float = 0.0) -> tuple:
        """
        Vectorised REINFORCE+PPO gradient for PhaseMLP over a mini-batch.

        Builds a super-batch of all (transition, active-IRS) panels, runs a
        single batched forward/backward, and returns the B-averaged gradient.

        Parameters
        ----------
        trans_list   : list of transition dicts (keys: 's_phase', 'phi', 'phase_idx')
        eff_adv_b    : (B,) float  PPO effective advantage per transition
        beta_entropy : float

        Returns
        -------
        L_pg  : float  mean(-eff_adv × log_π)  (diagnostic; B-averaged)
        L_ent : float  mean entropy contribution  (B-averaged)
        grads : dict   B-averaged gradient dict  ({} when no active IRS)
        """
        B = len(trans_list)

        panels_s   = []
        panels_idx = []
        sample_ids = []

        for b, trans in enumerate(trans_list):
            for m in _active_irs_from_phi(trans['phi']):
                panels_s.append(_layer_norm(trans['s_phase'][m]))
                panels_idx.append(trans['phase_idx'][m])
                sample_ids.append(b)

        if not panels_s:
            return 0.0, 0.0, {}

        T          = len(panels_s)
        sample_ids = np.array(sample_ids, dtype=int)   # (T,)
        X          = np.stack(panels_s)                # (T, d_s)
        idx_all    = np.stack(panels_idx)              # (T, N) int

        pre_acts, acts, logits = _mlp_forward_batch(X, self.Ws, self.bs)  # (T, N*L)
        probs_3d = _softmax_rows(
            logits.reshape(T * self.N, self.n_levels)
        ).reshape(T, self.N, self.n_levels)                    # (T, N, L)

        one_hot_3d = np.eye(self.n_levels)[idx_all]            # (T, N, L)
        eff_t      = eff_adv_b[sample_ids]                     # (T,) per-panel eff adv

        dL_pg_3d  = -eff_t[:, None, None] * (one_hot_3d - probs_3d)  # (T, N, L)
        log_p_3d  = np.log(probs_3d + 1e-10)                         # (T, N, L)
        H_3d      = -(probs_3d * log_p_3d).sum(axis=2, keepdims=True) # (T, N, 1)
        dL_ent_3d = beta_entropy * probs_3d * (log_p_3d + H_3d)      # (T, N, L)

        dL_dlogits = (dL_pg_3d + dL_ent_3d).reshape(T, -1)    # (T, N*L)

        # _mlp_backward_batch divides by T; scale by T/B to get B-normalised grads
        grads = _mlp_backward_batch(X, pre_acts, acts, self.Ws,
                                    dL_dlogits * (T / B))

        # ── losses (diagnostics, B-averaged) ──────────────────────────────
        log_p_elem   = log_p_3d[np.arange(T)[:, None],
                                 np.arange(self.N)[None, :],
                                 idx_all]                  # (T, N)
        log_p_panels = log_p_elem.sum(axis=1)              # (T,) log π per panel
        lp_b = np.zeros(B)
        np.add.at(lp_b, sample_ids, log_p_panels)
        L_pg = float(np.mean(-eff_adv_b * lp_b))

        H_per_panel = H_3d.sum(axis=(1, 2))                # (T,)
        ent_b = np.zeros(B)
        np.add.at(ent_b, sample_ids, H_per_panel)
        L_ent = float(-beta_entropy * ent_b.mean())

        return L_pg, L_ent, grads

    def apply_grads(self, grads: dict) -> None:
        """Apply pre-computed gradients. No-op when grads is empty (G=0)."""
        if not grads:
            return
        self.opt.step(self._params, grads)

    def update(self, s_phase_mat: np.ndarray, active_irs: np.ndarray,
               phase_idx: np.ndarray, advantage: float,
               beta_entropy: float = 0.0) -> tuple:
        """Compute gradients and apply in one call. Returns (L_pg, L_ent)."""
        L_pg, L_ent, grads = self.compute_grads(
            s_phase_mat, active_irs, phase_idx, advantage, beta_entropy)
        self.apply_grads(grads)
        return L_pg, L_ent

    # ── Parameter I/O ────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self._params.items()}

    def set_params(self, snapshot: dict) -> None:
        for k in self._params:
            if k in snapshot:
                self._params[k][:] = snapshot[k]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg_dict = {
            'd_s':      self.d_s,
            'M':        self.M,
            'N':        self.N,
            'n_levels': self.n_levels,
            'hidden':   self.hidden_sizes,
            'lr':       self.opt.lr,
        }
        with open(os.path.join(path, 'phase_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'phase_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        with open(os.path.join(path, 'phase_config.json')) as f:
            c = json.load(f)
        obj = cls(d_s=c['d_s'], M=c['M'], N=c['N'], n_levels=c['n_levels'],
                  hidden=c['hidden'], lr=c['lr'], seed=seed)
        obj.set_params(dict(np.load(os.path.join(path, 'phase_params.npz'))))
        return obj


# ── Power allocation MLP ───────────────────────────────────────────────────────

class PowerMLP:
    """
    Power allocation policy for G+1 common streams + K private streams.

    G = number of active IRS (those with ≥1 assigned user) — varies per step.
    The network has fixed output size M+1+K (max possible); only G+1+K slots
    are active per step.  w_c_vec is extracted as (G+1,) by keeping only the
    direct-group slot (index 0) and the G active-IRS slots.

    Input state
    -----------
    Effective channel magnitudes |h_eff[k]|, shape (K,).

    Output
    ------
    w_c_vec : (G+1,) float  — [direct, IRS_{active[0]}, …, IRS_{active[G-1]}]
    w_p     : (K,)   float  — private power per user
    Σ w_c_vec + Σ w_p = P_S by construction.

    active_irs_ids : sorted list of 1-based physical IRS gids with users.
                     Passed at runtime; determines G and the active mask.

    Parameters
    ----------
    d_s    : state dimension  (= K)
    K      : number of users
    M      : number of IRS panels  (sets maximum output size)
    P_S    : total power budget (W)
    hidden : int or list[int]
    """

    def __init__(self, d_s: int, K: int, M: int, P_S: float,
                 hidden=64, lr: float = 3e-4, seed: int = None):
        self.d_s = d_s
        self.K   = K
        self.M   = M
        self.P_S = P_S
        self._n_common = M + 1          # max common-stream slots (fixed weight dim)
        d_out = self._n_common + K

        rng = np.random.default_rng(seed)
        self.rng = rng

        hidden_sizes      = _parse_hidden(hidden)
        self.hidden_sizes = hidden_sizes
        self.Ws, self.bs  = _build_layers(d_s, d_out, hidden_sizes, rng)
        self._params      = _make_params(self.Ws, self.bs)
        self.opt          = _Adam(lr=lr)
        self.architecture = _arch_str(d_s, hidden_sizes, d_out)

    def _active_mask(self, active_irs_ids: list) -> np.ndarray:
        """Build (M+1,) bool mask: True for direct group + active IRS groups."""
        mask    = np.zeros(self._n_common, dtype=bool)
        mask[0] = True                          # direct group always active
        for gid in active_irs_ids:
            if 1 <= gid <= self.M:
                mask[gid] = True
        return mask

    def _masked_softmax(self, logits: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
        """Softmax over (M+1+K,) logits with -inf on inactive common slots."""
        logits_m = logits.copy()
        for g in range(self._n_common):
            if not mask[g]:
                logits_m[g] = -np.inf
        return _softmax_1d(logits_m)

    def _extract_wc(self, w_full: np.ndarray,
                    active_irs_ids: list) -> np.ndarray:
        """
        Extract (G+1,) w_c_vec from (M+1,) power slice.
        Order: [direct, active_irs_ids[0], active_irs_ids[1], …]
        """
        slots = [0] + list(active_irs_ids)
        return w_full[slots]

    def forward(self, s_t: np.ndarray,
                active_irs_ids: list) -> tuple:
        """
        Parameters
        ----------
        s_t            : (K,) float   |h_eff| magnitudes
        active_irs_ids : list[int]    1-based physical IRS gids with ≥1 user

        Returns
        -------
        w_c_vec : (G+1,) float   common-stream powers
        w_p     : (K,)   float   private powers
        fracs   : (M+1+K,)       full softmax fractions (for gradient bookkeeping)
        a_sel   : int            sampled action index into fracs (for PPO)
        """
        mask         = self._active_mask(active_irs_ids)
        s_t          = _layer_norm(s_t)
        _, _, logits = _mlp_forward(s_t, self.Ws, self.bs)
        fracs        = self._masked_softmax(logits, mask)
        w            = fracs * self.P_S

        w_c_full = w[:self._n_common]           # (M+1,)
        w_c_vec  = self._extract_wc(w_c_full, active_irs_ids)   # (G+1,)
        w_p      = w[self._n_common:].copy()    # (K,)

        valid_mask  = np.isfinite(fracs) & (fracs > 0)
        valid_idx   = np.where(valid_mask)[0]
        valid_probs = fracs[valid_idx] / fracs[valid_idx].sum()
        a_sel       = int(self.rng.choice(valid_idx, p=valid_probs))
        return w_c_vec, w_p, fracs, a_sel

    def compute_grads(self, s_t: np.ndarray,
                      active_irs_ids: list,
                      advantage: float,
                      beta_entropy: float = 0.0,
                      a_sel: int = None) -> tuple:
        """Returns (L_pg, L_ent, grads) without applying the update."""
        mask             = self._active_mask(active_irs_ids)
        s_t              = _layer_norm(s_t)
        pres, hs, logits = _mlp_forward(s_t, self.Ws, self.bs)
        probs            = self._masked_softmax(logits, mask)

        if a_sel is None:
            valid_mask  = np.isfinite(probs) & (probs > 0)
            valid_idx   = np.where(valid_mask)[0]
            valid_probs = probs[valid_idx] / probs[valid_idx].sum()
            a_sel       = int(self.rng.choice(valid_idx, p=valid_probs))

        L_pg = float(-advantage * np.log(probs[a_sel] + 1e-10))

        one_hot = np.zeros(self._n_common + self.K)
        one_hot[a_sel] = 1.0
        dL_pg_dlogits = -advantage * (one_hot - probs)

        log_probs      = np.log(probs + 1e-10)
        H_dist         = -float(np.sum(probs * log_probs))
        L_ent          = -beta_entropy * H_dist
        dL_ent_dlogits = beta_entropy * probs * (log_probs + H_dist)

        dL_dlogits = dL_pg_dlogits + dL_ent_dlogits
        for g in range(self._n_common):
            if not mask[g]:
                dL_dlogits[g] = 0.0

        grads = _mlp_backward(s_t, pres, hs, self.Ws, dL_dlogits)
        return L_pg, L_ent, grads

    def compute_log_prob(self, s_t: np.ndarray,
                         active_irs_ids: list,
                         a_sel: int) -> float:
        """Log π(a_sel|s) under current policy. Used for PPO ratio."""
        mask         = self._active_mask(active_irs_ids)
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)
        probs        = self._masked_softmax(logits, mask)
        return float(np.log(probs[a_sel] + 1e-10))

    def compute_log_prob_batch(self, trans_list: list) -> np.ndarray:
        """
        Vectorised log π(a_sel|s) for a mini-batch.

        Each transition dict must contain 's_power', 'active_irs_ids',
        and 'power_a_sel'.

        Returns
        -------
        lp_b : (B,) float  per-transition log-probability
        """
        B    = len(trans_list)
        X    = _layer_norm_batch(
            np.stack([t['s_power'] for t in trans_list])
        )                                                        # (B, d_s)
        _, _, logits = _mlp_forward_batch(X, self.Ws, self.bs)  # (B, d_out)

        lp_b = np.zeros(B)
        for b, trans in enumerate(trans_list):
            mask     = self._active_mask(trans['active_irs_ids'])
            probs    = self._masked_softmax(logits[b], mask)
            lp_b[b]  = float(np.log(probs[trans['power_a_sel']] + 1e-10))
        return lp_b

    def compute_grads_batch(self, trans_list: list,
                            eff_adv_b: np.ndarray,
                            beta_entropy: float = 0.0) -> tuple:
        """
        Vectorised PPO gradient for PowerMLP over a mini-batch.

        Batches the MLP forward/backward; the per-sample masked-softmax
        and gradient assembly loop is O(B × (M+K)) and is cheap.

        Returns (L_pg, L_ent, grads)  — all B-averaged.
        """
        B     = len(trans_list)
        d_out = self._n_common + self.K
        X     = _layer_norm_batch(
            np.stack([t['s_power'] for t in trans_list])
        )                                                             # (B, d_s)
        pre_acts, acts, logits = _mlp_forward_batch(X, self.Ws, self.bs)  # (B, d_out)

        dL_dlogits_b = np.zeros((B, d_out))
        L_pg  = 0.0
        L_ent = 0.0

        for b, trans in enumerate(trans_list):
            mask  = self._active_mask(trans['active_irs_ids'])
            probs = self._masked_softmax(logits[b], mask)
            a_sel = trans['power_a_sel']

            L_pg += float(-eff_adv_b[b] * np.log(probs[a_sel] + 1e-10))

            one_hot = np.zeros(d_out)
            one_hot[a_sel] = 1.0
            dL_pg_l   = -eff_adv_b[b] * (one_hot - probs)

            log_probs = np.log(probs + 1e-10)
            H         = -float(np.sum(probs * log_probs))
            L_ent    += -beta_entropy * H
            dL_ent_l  = beta_entropy * probs * (log_probs + H)

            dl = dL_pg_l + dL_ent_l
            for g in range(self._n_common):
                if not mask[g]:
                    dl[g] = 0.0
            dL_dlogits_b[b] = dl

        grads = _mlp_backward_batch(X, pre_acts, acts, self.Ws, dL_dlogits_b)
        return L_pg / B, L_ent / B, grads

    def apply_grads(self, grads: dict) -> None:
        self.opt.step(self._params, grads)

    def update(self, s_t: np.ndarray,
               active_irs_ids: list,
               advantage: float,
               beta_entropy: float = 0.0,
               a_sel: int = None) -> tuple:
        L_pg, L_ent, grads = self.compute_grads(
            s_t, active_irs_ids, advantage, beta_entropy, a_sel)
        self.apply_grads(grads)
        return L_pg, L_ent

    # ── Parameter I/O ────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self._params.items()}

    def set_params(self, snapshot: dict) -> None:
        for k in self._params:
            if k in snapshot:
                self._params[k][:] = snapshot[k]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg_dict = {
            'd_s':    self.d_s,
            'K':      self.K,
            'M':      self.M,
            'P_S':    self.P_S,
            'hidden': self.hidden_sizes,
            'lr':     self.opt.lr,
        }
        with open(os.path.join(path, 'power_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'power_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        with open(os.path.join(path, 'power_config.json')) as f:
            c = json.load(f)
        obj = cls(d_s=c['d_s'], K=c['K'], M=c['M'], P_S=c['P_S'],
                  hidden=c['hidden'], lr=c['lr'], seed=seed)
        obj.set_params(dict(np.load(os.path.join(path, 'power_params.npz'))))
        return obj


# ── Common-rate split MLP ──────────────────────────────────────────────────────

class CkMLP:
    """
    Common-rate allocation policy with per-group budget constraint.

    Input state
    -----------
    [D_k, R_p_k, R_c_g_k] for each user k, shape (3·K,):
      D_k      — traffic demand per user
      R_p_k    — private rate (from compute_rates_partial)
      R_c_g_k  — common rate available for user k's IRS group (0 for direct users)

    Output
    ------
    C_k : (K,) float  — within-group softmax fractions × R_c_g_k.
    alpha_k : (K,) float — raw within-group softmax fractions (for REINFORCE update).

    Parameters
    ----------
    d_s    : state dimension  (= 3·K)
    K      : number of users
    hidden : int or list[int]
    """

    def __init__(self, d_s: int, K: int,
                 hidden=64, lr: float = 3e-4, seed: int = None):
        self.d_s = d_s
        self.K   = K

        rng = np.random.default_rng(seed)
        self.rng = rng

        hidden_sizes      = _parse_hidden(hidden)
        self.hidden_sizes = hidden_sizes
        self.Ws, self.bs  = _build_layers(d_s, K, hidden_sizes, rng)
        self._params      = _make_params(self.Ws, self.bs)
        self.opt          = _Adam(lr=lr)
        self.architecture = _arch_str(d_s, hidden_sizes, K)

    @staticmethod
    def _build_groups(phi: np.ndarray) -> dict:
        groups: dict = {}
        for k, gid in enumerate(phi.astype(int)):
            groups.setdefault(int(gid), []).append(k)
        return groups

    def forward(self, s_t: np.ndarray,
                phi: np.ndarray,
                R_c_group: dict,
                K_active: int = None) -> tuple:
        """
        Parameters
        ----------
        s_t       : (3·K,) float
        phi       : (K,) int      — IRS assignment (0=direct, 1..M=IRS)
        R_c_group : dict {gid: float}
        K_active  : int or None   — number of active users

        Returns
        -------
        C_k       : (K,) float
        alpha_k   : (K,) float    within-group softmax fractions
        group_sel : dict {gid: int}
                    Per-group sampled local index (index into members list).
                    Stored in ep_buf; passed back to compute_grads/compute_log_prob
                    so PPO uses the action that was actually taken.
        """
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)

        C_k       = np.zeros(self.K)
        alpha_k   = np.ones(self.K) / max(self.K, 1)
        group_sel: dict = {}

        K_used = K_active if K_active is not None else self.K
        groups = self._build_groups(phi[:K_used])
        for gid, members in groups.items():
            R_c_g = float(R_c_group.get(gid, 0.0))
            if R_c_g < 1e-12:
                alpha_k[members] = 1.0 / max(len(members), 1)
                group_sel[gid]   = 0
                continue
            logits_g = logits[members]
            e        = np.exp(logits_g - logits_g.max())
            alpha_g  = e / e.sum()
            alpha_k[members] = alpha_g
            C_k[members]     = alpha_g * R_c_g
            # Sample representative member for policy gradient / PPO
            group_sel[gid] = int(self.rng.choice(len(members), p=alpha_g))

        return C_k, alpha_k, group_sel

    def compute_grads(self, s_t: np.ndarray,
                      phi: np.ndarray,
                      advantage: float,
                      beta_entropy: float = 0.0,
                      K_active: int = None,
                      group_sel: dict = None) -> tuple:
        """
        Returns (L_pg, L_ent, grads) without applying the update.

        Parameters
        ----------
        group_sel : dict {gid: int} or None
            Per-group sampled local index from forward() (PPO: stored action).
            If None, resamples from the current policy (backward-compat fallback).
        """
        s_t = _layer_norm(s_t)
        pres, hs, logits = _mlp_forward(s_t, self.Ws, self.bs)

        K_used      = K_active if K_active is not None else self.K
        groups      = self._build_groups(phi[:K_used])
        dL_dlogits  = np.zeros(self.K)
        L_pg_total  = 0.0
        L_ent_total = 0.0

        for gid, members in groups.items():
            if len(members) < 1:
                continue

            logits_g = logits[members]
            e        = np.exp(logits_g - logits_g.max())
            alpha_g  = e / e.sum()   # current-policy fractions

            # Use stored action or resample from current policy
            if group_sel is not None and gid in group_sel:
                idx_sel = int(group_sel[gid])
            else:
                idx_sel = int(self.rng.choice(len(members), p=alpha_g))

            L_pg_total += float(-advantage * np.log(alpha_g[idx_sel] + 1e-10))

            one_hot_g     = np.zeros(len(members))
            one_hot_g[idx_sel] = 1.0
            pg_per_member = -advantage * (one_hot_g - alpha_g)

            log_alpha_g    = np.log(alpha_g + 1e-10)
            H_g            = -float(np.sum(alpha_g * log_alpha_g))
            L_ent_total   += -beta_entropy * H_g
            ent_per_member = beta_entropy * alpha_g * (log_alpha_g + H_g)

            for i, k in enumerate(members):
                dL_dlogits[k] += pg_per_member[i] + ent_per_member[i]

        grads = _mlp_backward(s_t, pres, hs, self.Ws, dL_dlogits)
        return float(L_pg_total), float(L_ent_total), grads

    def compute_log_prob(self, s_t: np.ndarray,
                         phi: np.ndarray,
                         K_active: int,
                         group_sel: dict) -> float:
        """Log π(group_sel|s) under current policy. Used for PPO ratio."""
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)
        K_used  = K_active if K_active is not None else self.K
        groups  = self._build_groups(phi[:K_used])
        log_prob = 0.0
        for gid, members in groups.items():
            logits_g = logits[members]
            e        = np.exp(logits_g - logits_g.max())
            alpha_g  = e / e.sum()
            idx_sel  = int(group_sel.get(gid, 0))
            log_prob += float(np.log(alpha_g[idx_sel] + 1e-10))
        return log_prob

    def compute_log_prob_batch(self, trans_list: list, K: int) -> np.ndarray:
        """
        Vectorised log π(group_sel|s) for a mini-batch.

        Each transition dict must contain 's_ck', 'phi', and 'ck_group_sel'.

        Returns
        -------
        lp_b : (B,) float  per-transition log-probability
        """
        B = len(trans_list)
        X = _layer_norm_batch(
            np.stack([t['s_ck'] for t in trans_list])
        )                                                         # (B, d_s)
        _, _, logits = _mlp_forward_batch(X, self.Ws, self.bs)   # (B, K)

        lp_b = np.zeros(B)
        for b, trans in enumerate(trans_list):
            groups = self._build_groups(trans['phi'][:K])
            for gid, members in groups.items():
                logits_g = logits[b, members]
                e        = np.exp(logits_g - logits_g.max())
                alpha_g  = e / e.sum()
                idx_sel  = int(trans['ck_group_sel'].get(gid, 0))
                lp_b[b] += float(np.log(alpha_g[idx_sel] + 1e-10))
        return lp_b

    def compute_grads_batch(self, trans_list: list, K: int,
                            eff_adv_b: np.ndarray,
                            beta_entropy: float = 0.0) -> tuple:
        """
        Vectorised PPO gradient for CkMLP over a mini-batch.

        Batches the MLP forward/backward; the per-sample group-softmax
        and gradient assembly is O(B × G × K) and cheap compared to matmul.

        Returns (L_pg, L_ent, grads)  — all B-averaged.
        """
        B     = len(trans_list)
        X     = _layer_norm_batch(
            np.stack([t['s_ck'] for t in trans_list])
        )                                                              # (B, d_s)
        pre_acts, acts, logits = _mlp_forward_batch(X, self.Ws, self.bs)  # (B, K)

        dL_dlogits_b = np.zeros((B, self.K))
        L_pg  = 0.0
        L_ent = 0.0

        for b, trans in enumerate(trans_list):
            groups = self._build_groups(trans['phi'][:K])
            for gid, members in groups.items():
                logits_g = logits[b, members]
                e        = np.exp(logits_g - logits_g.max())
                alpha_g  = e / e.sum()
                idx_sel  = int(trans['ck_group_sel'].get(gid, 0))

                L_pg += float(-eff_adv_b[b] * np.log(alpha_g[idx_sel] + 1e-10))

                one_hot_g = np.zeros(len(members))
                one_hot_g[idx_sel] = 1.0
                pg_m  = -eff_adv_b[b] * (one_hot_g - alpha_g)

                log_a = np.log(alpha_g + 1e-10)
                H_g   = -float(np.sum(alpha_g * log_a))
                L_ent += -beta_entropy * H_g
                ent_m  = beta_entropy * alpha_g * (log_a + H_g)

                for i, k in enumerate(members):
                    dL_dlogits_b[b, k] += pg_m[i] + ent_m[i]

        grads = _mlp_backward_batch(X, pre_acts, acts, self.Ws, dL_dlogits_b)
        return L_pg / B, L_ent / B, grads

    def apply_grads(self, grads: dict) -> None:
        """Apply pre-computed gradients."""
        self.opt.step(self._params, grads)

    def update(self, s_t: np.ndarray,
               phi: np.ndarray,
               advantage: float,
               beta_entropy: float = 0.0,
               K_active: int = None,
               group_sel: dict = None) -> tuple:
        """Compute gradients and apply in one call. Returns (L_pg, L_ent)."""
        L_pg, L_ent, grads = self.compute_grads(
            s_t, phi, advantage, beta_entropy, K_active, group_sel)
        self.apply_grads(grads)
        return L_pg, L_ent

    # ── Parameter I/O ────────────────────────────────────────────────────

    def get_params(self) -> dict:
        return {k: v.copy() for k, v in self._params.items()}

    def set_params(self, snapshot: dict) -> None:
        for k in self._params:
            if k in snapshot:
                self._params[k][:] = snapshot[k]

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        cfg_dict = {
            'd_s':    self.d_s,
            'K':      self.K,
            'hidden': self.hidden_sizes,
            'lr':     self.opt.lr,
        }
        with open(os.path.join(path, 'ck_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'ck_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        with open(os.path.join(path, 'ck_config.json')) as f:
            c = json.load(f)
        obj = cls(d_s=c['d_s'], K=c['K'], hidden=c['hidden'], lr=c['lr'], seed=seed)
        obj.set_params(dict(np.load(os.path.join(path, 'ck_params.npz'))))
        return obj
