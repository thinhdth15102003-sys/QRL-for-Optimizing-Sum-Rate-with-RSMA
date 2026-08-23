"""
sub_actors.py
-------------
Classical MLP sub-policies for the three remaining action components.

Pipeline position
-----------------
  After the quantum actor produces the IRS assignment φ:

  1. PhaseMLP  — IRS phase shifts φ_{m,n} ∈ {0,..,n_levels-1}^{M×N}
                 State: c^SRU_{m,n,k} = conj(g_SR[m]) · g_RU[m,n,k] for assigned
                        pairs, zero otherwise. Shared-weight head applied PER
                        ELEMENT: one (2K + n_latent + K) row per (m,n), output
                        n_levels. Each element picks its own level.

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

from . import dirichlet as _dir


def _scatter(mask: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Place `vals` (len = mask.sum()) into a zero array of len(mask)."""
    out = np.zeros(mask.shape[0])
    out[mask] = vals
    return out




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


# ── Optional CuPy backend for the BATCH matmuls only (opt-in) ────────────────
#
# Set QRL_GPU_MLP=1 to run PhaseMLP's batched layer-norm / forward / backward on
# the GPU. Motivation is CPU RELIEF, not speed: PhaseMLP's compute_grads_batch is
# ~129 ms of the ~156 ms of sub-actor gradient work per minibatch, which is ~71%
# of a DNN run's PPO-update CPU (only ~8% of a VQC run's, where Python overhead
# around the quantum circuit dominates). Moving it off the CPU frees roughly a
# core per DNN run, so more runs fit on the 8-core box.
#
# ⚠ DELIBERATELY NARROW, to protect result quality:
#   · only the two BATCH methods are affected; forward()/_forward_irs and ALL
#     RNG stay on NumPy ⇒ the rollout is bit-identical to the CPU path, same
#     seed ⇒ same actions ⇒ same trajectory.
#   · float64 throughout — no precision trade (the 4090's FP64 is 1/64 of FP32,
#     so float32 would be far faster, but that is a quality trade and is not done).
#   · the small (T, N, L) softmax / one-hot / log work stays on CPU: 6k elements,
#     not worth a kernel launch.
# Gradients still differ from NumPy in the last ULPs (cuBLAS vs OpenBLAS
# summation order) — verified to ~1e-12 relative, i.e. rounding, not a bug.
_GPU_MLP = os.environ.get('QRL_GPU_MLP', '0') not in ('0', '', 'false', 'False')
_cp = None
if _GPU_MLP:
    try:
        import cupy as _cp
        _cp.cuda.Device(0).use()
    except Exception:                                   # no GPU → silent fallback
        _cp, _GPU_MLP = None, False


def gpu_mlp_enabled() -> bool:
    return bool(_GPU_MLP and _cp is not None)


def _gpu_forward(X: np.ndarray, Ws: list, bs: list):
    """layer-norm + MLP forward on GPU. Returns (gpu_state, logits as NumPy)."""
    Xg = _cp.asarray(X)
    mu = Xg.mean(axis=1, keepdims=True)
    sd = Xg.std(axis=1, keepdims=True) + 1e-6
    Xn = (Xg - mu) / sd
    Wg = [_cp.asarray(W) for W in Ws]
    bg = [_cp.asarray(b) for b in bs]
    pre_acts, acts = [], []
    x = Xn
    for W, b in zip(Wg[:-1], bg[:-1]):
        pre = x @ W + b
        h = _cp.maximum(0.0, pre)
        pre_acts.append(pre); acts.append(h)
        x = h
    logits = x @ Wg[-1] + bg[-1]
    return (Xn, pre_acts, acts, Wg), _cp.asnumpy(logits)


def _gpu_backward(state, dL_dlogits: np.ndarray) -> dict:
    """Backward on GPU for a state from _gpu_forward. Returns NumPy grads.
    Mirrors _mlp_backward_batch exactly, including the /B normalisation."""
    Xn, pre_acts, acts, Wg = state
    B = Xn.shape[0]
    n = len(Wg)
    grads = {}
    d_out = _cp.asarray(dL_dlogits) / B
    x_in = acts[-1] if acts else Xn
    grads[f'W{n-1}'] = _cp.asnumpy(x_in.T @ d_out)
    grads[f'b{n-1}'] = _cp.asnumpy(d_out.sum(axis=0))
    delta = d_out @ Wg[-1].T
    for i in range(n - 2, -1, -1):
        dpre = delta * (pre_acts[i] > 0)
        x_in = acts[i - 1] if i > 0 else Xn
        grads[f'W{i}'] = _cp.asnumpy(x_in.T @ dpre)
        grads[f'b{i}'] = _cp.asnumpy(dpre.sum(axis=0))
        if i > 0:
            delta = dpre @ Wg[i].T
    return grads


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
    Per-ELEMENT discrete phase-shift policy with shared weights.

    The same MLP processes each ACTIVE IRS independently, and within an IRS is
    applied to each of the N elements as a batch — so every element picks its
    own level from its own channel row (required now that g_RU is per-element).
    Inactive IRS (no users assigned after IRS-selection) are skipped — their
    phase_idx defaults to 0 and receives no gradient.

    Input per (IRS m, element n)
    ----------------------------
    c^SRU_{m,n,k} = conj(g_SR[m]) · g_RU[m,n,k]  for the users routed to m.
    Row = [Re(c), Im(c), z_t, routed-mask], shape (2·K + n_latent + K,).

    Parameters
    ----------
    d_s      : int             input dim per element = 2·K + n_latent + K
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
        # PER-ELEMENT head: the net is applied once per reflecting element with
        # SHARED weights, so it emits n_levels logits for that element (not N*L
        # logits for the whole panel). This is what makes phi_n a real decision:
        # each element sees its own cascade channel c_{m,n,k} and aligns to it.
        d_out         = n_levels      # logits per reflecting element

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
        """Forward for one IRS panel, batched over its N reflecting elements.

        s_m : (N, d_s) — row n is element n's own state (its cascade channel
              c_{m,n,k}, the shared latent, and the panel's user mask).
        Returns (idx (N,), log_prob, probs (N, n_levels)).
        """
        s_m = _layer_norm_batch(np.atleast_2d(s_m))
        _, _, logits = _mlp_forward_batch(s_m, self.Ws, self.bs)  # (N, n_levels)
        probs = _softmax_rows(logits)                             # (N, n_levels)
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
            # PER-ELEMENT batch: s_phase_mat[m] is (N, d_s), one row per element,
            # so the N elements form the batch for the shared-weight head.
            s_m  = _layer_norm_batch(np.atleast_2d(s_phase_mat[m]))
            pres, hs, logits = _mlp_forward_batch(s_m, self.Ws, self.bs)
            probs = _softmax_rows(logits)                                 # (N, n_levels)

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

            dL_dlogits = dL_pg + dL_ent                # (N, n_levels) — batched
            g_m = _mlp_backward_batch(s_m, pres, hs, self.Ws, dL_dlogits)

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
            _, _, logits = _mlp_forward_batch(
                _layer_norm_batch(np.atleast_2d(s_phase_mat[m])), self.Ws, self.bs)
            probs = _softmax_rows(logits)                      # (N, n_levels)
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

        panels_s   = []   # raw per-element inputs, each (N, d_s)
        panels_idx = []   # phase_idx per panel, each (N,) int
        sample_ids = []   # which transition each panel belongs to

        for b, trans in enumerate(trans_list):
            for m in _active_irs_from_phi(trans['phi']):
                panels_s.append(np.atleast_2d(trans['s_phase'][m]))   # (N, d_s)
                panels_idx.append(trans['phase_idx'][m])
                sample_ids.append(b)

        if not panels_s:
            return lp_b   # no active IRS in any transition

        T        = len(panels_s)
        X        = np.stack(panels_s)    # (T, N, d_s)
        idx_all  = np.stack(panels_idx)  # (T, N) int

        # The head is PER ELEMENT with shared weights, so flatten the element axis
        # into the batch: each of the T*N element rows is one sample.
        Xr = X.reshape(T * self.N, -1)
        if gpu_mlp_enabled():
            _, logits = _gpu_forward(Xr, self.Ws, self.bs)         # (T*N, L)
        else:
            Xb = _layer_norm_batch(Xr)                             # (T*N, d_s)
            _, _, logits = _mlp_forward_batch(Xb, self.Ws, self.bs)
        probs_3d = _softmax_rows(logits).reshape(T, self.N, self.n_levels)

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
                            beta_entropy: float = 0.0,
                            counterfactual: bool = False,
                            aux_w: float = 0.0) -> tuple:
        """
        Vectorised REINFORCE+PPO gradient for PhaseMLP over a mini-batch.

        aux_w — OPTIONAL continuous teacher (--phase-aux-weight). Adds
        w·(-log π(oracle_phase|s)) at EVERY update, using trans['phase_oracle'].
        Same idea as the assignment teacher: the supervised warm-up is applied
        once and then eroded by PPO (measured r154: coherent-gain alignment
        15.6% → 7.0% over 1300 ep), whereas this is always present so it cannot
        decay. Gradient form is identical to the PG term with advantage = w
        evaluated at the ORACLE index instead of the sampled one.
        Worth it because the phase gap is large once routing is fixed: on r177
        swapping in the oracle phase is +0.138 J = 42% of the whole remaining
        gap to AO.

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
        panels_or  = []   # oracle phase index per panel (for --phase-aux-weight)
        sample_ids = []
        panel_w    = []   # Δ7: per-panel occupancy weight (1.0 unless counterfactual)
        panel_hs   = []   # [--phase-aux-hspread] per-panel log channel spread

        for b, trans in enumerate(trans_list):
            act = _active_irs_from_phi(trans['phi'])
            if counterfactual and len(act) > 0 and 'q_pi' in trans:
                # Δ7 COUNTERFACTUAL: scale each active IRS's phase gradient by its
                # π_q EXPECTED OCCUPANCY (Σ_k π_q[k, m+1]) so phase credit tracks the
                # routing distribution smoothly instead of the jumpy sampled-active
                # set (F2 mitigation). Normalised to mean=1 over active IRS → overall
                # gradient magnitude unchanged (reduces to vanilla when occupancy uniform).
                occ   = np.asarray(trans['q_pi'])[:, 1:].sum(axis=0)   # (M,) expected #users per IRS
                w_act = np.array([occ[m] for m in act], dtype=float)
                w_act = w_act / (w_act.mean() + 1e-8)
            else:
                w_act = np.ones(len(act), dtype=float)
            for m, w in zip(act, w_act):
                panels_s.append(np.atleast_2d(trans['s_phase'][m]))   # (N, d_s)
                panels_idx.append(trans['phase_idx'][m])
                _po = trans.get('phase_oracle')
                panels_or.append(None if _po is None else _po[m])     # (N,) oracle idx
                sample_ids.append(b)
                panel_w.append(float(w))
                panel_hs.append(float(trans.get('log_hspread', 0.0)))

        if not panels_s:
            return 0.0, 0.0, {}

        T          = len(panels_s)
        sample_ids = np.array(sample_ids, dtype=int)   # (T,)
        X          = np.stack(panels_s)                # (T, N, d_s)
        idx_all    = np.stack(panels_idx)              # (T, N) int

        # PER-ELEMENT head with shared weights → flatten the element axis into the
        # batch, so the super-batch has T*N rows (one per reflecting element).
        Xr = X.reshape(T * self.N, -1)
        _gpu_state = None
        if gpu_mlp_enabled():
            _gpu_state, logits = _gpu_forward(Xr, self.Ws, self.bs)        # (T*N, L)
        else:
            Xb = _layer_norm_batch(Xr)                                     # (T*N, d_s)
            pre_acts, acts, logits = _mlp_forward_batch(Xb, self.Ws, self.bs)
        probs_3d = _softmax_rows(logits).reshape(T, self.N, self.n_levels)  # (T, N, L)

        one_hot_3d = np.eye(self.n_levels)[idx_all]            # (T, N, L)
        eff_t      = eff_adv_b[sample_ids] * np.asarray(panel_w)  # (T,) Δ7 occupancy-weighted

        dL_pg_3d  = -eff_t[:, None, None] * (one_hot_3d - probs_3d)  # (T, N, L)
        log_p_3d  = np.log(probs_3d + 1e-10)                         # (T, N, L)
        H_3d      = -(probs_3d * log_p_3d).sum(axis=2, keepdims=True) # (T, N, 1)
        dL_ent_3d = beta_entropy * probs_3d * (log_p_3d + H_3d)      # (T, N, L)

        # continuous phase teacher: same form as the PG term with advantage = w,
        # evaluated at the ORACLE index (see aux_w in the docstring)
        dL_aux_3d = 0.0
        if aux_w > 0.0 and all(p is not None for p in panels_or):
            or_all     = np.stack(panels_or)                          # (T, N) int
            one_hot_or = np.eye(self.n_levels)[or_all]                # (T, N, L)
            # [--phase-aux-hspread] REDISTRIBUTE the teacher, do not strengthen it.
            # Measured on 120 Case-2 states: swapping in the oracle phase is worth
            # +0.045 on the states the policy already wins and +0.583 on the 20%
            # it loses — more, there, than AO's own +0.434 lead. A uniform teacher
            # spends most of its weight where it buys +0.045, which is why fixing
            # the target (r280, n_sweeps=1) moved nothing. The losing states are
            # the high-spread ones (corr(delta, hspread) = -0.79), so tilt toward
            # them. Standardised WITHIN the batch and mean-preserving: total
            # teaching is unchanged, only its distribution moves. gamma=0 is
            # bit-identical to the uniform teacher.
            w_aux = aux_w
            g_hs = float(getattr(self, 'aux_hspread_gamma', 0.0) or 0.0)
            if g_hs > 0.0:
                hs = np.asarray(panel_hs, dtype=float)                # (T,)
                sd = hs.std()
                if sd > 1e-9:
                    t = (hs - hs.mean()) / sd
                    w_aux = aux_w * (1.0 + g_hs * np.tanh(t))[:, None, None]
            dL_aux_3d  = -w_aux * (one_hot_or - probs_3d)             # (T, N, L)

        dL_dlogits = (dL_pg_3d + dL_ent_3d + dL_aux_3d).reshape(T * self.N, self.n_levels)

        # _mlp_backward_batch divides by its ROW count, which is now T*N (one row per
        # reflecting element), not T. Scale by (T*N)/B so the result stays normalised
        # per transition exactly as before the per-element refactor.
        _dL_scaled = dL_dlogits * ((T * self.N) / B)
        if _gpu_state is not None:
            grads = _gpu_backward(_gpu_state, _dL_scaled)
        else:
            grads = _mlp_backward_batch(Xb, pre_acts, acts, self.Ws, _dL_scaled)

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
        params = dict(np.load(os.path.join(path, 'phase_params.npz')))
        # Pre-per-element checkpoints emitted ONE joint distribution over all N
        # elements (d_out = N*n_levels); the head is now shared and applied per
        # element (d_out = n_levels). The weights are not convertible.
        w_idx  = [int(k[1:]) for k in params if k.startswith('W') and k[1:].isdigit()]
        w_last = params[f'W{max(w_idx)}'] if w_idx else None
        if w_last is not None and w_last.shape[-1] == c['N'] * c['n_levels'] \
                and obj.Ws[-1].shape[-1] == c['n_levels']:
            raise ValueError(
                f"PhaseMLP at {path} is a PRE-PER-ELEMENT checkpoint: its output "
                f"layer is {w_last.shape[-1]} = N*n_levels ({c['N']}*{c['n_levels']}), "
                f"but the per-element head needs {c['n_levels']}. These weights "
                f"encode a policy for the OLD scalar-g_RU physics (one phase per "
                f"IRS) and cannot be converted — the run must be retrained. See "
                f"analysis/phase_oracle.py for what changed.")
        obj.set_params(params)
        return obj


# ── Power allocation MLP ───────────────────────────────────────────────────────

class PowerMLP:
    """
    Factored power-allocation policy (3-way independent softmax heads).

    Decomposes the budget decision into three semantically independent factors so
    each has its own gradient signal — fixes the gradient-bleed of a single
    softmax over M+1+K mixed slots that caused wp-concentration (Case2 r8).

      π_split   (2 logits)        : [common_total_frac, private_total_frac]
      π_common  (M+1 logits)      : how to split the common share across
                                    direct + IRS groups (inactive IRS masked).
      π_private (K logits)        : how to split the private share across K users.

    Budget invariant
    ----------------
      w_c_total = π_split[0] * P_S    →  w_c_full[g] = π_common[g] * w_c_total
      w_p_total = π_split[1] * P_S    →  w_p[k]      = π_private[k] * w_p_total
      Σ w_c_full + Σ w_p = P_S (by construction; π_common renormalised over active).

    Action / PPO
    ------------
    Treated as 3 independent action factors:
      log π(a) = log π_split(a_s) + log π_common(a_c) + log π_private(a_p)
    PPO uses one joint ratio (one A_eff per transition). Each axis backprops
    its own cross-entropy gradient: dL/dlogits_axis = -A_eff·(one_hot - probs).

    Entropy bonus
    -------------
    `beta_entropy` applies to all 3 axes; `beta_entropy_private_extra` adds on
    top for π_private ONLY. This is the lever for fighting wp-concentration
    without touching the common-vs-private split or per-group common share.

    Parameters
    ----------
    d_s    : state dimension
    K      : number of users
    M      : number of IRS panels (sets common output size M+1)
    P_S    : total power budget (W)
    hidden : int or list[int]
    """

    def __init__(self, d_s: int, K: int, M: int, P_S: float,
                 hidden=64, lr: float = 3e-4, seed: int = None,
                 power_fairness: float = 0.0,
                 power_priv_frac: float = 0.8,
                 power_fairness_priv: float = None):
        self.d_s = d_s
        self.K   = K
        self.M   = M
        self.P_S = P_S
        # ③ power-fairness α∈[0,1]: blend the EXECUTED power toward a QoS-safe
        # allocation (split + common-dist + private-dist) in forward(). 0 = learned head.
        #
        # α is applied PER AXIS: `power_fairness` drives split + common, while
        # `power_fairness_priv` drives the private distribution (defaults to the
        # same α → unchanged behaviour). They are separated because the two axes
        # want OPPOSITE things once C_k is allocated well: the Pareto measurement
        # (07-21, per-element physics) shows the reward-optimal point is a LOW
        # private fraction f with a CONCENTRATED private distribution — common
        # stream carries QoS, private power chases rate on the strong links.
        # Forcing private→uniform (α_priv high) caps sum-rate; see
        # per-element-physics-refactor / Pareto notes.
        #
        # power_priv_frac = target private split f. 0.8 was measured QoS-optimal
        # under EQUAL C_k; with a demand-filling C_k the optimum drops a long way
        # (≈0.68 for K5/M1, ≈0.28 for K10/M2 at P50/λ1.5).
        self.power_fairness  = float(max(0.0, min(1.0, power_fairness)))
        self.power_fairness_priv = (self.power_fairness
                                    if power_fairness_priv is None
                                    else float(max(0.0, min(1.0, power_fairness_priv))))
        self.power_priv_frac = float(max(0.0, min(1.0, power_priv_frac)))
        self._n_split   = 2
        self._n_common  = M + 1
        self._n_private = K
        d_out = self._n_split + self._n_common + self._n_private
        self._d_out = d_out

        self._sl_split   = slice(0, self._n_split)
        self._sl_common  = slice(self._n_split,
                                 self._n_split + self._n_common)
        self._sl_private = slice(self._n_split + self._n_common,
                                 self._n_split + self._n_common + self._n_private)

        rng = np.random.default_rng(seed)
        self.rng = rng

        hidden_sizes      = _parse_hidden(hidden)
        self.hidden_sizes = hidden_sizes
        self.Ws, self.bs  = _build_layers(d_s, d_out, hidden_sizes, rng)
        # [Case2 result_9 ep~250] Init SPLIT axis to constant π_split ≈ [0.57, 0.43]
        # at t=0 (gentle common-preference) by zeroing the output-layer weights for
        # split slots and biasing slot 0 by +0.3. Without state-dependent weights,
        # logits_split = [0.3, 0.0] always → π_split = [0.574, 0.426]. PG learns the
        # state-dependent weights from zero. Common/private slots keep He-init.
        #
        # ITERATION HISTORY (Case2 R_LoS=0.2):
        #   bias=+1.0 (result_9): π_split=[0.73,0.27] init. wc LOCKED ~70% throughout
        #     run (PG cannot drift it down). Per-user private starved (~3% P_S) →
        #     PhaseMLP IDLE worse than result_8 (ent ph 61% vs 56% max). Critic
        #     stability + PowerMLP no-idle ✅, but priority-2 PhaseMLP regressed.
        #   bias=+0.0 (smoke result_10): no nudge → wc collapsed to 7-15% within 60ep
        #     (single-softmax's implicit common-stream protection gone). ⚠ENTROPY-DOMINATED.
        #   bias=+0.3 (current): split between the two; π_split start at 0.57 leaves
        #     PG room to drift toward ~30-40% common (the result_8 healthy range).
        # Rationale: factoring removed the implicit common-stream protection of the
        # old shared-softmax; need SOME nudge but not a lock. PG can still drift either way.
        self.Ws[-1][:, 0:self._n_split] = 0.0
        self.bs[-1][0] = 0.3
        self.bs[-1][1] = 0.0
        self._params      = _make_params(self.Ws, self.bs)

        # ── PER-USER PRIVATE HEAD (permutation-equivariant, 2026-07-25) ──────────
        # A flat MLP [h_1..h_K] → [w_1..w_K] cannot learn w_k ∝ |h_k|: it must
        # discover K independent input→output alignments, and even with the
        # rescale-fix input and the clean-β teacher it stayed at 15% argmax
        # agreement over 10k episodes (r210-217). This shared per-user head — the
        # PhaseMLP per-element pattern applied to users — processes EACH user's own
        # scale-free channel features and emits ONE private logit, so "large |h_k|
        # → large logit" is a single shared map it can actually learn. Added as a
        # RESIDUAL to the flat MLP's private slots (flat path unchanged; if the
        # per-user head learns nothing we degrade to the old behaviour, no worse).
        self._pu_feat_dim = 3        # [h_k.real/hs, h_k.imag/hs, |h_k/hs|^2]
        pu_hidden = _parse_hidden([32, 16])
        self._pu_hidden = pu_hidden
        self.Wpu, self.bpu = _build_layers(self._pu_feat_dim, 1, pu_hidden, rng)
        for i, (W, b) in enumerate(zip(self.Wpu, self.bpu)):
            self._params[f'pu_W{i}'] = W
            self._params[f'pu_b{i}'] = b

        self.opt          = _Adam(lr=lr)
        self.architecture = _arch_str(d_s, hidden_sizes, d_out)

    # ── Per-user private head helpers ─────────────────────────────────────
    def _peruser_feat(self, s_t: np.ndarray) -> np.ndarray:
        """(K, 3) scale-free per-user channel features from s_power[:2K].
        s_power[:K]=h.real/hs, s_power[K:2K]=h.imag/hs already (see
        train._build_power_state), so magnitude ranking survives here."""
        hr = np.asarray(s_t[:self.K], dtype=float)
        hi = np.asarray(s_t[self.K:2 * self.K], dtype=float)
        return np.stack([hr, hi, hr * hr + hi * hi], axis=1)

    def _peruser_bias(self, s_t: np.ndarray) -> np.ndarray:
        """(K,) private-logit residual from the shared per-user head (no LN — the
        features are already scale-free; a per-user LN would erase magnitude)."""
        _, _, pu = _mlp_forward_batch(self._peruser_feat(s_t), self.Wpu, self.bpu)
        return pu[:, 0]

    def _full_logits(self, s_t: np.ndarray) -> np.ndarray:
        """Flat MLP logits with the per-user private residual added in."""
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)
        logits = logits.copy()
        logits[self._sl_private] = logits[self._sl_private] + self._peruser_bias(s_t)
        return logits

    def _peruser_feat_batch(self, S: np.ndarray) -> np.ndarray:
        """(B, K, 3) per-user features for a batch of raw states S (B, d_s)."""
        hr = S[:, :self.K]; hi = S[:, self.K:2 * self.K]
        return np.stack([hr, hi, hr * hr + hi * hi], axis=2)

    def _peruser_bias_batch(self, S: np.ndarray) -> np.ndarray:
        """(B, K) per-user private residual for a batch of states."""
        B = S.shape[0]
        feat = self._peruser_feat_batch(S).reshape(B * self.K, self._pu_feat_dim)
        _, _, pu = _mlp_forward_batch(feat, self.Wpu, self.bpu)   # (B*K, 1)
        return pu.reshape(B, self.K)

    def _peruser_grads_batch(self, S: np.ndarray, dL_private: np.ndarray) -> dict:
        """Backprop the private-logit residual through the shared per-user head.

        dL_private : (B, K) gradient of the loss w.r.t. each per-user residual.
        Scaled by K so the internal /(B·K) normalisation of _mlp_backward_batch
        yields a /B-normalised gradient (sum over users, mean over transitions),
        matching the flat MLP path. Returns {'pu_W{i}','pu_b{i}'} grads.
        """
        B = S.shape[0]
        feat = self._peruser_feat_batch(S).reshape(B * self.K, self._pu_feat_dim)
        pre, acts, _ = _mlp_forward_batch(feat, self.Wpu, self.bpu)
        dL = (dL_private.reshape(B * self.K, 1)) * float(self.K)
        g = _mlp_backward_batch(feat, pre, acts, self.Wpu, dL)
        return {f'pu_{k}': v for k, v in g.items()}

    # ── Mask & softmax helpers ────────────────────────────────────────────

    def _common_mask(self, active_irs_ids) -> np.ndarray:
        """(M+1,) bool: True for direct + active IRS groups."""
        mask    = np.zeros(self._n_common, dtype=bool)
        mask[0] = True
        for gid in active_irs_ids:
            if 1 <= gid <= self.M:
                mask[gid] = True
        return mask

    @staticmethod
    def _masked_softmax_1d(logits: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        x = logits.copy()
        x[~mask] = -np.inf
        return _softmax_1d(x)

    def _split_logits(self, logits: np.ndarray) -> tuple:
        return (logits[self._sl_split],
                logits[self._sl_common],
                logits[self._sl_private])

    def _probs_from_logits(self, logits: np.ndarray,
                           common_mask: np.ndarray) -> tuple:
        l_s, l_c, l_p = self._split_logits(logits)
        return (_softmax_1d(l_s),
                self._masked_softmax_1d(l_c, common_mask),
                _softmax_1d(l_p))

    @staticmethod
    def _extract_wc(w_full: np.ndarray,
                    active_irs_ids) -> np.ndarray:
        """Extract (G+1,) [direct, IRS_active...] from (M+1,) array."""
        slots = [0] + list(active_irs_ids)
        return w_full[slots]

    # ── Public forward ────────────────────────────────────────────────────

    # ── Dirichlet concentrations (the policy parameters) ──────────────────
    #
    # ⭐ 2026-07-21 PHANTOM-ACTION FIX. The action of this head is an ALLOCATION
    # on a simplex, so it is now drawn from a Dirichlet whose concentration the
    # network produces — and the DRAWN VECTOR IS WHAT GETS EXECUTED.
    #
    # Previously the head executed softmax(logits)·budget while PPO scored a
    # separately-sampled categorical INDEX that never entered the executed
    # power. The reward was therefore independent of the scored action, making
    # E_a[∇log π(a|s)] identically zero: the head could only be moved by the
    # entropy bonus. See RL/dirichlet.py for the measurement and the maths.

    def _conc(self, logits: np.ndarray, mask: np.ndarray) -> tuple:
        """
        (α_split, α_common_active, α_private) from raw logits.

        ③ power-fairness is applied HERE, in DISTRIBUTION space — blending the
        concentration, not the executed action. Writing
            α_eff = (1-a)·α + a·α₀·t          (α₀ = Σα, t = target mean)
        gives mean[x] = (1-a)·mean_learned + a·t, i.e. exactly the old blend
        semantics, while keeping sampled == executed. Blending the executed
        vector instead would re-introduce the phantom-action bug.
        """
        l_s, l_c, l_p = self._split_logits(logits)
        a_s = _dir.logits_to_conc(l_s)
        a_c = _dir.logits_to_conc(l_c[mask])
        a_p = _dir.logits_to_conc(l_p)

        a_fair = float(getattr(self, 'power_fairness', 0.0))
        a_priv = float(getattr(self, 'power_fairness_priv', a_fair))
        if a_fair > 0.0:
            f   = float(getattr(self, 'power_priv_frac', 0.8))
            n_c = a_c.shape[0]
            a_s = (1.0 - a_fair) * a_s + a_fair * a_s.sum() * np.array([1.0 - f, f])
            a_c = (1.0 - a_fair) * a_c + a_fair * a_c.sum() / n_c
        if a_priv > 0.0:
            a_p = (1.0 - a_priv) * a_p + a_priv * a_p.sum() / self._n_private
        return a_s, a_c, a_p

    def _dconc_blend(self, g_s, g_c, g_p):
        """
        Pull a gradient w.r.t. the BLENDED concentrations back to the raw ones.

        With α_eff = (1-a)·α + a·(Σα)·t the Jacobian is (1-a)·I + a·t·1ᵀ, so
            dL/dα_j = (1-a)·dL/dα_eff_j + a·Σ_i t_i·dL/dα_eff_i .
        """
        a_fair = float(getattr(self, 'power_fairness', 0.0))
        a_priv = float(getattr(self, 'power_fairness_priv', a_fair))
        if a_fair > 0.0:
            f   = float(getattr(self, 'power_priv_frac', 0.8))
            t_s = np.array([1.0 - f, f])
            g_s = (1.0 - a_fair) * g_s + a_fair * float(g_s @ t_s)
            n_c = g_c.shape[0]
            g_c = (1.0 - a_fair) * g_c + a_fair * float(g_c.sum()) / n_c
        if a_priv > 0.0:
            g_p = (1.0 - a_priv) * g_p + a_priv * float(g_p.sum()) / self._n_private
        return g_s, g_c, g_p

    # ── Public forward ────────────────────────────────────────────────────

    def forward(self, s_t: np.ndarray,
                active_irs_ids: list) -> tuple:
        """
        Returns
        -------
        w_c_vec : (G+1,) float
        w_p     : (K,)   float
        probs   : dict {'split','common','private'} — mean allocations (diagnostic)
        action  : dict {'split': (2,), 'common': (M+1,), 'private': (K,)} float
                  — the SAMPLED allocation vectors, which are exactly what is
                  executed below. 'common' is zero-padded at inactive slots.
        """
        logits = self._full_logits(s_t)
        mask = self._common_mask(active_irs_ids)
        a_s, a_c, a_p = self._conc(logits, mask)

        x_s = _dir.sample(self.rng, a_s)                 # [common_frac, priv_frac]
        x_c_act = _dir.sample(self.rng, a_c)             # over ACTIVE slots only
        x_p = _dir.sample(self.rng, a_p)                 # over K users

        x_c = np.zeros(self._n_common)
        x_c[mask] = x_c_act

        w_c_total = float(x_s[0]) * self.P_S
        w_p_total = float(x_s[1]) * self.P_S
        w_c_full  = x_c * w_c_total                                # (M+1,)
        w_p       = x_p * w_p_total                                # (K,)
        w_c_vec   = self._extract_wc(w_c_full, active_irs_ids)     # (G+1,)

        action = {'split': x_s, 'common': x_c, 'private': x_p}
        probs  = {'split':   a_s / a_s.sum(),
                  'common':  _scatter(mask, a_c / a_c.sum()),
                  'private': a_p / a_p.sum()}
        return w_c_vec, w_p, probs, action

    # ── Log-prob (single + batch) ─────────────────────────────────────────

    def compute_log_prob(self, s_t: np.ndarray,
                         active_irs_ids: list,
                         action) -> float:
        """Log π(a|s) = sum of 3 axis log-probs. Used for PPO ratio."""
        logits = self._full_logits(s_t)
        mask = self._common_mask(active_irs_ids)
        return self._logp_from_logits(logits, mask, action)

    def _logp_from_logits(self, logits, mask, action) -> float:
        """Σ of the 3 Dirichlet log-densities at the stored allocation vectors."""
        a_s, a_c, a_p = self._conc(logits, mask)
        return float(_dir.log_prob(action['split'], a_s)
                   + _dir.log_prob(np.asarray(action['common'])[mask], a_c)
                   + _dir.log_prob(action['private'], a_p))

    def compute_log_prob_batch(self, trans_list: list) -> np.ndarray:
        """
        Each transition dict must contain 's_power', 'active_irs_ids',
        and the 3 action fields 'power_a_split' / 'power_a_common' /
        'power_a_private' — now the sampled ALLOCATION VECTORS, not indices.
        """
        B = len(trans_list)
        S = np.stack([t['s_power'] for t in trans_list])
        X = _layer_norm_batch(S)
        _, _, logits = _mlp_forward_batch(X, self.Ws, self.bs)   # (B, d_out)
        logits = logits.copy()
        logits[:, self._sl_private] += self._peruser_bias_batch(S)   # (B, K) residual

        lp_b = np.zeros(B)
        for b, trans in enumerate(trans_list):
            mask = self._common_mask(trans['active_irs_ids'])
            lp_b[b] = self._logp_from_logits(
                logits[b], mask,
                {'split':   trans['power_a_split'],
                 'common':  trans['power_a_common'],
                 'private': trans['power_a_private']})
        return lp_b

    # ── Gradient (single + batch) ─────────────────────────────────────────

    def compute_grads(self, s_t: np.ndarray,
                      active_irs_ids: list,
                      advantage: float,
                      beta_entropy: float = 0.0,
                      action=None,
                      beta_entropy_private_extra: float = 0.0) -> tuple:
        """Single-transition gradient. Returns (L_pg, L_ent, grads)."""
        s_n              = _layer_norm(s_t)
        pres, hs, logits = _mlp_forward(s_n, self.Ws, self.bs)
        logits = logits.copy()
        logits[self._sl_private] = logits[self._sl_private] + self._peruser_bias(s_t)
        mask = self._common_mask(active_irs_ids)

        if action is None:                       # resample from the current policy
            a_s0, a_c0, a_p0 = self._conc(logits, mask)
            action = {'split':   _dir.sample(self.rng, a_s0),
                      'common':  _scatter(mask, _dir.sample(self.rng, a_c0)),
                      'private': _dir.sample(self.rng, a_p0)}

        dL, L_pg, L_ent = self._dlogits(
            logits, mask, action, float(advantage),
            beta_entropy, beta_entropy_private_extra)
        grads = _mlp_backward(s_n, pres, hs, self.Ws, dL)
        # per-user head residual (dL/dbias = dL/dprivate_logit); flat path unchanged
        grads.update(self._peruser_grads_batch(
            s_t[None, :], dL[self._sl_private][None, :]))
        return float(L_pg), float(L_ent), grads

    def _dlogits(self, logits, mask, action, adv,
                 beta_entropy, beta_entropy_private_extra,
                 aux_w: float = 0.0, aux_tgt: dict = None):
        """
        dL/dlogits for one transition under the Dirichlet policy.

        Chain: dL/dlogit = dL/dα_eff · (dα_eff/dα) · (dα/dlogit)
                         = _dconc_blend(dL/dα_eff) * sigmoid(logit)

        aux_w / aux_tgt — OPTIONAL continuous teacher (--power-aux-weight), the
        same device already used by CkMLP and PhaseMLP. Adds
            L_aux = -w · Σ_axis log p(x_oracle_axis | α_axis)
        where the target is the (β*, f*) allocation that maximises the design-view
        objective for this state. Power was the only head WITHOUT a teacher, and
        it accounted for ~90% of the remaining gap to the oracle while routing
        (taught) sat at 0% — see docs/Dk-Noise-Calibration.md.

        The teacher is evaluated on the BLENDED concentrations and pulled back
        through _dconc_blend, exactly like the PG term, so it teaches what is
        actually executed. A large power-fairness therefore damps the teacher as
        well — which is correct: with α high the head is pinned to uniform and
        no target can move it.
        """
        a_s, a_c, a_p = self._conc(logits, mask)
        x_s = np.asarray(action['split'],   dtype=float)
        x_c = np.asarray(action['common'],  dtype=float)[mask]
        x_p = np.asarray(action['private'], dtype=float)

        lp_s = _dir.log_prob(x_s, a_s)
        lp_c = _dir.log_prob(x_c, a_c)
        lp_p = _dir.log_prob(x_p, a_p)
        L_pg = -adv * (lp_s + lp_c + lp_p)

        # policy-gradient part: ∂(-adv·log π)/∂α = -adv · ∂log p/∂α
        g_s = -adv * _dir.dlogp_dconc(x_s, a_s)
        g_c = -adv * _dir.dlogp_dconc(x_c, a_c)
        g_p = -adv * _dir.dlogp_dconc(x_p, a_p)

        # entropy bonus: L_ent = -β·H, so ∂L_ent/∂α = -β·∂H/∂α
        beta_p = beta_entropy + beta_entropy_private_extra
        H_s, H_c, H_p = (_dir.entropy(a_s), _dir.entropy(a_c), _dir.entropy(a_p))
        L_ent = -(beta_entropy * (H_s + H_c) + beta_p * H_p)
        g_s += -beta_entropy * _dir.dentropy_dconc(a_s)
        g_c += -beta_entropy * _dir.dentropy_dconc(a_c)
        g_p += -beta_p       * _dir.dentropy_dconc(a_p)

        # continuous oracle teacher — same form as the PG term with "advantage =
        # aux_w", evaluated at the oracle allocation instead of the sampled one
        if aux_w > 0.0 and aux_tgt:
            for key, conc, gref in (('split', a_s, 's'),
                                    ('common', a_c, 'c'),
                                    ('private', a_p, 'p')):
                t = aux_tgt.get(key)
                if t is None:
                    continue
                t = np.asarray(t, dtype=float)
                if key == 'common':
                    t = t[mask]
                if t.shape != conc.shape:
                    continue
                t = np.maximum(t, _dir.X_MIN)
                t = t / t.sum()
                L_ent += -aux_w * _dir.log_prob(t, conc)
                g_aux = -aux_w * _dir.dlogp_dconc(t, conc)
                if   gref == 's': g_s = g_s + g_aux
                elif gref == 'c': g_c = g_c + g_aux
                else:             g_p = g_p + g_aux

        g_s, g_c, g_p = self._dconc_blend(g_s, g_c, g_p)

        # α = softplus(logit) + α_min  →  dα/dlogit = sigmoid(logit)
        l_s, l_c, l_p = self._split_logits(logits)
        dL = np.zeros(self._d_out)
        dL[self._sl_split]   = g_s * _dir.dconc_dlogits(l_s)
        # inactive common slots get no gradient (they are not in the Dirichlet)
        dL[self._sl_common]  = _scatter(mask, g_c * _dir.dconc_dlogits(l_c[mask]))
        dL[self._sl_private] = g_p * _dir.dconc_dlogits(l_p)
        return dL, L_pg, L_ent

    def compute_grads_batch(self, trans_list: list,
                            eff_adv_b: np.ndarray,
                            beta_entropy: float = 0.0,
                            beta_entropy_private_extra: float = 0.0,
                            aux_w: float = 0.0) -> tuple:
        """
        Vectorised PPO+entropy gradient over a mini-batch.
        Returns (L_pg, L_ent, grads, axis_stats)  — all B-averaged.

        axis_stats : per-axis health, used by the train.py diag panel.
            'share_split_max'   : max_i E[x_i] on the split axis    ∈ [1/2, 1]
            'share_common_max'  : ditto, common axis (per-sample G) ∈ [1/G', 1]
            'share_private_max' : ditto, private axis               ∈ [1/K, 1]
            'pi_split_common'   : mean common-share E[x_split[0]]   ∈ [0, 1]

        The share_*_max are the LARGEST MEAN ALLOCATION on each axis,
        max(α)/Σα. Flat = 1/n, fully concentrated = 1. Chosen over an entropy
        ratio because the Dirichlet entropy is DIFFERENTIAL (can be negative),
        which makes a normalised-entropy diagnostic change sign and mislead.
        ⚠ Not comparable to the pre-2026-07-21 categorical H/log(n) in old logs.
        """
        B = len(trans_list)
        S = np.stack([t['s_power'] for t in trans_list])
        X = _layer_norm_batch(S)
        pre_acts, acts, logits = _mlp_forward_batch(X, self.Ws, self.bs)  # (B, d_out)
        logits = logits.copy()
        logits[:, self._sl_private] += self._peruser_bias_batch(S)   # (B, K) residual

        dL_dlogits_b = np.zeros((B, self._d_out))
        L_pg  = 0.0
        L_ent = 0.0
        Hs_sum = Hcf_sum = Hp_sum = pi_s0_sum = 0.0

        for b, trans in enumerate(trans_list):
            mask   = self._common_mask(trans['active_irs_ids'])
            action = {'split':   trans['power_a_split'],
                      'common':  trans['power_a_common'],
                      'private': trans['power_a_private']}
            row, lpg, lent = self._dlogits(
                logits[b], mask, action, float(eff_adv_b[b]),
                beta_entropy, beta_entropy_private_extra,
                aux_w=aux_w, aux_tgt=trans.get('power_aux_tgt'))
            dL_dlogits_b[b] = row
            L_pg  += lpg
            L_ent += lent

            # axis-level diagnostics: largest MEAN share on each axis
            a_s, a_c, a_p = self._conc(logits[b], mask)
            Hs_sum += float(a_s.max() / a_s.sum())
            if int(mask.sum()) >= 2:
                Hcf_sum += float(a_c.max() / a_c.sum())
            Hp_sum    += float(a_p.max() / a_p.sum())
            pi_s0_sum += float(a_s[0] / a_s.sum())

        grads = _mlp_backward_batch(X, pre_acts, acts, self.Ws, dL_dlogits_b)
        # per-user head gets the private-logit gradient (residual: dL/dbias =
        # dL/dprivate_logit). The flat MLP above still gets the full dL unchanged.
        grads.update(self._peruser_grads_batch(S, dL_dlogits_b[:, self._sl_private]))
        axis_stats = {
            'share_split_max':   Hs_sum    / B,
            'share_common_max':  Hcf_sum   / B,
            'share_private_max': Hp_sum    / B,
            'pi_split_common':   pi_s0_sum / B,
        }
        return L_pg / B, L_ent / B, grads, axis_stats

    def apply_grads(self, grads: dict) -> None:
        self.opt.step(self._params, grads)

    def update(self, s_t: np.ndarray,
               active_irs_ids: list,
               advantage: float,
               beta_entropy: float = 0.0,
               action=None,
               beta_entropy_private_extra: float = 0.0) -> tuple:
        L_pg, L_ent, grads = self.compute_grads(
            s_t, active_irs_ids, advantage, beta_entropy, action,
            beta_entropy_private_extra)
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
            'factored': True,    # v2 marker: 3-way (split, common, private)
        }
        with open(os.path.join(path, 'power_config.json'), 'w') as f:
            json.dump(cfg_dict, f, indent=2)
        np.savez(os.path.join(path, 'power_params.npz'), **self.get_params())

    @classmethod
    def from_dir(cls, path: str, seed: int = None):
        with open(os.path.join(path, 'power_config.json')) as f:
            c = json.load(f)
        if not c.get('factored', False):
            raise ValueError(
                f"PowerMLP at {path} is a pre-factored (v1) checkpoint — "
                "incompatible output shape with the 3-way factored architecture. "
                "Train fresh from R_LoS=0.2.")
        obj = cls(d_s=c['d_s'], K=c['K'], M=c['M'], P_S=c['P_S'],
                  hidden=c['hidden'], lr=c['lr'], seed=seed)
        obj.set_params(dict(np.load(os.path.join(path, 'power_params.npz'))))
        return obj


# ── Common-rate split MLP ──────────────────────────────────────────────────────

# ⭐ ck-stability (2026-06-18): floor the within-group logit SPREAD so the C_k
# group-softmax cannot saturate to a one-hot. Without this, once β_entropy anneals
# to its floor the C_k softmax collapses to a degenerate one-hot (ent_ck→-0.000,
# ‖∇ck‖ explodes 7→1e5, clip-contained but broken → common-rate dumped on 1 user →
# QoS crash). Seen in result_40 (fresh+③) and result_44 (③ α=0.8). Flooring the
# spread at -CK_LOGIT_SPREAD keeps every member's fraction ≥ ~e^{-spread} → entropy
# bounded away from 0 and the softmax Jacobian (hence grad) stays bounded.
CK_LOGIT_SPREAD = 8.0

def _ck_group_softmax(logits_g: np.ndarray) -> np.ndarray:
    """Numerically-stable within-group softmax with a clamped logit spread."""
    z = logits_g - logits_g.max()
    np.maximum(z, -CK_LOGIT_SPREAD, out=z)
    e = np.exp(z)
    return e / e.sum()


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
        group_sel : dict {gid: (len(members),) float  |  None}
                    Per-group SAMPLED SHARE VECTOR — exactly the fractions used
                    to build C_k below. None marks a group whose common budget
                    was ~0, i.e. a deterministic (unsampled) group that must be
                    skipped by log-prob/gradient. Stored in ep_buf and passed
                    back to compute_grads/compute_log_prob.

        ⭐ 2026-07-21 PHANTOM-ACTION FIX (same defect as PowerMLP): this used to
        execute the softmax vector while PPO scored a separately-sampled member
        INDEX that never entered C_k, so the reward was independent of the
        scored action and the expected policy gradient was exactly zero. The
        shares are now drawn from a Dirichlet and the DRAW is what is executed.
        See RL/dirichlet.py.
        """
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)

        C_k       = np.zeros(self.K)
        alpha_k   = np.ones(self.K) / max(self.K, 1)
        group_sel: dict = {}

        K_used = K_active if K_active is not None else self.K
        groups = self._build_groups(phi[:K_used])
        for gid, members in groups.items():
            R_c_g = float(R_c_group.get(gid, 0.0))
            if R_c_g < 1e-12 or len(members) < 1:
                alpha_k[members] = 1.0 / max(len(members), 1)
                group_sel[gid]   = None        # nothing was sampled here
                continue
            conc_g = _dir.logits_to_conc(logits[members])
            x_g    = _dir.sample(self.rng, conc_g)
            alpha_k[members] = x_g
            C_k[members]     = x_g * R_c_g     # executed == sampled
            group_sel[gid]   = x_g

        return C_k, alpha_k, group_sel

    # ── shared per-group term (used by log-prob and both gradient paths) ──

    def _group_terms(self, logits, groups, group_sel):
        """Yield (members, conc_g, x_g) for every group that was actually
        sampled. Groups stored as None had a ~0 common budget and were
        deterministic — they contribute no log-prob and no gradient."""
        for gid, members in groups.items():
            if len(members) < 1:
                continue
            x_g = None if group_sel is None else group_sel.get(gid, None)
            if x_g is None:
                continue
            x_g = np.asarray(x_g, dtype=float)
            if x_g.shape[0] != len(members):
                continue                       # stale/mismatched buffer entry
            yield members, _dir.logits_to_conc(logits[members]), x_g

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
        group_sel : dict {gid: share vector | None}
            Per-group sampled shares from forward() (PPO: the stored action).
            Groups mapped to None had a ~0 budget and are skipped.
        """
        s_t = _layer_norm(s_t)
        pres, hs, logits = _mlp_forward(s_t, self.Ws, self.bs)

        K_used = K_active if K_active is not None else self.K
        groups = self._build_groups(phi[:K_used])
        dL_dlogits, L_pg_total, L_ent_total = self._dlogits(
            logits, groups, group_sel, float(advantage), beta_entropy)

        grads = _mlp_backward(s_t, pres, hs, self.Ws, dL_dlogits)
        return float(L_pg_total), float(L_ent_total), grads

    def _dlogits(self, logits, groups, group_sel, advantage, beta_entropy,
                 aux_w: float = 0.0, aux_tgt: dict = None):
        """
        dL/dlogits for one transition; shared by the single and batch paths.

        aux_w / aux_tgt — OPTIONAL auxiliary teacher (--ck-aux-weight). Adds
            L_aux = -w · log p(x_oracle | α)
        i.e. a maximum-likelihood pull toward the demand-fill allocation, on TOP
        of the PPO term. It is NOT a warm-up: it is present at every update, so
        PPO cannot erode it the way it eroded the phase warm-up (alignment
        15.6% → 7.0% over 1300 ep in result_154). The head still receives the
        full reward gradient and still adapts; the teacher only supplies the
        credit signal that a single shared advantage across four heads fails to
        deliver. Anneal w → 0 once the head tracks the target on its own.

        Same functional form as the PG term with "advantage = w" evaluated at
        the ORACLE share instead of the sampled one, so it reuses dlogp_dconc.
        """
        dL_dlogits  = np.zeros(self.K)
        L_pg_total  = 0.0
        L_ent_total = 0.0
        L_aux_total = 0.0
        for members, conc_g, x_g in self._group_terms(logits, groups, group_sel):
            L_pg_total += -advantage * _dir.log_prob(x_g, conc_g)
            g = -advantage * _dir.dlogp_dconc(x_g, conc_g)

            L_ent_total += -beta_entropy * _dir.entropy(conc_g)
            g += -beta_entropy * _dir.dentropy_dconc(conc_g)

            # α = softplus(logit) + α_min  →  dα/dlogit = sigmoid(logit)
            dL_dlogits[members] += g * _dir.dconc_dlogits(logits[members])

        if aux_w > 0.0 and aux_tgt:
            for gid, members in groups.items():
                t = aux_tgt.get(gid)
                if t is None or len(members) < 1:
                    continue
                t = np.asarray(t, dtype=float)
                if t.shape[0] != len(members):
                    continue
                conc_g = _dir.logits_to_conc(logits[members])
                L_aux_total += -aux_w * _dir.log_prob(t, conc_g)
                g_aux = -aux_w * _dir.dlogp_dconc(t, conc_g)
                dL_dlogits[members] += g_aux * _dir.dconc_dlogits(logits[members])
        return dL_dlogits, float(L_pg_total), float(L_ent_total + L_aux_total)

    def compute_log_prob(self, s_t: np.ndarray,
                         phi: np.ndarray,
                         K_active: int,
                         group_sel: dict) -> float:
        """Log π(group_sel|s) under current policy. Used for PPO ratio."""
        _, _, logits = _mlp_forward(_layer_norm(s_t), self.Ws, self.bs)
        K_used  = K_active if K_active is not None else self.K
        groups  = self._build_groups(phi[:K_used])
        return float(sum(_dir.log_prob(x_g, conc_g) for _, conc_g, x_g
                         in self._group_terms(logits, groups, group_sel)))

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
            lp_b[b] = sum(_dir.log_prob(x_g, conc_g) for _, conc_g, x_g
                          in self._group_terms(logits[b], groups,
                                               trans['ck_group_sel']))
        return lp_b

    def compute_grads_batch(self, trans_list: list, K: int,
                            eff_adv_b: np.ndarray,
                            beta_entropy: float = 0.0,
                            aux_w: float = 0.0) -> tuple:
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
            row, lpg, lent = self._dlogits(
                logits[b], groups, trans['ck_group_sel'],
                float(eff_adv_b[b]), beta_entropy,
                aux_w=aux_w, aux_tgt=trans.get('ck_aux_tgt'))
            dL_dlogits_b[b] = row
            L_pg  += lpg
            L_ent += lent

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
