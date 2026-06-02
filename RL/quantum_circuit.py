"""
quantum_circuit.py
------------------
Statevector quantum circuit simulation — dynamic n_qubits, GPU-accelerated.

Backend
-------
  CuPy (GPU) is used automatically when available; falls back to NumPy (CPU).

  Install CuPy for NVIDIA driver >= 595 / CUDA 12.x:
      pip install cupy-cuda12x

  Check which backend is active:
      from RL.quantum_circuit import GPU_BACKEND
      print(GPU_BACKEND)   # True = CuPy/GPU, False = NumPy/CPU

Circuit structure
-----------------
  |ψ_out⟩ = U_var(θ) · U_enc(z; λ) · H^⊗Nq |0⟩^⊗Nq

  U_enc : ∏_i  R_y(α_i) R_z(δ_i)
  U_var : ∏_ℓ  [ (optional U_enc re-upload)  ·  ∏_i R_y(θ_ℓi) R_z(θ_ℓi)  ·  U_ent ]
  U_ent : ∏_i  CZ_{i,i+1}    nearest-neighbour CZ chain

Observables
-----------
  ⟨Z_i⟩ (Nq) + ⟨Z_i Z_{i+1}⟩ (Nq-1)  →  o_hat ∈ R^{2·Nq-1}

Parameter-shift speedup
-----------------------
  param_shift_jacobian batches ALL ±SHIFT evaluations into a single forward
  pass so the GPU processes them in parallel.

  n_qubits=12, L=3 → 192 circuits / Jacobian call
  n_qubits=16, L=3 → 256 circuits / Jacobian call

  RTX 4090 runs all circuits simultaneously → ~50–200× faster than
  the sequential NumPy loop that was here previously.
"""

import numpy as np

# ── Backend selection ──────────────────────────────────────────────────────────

try:
    import cupy as _xp
    _xp.cuda.Device(0).use()   # verify a GPU is actually accessible
    GPU_BACKEND = True
except Exception:
    _xp = np
    GPU_BACKEND = False

SHIFT = np.pi / 2   # parameter-shift offset for rotation gates

# ── QC floating-point precision ───────────────────────────────────────────────
# complex64 (FP32) is 2-4× faster on GPU; negligible precision loss for RL.
# Change both to np.complex128 / np.float64 for full double precision.
QC_DTYPE  = np.complex64
_QC_RTYPE = np.float32

# ── Device helpers ─────────────────────────────────────────────────────────────

def _dev(arr: np.ndarray):
    """Move a NumPy array to the active device (GPU or CPU no-op)."""
    if GPU_BACKEND:
        return _xp.asarray(arr)
    return arr


def _cpu(arr) -> np.ndarray:
    """Move an array back to CPU NumPy (no-op if already NumPy)."""
    if GPU_BACKEND and isinstance(arr, _xp.ndarray):
        return arr.get()
    return np.asarray(arr)

# ── Qubit-index cache ─────────────────────────────────────────────────────────
# Precomputed pairs (x0, x1) for flat-index gate application.
# x0: all basis states with qubit bit = 0; x1 = x0 | stride.
# Eliminates reshape/moveaxis overhead — ~3.7× faster than tensor approach.

_QUBIT_IDX: dict = {}   # (nq, qubit) → (x0, x1) on device

def _qubit_idx(nq: int, qubit: int):
    key = (nq, qubit)
    if key not in _QUBIT_IDX:
        stride = 1 << (nq - 1 - qubit)
        x      = np.arange(1 << nq)
        x0_cpu = x[((x >> (nq - 1 - qubit)) & 1) == 0]
        x1_cpu = x0_cpu + stride
        _QUBIT_IDX[key] = (_dev(x0_cpu), _dev(x1_cpu))
    return _QUBIT_IDX[key]

_CZ_IDX: dict = {}   # (nq, ctrl, tgt) → x11 on device

def _cz_idx(nq: int, ctrl: int, tgt: int):
    key = (nq, ctrl, tgt)
    if key not in _CZ_IDX:
        x   = np.arange(1 << nq)
        x11 = x[((x >> (nq - 1 - ctrl)) & 1) & ((x >> (nq - 1 - tgt)) & 1)]
        _CZ_IDX[key] = _dev(x11)
    return _CZ_IDX[key]

# ── Cross-block topology (B3/B4/B1 — configurable at actor init time) ──────────
_EXTRA_CZ_PAIRS: tuple = ()   # B3: cross-block CZ bridges applied after NN chain per layer
_EXTRA_ZZ_PAIRS: tuple = ()   # B4: extra ⟨Z_i Z_j⟩ observables appended to NN ZZ
_FULL_ZZ_PAIRS:  tuple = ()   # B1: full ZZ set — when non-empty, REPLACES NN ZZ + extra ZZ
                               #     n_obs = nq + len(_FULL_ZZ_PAIRS)

# ── R1 structured readout (Δ2, 2026-06-01) ─────────────────────────────────────
# readout_mode == 'r1' → _obs_from_batch produces physics-structured observables
# aligned with the IRS-assignment action (user k → IRS m). Qubit layout (Option-A):
# user qubits 0..nq-M-1, IRS qubits nq-M..nq-1.  Observable order:
#   R1-a ⟨Z_u⟩  per user qubit (nq-M) | R1-b ⟨Z_u·Z_r⟩ user×IRS user-major (nq-M)·M
#   R1-c C_u = Σ_{u'≠u}⟨Z_u Z_u'⟩ cluster (nq-M) | R1-d ⟨Z_r⟩ per IRS (M)
# Total = (nq-M)(M+2)+M.  (Case2 nq12 M2 → 42; Case3 nq16 M4 → 76.)
# CAVEAT: R1-b/c carry real correlation only where qubits are entangled. Current
# circuit entangles via NN-chain CZ (+ extra_cz bridges); full R1-b value needs E1
# entanglement (Δ3, a follow-up). This core run tests R1 with EXISTING entanglement.
_READOUT_MODE: str = 'generic'   # 'generic' | 'r1'
_R1_M:         int = 0           # number of IRS qubits (= M) when r1 mode active


def configure_topology(extra_cz_pairs: tuple = (),
                       extra_zz_pairs: tuple = (),
                       full_zz_pairs:  tuple = (),
                       readout_mode:   str   = 'generic',
                       r1_m:           int   = 0) -> None:
    """
    Configure cross-block entanglement topology + readout mode.

    Call once at actor init (before any circuit evaluation) to set the
    module-global topology used by all circuit functions.

    extra_cz_pairs : (B3) sequence of (ctrl, tgt) pairs — CZ bridges applied after
                     the NN chain per variational layer.
    extra_zz_pairs : (B4) sequence of (qi, qj) pairs — extra ⟨Z_i Z_j⟩ observables
                     appended to NN ZZ.  N_QUANTUM += len(extra_zz_pairs).
    full_zz_pairs  : (B1) when non-empty, REPLACES the NN ZZ loop and extra_zz_pairs
                     entirely.  N_QUANTUM = nq + len(full_zz_pairs).
    readout_mode   : 'generic' (Z + NN-ZZ + extra/full) | 'r1' (structured per-action).
    r1_m           : number of IRS qubits (= M) when readout_mode == 'r1'.
    """
    global _EXTRA_CZ_PAIRS, _EXTRA_ZZ_PAIRS, _FULL_ZZ_PAIRS, _READOUT_MODE, _R1_M
    _EXTRA_CZ_PAIRS = tuple(extra_cz_pairs)
    _EXTRA_ZZ_PAIRS = tuple(extra_zz_pairs)
    _FULL_ZZ_PAIRS  = tuple(full_zz_pairs)
    _READOUT_MODE   = str(readout_mode)
    _R1_M           = int(r1_m)


def _n_obs(nq: int) -> int:
    """Number of observables given current topology settings."""
    if _READOUT_MODE == 'r1':
        nu = nq - _R1_M                  # user qubits
        return nu * (_R1_M + 2) + _R1_M  # R1-a + R1-b + R1-c + R1-d
    if _FULL_ZZ_PAIRS:
        return nq + len(_FULL_ZZ_PAIRS)
    return 2 * nq - 1 + len(_EXTRA_ZZ_PAIRS)


def _r1_pairs(nq: int, M: int):
    """Qubit-index lists for R1-b / R1-c observables (user qubits 0..nq-M-1)."""
    nu       = nq - M
    user_q   = list(range(nu))
    irs_q    = list(range(nu, nq))
    b_pairs  = [(u, r) for u in user_q for r in irs_q]          # (nq-M)·M, user-major
    c_groups = [[(u, up) for up in user_q if up != u] for u in user_q]
    return b_pairs, c_groups

# ── Flat-index gate application ───────────────────────────────────────────────
# All functions operate in-place on psi_b : (B, 2^nq) complex (on device).
# Using flat column indexing avoids reshape/moveaxis on high-dim tensors.

def _apply_h_flat(psi_b, qubit: int, nq: int) -> None:
    """Hadamard on qubit: in-place, all B circuits."""
    x0, x1  = _qubit_idx(nq, qubit)
    inv_r2  = 1.0 / np.sqrt(2)
    p0      = psi_b[:, x0].copy()
    p1      = psi_b[:, x1]
    psi_b[:, x0] = inv_r2 * (p0 + p1)
    psi_b[:, x1] = inv_r2 * (p0 - p1)


def _apply_ry_flat(psi_b, t_b, qubit: int, nq: int) -> None:
    """Ry(t_b) on qubit: in-place, per-circuit angles t_b (B,) on device."""
    x0, x1 = _qubit_idx(nq, qubit)
    c = _xp.cos(t_b / 2)[:, None]   # (B, 1)
    s = _xp.sin(t_b / 2)[:, None]
    p0 = psi_b[:, x0].copy()        # (B, dim//2)
    p1 = psi_b[:, x1]
    psi_b[:, x0] = c * p0 - s * p1
    psi_b[:, x1] = s * p0 + c * p1


def _apply_rz_flat(psi_b, t_b, qubit: int, nq: int) -> None:
    """Rz(t_b) on qubit: in-place, no allocation (diagonal gate)."""
    x0, x1 = _qubit_idx(nq, qubit)
    e = _xp.exp(0.5j * t_b)[:, None]   # (B, 1)
    psi_b[:, x0] *= _xp.conj(e)
    psi_b[:, x1] *= e


def _apply_cz_flat(psi_b, ctrl: int, tgt: int, nq: int) -> None:
    """CZ gate: in-place sign flip on |11⟩ states, no allocation."""
    x11 = _cz_idx(nq, ctrl, tgt)
    psi_b[:, x11] *= -1

# ── Core: batched circuit builder ─────────────────────────────────────────────

def _build_batch(alpha_b, delta_b, theta_y_b, theta_z_b,
                 n_var_layers: int, data_reuploading: bool):
    """
    Simulate B independent quantum circuits in one vectorised pass.

    Parameters (all on device)
    --------------------------
    alpha_b   : (B, nq)
    delta_b   : (B, nq)
    theta_y_b : (B, L, nq)
    theta_z_b : (B, L, nq)

    Returns
    -------
    psi_b : (B, 2**nq) complex  state vectors on device.
    """
    B   = alpha_b.shape[0]
    nq  = alpha_b.shape[1]
    dim = 1 << nq

    # Cast to configured QC precision (float32 → complex64 for GPU throughput)
    alpha_b   = _xp.asarray(alpha_b,   dtype=_QC_RTYPE)
    delta_b   = _xp.asarray(delta_b,   dtype=_QC_RTYPE)
    theta_y_b = _xp.asarray(theta_y_b, dtype=_QC_RTYPE)
    theta_z_b = _xp.asarray(theta_z_b, dtype=_QC_RTYPE)

    psi = _xp.zeros((B, dim), dtype=QC_DTYPE)
    psi[:, 0] = 1.0

    # H^⊗nq
    for i in range(nq):
        _apply_h_flat(psi, i, nq)

    # U_enc
    for i in range(nq):
        _apply_ry_flat(psi, alpha_b[:, i], i, nq)
        _apply_rz_flat(psi, delta_b[:, i], i, nq)

    # U_var: L variational layers
    for ell in range(n_var_layers):
        if data_reuploading and ell > 0:
            for i in range(nq):
                _apply_ry_flat(psi, alpha_b[:, i], i, nq)
                _apply_rz_flat(psi, delta_b[:, i], i, nq)
        for i in range(nq):
            _apply_ry_flat(psi, theta_y_b[:, ell, i], i, nq)
            _apply_rz_flat(psi, theta_z_b[:, ell, i], i, nq)
        for i in range(nq - 1):
            _apply_cz_flat(psi, i, i + 1, nq)
        for ctrl, tgt in _EXTRA_CZ_PAIRS:       # B3: cross-block entanglement bridges
            _apply_cz_flat(psi, ctrl, tgt, nq)

    return psi

# ── Observable computation ─────────────────────────────────────────────────────

def _obs_from_batch(psi_b, nq: int):
    """
    Compute Z and ZZ expectations for all B state vectors analytically.

    psi_b : (B, 2^nq) on device
    Returns (B, _n_obs(nq)) on device:
      B1 active               : (B, nq + len(_FULL_ZZ_PAIRS))
      B4 active, B1 inactive  : (B, 2*nq-1 + len(_EXTRA_ZZ_PAIRS))
      baseline                : (B, 2*nq-1)
    """
    B, dim = psi_b.shape
    probs  = _xp.abs(psi_b) ** 2                        # (B, dim)
    basis  = _xp.arange(dim, dtype=_xp.int32)           # (dim,)

    # Per-qubit Z eigenvalue sign vectors: S[i] = (1 - 2·bit_i(x)) ∈ {±1}^dim
    sign = [(1 - 2 * ((basis >> (nq - 1 - i)) & 1)) for i in range(nq)]

    z_exp = _xp.empty((B, nq), dtype=float)
    for i in range(nq):
        z_exp[:, i] = (probs * sign[i]).sum(axis=1)

    if _READOUT_MODE == 'r1':                            # Δ2: structured per-action readout
        M  = _R1_M
        nu = nq - M
        b_pairs, c_groups = _r1_pairs(nq, M)
        # R1-a: ⟨Z_u⟩ user qubits  | R1-d: ⟨Z_r⟩ IRS qubits
        r1a = z_exp[:, :nu]                              # (B, nu)
        r1d = z_exp[:, nu:]                              # (B, M)
        # R1-b: ⟨Z_u·Z_r⟩ user×IRS (user-major)
        r1b = _xp.empty((B, len(b_pairs)), dtype=float)
        for k, (u, r) in enumerate(b_pairs):
            r1b[:, k] = (probs * sign[u] * sign[r]).sum(axis=1)
        # R1-c: C_u = mean_{u'≠u} ⟨Z_u Z_u'⟩  (cluster aggregate, MEAN → bounded [-1,1])
        r1c = _xp.zeros((B, nu), dtype=float)
        for u, members in enumerate(c_groups):
            acc = _xp.zeros(B, dtype=float)
            for (_u, up) in members:
                acc = acc + (probs * sign[_u] * sign[up]).sum(axis=1)
            r1c[:, u] = acc / max(len(members), 1)
        return _xp.concatenate([r1a, r1b, r1c, r1d], axis=1)  # (B, (nu)(M+2)+M)

    if _FULL_ZZ_PAIRS:                                   # B1: full ZZ replaces NN ZZ + extra
        full = _xp.empty((B, len(_FULL_ZZ_PAIRS)), dtype=float)
        for k, (qi, qj) in enumerate(_FULL_ZZ_PAIRS):
            bit_i      = (basis >> (nq - 1 - qi)) & 1
            bit_j      = (basis >> (nq - 1 - qj)) & 1
            full[:, k] = (probs * (1 - 2 * bit_i) * (1 - 2 * bit_j)).sum(axis=1)
        return _xp.concatenate([z_exp, full], axis=1)   # (B, nq+len(_FULL_ZZ_PAIRS))

    zz_exp = _xp.empty((B, nq - 1), dtype=float)
    for i in range(nq - 1):
        bit_i        = (basis >> (nq - 1 - i)) & 1
        bit_j        = (basis >> (nq - 2 - i)) & 1
        zz_exp[:, i] = (probs * (1 - 2 * bit_i) * (1 - 2 * bit_j)).sum(axis=1)

    if _EXTRA_ZZ_PAIRS:                                  # B4: cross-block observables
        extra = _xp.empty((B, len(_EXTRA_ZZ_PAIRS)), dtype=float)
        for k, (qi, qj) in enumerate(_EXTRA_ZZ_PAIRS):
            bit_i = (basis >> (nq - 1 - qi)) & 1
            bit_j = (basis >> (nq - 1 - qj)) & 1
            extra[:, k] = (probs * (1 - 2 * bit_i) * (1 - 2 * bit_j)).sum(axis=1)
        return _xp.concatenate([z_exp, zz_exp, extra], axis=1)  # (B, 2*nq-1+n_extra)
    return _xp.concatenate([z_exp, zz_exp], axis=1)     # (B, 2*nq-1)

# ── Public API ─────────────────────────────────────────────────────────────────

def expectations_analytic(alpha: np.ndarray, delta: np.ndarray,
                           theta_y: np.ndarray, theta_z: np.ndarray,
                           n_var_layers: int = 1,
                           data_reuploading: bool = False) -> np.ndarray:
    """
    Exact Z / ZZ expectations from the state vector (no shot noise).
    Used during training for stable gradient computation.

    Returns o_hat ∈ [-1, 1]^{2·Nq-1}  as a CPU NumPy array.
    """
    nq    = len(alpha)
    psi_b = _build_batch(
        _dev(alpha[None]),   _dev(delta[None]),
        _dev(theta_y[None]), _dev(theta_z[None]),
        n_var_layers, data_reuploading)
    return _cpu(_obs_from_batch(psi_b, nq)[0])          # (2*nq-1,) CPU array


def expectations_shots(alpha: np.ndarray, delta: np.ndarray,
                        theta_y: np.ndarray, theta_z: np.ndarray,
                        n_shots: int = 1750,
                        rng: np.random.Generator = None,
                        n_var_layers: int = 1,
                        data_reuploading: bool = False) -> np.ndarray:
    """
    Shot-based Z / ZZ expectations (finite-sample noise).
    State vector computed on GPU; sampling done on CPU.

    Returns o_hat ∈ [-1, 1]^{2·Nq-1}  as a CPU NumPy array.
    """
    nq  = len(alpha)
    dim = 1 << nq
    if rng is None:
        rng = np.random.default_rng()

    psi_b = _build_batch(
        _dev(alpha[None]),   _dev(delta[None]),
        _dev(theta_y[None]), _dev(theta_z[None]),
        n_var_layers, data_reuploading)
    psi = _cpu(psi_b[0])                                # bring state vector to CPU

    probs  = np.abs(psi) ** 2
    probs  = np.clip(probs, 0, None);  probs /= probs.sum()
    samples = rng.choice(dim, size=n_shots, p=probs)

    # Per-qubit Z eigenvalue signs per shot: s[i] = (1 - 2·bit_i) ∈ {±1}^n_shots
    s_shot = [(1 - 2 * ((samples >> (nq - 1 - i)) & 1)) for i in range(nq)]
    z_exp  = np.array([s_shot[i].mean() for i in range(nq)])

    if _READOUT_MODE == 'r1':                            # Δ2: structured per-action readout
        M  = _R1_M
        nu = nq - M
        b_pairs, c_groups = _r1_pairs(nq, M)
        r1a = z_exp[:nu]
        r1d = z_exp[nu:]
        r1b = np.array([(s_shot[u] * s_shot[r]).mean() for (u, r) in b_pairs])
        r1c = np.array([
            float(np.mean([(s_shot[_u] * s_shot[up]).mean() for (_u, up) in members]))
            if members else 0.0
            for members in c_groups
        ])
        return np.concatenate([r1a, r1b, r1c, r1d])

    if _FULL_ZZ_PAIRS:                                   # B1: full ZZ replaces NN ZZ + extra
        full = np.empty(len(_FULL_ZZ_PAIRS))
        for k, (qi, qj) in enumerate(_FULL_ZZ_PAIRS):
            bit_i   = (samples >> (nq - 1 - qi)) & 1
            bit_j   = (samples >> (nq - 1 - qj)) & 1
            full[k] = ((1 - 2 * bit_i) * (1 - 2 * bit_j)).mean()
        return np.concatenate([z_exp, full])

    zz_exp = np.empty(nq - 1)
    for i in range(nq - 1):
        bit_i     = (samples >> (nq - 1 - i)) & 1
        bit_j     = (samples >> (nq - 2 - i)) & 1
        zz_exp[i] = ((1 - 2 * bit_i) * (1 - 2 * bit_j)).mean()

    if _EXTRA_ZZ_PAIRS:                                  # B4: cross-block observables
        extra = np.empty(len(_EXTRA_ZZ_PAIRS))
        for k, (qi, qj) in enumerate(_EXTRA_ZZ_PAIRS):
            bit_i = (samples >> (nq - 1 - qi)) & 1
            bit_j = (samples >> (nq - 1 - qj)) & 1
            extra[k] = ((1 - 2 * bit_i) * (1 - 2 * bit_j)).mean()
        return np.concatenate([z_exp, zz_exp, extra])
    return np.concatenate([z_exp, zz_exp])


def expectations_analytic_batch(alpha_b: np.ndarray, delta_b: np.ndarray,
                                theta_y: np.ndarray, theta_z: np.ndarray,
                                n_var_layers: int = 1,
                                data_reuploading: bool = False) -> np.ndarray:
    """
    Exact Z / ZZ expectations for B_s samples in one batched GPU pass.

    alpha_b, delta_b : (B_s, nq)  — per-sample encoding angles
    theta_y, theta_z : (L, nq)    — shared variational parameters

    Returns o_hat_b ∈ [-1, 1]^{B_s × (2·Nq-1)} as a CPU NumPy array.
    """
    B_s, nq = alpha_b.shape
    theta_y_b = np.tile(theta_y[np.newaxis], (B_s, 1, 1))
    theta_z_b = np.tile(theta_z[np.newaxis], (B_s, 1, 1))
    psi_b = _build_batch(
        _dev(alpha_b), _dev(delta_b),
        _dev(theta_y_b), _dev(theta_z_b),
        n_var_layers, data_reuploading)
    return _cpu(_obs_from_batch(psi_b, nq))   # (B_s, 2*nq-1)


def param_shift_jacobian(alpha: np.ndarray, delta: np.ndarray,
                          theta_y: np.ndarray, theta_z: np.ndarray,
                          wrt: str,
                          n_var_layers: int = 1,
                          data_reuploading: bool = False) -> np.ndarray:
    """
    Parameter-shift Jacobian  J = ∂o_hat / ∂p.

    All ±SHIFT circuit evaluations are issued as a SINGLE batched forward
    pass — the GPU evaluates them all in parallel.

    wrt : 'alpha' | 'delta'            → J shape (2·Nq-1, Nq)
          'theta_y' | 'theta_z'        → J shape (2·Nq-1, L·Nq)

    Returns J as a CPU NumPy array (same API as before).
    """
    nq    = len(alpha)
    n_obs = _n_obs(nq)

    # ── Build batch of all ±SHIFT parameter configurations ────────────────────
    if wrt in ('alpha', 'delta'):
        n_params  = nq
        B         = 2 * n_params

        alpha_b   = np.tile(alpha,   (B, 1))        # (B, nq)
        delta_b   = np.tile(delta,   (B, 1))
        theta_y_b = np.tile(theta_y, (B, 1, 1))     # (B, L, nq)
        theta_z_b = np.tile(theta_z, (B, 1, 1))

        target = alpha_b if wrt == 'alpha' else delta_b
        for j in range(n_params):
            target[2*j,     j] += SHIFT
            target[2*j + 1, j] -= SHIFT

    else:   # 'theta_y' | 'theta_z'
        n_params  = n_var_layers * nq
        B         = 2 * n_params

        alpha_b   = np.tile(alpha,   (B, 1))
        delta_b   = np.tile(delta,   (B, 1))
        theta_y_b = np.tile(theta_y, (B, 1, 1))
        theta_z_b = np.tile(theta_z, (B, 1, 1))

        target = theta_y_b if wrt == 'theta_y' else theta_z_b
        for ell in range(n_var_layers):
            for j in range(nq):
                idx = ell * nq + j
                target[2*idx,     ell, j] += SHIFT
                target[2*idx + 1, ell, j] -= SHIFT

    # ── Single batched GPU pass ───────────────────────────────────────────────
    psi_b = _build_batch(
        _dev(alpha_b), _dev(delta_b),
        _dev(theta_y_b), _dev(theta_z_b),
        n_var_layers, data_reuploading)
    o_b = _cpu(_obs_from_batch(psi_b, nq))             # (B, n_obs) on CPU

    # ── Assemble Jacobian ─────────────────────────────────────────────────────
    J = np.zeros((n_obs, n_params))
    for k in range(n_params):
        J[:, k] = (o_b[2*k] - o_b[2*k + 1]) / 2.0

    return J


def param_shift_jacobian_all_batch(alpha_b: np.ndarray, delta_b: np.ndarray,
                                    theta_y: np.ndarray, theta_z: np.ndarray,
                                    n_var_layers: int = 1,
                                    data_reuploading: bool = False):
    """
    Compute all four Jacobians (theta_y, theta_z, alpha, delta) in ONE GPU pass.

    Replaces four sequential per-wrt Jacobian computations with a single
    _build_batch invocation of size B_s × 192 circuits, eliminating three
    redundant GPU kernel launches and large array allocations.

    Returns
    -------
    J_ty_b : (B_s, n_obs, L·nq)
    J_tz_b : (B_s, n_obs, L·nq)
    J_a_b  : (B_s, n_obs, nq)
    J_d_b  : (B_s, n_obs, nq)
    """
    B_s, nq = alpha_b.shape
    L        = n_var_layers
    n_obs    = _n_obs(nq)
    n_p_list = [L * nq, L * nq, nq, nq]          # theta_y, theta_z, alpha, delta
    wrt_list = ['theta_y', 'theta_z', 'alpha', 'delta']

    # ── Build one shifted-parameter array per wrt type (CPU) ──────────────────
    segs = []
    for wrt, n_p in zip(wrt_list, n_p_list):
        B   = B_s * 2 * n_p
        a   = np.repeat(alpha_b, 2 * n_p, axis=0)
        d   = np.repeat(delta_b, 2 * n_p, axis=0)
        ty  = np.tile(theta_y[np.newaxis], (B, 1, 1))
        tz  = np.tile(theta_z[np.newaxis], (B, 1, 1))

        s_arr = np.arange(B_s)
        k_arr = np.arange(n_p)
        plus_rows  = (s_arr[:, None] * (2 * n_p) + 2 * k_arr[None, :]).ravel()
        minus_rows = plus_rows + 1

        if wrt == 'theta_y':
            er = np.tile(k_arr // nq, B_s);  jr = np.tile(k_arr % nq, B_s)
            ty[plus_rows, er, jr] += SHIFT;  ty[minus_rows, er, jr] -= SHIFT
        elif wrt == 'theta_z':
            er = np.tile(k_arr // nq, B_s);  jr = np.tile(k_arr % nq, B_s)
            tz[plus_rows, er, jr] += SHIFT;  tz[minus_rows, er, jr] -= SHIFT
        elif wrt == 'alpha':
            jr = np.tile(k_arr, B_s)
            a[plus_rows, jr] += SHIFT;  a[minus_rows, jr] -= SHIFT
        else:
            jr = np.tile(k_arr, B_s)
            d[plus_rows, jr] += SHIFT;  d[minus_rows, jr] -= SHIFT

        segs.append((a, d, ty, tz, n_p))

    # ── Concatenate all segments → single super-batch (CPU) ───────────────────
    alpha_all   = np.concatenate([s[0] for s in segs], axis=0)
    delta_all   = np.concatenate([s[1] for s in segs], axis=0)
    theta_y_all = np.concatenate([s[2] for s in segs], axis=0)
    theta_z_all = np.concatenate([s[3] for s in segs], axis=0)

    # ── ONE GPU forward pass ───────────────────────────────────────────────────
    psi_b = _build_batch(
        _dev(alpha_all), _dev(delta_all),
        _dev(theta_y_all), _dev(theta_z_all),
        n_var_layers, data_reuploading)
    o_b = _cpu(_obs_from_batch(psi_b, nq))          # (B_total, n_obs)

    # ── Parse each Jacobian from the output block ──────────────────────────────
    Js = []
    offset = 0
    for n_p in n_p_list:
        seg_len = B_s * 2 * n_p
        o_seg   = o_b[offset: offset + seg_len]     # (B_s * 2 * n_p, n_obs)
        o_4d    = o_seg.reshape(B_s, n_p, 2, n_obs)
        J_b     = (o_4d[:, :, 0, :] - o_4d[:, :, 1, :]) / 2.0
        Js.append(J_b.transpose(0, 2, 1))           # (B_s, n_obs, n_p)
        offset += seg_len

    return tuple(Js)   # (J_ty_b, J_tz_b, J_a_b, J_d_b)


def spsa_jacobian_all_batch(alpha_b: np.ndarray, delta_b: np.ndarray,
                             theta_y: np.ndarray, theta_z: np.ndarray,
                             n_var_layers: int = 1,
                             data_reuploading: bool = False,
                             spsa_epsilon: float = 0.1,
                             n_reps: int = 4,
                             rng: np.random.Generator = None):
    """
    SPSA estimate of all four Jacobians (theta_y, theta_z, alpha, delta).
    Drop-in replacement for param_shift_jacobian_all_batch with 32× fewer circuits.

    Circuit count comparison (nq=16, L=3, B_s=64):
      param_shift_jacobian_all_batch : 64 × 256 = 16,384 circuits  (8.6 GB state)
      spsa_jacobian_all_batch n_reps=4:  2 ×   4 × 64 =    512 circuits  (268 MB state)
      Speedup                         : 32×

    Algorithm
    ---------
    Each repetition:
      1. Sample Rademacher Δ[s] ∈ {±1}^n_total per sample  (n_total = 2*(L+1)*nq)
         Split into blocks: Δ_ty, Δ_tz, Δ_a, Δ_d
      2. Build (θ ± ε·Δ) parameter arrays for all B_s samples
      3. ONE GPU forward pass: 2*B_s circuits (+ and − perturbed)
      4. SPSA Jacobian estimate:
             J[s, o, k] ≈ (o_+[s,o] − o_−[s,o]) / (2ε) × Δ[s,k]
         Unbiased: E[J_SPSA[s,o,k]] = ∂o[s,o]/∂param_k  for Rademacher Δ.
    Estimates are averaged over n_reps repetitions (variance ∝ 1/n_reps).

    Returns  (same shapes as param_shift_jacobian_all_batch)
    -------
    J_ty_b : (B_s, n_obs, L*nq)
    J_tz_b : (B_s, n_obs, L*nq)
    J_a_b  : (B_s, n_obs, nq)
    J_d_b  : (B_s, n_obs, nq)
    """
    if rng is None:
        rng = np.random.default_rng()

    B_s, nq = alpha_b.shape
    L        = n_var_layers
    n_obs    = _n_obs(nq)
    n_ty     = L * nq       # theta_y params per sample
    n_tz     = L * nq       # theta_z params per sample
    n_a      = nq            # alpha params per sample
    n_d      = nq            # delta params per sample
    n_total  = n_ty + n_tz + n_a + n_d   # = 2*(L+1)*nq

    # Accumulate Jacobian estimates across reps
    J_ty = np.zeros((B_s, n_obs, n_ty), dtype=np.float32)
    J_tz = np.zeros((B_s, n_obs, n_tz), dtype=np.float32)
    J_a  = np.zeros((B_s, n_obs, n_a),  dtype=np.float32)
    J_d  = np.zeros((B_s, n_obs, n_d),  dtype=np.float32)

    # Base theta arrays tiled once (CPU); reused across reps
    ty_base = np.tile(theta_y[np.newaxis], (B_s, 1, 1)).astype(np.float32)  # (B_s, L, nq)
    tz_base = np.tile(theta_z[np.newaxis], (B_s, 1, 1)).astype(np.float32)
    a_base  = alpha_b.astype(np.float32)
    d_base  = delta_b.astype(np.float32)
    eps     = float(spsa_epsilon)

    for _ in range(n_reps):
        # Rademacher {-1, +1} perturbations: (B_s, n_total)
        D = (rng.integers(0, 2, size=(B_s, n_total)) * 2 - 1).astype(np.float32)
        D_ty = D[:, :n_ty]
        D_tz = D[:, n_ty : n_ty + n_tz]
        D_a  = D[:, n_ty + n_tz : n_ty + n_tz + n_a]
        D_d  = D[:, n_ty + n_tz + n_a :]

        # Build + and - parameter arrays (CPU, cheap)
        ty_p = ty_base + eps * D_ty.reshape(B_s, L, nq)
        ty_m = ty_base - eps * D_ty.reshape(B_s, L, nq)
        tz_p = tz_base + eps * D_tz.reshape(B_s, L, nq)
        tz_m = tz_base - eps * D_tz.reshape(B_s, L, nq)
        a_p  = a_base  + eps * D_a
        a_m  = a_base  - eps * D_a
        d_p  = d_base  + eps * D_d
        d_m  = d_base  - eps * D_d

        # Concatenate + and - → single batch of 2*B_s circuits
        alpha_all = np.concatenate([a_p,  a_m],  axis=0)   # (2*B_s, nq)
        delta_all = np.concatenate([d_p,  d_m],  axis=0)
        ty_all    = np.concatenate([ty_p, ty_m], axis=0)   # (2*B_s, L, nq)
        tz_all    = np.concatenate([tz_p, tz_m], axis=0)

        # ONE GPU forward pass: 2*B_s circuits
        psi_b = _build_batch(
            _dev(alpha_all), _dev(delta_all),
            _dev(ty_all),    _dev(tz_all),
            n_var_layers, data_reuploading)
        o_b = _cpu(_obs_from_batch(psi_b, nq))    # (2*B_s, n_obs)

        o_p = o_b[:B_s].astype(np.float32)        # (B_s, n_obs)
        o_m = o_b[B_s:].astype(np.float32)

        # SPSA finite difference: diff[s,o] = (o_+[s,o] - o_-[s,o]) / (2*ε)
        diff = (o_p - o_m) / (2.0 * eps)          # (B_s, n_obs)

        # Jacobian estimate: J[s,o,k] += diff[s,o] * Δ[s,k]
        # Unbiased because E[Δ_j * Δ_k] = δ_{jk} for independent Rademacher Δ
        J_ty += diff[:, :, None] * D_ty[:, None, :]   # (B_s, n_obs, n_ty)
        J_tz += diff[:, :, None] * D_tz[:, None, :]
        J_a  += diff[:, :, None] * D_a[:, None, :]
        J_d  += diff[:, :, None] * D_d[:, None, :]

    inv_reps = 1.0 / n_reps
    return (J_ty * inv_reps,
            J_tz * inv_reps,
            J_a  * inv_reps,
            J_d  * inv_reps)
