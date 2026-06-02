"""Δ1 gradcheck: analytic dL/d{gamma_enc, beta_enc, lam_y} vs finite-difference.
Uses compute_grads (single, exact param-shift) with data_reuploading=False so
param-shift is exact → isolates the affine+LN+lam chain-rule correctness.
Also confirms λ gradient is NON-ZERO (the unfreeze goal)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import RL.quantum_circuit as qc
# Use float64 circuit for a clean gradcheck (FP32 noise ~0.1 in finite-diff otherwise)
qc.QC_DTYPE = np.complex128
qc._QC_RTYPE = np.float64
from params import make_config
from RL.quantum_actor import QuantumActor

cfg = make_config()                       # Case 2: K=10, M=2, nq=12
act = QuantumActor(
    cfg=cfg, n_qubits=12, n_latent=24,
    n_hidden_ae=[128, 64], n_hidden_post=[256, 128], n_var_layers=2,
    n_shots=1500, lr_ae=1e-4, lr_qc=1e-4, lr_xi=3e-4,
    data_reuploading=False,               # exact param-shift for clean gradcheck
    ae_pretrain_lr=1e-3, spsa_n_reps=0,
    extra_cz_pairs=((0, 10), (5, 11)), extra_zz_pairs=((0, 10), (5, 11)),
    seed=0)

rng = np.random.default_rng(1)
s_t = rng.standard_normal(act.d_s)
phi = rng.integers(0, act.n_choices, act.K)
adv = 0.7
beta_ent = 0.001

def loss_qc():
    """Recompute the policy+entropy loss (the part quantum params see)."""
    Lpg, Lae, Lent, *_ = act.compute_grads(s_t, phi, adv, beta_ent)
    return Lpg + Lent     # AE loss does NOT depend on gamma/beta/lam

# analytic grads
Lpg, Lae, Lent, g_ae, g_qc, g_xi = act.compute_grads(s_t, phi, adv, beta_ent)
print(f"keys in g_qc: {sorted(g_qc.keys())}")
assert 'gamma_enc' in g_qc and 'beta_enc' in g_qc

eps = 1e-6
def fd(param, idx):
    flat = param.reshape(-1)
    old = flat[idx]
    flat[idx] = old + eps; lp = loss_qc()
    flat[idx] = old - eps; lm = loss_qc()
    flat[idx] = old
    return (lp - lm) / (2 * eps)

for name, arr, gA in [('beta_enc', act.beta_enc, g_qc['beta_enc']),
                      ('gamma_enc', act.gamma_enc, g_qc['gamma_enc']),
                      ('lam_y', act.lam_y, g_qc['lam_y'])]:
    idxs = [0, len(arr.reshape(-1)) // 2, len(arr.reshape(-1)) - 1]
    errs = []
    for i in idxs:
        gN = fd(arr, i)
        gAi = gA.reshape(-1)[i]
        denom = max(abs(gN), abs(gAi), 1e-8)
        errs.append(abs(gN - gAi) / denom)
        print(f"  {name}[{i}]: analytic={gAi:+.3e}  fd={gN:+.3e}  relerr={errs[-1]:.2e}")
    assert max(errs) < 1e-2, f"{name} gradcheck FAILED (relerr {max(errs):.2e})"
    print(f"  {name}: max relerr {max(errs):.2e}  ✓")

# ── λ gradient non-zero check (the unfreeze goal) ──
lam_mag = np.abs(g_qc['lam_y']).mean()
lam_net = np.abs(g_qc['lam_y'].mean())
print(f"\nλ_y grad: mean|g|={lam_mag:.3e}  |mean g|={lam_net:.3e}")
print("(single-sample: both nonzero. Batch DC-recovery via β verified separately at train.)")
assert lam_mag > 1e-8, "λ gradient is zero — unfreeze failed!"

print("\n✅ Δ1 gradcheck: gamma_enc, beta_enc, lam_y chain-rule correct + λ grad nonzero")
