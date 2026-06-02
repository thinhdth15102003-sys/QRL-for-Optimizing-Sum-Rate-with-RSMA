"""V4 for Δ4 SoftmaxPQC head + Δ2 R1 wiring + Δ5 bypass: shape, gradcheck, integration.
Full architecture-2 actor (readout_mode='r1', softmax_head=True). float64 + reup off
+ exact param-shift for clean head gradcheck."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import RL.quantum_circuit as qc
qc.QC_DTYPE = np.complex128
qc._QC_RTYPE = np.float64
from params import make_config
from RL.quantum_actor import QuantumActor

cfg = make_config()    # Case 2: K=10, M=2
act = QuantumActor(
    cfg=cfg, n_qubits=12, n_latent=24,
    n_hidden_ae=[128, 64], n_hidden_post=[256, 128], n_var_layers=2,
    n_shots=1500, lr_ae=1e-4, lr_qc=1e-4, lr_xi=3e-4,
    data_reuploading=False, ae_pretrain_lr=1e-3, spsa_n_reps=0,
    extra_cz_pairs=((0, 10), (5, 11)),
    readout_mode='r1', softmax_head=True, softmax_beta_init=1.0,
    seed=0)

# ── V3 shapes ──
nu = 12 - 2
n_obs_exp = nu * (2 + 2) + 2   # (nq-M)(M+2)+M = 42
print(f"N_QUANTUM={act.N_QUANTUM}  expect {n_obs_exp}")
assert act.N_QUANTUM == n_obs_exp
nOut = act.K * act.n_choices    # 10*3 = 30
print(f"W_sm shape={act.W_sm.shape}  expect ({24 + n_obs_exp}, {nOut})")
assert act.W_sm.shape == (24 + n_obs_exp, nOut)
assert act.beta_temp.shape == (1,)
n_head_params = act.W_sm.size + act.b_sm.size + 1
print(f"head params = {n_head_params}  (vs classical MLP ~50K)")

# ── forward runs ──
rng = np.random.default_rng(1)
s_t = rng.standard_normal(act.d_s)
phi, lp, info = act.forward(s_t)
print(f"forward: phi={phi}  logprob={lp:.3f}  o_hat shape={info['o_hat'].shape}")
assert phi.shape == (act.K,) and info['o_hat'].shape == (n_obs_exp,)

# ── V4 head gradcheck: W_sm, b_sm, beta_temp via compute_grads (exact) ──
adv = 0.7; beta_ent = 0.001
phi_a = rng.integers(0, act.n_choices, act.K)
def loss_pg():
    Lpg, Lae, Lent, *_ = act.compute_grads(s_t, phi_a, adv, beta_ent)
    return Lpg + Lent
Lpg, Lae, Lent, g_ae, g_qc, g_xi = act.compute_grads(s_t, phi_a, adv, beta_ent)
print(f"\ng_xi keys: {sorted(g_xi.keys())}  (expect W_sm, b_sm, beta_temp)")
assert set(g_xi.keys()) == {'W_sm', 'b_sm', 'beta_temp'}

eps = 1e-6
def fd(arr, i):
    f = arr.reshape(-1); old = f[i]
    f[i] = old + eps; lp = loss_pg(); f[i] = old - eps; lm = loss_pg(); f[i] = old
    return (lp - lm) / (2 * eps)
for name, arr, gA in [('beta_temp', act.beta_temp, g_xi['beta_temp']),
                      ('b_sm', act.b_sm, g_xi['b_sm']),
                      ('W_sm', act.W_sm, g_xi['W_sm'])]:
    n = arr.size
    idxs = [0, n // 2, n - 1]
    errs = []
    for i in idxs:
        gN = fd(arr, i); gAi = gA.reshape(-1)[i]
        errs.append(abs(gN - gAi) / max(abs(gN), abs(gAi), 1e-8))
    print(f"  {name}: max relerr {max(errs):.2e}")
    assert max(errs) < 1e-2, f"{name} gradcheck FAILED"

# ── also confirm λ + affine still gradcheck through the new head ──
for name, arr, gA in [('lam_y', act.lam_y, g_qc['lam_y']),
                      ('beta_enc', act.beta_enc, g_qc['beta_enc'])]:
    errs = [abs(fd(arr, i) - gA.reshape(-1)[i]) / max(abs(gA.reshape(-1)[i]), 1e-8)
            for i in [0, arr.size // 2]]
    print(f"  {name} (through SoftmaxPQC head): max relerr {max(errs):.2e}")
    assert max(errs) < 1e-2

print("\n✅ Δ4 SoftmaxPQC head + Δ2 R1 + Δ5 bypass: shapes, gradcheck (head+λ), forward OK")
