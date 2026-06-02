"""Confirm the training batch path (compute_logprobs_grads_batch) runs with Δ1
affine + produces finite gamma_enc/beta_enc grads. FP32 + SPSA (real config)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
from params import make_config
from RL.quantum_actor import QuantumActor

cfg = make_config()
act = QuantumActor(
    cfg=cfg, n_qubits=12, n_latent=24,
    n_hidden_ae=[128, 64], n_hidden_post=[256, 128], n_var_layers=3,
    n_shots=1500, lr_ae=1e-4, lr_qc=1e-4, lr_xi=3e-4,
    data_reuploading=True, ae_pretrain_lr=1e-3, spsa_n_reps=16,
    extra_cz_pairs=((0, 10), (5, 11)), extra_zz_pairs=((0, 10), (5, 11)),
    seed=0)

rng = np.random.default_rng(2)
B_s = 8
s_list   = [rng.standard_normal(act.d_s) for _ in range(B_s)]
phi_list = [rng.integers(0, act.n_choices, act.K) for _ in range(B_s)]
lp_old   = np.zeros(B_s)
advs     = rng.standard_normal(B_s)

out = act.compute_logprobs_grads_batch(
    s_list, phi_list, lp_old, advs, 0.2, 0.5, 0.001)
# returns: l_q_arr, L_ae, L_ent, g_ae, g_qc, g_xi, clip_q, kl_q
l_q_arr, L_ae, L_ent, g_ae, g_qc, g_xi, clip_q, kl_q = out
print(f"batch method ran. g_qc keys: {sorted(g_qc.keys())}")
for k in ('gamma_enc', 'beta_enc', 'lam_y', 'lam_z'):
    g = g_qc[k]
    print(f"  {k}: shape={g.shape} mean|g|={np.abs(g).mean():.3e} finite={np.isfinite(g).all()}")
    assert np.isfinite(g).all(), f"{k} has non-finite grads!"
assert g_qc['gamma_enc'].shape == (act.N_LATENT,)
assert g_qc['beta_enc'].shape == (act.N_LATENT,)
assert np.abs(g_qc['lam_y']).mean() > 1e-9, "lam_y batch grad ~0!"

# Apply grads once → confirm params move (opt step works on new keys)
lam0 = act.lam_y.copy(); beta0 = act.beta_enc.copy()
act.apply_grads(g_ae, g_qc, g_xi)
print(f"\nafter 1 opt step:  Δλ_y|max|={np.abs(act.lam_y-lam0).max():.2e}  "
      f"Δβ_enc|max|={np.abs(act.beta_enc-beta0).max():.2e}")
assert np.abs(act.lam_y - lam0).max() > 0, "lam_y did not move!"
assert np.abs(act.beta_enc - beta0).max() > 0, "beta_enc did not move!"

# save/load round-trip includes gamma/beta
import tempfile, shutil, os
tmp = tempfile.mkdtemp()
try:
    act.save(tmp)
    act2 = QuantumActor.from_dir(tmp, cfg=cfg, seed=1)
    assert np.allclose(act2.gamma_enc, act.gamma_enc), "gamma_enc not persisted!"
    assert np.allclose(act2.beta_enc, act.beta_enc), "beta_enc not persisted!"
    print("save/load round-trip: gamma_enc, beta_enc persisted ✓")
finally:
    shutil.rmtree(tmp)

print("\n✅ Δ1 batch training path OK: runs, finite grads, params move, persists")
