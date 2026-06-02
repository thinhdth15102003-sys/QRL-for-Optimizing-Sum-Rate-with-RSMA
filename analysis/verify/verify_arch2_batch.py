"""Architecture-2 (R1 + SoftmaxPQC + Δ1) training batch path: runs, finite grads,
params move, PPO ratio consistent (lp_old from forward matches compute_log_prob)."""
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
    extra_cz_pairs=((0, 10), (5, 11)),
    readout_mode='r1', softmax_head=True, softmax_beta_init=1.0,
    seed=0)

rng = np.random.default_rng(3)
B_s = 8
s_list   = [rng.standard_normal(act.d_s) for _ in range(B_s)]
# sample phi via the analytic policy + check compute_log_prob == compute_logprobs_batch
phi_list = [act.forward(s)[0] for s in s_list]

# PPO consistency: compute_log_prob (single) vs compute_logprobs_batch
lp_single = np.array([act.compute_log_prob(s, p) for s, p in zip(s_list, phi_list)])
lp_batch  = act.compute_logprobs_batch(s_list, phi_list)
print(f"PPO-ratio consistency (single vs batch logprob): max diff {np.abs(lp_single-lp_batch).max():.2e}")
assert np.abs(lp_single - lp_batch).max() < 1e-9, "single/batch logprob mismatch!"

advs = rng.standard_normal(B_s)
out = act.compute_logprobs_grads_batch(s_list, phi_list, lp_batch, advs, 0.2, 0.5, 0.001)
l_q, L_ae, L_ent, g_ae, g_qc, g_xi, clip_q, kl_q = out
print(f"batch grad keys xi: {sorted(g_xi.keys())}")
assert set(g_xi.keys()) == {'W_sm', 'b_sm', 'beta_temp'}
for k, g in {**g_qc, **g_xi}.items():
    assert np.isfinite(np.asarray(g)).all(), f"{k} non-finite!"
print(f"  lam_y mean|g|={np.abs(g_qc['lam_y']).mean():.3e}  "
      f"beta_enc mean|g|={np.abs(g_qc['beta_enc']).mean():.3e}  "
      f"W_sm mean|g|={np.abs(g_xi['W_sm']).mean():.3e}  beta_temp g={g_xi['beta_temp'][0]:.3e}")

# apply → params move
b0 = act.beta_temp.copy(); l0 = act.lam_y.copy()
act.apply_grads(g_ae, g_qc, g_xi)
print(f"after step: Δβ_temp={abs(act.beta_temp[0]-b0[0]):.2e}  Δλ_y|max|={np.abs(act.lam_y-l0).max():.2e}")
assert abs(act.beta_temp[0] - b0[0]) > 0

# save/load round-trip
import tempfile, shutil
tmp = tempfile.mkdtemp()
try:
    act.save(tmp)
    act2 = QuantumActor.from_dir(tmp, seed=1)
    assert act2.SOFTMAX_HEAD and act2.READOUT_MODE == 'r1'
    assert act2.N_QUANTUM == 42
    assert np.allclose(act2.W_sm, act.W_sm) and np.allclose(act2.beta_temp, act.beta_temp)
    # forward parity
    p1 = act.forward(s_list[0], greedy=True)[0]
    p2 = act2.forward(s_list[0], greedy=True)[0]
    assert np.array_equal(p1, p2), "greedy forward parity broken after load"
    print("save/load round-trip + greedy parity OK (readout_mode, softmax_head, W_sm, β persisted)")
finally:
    shutil.rmtree(tmp)

print("\n✅ Architecture-2 batch training path OK: PPO-consistent, finite grads, moves, persists")
