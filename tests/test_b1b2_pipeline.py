"""
test_b1b2_pipeline.py
---------------------
Full pipeline test for B1 (full ZZ observables) + B2 (dual-branch AE).

Checks
------
  1. Case 2 (K=10, M=2, nq=12, n_latent=24) with B4 extra_zz_pairs
  2. d_s = K*(M+2)+2*M = 44 (affinity ‖ D_k ‖ |g_SU| ‖ irs_feat)
  3. extract_state() shape (44,)
  4. forward() end-to-end (o_hat, z_t shapes)
  5. compute_grads()       single-sample backward
  6. compute_grads_batch() mini-batch backward — no shape errors
  7. compute_logprobs_grads_batch() fused PPO path
  8. pretrain_ae_step() AE hot-start
  9. B1 mode: full_zz_pairs active — N_QUANTUM = nq + 29 = 41
 10. B1 compute_grads_batch shapes

Run from project root:
    python test_b1b2_pipeline.py
"""

# ── path bootstrap: make project root importable when run as script ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import sys, time
sys.path.insert(0, r'C:\Project\IRS-assisted RSMA Quantum-RL')

import numpy as np
from RL.quantum_actor import QuantumActor
from RL.quantum_circuit import GPU_BACKEND

PASS  = "\033[32mPASS\033[0m"
FAIL  = "\033[31mFAIL\033[0m"

# ── Case 2 architecture dimensions ──────────────────────────────────────────
K, M, nq, n_latent = 10, 2, 12, 24
B_s = 8   # mini-batch size for batch tests
N_HIDDEN_AE   = [256, 128, 64]
N_HIDDEN_POST = [256, 128]

# d_aff = K*(M+2) + 2*M  (affinity ‖ D_k ‖ |g_SU| ‖ irs_feat)
D_AFF = K * (M + 2) + 2 * M

# B4: N_QUANTUM = 2*nq-1 + len(extra_zz)
EXTRA_ZZ = ((0, 10), (5, 11))         # Case 2 cross-block pairs
EXTRA_CZ = ((0, 10), (5, 11))
N_Q_B4 = 2 * nq - 1 + len(EXTRA_ZZ)  # 25

# B1: full_zz — (nq-M-1) user-NN + (nq-M)*M cross pairs
_nu = nq - M  # 10
FULL_ZZ = (tuple((i, i+1) for i in range(_nu - 1))   # 9 user-NN
           + tuple((u, nq - M + j)
                   for u in range(_nu) for j in range(M)))  # 20 cross
N_Q_B1  = nq + len(FULL_ZZ)           # 12 + 29 = 41

print(f"GPU backend: {GPU_BACKEND}")
print(f"D_AFF={D_AFF}  N_Q_B4={N_Q_B4}  N_Q_B1={N_Q_B1}  len(FULL_ZZ)={len(FULL_ZZ)}")
print()


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Cfg:
    """Minimal config stand-in."""
    pass

def make_cfg(k, m):
    c = _Cfg(); c.K = k; c.M = m
    return c

def mock_obs(rng):
    return {
        'g_SR': rng.standard_normal(M) + 1j * rng.standard_normal(M),
        'g_RU': rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K)),
        'g_SU': rng.standard_normal(K) + 1j * rng.standard_normal(K),
    }

def make_actor(full_zz=(), extra_cz=EXTRA_CZ, extra_zz=EXTRA_ZZ,
               seed=0, n_shots=50, spsa_n_reps=0):
    return QuantumActor(
        cfg=make_cfg(K, M),
        n_qubits=nq, n_latent=n_latent,
        n_hidden_ae=N_HIDDEN_AE, n_hidden_post=N_HIDDEN_POST,
        n_var_layers=3, n_shots=n_shots,
        lr_ae=1e-4, lr_qc=1e-4, lr_xi=3e-4,
        data_reuploading=True,
        ae_pretrain_lr=1e-3,
        spsa_n_reps=spsa_n_reps,
        extra_cz_pairs=extra_cz,
        extra_zz_pairs=extra_zz,
        full_zz_pairs=full_zz,
        seed=seed,
    )

def check(name, cond, detail=''):
    if cond:
        print(f"  [{PASS}] {name}")
    else:
        print(f"  [{FAIL}] {name}  {detail}")
    return cond

errors = []

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: B4 mode (default)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 1 — B4 mode (Case 2, extra_zz_pairs)")
print("=" * 60)

rng = np.random.default_rng(0)
actor = make_actor()  # B4 default

# 1a. dimension checks
ok = check("d_s == D_AFF", actor.d_s == D_AFF,
           f"got {actor.d_s}")
if not ok: errors.append("d_s")

ok = check("N_QUANTUM == N_Q_B4", actor.N_QUANTUM == N_Q_B4,
           f"got {actor.N_QUANTUM}")
if not ok: errors.append("N_QUANTUM B4")

# 1b. extract_state shape
obs = mock_obs(rng)
demand = np.full(K, 0.1)
blocked = np.zeros(K, dtype=bool)
s_t = actor.extract_state(obs, demand, blocked)
ok = check("extract_state shape", s_t.shape == (D_AFF,),
           f"got {s_t.shape}")
if not ok: errors.append("extract_state")

# 1c. forward
phi, lp, info = actor.forward(s_t)
ok = check("forward z_t shape",   info['z_t'].shape == (n_latent,))
ok = check("forward o_hat shape", info['o_hat'].shape == (N_Q_B4,),
           f"got {info['o_hat'].shape}")
ok = check("forward phi shape",   phi.shape == (K,))
if not ok: errors.append("forward B4")

# 1d. compute_grads (single-sample)
t0 = time.time()
L_pg, L_ae, L_ent, g_ae, g_qc, g_xi = actor.compute_grads(
    s_t, phi, advantage=1.0, ae_weight=0.5, beta_entropy=0.05)
dt = time.time() - t0
ok = check("compute_grads scalars",
           np.isfinite(L_pg) and np.isfinite(L_ae) and np.isfinite(L_ent),
           f"pg={L_pg:.3f} ae={L_ae:.3f} ent={L_ent:.3f}")
ok = check("compute_grads shapes",
           g_qc['theta_y'].shape == (3, nq) and g_qc['lam_y'].shape == (nq,))
print(f"    (single-sample compute_grads: {dt:.2f}s)")

# 1e. compute_grads_batch
s_t_list   = [actor.extract_state(mock_obs(rng), demand, blocked) for _ in range(B_s)]
phi_list   = [actor.forward(s)[0] for s in s_t_list]
adv_arr    = rng.standard_normal(B_s)

t0 = time.time()
out = actor.compute_grads_batch(s_t_list, phi_list, adv_arr,
                                ae_weight=0.5, beta_entropy=0.05)
dt = time.time() - t0
L_pg_b, L_ae_b, L_ent_b, g_ae_b, g_qc_b, g_xi_b = out
ok = check("compute_grads_batch scalars",
           np.isfinite(L_pg_b) and np.isfinite(L_ae_b) and np.isfinite(L_ent_b))
ok = check("compute_grads_batch theta_y shape",
           g_qc_b['theta_y'].shape == (3, nq))
print(f"    (batch={B_s} compute_grads_batch: {dt:.2f}s)")

# 1f. compute_logprobs_grads_batch (fused PPO path)
lp_old = np.array([actor.compute_log_prob(s, p)
                   for s, p in zip(s_t_list, phi_list)])
t0 = time.time()
out_fused = actor.compute_logprobs_grads_batch(
    s_t_list, phi_list, lp_old, adv_arr,
    ppo_epsilon=0.2, ae_weight=0.5, beta_entropy=0.05)
dt = time.time() - t0
l_q, L_ae_f, L_ent_f, g_ae_f, g_qc_f, g_xi_f, clip_q_f, kl_q_f = out_fused
ok = check("fused PPO l_q shape",   l_q.shape == (B_s,))
ok = check("fused PPO ae finite",   np.isfinite(L_ae_f))
print(f"    (fused PPO compute_logprobs_grads_batch: {dt:.2f}s)")

# 1g. pretrain_ae_step
L_ae_pre = actor.pretrain_ae_step(s_t)
ok = check("pretrain_ae_step finite", np.isfinite(L_ae_pre),
           f"L_ae={L_ae_pre:.3f}")

# 1h. apply_grads (no crash)
try:
    actor.apply_grads(g_ae_b, g_qc_b, g_xi_b)
    check("apply_grads no crash", True)
except Exception as e:
    check("apply_grads no crash", False, str(e))
    errors.append("apply_grads")

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: B1 mode (full_zz_pairs)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 2 — B1 mode (full_zz_pairs, no extra_zz)")
print("=" * 60)

actor_b1 = make_actor(full_zz=FULL_ZZ, extra_cz=EXTRA_CZ, extra_zz=())

ok = check("N_QUANTUM == N_Q_B1", actor_b1.N_QUANTUM == N_Q_B1,
           f"got {actor_b1.N_QUANTUM}")
if not ok: errors.append("N_QUANTUM B1")

ok = check("d_s unchanged", actor_b1.d_s == D_AFF)

# forward B1
phi_b1, lp_b1, info_b1 = actor_b1.forward(s_t)
ok = check("forward B1 o_hat shape", info_b1['o_hat'].shape == (N_Q_B1,),
           f"got {info_b1['o_hat'].shape}")

# compute_grads B1 single
L_pg1, L_ae1, L_ent1, g_ae1, g_qc1, g_xi1 = actor_b1.compute_grads(
    s_t, phi_b1, advantage=1.0, ae_weight=0.5, beta_entropy=0.05)
ok = check("B1 compute_grads finite",
           np.isfinite(L_pg1) and np.isfinite(L_ae1))

# compute_grads_batch B1
phi_list_b1 = [actor_b1.forward(s)[0] for s in s_t_list]
t0 = time.time()
out_b1 = actor_b1.compute_grads_batch(s_t_list, phi_list_b1, adv_arr,
                                      ae_weight=0.5, beta_entropy=0.05)
dt = time.time() - t0
L_pg_b1, L_ae_b1, L_ent_b1, g_ae_b1, g_qc_b1, g_xi_b1 = out_b1
ok = check("B1 compute_grads_batch finite",
           np.isfinite(L_pg_b1) and np.isfinite(L_ae_b1))
ok = check("B1 theta_y grad shape", g_qc_b1['theta_y'].shape == (3, nq))
print(f"    (batch={B_s} B1 compute_grads_batch: {dt:.2f}s)")

# fused PPO path B1
lp_old_b1 = np.array([actor_b1.compute_log_prob(s, p)
                      for s, p in zip(s_t_list, phi_list_b1)])
out_fused_b1 = actor_b1.compute_logprobs_grads_batch(
    s_t_list, phi_list_b1, lp_old_b1, adv_arr,
    ppo_epsilon=0.2, ae_weight=0.5, beta_entropy=0.05)
l_q_b1 = out_fused_b1[0]
ok = check("B1 fused PPO shape", l_q_b1.shape == (B_s,))

print()

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: SPSA mode (B4, spsa_n_reps=2)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TEST 3 — SPSA gradient estimator (B4, n_reps=2)")
print("=" * 60)

actor_spsa = make_actor(spsa_n_reps=2)
out_spsa = actor_spsa.compute_grads_batch(
    s_t_list, phi_list, adv_arr, ae_weight=0.5, beta_entropy=0.05)
L_pg_sp, L_ae_sp, _, g_ae_sp, g_qc_sp, _ = out_spsa
ok = check("SPSA batch finite", np.isfinite(L_pg_sp) and np.isfinite(L_ae_sp))
ok = check("SPSA theta_y shape", g_qc_sp['theta_y'].shape == (3, nq))

print()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
if not errors:
    print(f"\033[32mALL TESTS PASSED\033[0m")
else:
    print(f"\033[31mFAILED: {errors}\033[0m")
    sys.exit(1)
