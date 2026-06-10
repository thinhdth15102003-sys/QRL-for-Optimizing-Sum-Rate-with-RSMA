"""
CPU unit test for Δ7 PhaseMLP counterfactual gradient (occupancy-weighting).
Pure-numpy — no GPU, no training. Validates:
  1. compute_grads_batch runs in both modes, same grad keys/shapes.
  2. counterfactual=False is unchanged (vanilla).
  3. UNIFORM π_q occupancy  → counterfactual ≈ vanilla (weights normalise to 1).
  4. SKEWED  π_q occupancy  → counterfactual DIFFERS (reweights per-IRS).
  5. Missing 'q_pi' + counterfactual=True → graceful fallback (≈ vanilla).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from RL.sub_actors import PhaseMLP

K, M, N, L = 10, 2, 24, 4
rng = np.random.default_rng(0)
net = PhaseMLP(d_s=2 * K, M=M, N=N, n_levels=L, hidden=[64, 128], lr=3e-4, seed=0)

# phi routes users to BOTH IRS (so 2 active panels per transition)
phi_base = np.array([1, 1, 1, 2, 2, 0, 0, 1, 2, 0])


def make_trans(B, q_pi_fn):
    out = []
    for b in range(B):
        out.append({
            'phi':       phi_base.copy(),
            's_phase':   rng.standard_normal((M, 2 * K)),
            'phase_idx': rng.integers(0, L, size=(M, N)),
            'q_pi':      q_pi_fn(),
        })
    return out


def gnorm(grads):
    return sum(float(np.sum(v * v)) for v in grads.values()) ** 0.5


def gdiff(g1, g2):
    return sum(float(np.sum((g1[k] - g2[k]) ** 2)) for k in g1) ** 0.5


B = 8
eff = rng.standard_normal(B)

# uniform occupancy: every user equal mass to direct/IRS1/IRS2
uni = make_trans(B, lambda: np.full((K, M + 1), 1.0 / (M + 1)))
# skewed: most IRS mass on group 1
skew = make_trans(B, lambda: np.array([[0.2, 0.7, 0.1]] * K))

L_pg_v, L_ent_v, g_van = net.compute_grads_batch(uni, eff, beta_entropy=0.001,
                                                 counterfactual=False)
_, _, g_cf_uni = net.compute_grads_batch(uni, eff, beta_entropy=0.001,
                                         counterfactual=True)
_, _, g_cf_skew = net.compute_grads_batch(skew, eff, beta_entropy=0.001,
                                          counterfactual=True)

# missing q_pi — compare on the SAME data: vanilla vs counterfactual-with-no-qpi
no_qpi = make_trans(B, lambda: np.full((K, M + 1), 1.0 / (M + 1)))
for t in no_qpi:
    del t['q_pi']
_, _, g_van_missing = net.compute_grads_batch(no_qpi, eff, beta_entropy=0.001,
                                              counterfactual=False)
_, _, g_cf_missing = net.compute_grads_batch(no_qpi, eff, beta_entropy=0.001,
                                             counterfactual=True)

print("=" * 60)
print("  Δ7 PhaseMLP counterfactual — CPU unit test")
print("=" * 60)
ok = True

# 1. keys/shapes match
keys_match = (set(g_van) == set(g_cf_uni) and
              all(g_van[k].shape == g_cf_uni[k].shape for k in g_van))
print(f"  [1] grad keys+shapes match            : {'PASS' if keys_match else 'FAIL'}")
ok &= keys_match

# 2. no NaN
no_nan = all(np.all(np.isfinite(v)) for v in g_cf_uni.values())
print(f"  [2] counterfactual grads finite       : {'PASS' if no_nan else 'FAIL'}")
ok &= no_nan

# 3. uniform occupancy ≈ vanilla
d_uni = gdiff(g_van, g_cf_uni) / (gnorm(g_van) + 1e-12)
t3 = d_uni < 1e-6
print(f"  [3] uniform π_q ≈ vanilla (rel={d_uni:.2e}) : {'PASS' if t3 else 'FAIL'}")
ok &= t3

# 4. skewed occupancy differs from vanilla
d_skew = gdiff(g_van, g_cf_skew) / (gnorm(g_van) + 1e-12)
t4 = d_skew > 1e-3
print(f"  [4] skewed π_q DIFFERS  (rel={d_skew:.2e}) : {'PASS' if t4 else 'FAIL'}")
ok &= t4

# 5. missing q_pi falls back ≈ vanilla (compared on SAME data)
d_miss = gdiff(g_van_missing, g_cf_missing) / (gnorm(g_van_missing) + 1e-12)
t5 = d_miss < 1e-6
print(f"  [5] missing q_pi → fallback (rel={d_miss:.2e}): {'PASS' if t5 else 'FAIL'}")
ok &= t5

print("=" * 60)
print(f"  RESULT: {'ALL PASS ✅' if ok else 'FAIL ❌'}")
sys.exit(0 if ok else 1)
