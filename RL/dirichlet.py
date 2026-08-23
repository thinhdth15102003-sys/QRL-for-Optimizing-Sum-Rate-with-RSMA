"""
RL/dirichlet.py
---------------
Dirichlet policy primitives for the SIMPLEX-VALUED sub-actors (power split /
common-distribution / private-distribution / per-group C_k shares).

WHY THIS EXISTS
---------------
PowerMLP and CkMLP allocate a budget across slots, i.e. their action lives on a
probability simplex. The pre-2026-07-21 implementation sampled a CATEGORICAL
INDEX for the PPO log-prob while EXECUTING the softmax probability vector:

    w_p       = softmax(logits) * budget          # executed
    a_private = rng.choice(K, p=softmax(logits))  # scored by PPO

The executed allocation did not depend on the sampled index, so the reward was
independent of the action PPO was scoring. For a fixed state that makes the
policy gradient exactly zero in expectation:

    E_a[∇ log π(a|s)] = -(p - p) = 0

(measured: ‖mean grad‖/‖per-sample grad‖ = 0.0093 over 20k samples at a fixed
state and fixed advantage, versus 0.0071 for pure noise). Those heads could
only ever be moved by the entropy bonus — which is why power "was not a lever"
and why C_k logits random-walked into explosions.

THE FIX
-------
Make the SAMPLED action BE the executed allocation: draw the allocation vector
itself from Dirichlet(α), with α produced by the network. The Dirichlet has an
analytic log-density and entropy, so PPO's score-function gradient works
unchanged — and now the reward genuinely depends on the sampled action.

Gumbel-softmax / reparameterisation is NOT usable here: it needs
∂reward/∂action, and the rate model (CSI/rate.py) is non-differentiable numpy.

PARAMETERISATION
----------------
    α_i = softplus(z_i) + α_min           (α_i > 0 required)
    mean[x_i] = α_i / α_0,  α_0 = Σ α_i

So the network's logits still control the MEAN allocation much as the old
softmax did, while α_0 controls spread (large α_0 → concentrated around the
mean → low entropy). Entropy annealing acts on α_0 naturally.

FORMULAS (all implemented and numerically gradient-checked in tests)
    log p(x|α) = logΓ(α_0) − Σ logΓ(α_i) + Σ (α_i − 1) log x_i
    ∂log p/∂α_i = ψ(α_0) − ψ(α_i) + log x_i
    H(α)        = log B(α) + (α_0 − k)ψ(α_0) − Σ (α_i − 1)ψ(α_i)
    ∂H/∂α_i     = (α_0 − k)ψ'(α_0) − (α_i − 1)ψ'(α_i)
    ∂α_i/∂z_i   = sigmoid(z_i)
"""

import numpy as np
from scipy.special import digamma, gammaln, polygamma

# Floor on each concentration. Keeps α > 0, bounds the log-density, and stops a
# collapsing head from driving the distribution onto the simplex corners.
ALPHA_MIN = 0.10
# Sampled coordinates are clipped away from 0 before log(): a Dirichlet draw can
# underflow to exactly 0 for small α, which would make log p = -inf.
X_MIN = 1e-9


# ── parameterisation ─────────────────────────────────────────────────────────

def softplus(z: np.ndarray) -> np.ndarray:
    """log(1+e^z), overflow-safe."""
    return np.logaddexp(0.0, z)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def logits_to_conc(z: np.ndarray, alpha_min: float = ALPHA_MIN) -> np.ndarray:
    """α = softplus(z) + α_min  (elementwise, α > 0)."""
    return softplus(z) + alpha_min


def dconc_dlogits(z: np.ndarray) -> np.ndarray:
    """∂α/∂z = sigmoid(z)."""
    return sigmoid(z)


# ── sampling ─────────────────────────────────────────────────────────────────

def sample(rng, alpha: np.ndarray) -> np.ndarray:
    """
    Draw x ~ Dirichlet(α), clipped away from the boundary and renormalised so
    log x is always finite. Shape follows `alpha`; last axis is the simplex.
    """
    x = rng.dirichlet(np.asarray(alpha, dtype=float))
    x = np.maximum(x, X_MIN)
    return x / x.sum(axis=-1, keepdims=True)


# ── log-density ──────────────────────────────────────────────────────────────

def log_prob(x: np.ndarray, alpha: np.ndarray) -> float:
    """log p(x | α) for a single simplex point."""
    x = np.maximum(np.asarray(x, dtype=float), X_MIN)
    a = np.asarray(alpha, dtype=float)
    a0 = a.sum()
    return float(gammaln(a0) - gammaln(a).sum() + ((a - 1.0) * np.log(x)).sum())


def log_prob_batch(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Row-wise log p(X[b] | A[b]) for (B, k) inputs → (B,)."""
    X = np.maximum(np.asarray(X, dtype=float), X_MIN)
    A = np.asarray(A, dtype=float)
    a0 = A.sum(axis=-1)
    return (gammaln(a0) - gammaln(A).sum(axis=-1)
            + ((A - 1.0) * np.log(X)).sum(axis=-1))


def dlogp_dconc(x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """∂ log p(x|α) / ∂α_i = ψ(α_0) − ψ(α_i) + log x_i.  Shape = alpha."""
    x = np.maximum(np.asarray(x, dtype=float), X_MIN)
    a = np.asarray(alpha, dtype=float)
    a0 = a.sum(axis=-1, keepdims=True)
    return digamma(a0) - digamma(a) + np.log(x)


# ── entropy ──────────────────────────────────────────────────────────────────

def entropy(alpha: np.ndarray) -> float:
    """H(Dirichlet(α)) for a single α."""
    a = np.asarray(alpha, dtype=float)
    a0, k = a.sum(), a.shape[-1]
    log_B = gammaln(a).sum() - gammaln(a0)
    return float(log_B + (a0 - k) * digamma(a0)
                 - ((a - 1.0) * digamma(a)).sum())


def entropy_batch(A: np.ndarray) -> np.ndarray:
    """Row-wise entropy for (B, k) → (B,)."""
    A = np.asarray(A, dtype=float)
    a0 = A.sum(axis=-1)
    k = A.shape[-1]
    log_B = gammaln(A).sum(axis=-1) - gammaln(a0)
    return (log_B + (a0 - k) * digamma(a0)
            - ((A - 1.0) * digamma(A)).sum(axis=-1))


def dentropy_dconc(alpha: np.ndarray) -> np.ndarray:
    """∂H/∂α_i = (α_0 − k)ψ'(α_0) − (α_i − 1)ψ'(α_i).  Shape = alpha."""
    a = np.asarray(alpha, dtype=float)
    a0 = a.sum(axis=-1, keepdims=True)
    k = a.shape[-1]
    tri_a0 = polygamma(1, a0)
    tri_ai = polygamma(1, a)
    return (a0 - k) * tri_a0 - (a - 1.0) * tri_ai


# ── convenience: full logits → (α, sample, logp) in one step ─────────────────

def sample_from_logits(rng, z: np.ndarray, alpha_min: float = ALPHA_MIN):
    """Returns (x, alpha, log_prob) for logits z."""
    a = logits_to_conc(z, alpha_min)
    x = sample(rng, a)
    return x, a, log_prob(x, a)
