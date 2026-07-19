"""probe_variable_b_seats.py — VARIABLE-B admission capacity (seat law) MC validation.

Theory (docs/IRS-Group-Capacity-Ablation.md §3, 2026-07-18):
  Group of n members (Γ_i = Γ̄_B · X_i, X~Exp(1)) is QoS-feasible iff
      Σ_i ς_i ≤ R_c(min_i Γ_i),   ς_i = (D − R_p(Γ_i))⁺        [general Σ-shortfall bound §1]
  n*(draw) = max feasible n.  With B0 seats already occupied (exchangeable members),
  additional admissions n_add*(B0) = n* − B0  →  SEAT LAW: ∂n_add*/∂B0 = −1 (homogeneous).
  Heterogeneous best-first candidates (each next mean worse) break the law: n_add* falls FASTER.

Rates (§1 equal-power reference, ρ=0.2, K=10, G+1=2 groups):
  x = α(1−ρ) = 0.08,  y = β_g ρ = 0.1,  c_p = 1 − x − y = 0.82
  R_c(Γ) = log2(1+Γ) − log2(1+(1−y)Γ)
  R_p(Γ) = log2(1+(1−y)Γ) − log2(1+c_p·Γ)

Pure numpy — no env dependency (Γ-form ≡ code already validated to 1e-16 by
analysis/probe_bstar_confirm.py rig; this probe validates the NEW seat-law layer).
Run:  python3.11 analysis/probe_variable_b_seats.py
"""
import numpy as np

X_FRAC, Y_FRAC = 0.08, 0.1
C_P = 1.0 - X_FRAC - Y_FRAC
D_K = 0.1
N_MAX = 8
N_MC = 20000
RNG = np.random.default_rng(42)

# regime medians from worksheet §2(2) (probe_regime_check, N=24 reference)
REGIMES = {"P50": (703.0, 14.0), "P40": (70.0, 1.4), "P30": (7.0, 0.14)}


def r_c(g):
    return np.log2(1 + g) - np.log2(1 + (1 - Y_FRAC) * g)


def r_p(g):
    return np.log2(1 + (1 - Y_FRAC) * g) - np.log2(1 + C_P * g)


def n_star_per_draw(gbar_seq, n_mc=N_MC):
    """gbar_seq: per-slot mean SNR (len N_MAX, admission order). Returns n*(draw) array."""
    k = len(gbar_seq)
    x = RNG.exponential(size=(n_mc, k)) * np.asarray(gbar_seq)[None, :]
    rp = r_p(x)
    sf = np.maximum(0.0, D_K - rp)                      # shortfall ς_i
    cum_sf = np.cumsum(sf, axis=1)                      # Σ_{i≤n} ς_i
    run_min = np.minimum.accumulate(x, axis=1)          # min_{i≤n} Γ_i
    feas = cum_sf <= r_c(run_min)                       # feasible(n), nested prefix
    nstar = np.where(feas.any(axis=1),
                     np.max(np.where(feas, np.arange(1, k + 1)[None, :], 0), axis=1), 0)
    return nstar


def mode(a):
    v, c = np.unique(a, return_counts=True)
    return int(v[np.argmax(c)])


print(f"MC draws = {N_MC} · n_max = {N_MAX} · D = {D_K} · (x, y) = ({X_FRAC}, {Y_FRAC})")

print("\n== 1) HOMOGENEOUS n* (all slots Γ̄_B) + N*-scaling (Γ̄ × (N*/24)²) ==")
print(f"{'regime':6} {'N*':>4} {'Γ̄_B':>9} | n* mode/mean | seat check n_add*(B0)=n*−B0 (B0=0..3)")
for reg, (gbar_b, gbar_dir) in REGIMES.items():
    for nstar_el, scale in ((12, 0.25), (24, 1.0), (48, 4.0)):
        gb = gbar_b * scale
        ns = n_star_per_draw([gb] * N_MAX)
        adds = [f"{max(mode(ns) - b0, 0)}" for b0 in range(4)]
        print(f"{reg:6} {nstar_el:>4} {gb:>9.1f} | {mode(ns)} / {ns.mean():.2f}   | " + " ".join(adds))

print("\n== 2) RATE-SIDE sign (corner rule): P[R_p(Γ_B·X) > R_p(Γ_dir·X')] ==")
for reg, (gbar_b, gbar_dir) in REGIMES.items():
    xb = RNG.exponential(size=N_MC) * gbar_b
    xd = RNG.exponential(size=N_MC) * gbar_dir
    frac = float(np.mean(r_p(xb) > r_p(xd)))
    print(f"{reg:6}: frac positive = {frac:.3f}  (>0.5 → fill-to-cliff; <0.5 → B*_rate=0)")

print("\n== 3) HETEROGENEOUS best-first (means Γ̄_B·[1, .5, .25, .125, ...]) — seat law BREAKS ==")
print(f"{'regime':6} {'B0':>3} | n_add* mode/mean | homogeneous-law prediction (n*_hom − B0)")
for reg, (gbar_b, gbar_dir) in REGIMES.items():
    means = [gbar_b * (0.5 ** j) for j in range(N_MAX)]
    ns_het = n_star_per_draw(means)
    ns_hom = n_star_per_draw([gbar_b] * N_MAX)
    for b0 in range(4):
        n_add = np.maximum(ns_het - b0, 0)
        pred = max(mode(ns_hom) - b0, 0)
        print(f"{reg:6} {b0:>3} | {mode(n_add)} / {n_add.mean():.2f}        | {pred}")

print("\n== 4) SANITY: E[m_n] MC vs exact 1/Λ = Γ̄_B/n (homogeneous) ==")
gb = REGIMES["P40"][0]
x = RNG.exponential(size=(N_MC, N_MAX)) * gb
rm = np.minimum.accumulate(x, axis=1)
for n in (1, 2, 4, 8):
    print(f"n={n}: MC E[m]={rm[:, n-1].mean():9.3f}  exact={gb/n:9.3f}")
