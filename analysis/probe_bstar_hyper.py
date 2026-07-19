"""probe_bstar_hyper.py — B*/n* vs HYPERPARAMETERS: P_S x N x C_k-policy x drop-percent d.

Extends probe_variable_b_seats.py (2026-07-18) per user request 07-19:
big-picture map of group capacity vs the hypers that enter it.

Model (docs/IRS-Group-Capacity-Ablation.md §1, equal-power reference x=0.08, y=0.1):
  Γ_i = Γ̄(P_S, N) · X_i,  X~Exp(1) iid,  Γ̄ ∝ N² · P_S_linear  (anchor: Γ̄=703 @P50,N24; Γ_dir=14)
  R_c(Γ) = log2(1+Γ) − log2(1+(1−y)Γ)          [budget; saturates at log2(1/(1−y))]
  R_p(Γ) = log2(1+(1−y)Γ) − log2(1+c_p Γ)      [private; saturates]
  ς_i = (D − R_p(Γ_i))⁺                         [shortfall]

C_k policies (the LAST decision variable in the chain):
  general LP  : feasible(n) ⟺ Σ_{i≤n} ς_i ≤ R_c(Γ_min)     [ASSUME-optimal, exact by construction]
  equal-share : feasible(n) ⟺ n · max_{i≤n} ς_i ≤ R_c(Γ_min) [C_k = R_c/n, binds at worst member]

Drop-percent d (corner rule, ties to capacity band [R*(0), R*(d)]):
  may drop up to ⌈d·n⌉ worst-Γ members → served min = (⌈dn⌉+1)-th order statistic;
  Σς / max ς taken over SERVED members only.
  Homogeneous Exp check: E[X_(j+1)] = Γ̄ · Σ_{i=0}^{j} 1/(n−i).

Run:  python3.11 analysis/probe_bstar_hyper.py
"""
import numpy as np

X_FRAC, Y_FRAC = 0.08, 0.1
C_P = 1.0 - X_FRAC - Y_FRAC
D_K = 0.1
N_MAX = 16          # > K để thấy trần lý thuyết không bị censor
N_MC = 20000
RNG = np.random.default_rng(42)

GBAR_P50_N24, GDIR_P50 = 703.0, 14.0
PS_SCALE = {"P50": 1.0, "P40": 0.1, "P30": 0.01}
N_SCALE = {12: 0.25, 24: 1.0, 48: 4.0}
DROPS = [0.0, 0.1, 0.2, 0.3]


def r_c(g):
    return np.log2(1 + g) - np.log2(1 + (1 - Y_FRAC) * g)


def r_p(g):
    return np.log2(1 + (1 - Y_FRAC) * g) - np.log2(1 + C_P * g)


def n_star(gbar, d, policy, n_mc=N_MC):
    """n*(draw) = max feasible n ≤ N_MAX, dropping ⌈d·n⌉ worst-Γ members."""
    x = RNG.exponential(size=(n_mc, N_MAX)) * gbar
    xs = np.sort(x, axis=1)                       # ascending per draw (prefix n = first n? NO)
    # membership is the first-n draws (admission order irrelevant for iid); for each n use
    # the n FIRST columns of the unsorted draw, then sort within.
    nstar = np.zeros(n_mc, dtype=int)
    for n in range(1, N_MAX + 1):
        sub = np.sort(x[:, :n], axis=1)           # served candidates ascending
        j = int(np.ceil(d * n)) if d > 0 else 0
        j = min(j, n - 1)                         # phải serve ≥1
        served = sub[:, j:]                       # drop j worst-Γ
        gmin = served[:, 0]
        sf = np.maximum(0.0, D_K - r_p(served))
        if policy == "general":
            feas = sf.sum(axis=1) <= r_c(gmin)
        else:                                     # equal-share
            m = served.shape[1]
            feas = m * sf.max(axis=1) <= r_c(gmin)
        nstar = np.where(feas, n, nstar)
    return nstar


def mode(a):
    v, c = np.unique(a, return_counts=True)
    return int(v[np.argmax(c)])


print(f"MC={N_MC} n_max={N_MAX} D={D_K} (x,y)=({X_FRAC},{Y_FRAC})  |  R_c(inf)={np.log2(1/(1-Y_FRAC)):.4f}")

print("\n== 1) SANITY: E[served-min] MC vs exact Γ̄·Σ_{i=0}^{j} 1/(n−i)  (P40, N24, n=8) ==")
gb = GBAR_P50_N24 * PS_SCALE["P40"]
x8 = np.sort(RNG.exponential(size=(N_MC, 8)) * gb, axis=1)
for d in DROPS:
    j = min(int(np.ceil(d * 8)), 7)
    exact = gb * sum(1.0 / (8 - i) for i in range(j + 1))
    print(f"  d={d:.1f} (j={j}): MC={x8[:, j].mean():8.3f}  exact={exact:8.3f}")

print("\n== 2) MAIN GRID: n* (mode/mean) per P_S x N x policy x d ==")
hdr = "  ".join(f"d={d:.0%}" for d in DROPS)
print(f"{'P_S':4} {'N':>3} {'policy':11} | {hdr}   (two-point n*_tp)")
for ps, s_ps in PS_SCALE.items():
    for n_el, s_n in N_SCALE.items():
        gbar = GBAR_P50_N24 * s_ps * s_n
        sf_tp = max(0.0, D_K - r_p(np.array([gbar]))[0])
        ntp = int(r_c(np.array([gbar]))[0] / sf_tp) if sf_tp > 0 else N_MAX
        for pol in ("general", "equal-share"):
            cells = []
            for d in DROPS:
                ns = n_star(gbar, d, pol)
                cells.append(f"{mode(ns):2d}/{ns.mean():5.2f}")
            print(f"{ps:4} {n_el:>3} {pol:11} | " + "  ".join(cells) + f"   (tp={min(ntp, N_MAX)})")

print("\n== 3) C_k-POLICY GAP vs GROUP SIZE (P40, N24, d=0): P[feasible(n)] per policy ==")
gb = GBAR_P50_N24 * 0.1
x = RNG.exponential(size=(N_MC, N_MAX)) * gb
print(f"{'n':>3} {'P_gen':>7} {'P_eq':>7} {'gap':>7}  E[max ς]/E[Σς/n]")
for n in (2, 4, 6, 8, 12, 16):
    sub = np.sort(x[:, :n], axis=1)
    gmin = sub[:, 0]
    sf = np.maximum(0.0, D_K - r_p(sub))
    p_gen = float(np.mean(sf.sum(axis=1) <= r_c(gmin)))
    p_eq = float(np.mean(n * sf.max(axis=1) <= r_c(gmin)))
    ratio = float(sf.max(axis=1).mean() / max(sf.mean(), 1e-12))
    print(f"{n:>3} {p_gen:7.3f} {p_eq:7.3f} {p_gen-p_eq:7.3f}  {ratio:6.2f}")
