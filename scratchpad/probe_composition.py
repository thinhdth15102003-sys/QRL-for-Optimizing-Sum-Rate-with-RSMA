"""Composition capacity B* = max FREE users admittable to an IRS group of A BLOCKED users,
at QoS level q, general C_k policy. Empirical bootstrap from measured Γ pools (gbar_pools.npz):
  - A blocked ~ resample(P_blk)·scale        (strong, near IRS)
  - free candidates ~ resample(P_free)·scale, admitted BEST-FIRST (only beneficial: Γ_irs>Γ_dir)
QoS q: serve strongest ⌈q·n⌉ of the n=A+B group (drop weakest ⌊(1-q)n⌋), general feasible iff
  Σ_served ς ≤ R_c(Γ_min^served),  ς=(D-R_p)^+.  Γ ∝ P_S (pools measured @P50/N24)."""
import os, numpy as np

D, X, Y = 0.1, 0.08, 0.1
CP = 1.0 - X - Y
r_c = lambda g: np.log2(1 + g) - np.log2(1 + (1 - Y) * g)
r_p = lambda g: np.log2(1 + (1 - Y) * g) - np.log2(1 + CP * g)

pk = np.load(os.path.join(os.path.dirname(__file__), 'gbar_pools.npz'))
P_BLK, P_FREE = pk['P_blk'], pk['P_free']          # measured @P50
PS = {"70": 100.0, "60": 10.0, "50": 1.0, "40": 0.1, "30": 0.01}
QOS = [0.95, 0.90, 0.80, 0.70]
A_LIST = [2, 4, 6, 8]
K, N_MC = 10, 20000            # total users; group A blocked + B free ≤ K (supply cap)
RNG = np.random.default_rng(42)


def bstar(A, scale, q):
    b_cap = K - A                # can't admit more free than free users exist
    blk = RNG.choice(P_BLK, size=(N_MC, A)) * scale
    fre = -np.sort(-(RNG.choice(P_FREE, size=(N_MC, b_cap)) * scale), axis=1)  # desc = best-first
    best = np.zeros(N_MC, dtype=int)
    for B in range(0, b_cap + 1):
        grp = blk if B == 0 else np.concatenate([blk, fre[:, :B]], axis=1)
        n = A + B
        srt = np.sort(grp, axis=1)                 # ascending
        jdrop = int(np.floor((1 - q) * n))         # serve strongest ⌈qn⌉
        served = srt[:, jdrop:]
        sf = np.maximum(0.0, D - r_p(served))
        feas = sf.sum(axis=1) <= r_c(served[:, 0])
        best = np.where(feas, B, best)
    return best.mean()


print(f"pools: BLK n={len(P_BLK)} med={np.median(P_BLK):.0f} | FREE-benef n={len(P_FREE)} med={np.median(P_FREE):.0f}")
print(f"B* = mean max free (MC={N_MC}, cap B≤K-A with K={K}, general policy)\n")
hdr = "  ".join(f"QoS{int(q*100)}" for q in QOS)
for A in A_LIST:
    print(f"--- A={A} blocked ---   {hdr}")
    for ps, sc in PS.items():
        cells = "  ".join(f"{bstar(A, sc, q):5.1f}" for q in QOS)
        print(f"  P_S={ps:>2}dBm : {cells}")
    print()
