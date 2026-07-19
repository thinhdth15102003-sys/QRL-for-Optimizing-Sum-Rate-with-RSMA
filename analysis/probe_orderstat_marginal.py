#!/usr/bin/env python3
"""
probe_orderstat_marginal.py — §2b.5 TODO (i): validate the E_1 closed form of the
expected marginal E[Δ_{B→B+1}] (doc §2b.4) against Monte-Carlo on the §1 Γ-model.

Model (two-point MEANS, per-realization Rayleigh draws — §2b.1):
  IRS group: A blocked draws Γ = Γ̄_A·X + B free draws Γ = Γ̄_B·X, X~Exp(1) i.i.d.
  Direct:    deterministic Γ_dir (g_SU has no CN(0,1) factor in the model).
  Equal-power reference: α_k=1/K, β_g=1/(G+1), ρ=0.2  (§4 reference point).

Closed forms checked (all from §2b):
  E[ln(1+cX)] = e^{λ/c}E_1(λ/c),  X~Exp(rate λ)
  m_B ~ Exp(Λ_B), Λ_B = A/Γ̄_A + B/Γ̄_B,  E[m_B]=1/Λ_B
  P(new free draw becomes min) = (1/Γ̄_B)/Λ_{B+1}
  E[Δ_{B→B+1}] = (E[R_p(Γ_new)] − R_p(Γ_dir)) + (E[R_c(m_{B+1})] − E[R_c(m_B)])

Γ̄ inputs = §3 regime-check medians (Case-2 geometry, N=24):
  P50: Γ̄_irs=703,  Γ_dir=14   ·  P40: 70, 1.4  ·  P30: 7, 0.14
"""
import numpy as np

# ── config (mirrors §4.3 spot-check shape: K=5, A=2 blocked, sweep B) ─────────
K, A   = 5, 2
RHO    = 0.2
ALPHA  = 1.0 / K          # equal private split
BETA   = 0.5              # 2 groups (direct + 1 IRS) → β_g = 1/2
N_MC   = 400_000
RNG    = np.random.default_rng(7)

C2 = 1.0 - BETA * RHO                      # 0.9   (common Möbius factor)
CP = 1.0 - ALPHA * (1 - RHO) - BETA * RHO  # 0.74  (private Möbius factor)

REGIMES = {50: (703.0, 14.0), 40: (70.0, 1.4), 30: (7.0, 0.14)}

# ── rate functions (§1 / notation anchor) ─────────────────────────────────────
def R_p(g):  return np.log2(1 + C2 * g) - np.log2(1 + CP * g)
def R_c(g):  return np.log2(1 + g) - np.log2(1 + C2 * g)

# ── e^x E1(x), overflow-safe ──────────────────────────────────────────────────
def expE1(x):
    from scipy.special import exp1
    x = np.asarray(x, float)
    out = np.empty_like(x)
    lo = x <= 30
    out[lo] = np.exp(x[lo]) * exp1(x[lo])
    xh = x[~lo]                      # asymptotic: e^x E1(x) ~ 1/x Σ (-1)^n n!/x^n
    out[~lo] = (1/xh) * (1 - 1/xh + 2/xh**2 - 6/xh**3 + 24/xh**4)
    return out

def E_ln1p_over_ln2(lam, c):
    """E[log2(1+cX)], X~Exp(rate lam)."""
    return expE1(np.array([lam / c]))[0] / np.log(2)

def E_Rp(gbar):
    lam = 1.0 / gbar
    return E_ln1p_over_ln2(lam, C2) - E_ln1p_over_ln2(lam, CP)

def E_Rc_min(Lam):
    return E_ln1p_over_ln2(Lam, 1.0) - E_ln1p_over_ln2(Lam, C2)

print(f"cfg: K={K} A={A} rho={RHO} alpha={ALPHA} beta={BETA}  c2={C2} cp={CP}  MC n={N_MC}")
print(f"{'P_S':>4} {'B->B+1':>7} | {'E[dLt] form':>12} {'E[dLt] MC':>12} {'MC 3sig':>8} | "
      f"{'P(drag) form':>12} {'MC':>7} | {'E[m] form':>10} {'MC':>10}")

for ratio in (1.0, 0.05):   # Γ̄_B / Γ̄_A : same-quality free vs far free (α=3.5 ⇒ ~2.4× farther)
    print(f"--- Γ̄_B = {ratio}·Γ̄_A ---")
    for PS, (gI, gD) in REGIMES.items():
        gB = ratio * gI
        for B in (0, 1, 2):
            # ---- closed form -------------------------------------------------
            LamB  = A / gI + B / gB
            LamB1 = A / gI + (B + 1) / gB
            e_form = (E_Rp(gB) - R_p(gD)) + (E_Rc_min(LamB1) - E_Rc_min(LamB))
            p_form = (1 / gB) / LamB1
            m_form = 1 / LamB1
            # ---- Monte-Carlo on the same model -------------------------------
            blocked = gI * RNG.exponential(size=(N_MC, A))
            frees   = gB * RNG.exponential(size=(N_MC, B)) if B else np.full((N_MC, 0), np.inf)
            new     = gB * RNG.exponential(size=N_MC)
            m_old = np.hstack([blocked, frees]).min(axis=1)
            m_new = np.minimum(m_old, new)
            d  = (R_p(new) - R_p(gD)) + (R_c(m_new) - R_c(m_old))
            e_mc, sig = d.mean(), 3 * d.std() / np.sqrt(N_MC)
            p_mc = (new < m_old).mean()
            m_mc = m_new.mean()
            ok = "OK " if abs(e_form - e_mc) <= sig else "***"
            print(f"{PS:>4} {B}->{B+1:<4} | {e_form:>12.5f} {e_mc:>12.5f} {sig:>8.5f} | "
                  f"{p_form:>12.3f} {p_mc:>7.3f} | {m_form:>10.3f} {m_mc:>10.3f}  {ok}")

# reference rows (doc §4.3 empirical spot-check, env draws N=24):
print("\n§4.3 spot-check (env, for sign/shape comparison — not same Γ̄):")
print("  P50: d01=-0.007 d12=-0.004 | P40: d01=+0.001 d12=-0.004 | P30: d01=+0.006 d12=-0.009")

# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — HETEROGENEOUS BEST-FIRST + FINAL B* = min(B*_rate, B*_QoS)   (§2c)
#   Free pool with heterogeneous means (near/mid/far: Γ̄_b = r·Γ̄_A), admitted
#   best-first. Cumulative argmax over B (incl. group-0-empty cliff at B=3)
#   → B*_rate; per-draw IRS-side QoS bound → B*_QoS; B* = min.
# ══════════════════════════════════════════════════════════════════════════════
D_K     = 0.10
RATIOS  = (0.3, 0.05, 0.01)          # best-first pool: ~1.4x / 2.4x / 3.7x farther (α=3.5)
N_MC2   = 200_000

print("\n═══ PART 3: heterogeneous best-first (pool r=0.3/0.05/0.01) ═══")
print(f"{'P_S':>4} | {'E[d1] f/mc':>18} {'E[d2] f/mc':>18} {'E[d3+cliff] f/mc':>18} | "
      f"{'B*rate':>6} {'B*QoS':>6} {'B*':>4} (means)")

for PS, (gI, gD) in REGIMES.items():
    gBs   = [r * gI for r in RATIOS]
    cliff = R_c(gD)                                   # lost group-0 common (fixed weights)
    # ---- closed-form marginal sequence (admission j = 1..3, best-first) ------
    e_form = []
    Lam = A / gI
    for j, gb in enumerate(gBs):
        Lam1 = Lam + 1 / gb
        e = (E_Rp(gb) - R_p(gD)) + (E_Rc_min(Lam1) - E_Rc_min(Lam))
        if j == len(gBs) - 1:
            e -= cliff                                # last direct leaves → group 0 vanishes
        e_form.append(e); Lam = Lam1
    # ---- MC: realized cumulative + per-draw thresholds ------------------------
    blocked = gI * RNG.exponential(size=(N_MC2, A))
    frees   = np.stack([gb * RNG.exponential(size=N_MC2) for gb in gBs], axis=1)
    d_mc, cum = [], np.zeros((N_MC2, 4))              # cum[:,B] = ΔR_tot after B admits
    m_old = blocked.min(axis=1)
    s_blk = np.clip(D_K - R_p(blocked), 0, None).sum(axis=1)   # blocked shortfall mass
    qos_eq  = np.ones((N_MC2, 4), bool)               # equal-share  C_k = R_c/n
    qos_gen = np.ones((N_MC2, 4), bool)               # general share: Σ s_k ≤ R_c(m)
    qos_eq[:, 0]  = R_p(m_old) + R_c(m_old) / A >= D_K
    qos_gen[:, 0] = s_blk <= R_c(m_old)
    S = s_blk.copy()
    for j in range(3):
        new   = frees[:, j]
        m_new = np.minimum(m_old, new)
        d = (R_p(new) - R_p(gD)) + (R_c(m_new) - R_c(m_old))
        if j == 2:
            d = d - cliff
        d_mc.append(d)
        cum[:, j + 1] = cum[:, j] + d
        S = S + np.clip(D_K - R_p(new), 0, None)
        qos_eq[:, j + 1]  = R_p(m_new) + R_c(m_new) / (A + j + 1) >= D_K
        qos_gen[:, j + 1] = S <= R_c(m_new)
        m_old = m_new
    b_rate = cum.argmax(axis=1)                       # cumulative max (NOT greedy-stop)
    def maxfeas(q):                                    # largest feasible B
        b = np.where(q, np.arange(4), -1).max(axis=1)
        return np.maximum(b, 0)
    b_eq, b_gen = maxfeas(qos_eq), maxfeas(qos_gen)
    bs_eq, bs_gen = np.minimum(b_rate, b_eq), np.minimum(b_rate, b_gen)
    fm = " ".join(f"{f:+.4f}/{np.mean(m):+.4f}" for f, m in zip(e_form, d_mc))
    print(f"{PS:>4} | {fm} |")
    print(f"     B*rate mode={np.bincount(b_rate).argmax()} mean={b_rate.mean():.2f} | "
          f"QoS eq-share mode={np.bincount(b_eq).argmax()} mean={b_eq.mean():.2f} → B*={np.bincount(bs_eq).argmax()} | "
          f"QoS GENERAL mode={np.bincount(b_gen).argmax()} mean={b_gen.mean():.2f} → B*={np.bincount(bs_gen).argmax()}")
    print(f"     cliff R_c(Γ_dir)={cliff:.3f}  (§4.3 env cliff ref: P50 0.137 / P40 0.118 / P30 0.041)")
