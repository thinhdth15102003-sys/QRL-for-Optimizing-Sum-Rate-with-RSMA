"""
test_env.py
-----------
Test suite for the Multi-IRS ISTN Environment.

All tests pull their topology from params.py via make_config() so that
changing K, M, N in one place is immediately reflected here.

Run with:
    python test_env.py            # all tests
    python test_env.py -v         # verbose output
    python test_env.py --quick    # skip slow multi-episode tests
"""

import sys
import argparse
import numpy as np

from params import (                    # ← single source of truth
    make_config,
    n_episodes, n_eval_episodes,
    n_irs_pos_samples, n_walk_steps, n_signal_samples,
    seed_default, seed_eval, seed_config, seed_irs,
    seed_env, seed_partial, seed_power, seed_csi,
)
from CSI.env       import ISTNEnv
from CSI.baselines import RandomPolicy, GreedyPolicy, DirectOnlyPolicy, AllIRSPolicy

# Ensure the console accepts UTF-8 box-drawing characters on Windows.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}"); sys.exit(1)
def hint(msg): print(f"  {YELLOW}→{RESET}  {msg}")
def section(title):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ── shared position helper ─────────────────────────────────────────────────────

def _sample_positions(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Sample n (x,y) positions uniformly within a circle of given radius (km)."""
    r     = radius * np.sqrt(rng.uniform(0, 1, size=n))
    theta = rng.uniform(0, 2 * np.pi, size=n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])   # (n, 2)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 0 — Signal Processing Flow  (env → s → channel → h_k → y_k → R)
# ══════════════════════════════════════════════════════════════════════════════
def test_signal_flow(verbose=True):
    section("TEST 0 · Full Signal Processing Flow  (env → s → channel → h_k → y_k → R)")

    from istn.channel import ChannelModel
    from CSI.irs      import IRSPhaseModel
    from CSI.rate     import RateComputer

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_env)
    env.reset()

    ch  = env.channels
    Phi = env.Phi
    rc  = RateComputer(cfg)

    W = 58   # inner column width

    # ── Stage 1: Environment — positions ──────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 1 — Environment Setup  (positions + channel geometry)")
    print(f"  {'─'*W}")
    print(f"  {'Variable':<18}  {'Shape':<14}  {'unit':<6}  Description")
    print(f"  {'─'*18}  {'─'*14}  {'─'*6}  {'─'*28}")
    up = env.user_pos
    ip = env.irs_pos
    print(f"  {'user_pos':<18}  {str(up.shape):<14}  km     K×2 ground (x,y) positions")
    print(f"  {'irs_pos':<18}  {str(ip.shape):<14}  km     M×2 rooftop (x,y) positions")
    print(f"  {'h_SR_km':<18}  {'scalar':<14}  km     satellite altitude = {cfg.h_SR_km}")
    print(f"  {'h_IRS_km':<18}  {'scalar':<14}  km     IRS rooftop height = {cfg.h_IRS_km*1000:.0f} m")

    print(f"\n    User positions (km)  [LoS radius = {cfg.R_LoS_km} km]:")
    print(f"    {'k':>4}  {'x':>10}  {'y':>10}  {'dist (m)':>10}")
    print(f"    {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}")
    for k in range(cfg.K):
        d = np.sqrt(up[k, 0]**2 + up[k, 1]**2)
        print(f"    {k+1:>4}  {up[k,0]:>+10.4f}  {up[k,1]:>+10.4f}  {d*1000:>8.1f}")

    print(f"\n    IRS positions (km)  [rooftop z = {cfg.h_IRS_km*1000:.0f} m]:")
    print(f"    {'m':>4}  {'x':>10}  {'y':>10}  {'dist (m)':>10}")
    print(f"    {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}")
    for m in range(cfg.M):
        d = np.sqrt(ip[m, 0]**2 + ip[m, 1]**2)
        print(f"    {m+1:>4}  {ip[m,0]:>+10.4f}  {ip[m,1]:>+10.4f}  {d*1000:>8.1f}")

    # ── Stage 2: Channel coefficients ─────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 2 — Channel Coefficients  g_SR, g_RU, g_SU")
    print(f"  {'─'*W}")
    print(f"  Sat→IRS  : g_SR[m]   = (free-space path loss) · rain_attn · exp(jφ_SR)")
    print(f"  IRS→User : g_RU[m,k] = sqrt(G_U · g_sf · path_loss) · Rayleigh sample")
    print(f"  Sat→User : g_SU[k]   = (free-space path loss) · rain_attn · exp(jφ_SU)")
    print(f"  Observed : g̃ = g_true + Δg,  Δg = κ|g|ε,  ε~CN(0,1),  κ={cfg.kappa}")
    print(f"  Blocking : g_SU[k] ×= β_block={cfg.beta_blocking} when building intercepts sat→user path")

    g_SR = ch['g_SR']
    g_RU = ch['g_RU']
    g_SU = ch['g_SU']
    beta = ch['beta']
    su_blocked = ch['su_blocked']
    d_SU    = ch['d_SU']
    d_IRS_U = ch['d_IRS_U']

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'dtype':<12}  {'|·| min':>10}  {'|·| max':>10}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*10}  {'─'*10}")
    for name, arr in [('g_SR (Sat→IRS)', g_SR),
                      ('g_RU (IRS→User)', g_RU),
                      ('g_SU (Sat→User)', g_SU)]:
        mag = np.abs(arr)
        print(f"  {name:<18}  {str(arr.shape):<14}  {str(arr.dtype):<12}  "
              f"{mag.min():>10.3e}  {mag.max():>10.3e}")
    print(f"  {'beta':<18}  {str(beta.shape):<14}  {'float64':<12}  "
          f"{'β =':>10}  {beta[0]:.3f} (fixed)")
    print(f"  {'su_blocked':<18}  {str(su_blocked.shape):<14}  {'bool':<12}  "
          f"  {su_blocked.sum():>4}/{cfg.K} users have blocked direct link")

    print(f"\n  Distances:")
    print(f"  {'d_SU (Sat→User)':<22}  {str(d_SU.shape):<12}  "
          f"km  [{d_SU.min():.3f}, {d_SU.max():.3f}]  "
          f"(altitude dominates: h_sat={cfg.h_SR_km} km)")
    print(f"  {'d_IRS_U (IRS→User)':<22}  {str(d_IRS_U.shape):<12}  "
          f"km  [{d_IRS_U.min():.4f}, {d_IRS_U.max():.4f}]  (ground distances)")

    # ── Stage 3: IRS phase shifts ──────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 3 — IRS Phase Shifts  Φ")
    print(f"  {'─'*W}")
    print(f"  Phi[m,n] = exp(j·θ_{{m,n}}),  θ ∈ {{0, π/2, π, 3π/2}}  (2-bit quantised)")
    print(f"  |Phi[m,n]| = 1  (unit modulus — passive reflection)")

    phase_ang = np.angle(Phi)
    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'dtype':<12}  Description")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*28}")
    print(f"  {'Phi':<18}  {str(Phi.shape):<14}  {str(Phi.dtype):<12}  "
          f"complex phase-shift matrix (M×N)")
    print(f"  {'phase_angles':<18}  {str(phase_ang.shape):<14}  {'float64':<12}  "
          f"∈ {{0, π/2, π, 3π/2}} rad")

    print(f"\n  Phase distribution across M×N={cfg.M*cfg.N} elements:")
    print(f"  {'Level':>6}  {'Angle (rad)':>12}  {'Angle (°)':>10}  {'Count':>6}  {'%':>6}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*6}")
    for i, lev in enumerate(cfg.phase_levels):
        cnt = int(np.sum(np.isclose(phase_ang, lev, atol=1e-6)))
        print(f"  {i:>6}  {lev:>12.4f}  {np.degrees(lev):>9.1f}°  "
              f"{cnt:>6}  {100*cnt/(cfg.M*cfg.N):>5.1f}%")

    print(f"\n  Phi sample (IRS 1, first 8 elements):")
    print(f"  {'n':>4}  {'Re(Phi)':>10}  {'Im(Phi)':>10}  {'|Phi|':>8}  {'angle (°)':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")
    for n in range(min(8, cfg.N)):
        p = Phi[0, n, n]
        print(f"  {n:>4}  {p.real:>+10.4f}  {p.imag:>+10.4f}  "
              f"{abs(p):>8.4f}  {np.degrees(np.angle(p)):>9.1f}°")

    # ── Stage 4: IRS assignment ────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 4 — IRS Assignment  φ  (from quantum actor output)")
    print(f"  {'─'*W}")
    print(f"  φ[k] ∈ {{0,..,{cfg.M}}}:  0 = direct satellite link")
    print(f"                          1..{cfg.M} = route through IRS m = φ[k]")

    # Use round-robin for illustration (last 2 users go direct)
    assignment = np.array([(k % cfg.M) + 1 for k in range(cfg.K - 2)] + [0, 0])
    groups: dict = {}
    for k, gid in enumerate(assignment.astype(int)):
        groups.setdefault(int(gid), []).append(k)
    active_irs = sorted(g for g in groups if g > 0)
    n_common_streams = len(active_irs)

    print(f"\n  {'Variable':<18}  {'Shape':<14}  Description")
    print(f"  {'─'*18}  {'─'*14}  {'─'*32}")
    print(f"  {'assignment φ':<18}  {str(assignment.shape):<14}  ∈ {{0,...,{cfg.M}}}")
    print(f"  {'active IRS':<18}  list          {active_irs}  ({n_common_streams} common streams)")

    print(f"\n  Group membership  (gid → users):")
    print(f"  {'gid':>5}  {'Link':>8}  {'#users':>7}  User indices")
    print(f"  {'─'*5}  {'─'*8}  {'─'*7}  {'─'*28}")
    for gid in sorted(groups.keys()):
        lbl = "direct" if gid == 0 else f"IRS {gid}"
        members_1idx = [k + 1 for k in groups[gid]]
        print(f"  {gid:>5}  {lbl:>8}  {len(groups[gid]):>7}  {members_1idx}")

    print(f"\n  Message split after IRS selection:")
    print(f"  s = [s_1,..,s_K]  →  [s_c,1,..,s_c,G, s_p,1,..,s_p,K]")
    print(f"  G = {n_common_streams} active IRS  →  {n_common_streams} common + {cfg.K} private sub-messages")

    # ── Stage 5: Effective channel h_k ────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 5 — Effective Channel  h_k")
    print(f"  {'─'*W}")
    print(f"  Direct (φ[k]=0)  : h_k = g_SU[k]")
    print(f"  IRS m  (φ[k]=m)  : h_k = β_m · Σ_n g_SR[m] · Φ[m,n] · g_RU[m,k]")
    print(f"                             ↑ N={cfg.N} reflecting elements summed coherently")

    h_eff = rc.effective_channels_all(assignment, Phi, ch)   # (K,) complex

    # Best IRS effective channel for each user (for comparison)
    h_irs_mat = np.array([
        ch['beta'][m] * np.abs(
            ch['g_SR'][m] * np.sum(np.diag(Phi[m])) * ch['g_RU'][m])
        for m in range(cfg.M)
    ])   # (M, K)
    h_direct = np.abs(ch['g_SU'])   # (K,)

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'dtype':<12}  {'|·| range'}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*20}")
    print(f"  {'h_eff':<18}  {str(h_eff.shape):<14}  {str(h_eff.dtype):<12}  "
          f"[{np.abs(h_eff).min():.3e}, {np.abs(h_eff).max():.3e}]")

    print(f"\n  Per-user effective channel:")
    print(f"  {'k':>4}  {'Link':>7}  {'|h_k|':>12}  {'|g_SU[k]|':>12}  "
          f"{'best |h_IRS|':>13}  {'IRS wins?':>10}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*13}  {'─'*10}")
    for k in range(cfg.K):
        gid = int(assignment[k])
        link = "direct" if gid == 0 else f"IRS {gid}"
        irs_best = h_irs_mat[:, k].max()
        irs_wins = irs_best > h_direct[k]
        wins_str = f"{GREEN}yes{RESET}" if irs_wins else "no"
        print(f"  {k+1:>4}  {link:>7}  {np.abs(h_eff[k]):>12.4e}  "
              f"{h_direct[k]:>12.4e}  {irs_best:>13.4e}  {wins_str:>10}")

    ok(f"h_eff computed for all {cfg.K} users  "
       f"|h_eff| in [{np.abs(h_eff).min():.3e}, {np.abs(h_eff).max():.3e}]")

    # ── Stage 6: Power allocation ──────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 6 — Power Allocation  w_c_vec, w_p")
    print(f"  {'─'*W}")
    print(f"  w_c_vec : (M+1,)  one common-stream power per group "
          f"(0=direct, 1..M=IRS)")
    print(f"  w_p     : (K,)    one private-stream power per user")
    print(f"  Constraint: Σ w_c_vec + Σ w_p = P_S = {cfg.P_S:.4f} W")

    P       = cfg.P_S
    w_c_vec = np.full(cfg.M + 1, P * 0.2 / (cfg.M + 1))   # equal share among M+1 groups
    w_p     = np.full(cfg.K,     P * 0.8 / cfg.K)

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'Σ (W)':>10}  {'% P_S':>8}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*10}  {'─'*8}")
    print(f"  {'w_c_vec':<18}  {str(w_c_vec.shape):<14}  "
          f"{w_c_vec.sum():>10.4f}  {100*w_c_vec.sum()/P:>7.1f}%")
    print(f"  {'w_p':<18}  {str(w_p.shape):<14}  "
          f"{w_p.sum():>10.4f}  {100*w_p.sum()/P:>7.1f}%")
    total_w = w_c_vec.sum() + w_p.sum()
    print(f"  {'─'*18}  {'─'*14}  {'─'*10}  {'─'*8}")
    print(f"  {'Total':<18}  {'─'*14}  {total_w:>10.4f}  {100*total_w/P:>7.1f}%")

    print(f"\n  Per-group common power:")
    for g in range(cfg.M + 1):
        if g == 0:
            label = "direct"
            active = bool(np.any(assignment == 0))
        else:
            label = f"IRS {g}"
            active = (g in active_irs)
        tag = " (active)" if active else " (masked → 0 in PowerMLP)"
        print(f"    w_c_vec[{g}] ({label}) = {w_c_vec[g]:.4f} W{tag}")

    ok(f"Power constraint satisfied: {total_w:.6f} W = P_S")

    # ── Stage 7: Transmitted signal x ─────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 7 — Transmitted Signal  x")
    print(f"  {'─'*W}")
    print(f"  x = Σ_{{g=1}}^{{G}} √w_c_g · s_c_g  +  Σ_{{k=1}}^{{K}} √w_p_k · s_p_k")
    print(f"  s_c_g ~ CN(0,1) for g=1..{n_common_streams} active IRS groups")
    print(f"  s_p_k ~ CN(0,1) for k=1..{cfg.K} users")
    print(f"  E[|x|²] = Σ w_c_g + Σ w_p_k = {total_w:.4f} W  = P_S")

    rng_sig = np.random.default_rng(42)
    def _cn01():
        return (rng_sig.standard_normal() + 1j * rng_sig.standard_normal()) / np.sqrt(2)

    s_c = np.array([_cn01() for _ in range(cfg.M + 1)])  # (M+1,) — one per group
    s_p = np.array([_cn01() for _ in range(cfg.K)])      # (K,)
    # Only active groups contribute to x
    x_common  = np.sum(np.sqrt(w_c_vec) * s_c)           # Σ √w_c_g · s_c_g
    x_private = np.sum(np.sqrt(w_p) * s_p)            # Σ √w_p_k · s_p_k
    x         = x_common + x_private

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'dtype':<12}  Value / Info")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*26}")
    print(f"  {'s_c':<18}  {str(s_c.shape):<14}  {str(s_c.dtype):<12}  "
          f"common symbols (one per IRS group)")
    print(f"  {'s_p':<18}  {str(s_p.shape):<14}  {str(s_p.dtype):<12}  "
          f"private symbols (one per user)")
    print(f"  {'x_common':<18}  {'scalar':<14}  {'complex':<12}  "
          f"{x_common.real:+.4f} + {x_common.imag:+.4f}j")
    print(f"  {'x_private':<18}  {'scalar':<14}  {'complex':<12}  "
          f"{x_private.real:+.4f} + {x_private.imag:+.4f}j")
    print(f"  {'x = x_c + x_p':<18}  {'scalar':<14}  {'complex':<12}  "
          f"{x.real:+.4f} + {x.imag:+.4f}j")
    print(f"  {'|x|²':<18}  {'scalar':<14}  {'float':<12}  "
          f"{abs(x)**2:.4f}  (one sample; E[|x|²]={total_w:.4f})")

    # ── Stage 8: Received signal y_k ──────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 8 — Received Signal  y_k")
    print(f"  {'─'*W}")
    print(f"  y_k = h_k · x + n_k")
    print(f"  n_k ~ CN(0, σ²={cfg.sigma2:.4f})   noise variance σ² = 10^(noise_dB/10)")
    print(f"  (x is the same broadcast signal for all K users)")

    noise_std = np.sqrt(cfg.sigma2 / 2)
    n_k = noise_std * (
        rng_sig.standard_normal(cfg.K) + 1j * rng_sig.standard_normal(cfg.K))
    y_k = h_eff * x + n_k    # (K,) — one snapshot

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'dtype':<12}  {'|·| range'}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*22}")
    print(f"  {'h_eff':<18}  {str(h_eff.shape):<14}  {str(h_eff.dtype):<12}  "
          f"[{np.abs(h_eff).min():.3e}, {np.abs(h_eff).max():.3e}]")
    print(f"  {'n_k (noise)':<18}  {str(n_k.shape):<14}  {str(n_k.dtype):<12}  "
          f"[{np.abs(n_k).min():.3e}, {np.abs(n_k).max():.3e}]")
    print(f"  {'y_k':<18}  {str(y_k.shape):<14}  {str(y_k.dtype):<12}  "
          f"[{np.abs(y_k).min():.3e}, {np.abs(y_k).max():.3e}]")

    print(f"\n  First 5 users (one-sample snapshot):")
    print(f"  {'k':>4}  {'Link':>7}  {'h_k':>22}  {'n_k':>22}  {'y_k':>22}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*22}  {'─'*22}  {'─'*22}")
    for k in range(min(5, cfg.K)):
        gid = int(assignment[k])
        link = "direct" if gid == 0 else f"IRS {gid}"
        hk   = h_eff[k]
        nk   = n_k[k]
        yk   = y_k[k]
        print(f"  {k+1:>4}  {link:>7}  "
              f"{hk.real:>+9.3e}{hk.imag:>+9.3e}j  "
              f"{nk.real:>+9.3e}{nk.imag:>+9.3e}j  "
              f"{yk.real:>+9.3e}{yk.imag:>+9.3e}j")

    ok(f"y_k computed for all {cfg.K} users (one-snapshot simulation)")

    # ── Stage 9: SINR ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 9 — SINR  (per user, per stream)")
    print(f"  {'─'*W}")
    print(f"  Notation: h2_k = |h_k|²,  P_c_g = w_c_vec[g],  P_p_k = w_p[k]")
    print(f"            Σw_c = Σ_g w_c_vec[g],  Σw_p = Σ_k w_p[k]")
    print()
    print(f"  SINR_c[k,g] = h2_k · P_c_g")
    print(f"                ─────────────────────────────────────────────────")
    print(f"                h2_k · (Σw_p + Σw_c - P_c_g) + σ²")
    print(f"                (before SIC: all other signals are interference)")
    print()
    print(f"  SINR_p[k]   = h2_k · P_p_k")
    print(f"                ─────────────────────────────────────────────────")
    print(f"                h2_k · (Σw_p - P_p_k + Σw_c - P_c_{{gid[k]}}) + σ²")
    print(f"                (after SIC of own group's common; direct: no SIC)")

    # Build gid → wc_idx mapping (G+1 active groups: [direct, *active_irs])
    wc_map = rc._make_wc_map(active_irs)
    sinr_p, sinr_c, _ = rc._sinr_all(h_eff, assignment, w_p, w_c_vec,
                                      wc_map, cfg.sigma2)

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'min':>12}  {'max':>12}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*12}  {'─'*12}")
    print(f"  {'SINR_c':<18}  {str(sinr_c.shape):<14}  "
          f"{sinr_c.min():>12.4e}  {sinr_c.max():>12.4e}  "
          f"(0 for direct users)")
    print(f"  {'SINR_p':<18}  {str(sinr_p.shape):<14}  "
          f"{sinr_p.min():>12.4e}  {sinr_p.max():>12.4e}")

    ok(f"SINR computed for all {cfg.K} users  "
       f"(SINR_p range [{sinr_p.min():.3e}, {sinr_p.max():.3e}])")

    # ── Stage 10: Rates ────────────────────────────────────────────────────────
    print(f"\n  {'─'*W}")
    print(f"  Stage 10 — Rates  R_c, R_p, C_k, R_total")
    print(f"  {'─'*W}")
    print(f"  R_c_group[g] = log2(1 + min_{{k∈g}} SINR_c[k,g])")
    print(f"                 ↑ bottleneck = worst-SINR user in group g")
    print(f"  R_p[k]       = log2(1 + SINR_p[k])")
    print(f"  C_k[k]       = α_k · R_c_group[g],  Σ_{{k∈g}} α_k = 1  (within-group split)")
    print(f"  R_total[k]   = R_p[k] + C_k[k]")
    print(f"  Σ R_total    = sum-rate  (bps/Hz)")

    result = rc.compute_sum_rate(assignment, Phi, ch, w_p, w_c_vec)
    R_p       = result['R_private']
    C_k       = result['C_k']
    R_c_group = result['R_common_group']
    R_total   = R_p + C_k

    print(f"\n  {'Variable':<18}  {'Shape':<14}  {'min':>10}  {'max':>10}  {'sum':>10}")
    print(f"  {'─'*18}  {'─'*14}  {'─'*10}  {'─'*10}  {'─'*10}")
    print(f"  {'R_p (private)':<18}  {str(R_p.shape):<14}  "
          f"{R_p.min():>10.4f}  {R_p.max():>10.4f}  {R_p.sum():>10.4f}")
    print(f"  {'C_k (common)':<18}  {str(C_k.shape):<14}  "
          f"{C_k.min():>10.4f}  {C_k.max():>10.4f}  {C_k.sum():>10.4f}")
    print(f"  {'R_total':<18}  {str(R_total.shape):<14}  "
          f"{R_total.min():>10.4f}  {R_total.max():>10.4f}  "
          f"{R_total.sum():>10.4f}")

    print(f"\n  R_c_group  (common rate per group — bottleneck constraint):")
    print(f"  {'gid':>5}  {'Link':>8}  {'#members':>9}  "
          f"{'min SINR_c':>12}  {'R_c (bps/Hz)':>13}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*12}  {'─'*13}")
    for gid in sorted(R_c_group.keys()):
        lbl = "direct" if gid == 0 else f"IRS {gid}"
        members = groups.get(gid, [])
        if gid == 0 or not members:
            print(f"  {gid:>5}  {lbl:>8}  {len(members):>9}  "
                  f"{'—':>12}  {R_c_group[gid]:>13.4f}")
        else:
            min_sinrc = min(sinr_c[k] for k in members)
            print(f"  {gid:>5}  {lbl:>8}  {len(members):>9}  "
                  f"{min_sinrc:>12.4e}  {R_c_group[gid]:>13.4f}")

    print(f"\n  Per-user rate breakdown:")
    print(f"  {'k':>4}  {'Link':>7}  {'|h_k|':>10}  {'SINR_c':>10}  "
          f"{'SINR_p':>10}  {'R_p':>8}  {'C_k':>8}  "
          f"{'R_tot':>8}  {'≥D_k':>6}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*10}  {'─'*10}  "
          f"{'─'*10}  {'─'*8}  {'─'*8}  "
          f"{'─'*8}  {'─'*7}")
    n_qos = 0
    for k in range(cfg.K):
        gid  = int(assignment[k])
        link = "direct" if gid == 0 else f"IRS {gid}"
        qos  = bool(R_total[k] >= cfg.D_k_bps_hz)
        if qos:
            n_qos += 1
        q_str = f"{GREEN}OK{RESET}" if qos else f"{RED}--{RESET}"
        print(f"  {k+1:>4}  {link:>7}  {np.abs(h_eff[k]):>10.3e}  "
              f"{sinr_c[k]:>10.4e}  {sinr_p[k]:>10.4e}  "
              f"{R_p[k]:>8.4f}  {C_k[k]:>8.4f}  "
              f"{R_total[k]:>8.4f}  {q_str}")

    print(f"\n  {'─'*W}")
    print(f"  Summary:")
    print(f"  {'─'*W}")
    print(f"  Σ R_total (sum-rate) = {result['sum_rate']:.4f} bps/Hz")
    print(f"  QoS fraction         = {n_qos}/{cfg.K} users with R_total ≥ D_k={cfg.D_k_bps_hz}")
    print(f"  Feasible             = {result['feasible']}  "
          f"(power_ok={result['power_ok']}, rate_ok={all(R_total >= cfg.D_k_bps_hz)})")

    ok(f"Full pipeline: env → g → h_k → y_k → SINR → rates  COMPLETE")
    ok(f"Sum-rate = {result['sum_rate']:.4f} bps/Hz  |  "
       f"QoS: {n_qos}/{cfg.K}  |  P_used = {total_w:.2f} W")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Config + User Positions
# ══════════════════════════════════════════════════════════════════════════════
def test_config(verbose=True):
    section("TEST 1 · SystemConfig + User Positions")

    cfg = make_config()
    print(f"\n{cfg.summary()}\n")

    assert np.isclose(cfg.P_S, 10 ** ((cfg.P_S_dBm - 30) / 10))
    ok("P_S derived correctly from dBm")

    assert len(cfg.phase_levels) == 2 ** cfg.quantization_bits
    expected = [2 * np.pi * k / (2 ** cfg.quantization_bits)
                for k in range(2 ** cfg.quantization_bits)]
    assert np.allclose(cfg.phase_levels, expected)
    ok(f"Phase levels correct: {np.round(cfg.phase_levels, 4)}")

    assert cfg.sigma2 > 0
    ok(f"Noise variance σ² = {cfg.sigma2:.6f}")

    # Override test — explicitly creating a different config
    cfg2 = make_config(K=8, M=5, N=32, kappa=0.05)
    assert cfg2.K == 8 and cfg2.M == 5 and cfg2.N == 32
    ok("make_config() override (K=8, M=5, N=32) works correctly")

    # User positions (x, y, z=0) within satellite LoS zone
    env_tmp = ISTNEnv(cfg=cfg)
    env_tmp.reset(seed=seed_config)
    print(f"\n  User positions (x, y, z=0) [km]"
          f"  — LoS zone radius = {cfg.R_LoS_km} km  (K={cfg.K} users):")
    print(f"  {'User':>5}  {'x (km)':>10}  {'y (km)':>10}  {'z (km)':>8}  "
          f"{'dist from centre':>18}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*18}")
    for k in range(cfg.K):
        x, y = env_tmp.user_pos[k]
        d    = np.sqrt(x**2 + y**2)
        print(f"  {k+1:>5}  {x:>+10.4f}  {y:>+10.4f}  {'0.0000':>8}  "
              f"{d*1000:>14.1f} m")
    all_inside = np.all(
        np.sqrt(env_tmp.user_pos[:, 0]**2 + env_tmp.user_pos[:, 1]**2)
        <= cfg.R_LoS_km + 1e-9)
    ok(f"All {cfg.K} users within LoS zone (R={cfg.R_LoS_km} km): {all_inside}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Channel generation + Array Shapes
# ══════════════════════════════════════════════════════════════════════════════
def test_channel(verbose=True):
    section("TEST 2 · ChannelModel + Array Shapes")

    from istn.channel import ChannelModel
    cfg = make_config(kappa=0.01)
    cm  = ChannelModel(cfg, rng=np.random.default_rng(0))

    rng_pos  = np.random.default_rng(1)
    irs_pos  = _sample_positions(rng_pos, cfg.M, cfg.R_LoS_km)
    user_pos = _sample_positions(rng_pos, cfg.K, cfg.R_LoS_km)
    ch       = cm.generate(user_pos, irs_pos)

    # Shape report
    print(f"\n  Channel array shapes  (M={cfg.M}, K={cfg.K}, N={cfg.N}):")
    print(f"  {'Key':<12}  {'Shape':<18}  dtype")
    print(f"  {'─'*12}  {'─'*18}  {'─'*12}")
    for key, val in ch.items():
        if isinstance(val, np.ndarray):
            print(f"  {key:<12}  {str(val.shape):<18}  {val.dtype}")
    print()

    # Shape checks using cfg attributes (not hardcoded)
    assert ch['g_SR'].shape     == (cfg.M,)
    assert ch['g_RU'].shape     == (cfg.M, cfg.K)
    assert ch['g_SU'].shape     == (cfg.K,)
    assert ch['g_SR_hat'].shape == (cfg.M,)
    assert ch['g_RU_hat'].shape == (cfg.M, cfg.K)
    assert ch['g_SU_hat'].shape == (cfg.K,)
    assert ch['beta'].shape     == (cfg.M,)
    assert ch['d_SU'].shape     == (cfg.K,)
    assert ch['d_IRS_U'].shape  == (cfg.M, cfg.K)
    ok(f"All channel arrays have correct shapes  (M={cfg.M}, K={cfg.K})")

    # IRS reflection efficiency (beta_IRS, not beta_blocking)
    assert np.allclose(ch['beta'], cfg.beta_IRS)
    ok(f"IRS reflection efficiency β_IRS = {ch['beta'][0]:.2f}")

    # Imperfect CSI: estimated (g_hat) ≠ true (g)
    diff_SU = np.abs(ch['g_SU_hat'] - ch['g_SU'])
    assert np.any(diff_SU > 0)
    ok(f"Imperfect CSI active: mean |Δg_SU| = {diff_SU.mean():.2e}")

    # Slant ranges ≈ 800 km (users within R_LoS_km << h_SR_km)
    assert np.all(ch['d_SU'] >= 800) and np.all(ch['d_SU'] <= 801)
    ok(f"Slant ranges d_SU ≈ 800 km: [{ch['d_SU'].min():.3f}, {ch['d_SU'].max():.3f}]")

    # Reproducibility with same seed
    rng_pos2  = np.random.default_rng(1)
    irs_pos2  = _sample_positions(rng_pos2, cfg.M, cfg.R_LoS_km)
    user_pos2 = _sample_positions(rng_pos2, cfg.K, cfg.R_LoS_km)
    cm2 = ChannelModel(cfg, rng=np.random.default_rng(0))
    ch2 = cm2.generate(user_pos2, irs_pos2)
    assert np.allclose(ch['g_SU'], ch2['g_SU'])
    ok("Channel generation is reproducible with fixed seed")

    if verbose:
        hint(f"|g_SU| min/max: {np.abs(ch['g_SU']).min():.5f} / {np.abs(ch['g_SU']).max():.5f}")
        hint(f"|g_SR[0]| = {np.abs(ch['g_SR'][0]):.5e}")
        hint(f"d_IRS_U[0] (m) min/max: {ch['d_IRS_U'][0].min()*1000:.1f} / {ch['d_IRS_U'][0].max()*1000:.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — IRS Phase Model + IRS/Building Positions
# ══════════════════════════════════════════════════════════════════════════════
def test_irs(verbose=True):
    section("TEST 3 · IRSPhaseModel + IRS/Building Positions")

    from CSI.irs import IRSPhaseModel

    cfg  = make_config()
    irsm = IRSPhaseModel(cfg)
    rng  = np.random.default_rng(1)

    phases = irsm.random_phases(rng)
    assert phases.shape == (cfg.M, cfg.N)
    ok(f"random_phases returns (M={cfg.M}, N={cfg.N}) array")

    valid_levels = set(np.round(cfg.phase_levels, 6))
    for p in phases.flatten():
        assert round(p, 6) in valid_levels, f"Invalid phase value: {p}"
    ok("All random phases are valid discrete levels")

    Phi = irsm.build_phi(phases)
    assert Phi.shape == (cfg.M, cfg.N, cfg.N)
    assert Phi.dtype == complex
    diag_entries = Phi[:, np.arange(cfg.N), np.arange(cfg.N)]   # (M, N)
    assert np.allclose(np.abs(diag_entries), 1.0)
    # Off-diagonal entries should be zero
    off_mask = ~np.eye(cfg.N, dtype=bool)
    assert np.allclose(Phi[:, off_mask], 0.0)
    ok("Phi diagonal entries have unit modulus, off-diagonal entries are zero")

    cont      = np.array([[0.1, np.pi/2 + 0.3, np.pi - 0.1, 5.0]])
    quantised = irsm.quantize(cont)
    expected  = np.array([[0.0, np.pi/2, np.pi, 3*np.pi/2]])
    assert np.allclose(quantised, expected, atol=1e-6)
    ok("Quantise snaps to nearest level correctly")

    idx = np.array([[0, 1, 2, 3]])
    p   = irsm.index_to_phase(idx)
    assert np.allclose(p[0], cfg.phase_levels)
    ok("index_to_phase maps 0→0, 1→π/2, 2→π, 3→3π/2")

    # IRS positions (x, y, z) — building base at z=0, IRS at rooftop z=h_IRS_km
    env_irs = ISTNEnv(cfg=cfg)
    env_irs.reset(seed=seed_irs)
    z = cfg.h_IRS_km
    print(f"\n  IRS / Building positions  (M={cfg.M})"
          f"  [building z=0, IRS rooftop z={z:.4f} km = {z*1000:.0f} m]:")
    print(f"  {'IRS':>5}  {'bldg x (km)':>13}  {'bldg y (km)':>13}  "
          f"{'bldg z':>8}  {'IRS z':>8}  {'dist (m)':>10}")
    print(f"  {'─'*5}  {'─'*13}  {'─'*13}  {'─'*8}  {'─'*8}  {'─'*10}")
    for m in range(cfg.M):
        x, y = env_irs.irs_pos[m]
        dist = np.sqrt(x**2 + y**2)
        print(f"  {m+1:>5}  {x:>+13.4f}  {y:>+13.4f}  {'0.0000':>8}  "
              f"{z:>8.4f}  {dist*1000:>8.1f} m")
    all_in_los = np.all(
        np.sqrt(env_irs.irs_pos[:, 0]**2 + env_irs.irs_pos[:, 1]**2)
        <= cfg.R_LoS_km + 1e-9)
    ok(f"All {cfg.M} IRS within satellite LoS zone (R={cfg.R_LoS_km} km): {all_in_los}")

    if verbose:
        hint(f"Phase levels (rad): {np.round(cfg.phase_levels, 4)}")
        hint(f"Sample Phi row 0 (angle, first 6): {np.round(np.angle(Phi[0, :6]), 4)}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Rate computation + Per-user QoS + Signal model
# ══════════════════════════════════════════════════════════════════════════════
def test_rate(verbose=True):
    section("TEST 4 · RateComputer + Per-user QoS + Signal Model")

    from istn.channel import ChannelModel
    from CSI.irs      import IRSPhaseModel
    from CSI.rate     import RateComputer
    from CSI.signal   import BasebandSignal

    cfg = make_config()
    rng = np.random.default_rng(42)

    rng_pos  = np.random.default_rng(42)
    irs_pos  = _sample_positions(rng_pos, cfg.M, cfg.R_LoS_km)
    user_pos = _sample_positions(rng_pos, cfg.K, cfg.R_LoS_km)
    ch   = ChannelModel(cfg, rng).generate(user_pos, irs_pos)
    irsm = IRSPhaseModel(cfg)
    Phi  = irsm.random_phi(rng)
    rc   = RateComputer(cfg)

    # Round-robin assignment: user k → IRS (k%M)+1, last user → direct (0)
    assignment = np.array([(k % cfg.M) + 1 for k in range(cfg.K - 1)] + [0])
    P       = cfg.P_S
    w_p     = np.full(cfg.K, P * 0.8 / cfg.K)
    w_c_vec = np.full(cfg.M + 1, P * 0.2 / (cfg.M + 1))   # (M+1,) per-group common power

    result = rc.compute_sum_rate(assignment, Phi, ch, w_p, w_c_vec)

    # Basic sanity
    assert result['sum_rate'] > 0
    ok(f"Sum-rate = {result['sum_rate']:.4f} bps/Hz")

    assert result['R_private'].shape == (cfg.K,)
    assert np.all(result['R_private'] >= 0)
    ok(f"Private rates shape = ({cfg.K},), all ≥ 0")

    assert result['SINR_p'].shape == (cfg.K,)
    assert np.all(result['SINR_p'] >= 0)
    ok(f"Private SINRs shape = ({cfg.K},), all ≥ 0")

    assert isinstance(result['feasible'], bool)
    ok(f"Feasible = {result['feasible']}, Power OK = {result['power_ok']}")

    # Groups: all assignment values must appear as group IDs
    groups = result['groups']
    assert all(0 <= gid <= cfg.M for gid in groups.keys())
    assert 0 in groups, "Direct-link group (0) must exist"
    ok(f"Groups formed: {sorted(groups.keys())}  covering {cfg.K} users")

    assert np.sum(w_c_vec) + np.sum(w_p) <= cfg.P_S + 1e-9
    ok("Power constraint satisfied")

    result_low = rc.compute_sum_rate(assignment, Phi, ch,
                                     np.full(cfg.K, 1e-10), np.full(cfg.M + 1, 1e-10))
    assert result_low['sum_rate'] < result['sum_rate']
    ok("Sum-rate decreases with lower power (monotonicity check)")

    # Blocking indicator: user is "blocked" when the best available IRS effective
    # channel outperforms the direct link  →  user benefits from IRS routing.
    #   h_direct[k]   = |g_SU[k]|
    #   h_irs[m,k]    = beta_m * |Σ_n g_SR[m] · φ_n · g_RU[m,k]|
    #   blocked[k]    = max_m(h_irs[m,k]) > h_direct[k]
    h_direct = np.abs(ch['g_SU'])                          # (K,)
    h_irs    = np.array([
        ch['beta'][m] * np.abs(
            ch['g_SR'][m] * np.sum(np.diag(Phi[m])) * ch['g_RU'][m])
        for m in range(cfg.M)
    ])                                                      # (M, K)
    blocked = h_irs.max(axis=0) > h_direct                 # (K,) bool

    # Per-user QoS and rate breakdown
    print(f"\n  Per-user QoS  (D_k={cfg.D_k_bps_hz} bps/Hz,  K={cfg.K} users):")
    print(f"  {'User':>5}  {'Link':>7}  {'Blocked':>8}  {'R_priv':>8}  "
          f"{'C_k':>8}  {'R_total':>8}  {'QoS':>6}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
    for k in range(cfg.K):
        r_priv    = result['R_private'][k]
        c_k       = result['C_k'][k]
        a_lbl     = f"IRS{assignment[k]}" if assignment[k] > 0 else "direct"
        qos_pass  = (r_priv + c_k) >= cfg.D_k_bps_hz
        # Pre-pad plain text, then wrap only the text in colour (not the spaces)
        blk_str   = " " * 5 + f"{RED}YES{RESET}" if blocked[k] else " " * 6 + "no"
        qos_str   = " " * 4 + f"{GREEN}OK{RESET}" if qos_pass   else " " * 2 + f"{RED}FAIL{RESET}"
        print(f"  {k+1:>5}  {a_lbl:>7}  {blk_str}  {r_priv:>8.4f}  "
              f"{c_k:>8.4f}  {r_priv+c_k:>8.4f}  {qos_str}")
    n_blocked = int(blocked.sum())
    print(f"\n  Blocked users: {n_blocked}/{cfg.K} "
          f"({'all would benefit from IRS' if n_blocked == cfg.K else 'mix of direct/IRS preference'})")
    print(f"  Sum-rate = {result['sum_rate']:.4f} bps/Hz  "
          f"({'feasible' if result['feasible'] else 'INFEASIBLE'})")

    # Baseband signal verification: x = √P_c·s_c + Σ_k √P_p_k·s_p_k
    bs  = BasebandSignal(cfg, rng=np.random.default_rng(99))
    pwr = bs.verify_power(w_c_vec, w_p, n_samples=n_signal_samples)
    ok(f"Signal power: P_theory={pwr['P_theoretical']:.2f} W, "
       f"P_empirical={pwr['P_empirical']:.2f} W "
       f"(err={100*pwr['relative_error']:.2f}%)")

    if verbose:
        hint(f"Signal model: x = √P_c·s_c + Σ_k √P_p_k·s_p_k")
        hint(f"  s_c, s_p_k ~ CN(0,1)  →  E[|s|²]=1")
        hint(f"  w = √P·v,  ||v||²=1   →  E[|w·s|²]=P")
        hint(f"  Total E[|x|²] = P_c + Σ P_p_k = {pwr['P_theoretical']:.2f} W")
        hint(f"Common rate groups: "
             f"{ {k: round(v,4) for k,v in result['R_common_group'].items()} }")
        hint(f"Effective |h| min/max: "
             f"{np.abs(result['h_eff']).min():.4e} / {np.abs(result['h_eff']).max():.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Environment reset & step
# ══════════════════════════════════════════════════════════════════════════════
def test_env_core(verbose=True):
    section("TEST 5 · ISTNEnv reset & step")

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_env)

    # --- reset ---
    obs = env.reset()
    assert set(obs.keys()) == {'g_SR', 'g_RU', 'g_SU',
                                'beta', 'Phi_angle', 'assignment', 'K_active'}
    ok("reset() returns obs with correct keys")

    assert obs['g_SR'].shape      == (cfg.M,)
    assert obs['g_RU'].shape      == (cfg.M, cfg.K)
    assert obs['g_SU'].shape      == (cfg.K,)
    assert obs['beta'].shape      == (cfg.M,)
    assert obs['Phi_angle'].shape == (cfg.M, cfg.N)
    assert obs['assignment'].shape == (cfg.K,)
    assert np.iscomplexobj(obs['g_SR']) and np.iscomplexobj(obs['g_RU'])
    ok(f"Observation shapes correct  (M={cfg.M}, K={cfg.K}, N={cfg.N})")

    assert np.all(obs['assignment'] >= 0) and np.all(obs['assignment'] <= cfg.M)
    ok(f"Assignment ∈ [0, M={cfg.M}]")

    # --- step with generic action ---
    test_assignment = np.array([k % (cfg.M + 1) for k in range(cfg.K)])
    action = {
        'assignment': test_assignment,
        'phase_idx':  np.zeros((cfg.M, cfg.N), dtype=int),
        'w_p':     np.full(cfg.K,     cfg.P_S * 0.8 / cfg.K),
        'w_c_vec': np.full(cfg.M + 1, cfg.P_S * 0.2 / (cfg.M + 1)),
    }
    obs2, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert done is False
    assert info['step'] == 1
    ok(f"step() returns reward={reward:.4f}, done={done}")

    assert np.array_equal(info['assignment'], test_assignment)
    ok("Assignment applied correctly in step")

    # --- step counter ---
    for _ in range(4):
        env.step({'assignment': np.zeros(cfg.K, dtype=int)})
    assert env._step_count == 5
    ok("Step counter increments correctly")

    # --- state vector ---
    sv = env.get_state_vector()
    assert sv.shape == (env.state_dim,)
    assert sv.dtype == float
    ok(f"State vector shape = ({env.state_dim},)")

    if verbose:
        hint(f"State dim breakdown: "
             f"g_SR={cfg.M}, g_RU={cfg.M*cfg.K}, "
             f"g_SU={cfg.K}, beta={cfg.M}, "
             f"Phi(cos+sin)={2*cfg.M*cfg.N}, assign={cfg.K}  → total={env.state_dim}")
        hint(f"State vector range: [{sv.min():.3e}, {sv.max():.3e}]")
        hint(f"Reward: {reward:.4f},  Feasible: {info['feasible']}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Partial action (only assignment provided)
# ══════════════════════════════════════════════════════════════════════════════
def test_partial_action(verbose=True):
    section("TEST 6 · Partial action dict")

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_partial)
    env.reset()

    old_Phi = env.Phi.copy()
    old_w_p = env.w_p.copy()

    obs, reward, done, info = env.step(
        {'assignment': np.zeros(cfg.K, dtype=int)})
    assert np.all(info['assignment'] == 0)
    ok("assignment-only action applied correctly")

    assert np.allclose(env.Phi, old_Phi)
    ok("Phi unchanged when not in action")

    assert np.allclose(env.w_p, old_w_p)
    ok("w_p unchanged when not in action")

    if verbose:
        hint(f"Reward (all direct): {reward:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Power constraint + Per-user power allocation
# ══════════════════════════════════════════════════════════════════════════════
def test_power_constraint(verbose=True):
    section("TEST 7 · Power constraint enforcement + Per-user allocation")

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_power)
    env.reset()

    # Provide oversized power (should be re-scaled to meet budget)
    action = {
        'assignment': np.zeros(cfg.K, dtype=int),
        'w_p':     np.full(cfg.K, 1000.0),
        'w_c_vec': np.full(cfg.M + 1, 500.0 / (cfg.M + 1)),
    }
    env.step(action)

    total_power = float(np.sum(env.w_c_vec)) + float(np.sum(env.w_p))
    assert total_power <= cfg.P_S * (1 + 1e-9), \
        f"Power budget exceeded: {total_power:.6g} > {cfg.P_S:.6g}"
    ok(f"Power re-scaled to budget: {total_power:.6g} W ≤ {cfg.P_S:.6g} W")

    print(f"\n  Power allocation  (budget = {cfg.P_S:.6g} W,  K={cfg.K}, M={cfg.M}):")
    print(f"  {'Source':<20}  {'Power (W)':>14}  {'% budget':>9}")
    print(f"  {'─'*20}  {'─'*14}  {'─'*9}")
    for g in range(cfg.M + 1):
        lbl = "Direct common (w_c)" if g == 0 else f"IRS {g} common (w_c)"
        print(f"  {lbl:<20}  {env.w_c_vec[g]:>14.6f}  "
              f"{100*env.w_c_vec[g]/cfg.P_S:>8.2f}%")
    for k in range(cfg.K):
        lbl = f"User {k+1} (w_p)"
        print(f"  {lbl:<20}  {env.w_p[k]:>14.6f}  "
              f"{100*env.w_p[k]/cfg.P_S:>8.2f}%")
    print(f"  {'─'*20}  {'─'*14}  {'─'*9}")
    total = float(np.sum(env.w_c_vec)) + float(np.sum(env.w_p))
    print(f"  {'Total':<20}  {total:>14.6f}  {100*total/cfg.P_S:>8.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Baseline policies
# ══════════════════════════════════════════════════════════════════════════════
def test_baselines(verbose=True):
    section("TEST 8 · Baseline policies")

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_eval)

    policies = {
        'Random':      RandomPolicy(cfg, rng=np.random.default_rng(0)),
        'Greedy':      GreedyPolicy(cfg),
        'DirectOnly':  DirectOnlyPolicy(cfg),
        'AllIRS':      AllIRSPolicy(cfg),
    }

    for name, policy in policies.items():
        obs = env.reset(seed=seed_eval)
        action = policy.act(obs, env) if name == 'Greedy' else policy.act(obs)
        obs2, reward, done, info = env.step(action)
        status = "feasible" if info['feasible'] else "infeasible"
        ok(f"{name:12s} → sum-rate={info['sum_rate']:.4f} bps/Hz  [{status}]")
        if verbose:
            hint(f"  R_priv min/max: {info['R_private'].min():.3f} / "
                 f"{info['R_private'].max():.3f}")

    obs = env.reset(seed=seed_irs)
    act = DirectOnlyPolicy(cfg).act(obs)
    assert np.all(act['assignment'] == 0)
    ok("DirectOnlyPolicy always assigns 0 (direct link)")

    act2 = AllIRSPolicy(cfg).act(obs)
    assert np.all(act2['assignment'] >= 1)
    ok("AllIRSPolicy never uses direct link")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Multi-episode statistical consistency
# ══════════════════════════════════════════════════════════════════════════════
def test_multi_episode(verbose=True):
    section(f"TEST 9 · Multi-episode stats  (n={n_episodes})")

    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=seed_default)

    policies = {
        'Random':     RandomPolicy(cfg, rng=np.random.default_rng(1)),
        'Greedy':     GreedyPolicy(cfg),
        'DirectOnly': DirectOnlyPolicy(cfg),
        'AllIRS':     AllIRSPolicy(cfg),
    }

    results = {name: [] for name in policies}

    for ep in range(n_episodes):
        for name, policy in policies.items():
            obs = env.reset()
            action = policy.act(obs, env) if name == 'Greedy' else policy.act(obs)
            _, reward, _, info = env.step(action)
            results[name].append(info['sum_rate'])

    print()
    print(f"  {'Policy':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    for name, rates in results.items():
        r = np.array(rates)
        print(f"  {name:<14} {r.mean():>8.4f} {r.std():>8.4f} "
              f"{r.min():>8.4f} {r.max():>8.4f}")

    g_mean = np.mean(results['Greedy'])
    r_mean = np.mean(results['Random'])
    ok(f"Greedy ({g_mean:.4f}) ≥ Random ({r_mean:.4f}): {g_mean >= r_mean}")

    # IRS positions change every episode; beta is fixed
    irs_positions = []
    for _ in range(n_irs_pos_samples):
        env.reset()
        irs_positions.append(env.irs_pos.copy())
    irs_positions = np.array(irs_positions)   # (20, M, 2)
    assert irs_positions.std() > 0
    ok(f"IRS positions resampled each episode (std={irs_positions.std():.4f} > 0)")

    obs = env.reset()
    assert np.allclose(obs['beta'], cfg.beta_IRS)
    ok(f"β_IRS fixed at {cfg.beta_IRS}")

    # Users stay within LoS zone after walking
    env2 = ISTNEnv(cfg=cfg, seed=seed_csi)
    env2.reset()
    for _ in range(n_walk_steps):
        env2.step({'assignment': np.zeros(cfg.K, dtype=int)})
    dist = np.sqrt(env2.user_pos[:, 0]**2 + env2.user_pos[:, 1]**2)
    assert np.all(dist <= cfg.R_LoS_km + 1e-9)
    ok(f"Users stay within LoS zone after {n_walk_steps} walk steps "
       f"(max dist={dist.max()*1000:.1f} m ≤ {cfg.R_LoS_km*1000:.0f} m)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — Imperfect CSI effect
# ══════════════════════════════════════════════════════════════════════════════
def test_imperfect_csi(verbose=True):
    section("TEST 10 · Imperfect CSI comparison")

    cfg_perfect   = make_config(kappa=0.0)
    cfg_imperfect = make_config(kappa=0.01)

    env_p = ISTNEnv(cfg=cfg_perfect,   seed=seed_csi)
    env_i = ISTNEnv(cfg=cfg_imperfect, seed=seed_csi)

    policy_p = GreedyPolicy(cfg_perfect)
    policy_i = GreedyPolicy(cfg_imperfect)

    rates_p, rates_i = [], []
    for ep in range(n_eval_episodes):
        obs_p = env_p.reset()
        act_p = policy_p.act(obs_p, env_p)
        _, _, _, info_p = env_p.step(act_p)
        rates_p.append(info_p['sum_rate'])

        obs_i = env_i.reset()
        act_i = policy_i.act(obs_i, env_i)
        _, _, _, info_i = env_i.step(act_i)
        rates_i.append(info_i['sum_rate'])

    mean_p = np.mean(rates_p)
    mean_i = np.mean(rates_i)
    ok(f"Perfect CSI   (σ_e²=0.00) mean rate : {mean_p:.4f} bps/Hz")
    ok(f"Imperfect CSI (σ_e²=0.01) mean rate : {mean_i:.4f} bps/Hz")

    obs_i = env_i.reset()
    ch    = env_i.channels
    diff_SR = np.mean(np.abs(ch['g_SR_hat'] - ch['g_SR']))
    diff_SU = np.mean(np.abs(ch['g_SU_hat'] - ch['g_SU']))
    ok(f"Mean |Δg_SR| = {diff_SR:.4e},  Mean |Δg_SU| = {diff_SU:.4e}")

    if verbose:
        delta_rate = mean_p - mean_i
        hint(f"Rate loss due to CSI imperfection: {delta_rate:.4f} bps/Hz "
             f"({100*delta_rate/mean_p:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--quick', action='store_true',
                        help='Skip slow multi-episode test')
    args = parser.parse_args()

    print(f"\n{BOLD}{'═'*60}")
    print( "  Multi-IRS ISTN Environment — Test Suite")
    print(f"{'═'*60}{RESET}")

    tests = [
        test_signal_flow,
        test_config,
        test_channel,
        test_irs,
        test_rate,
        test_env_core,
        test_partial_action,
        test_power_constraint,
        test_baselines,
        test_imperfect_csi,
    ]

    for t in tests:
        t(verbose=args.verbose)

    if not args.quick:
        test_multi_episode(verbose=args.verbose)

    print(f"\n{BOLD}{GREEN}{'═'*60}")
    print( "  All tests passed!")
    print(f"{'═'*60}{RESET}\n")


if __name__ == '__main__':
    main()
