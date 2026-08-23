"""Measure empirical IRS-route SNR pools for the composition-capacity table.
  P_blk = Γ_irs of BLOCKED users (near IRS).
  P_free = Γ_irs of FREE users that BENEFIT from the IRS (Γ_irs > Γ_dir).
Γ = |h_eff|²·P_S/σ² (power-independent channel SNR). Each user kept on its NEAREST IRS.
Measured at P50/N24, geom 1/1; probe scales Γ ∝ P_S. Saved to scratchpad/gbar_pools.npz."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from params import make_config
from CSI.env import ISTNEnv

cfg = make_config(K=10, M=2, N=24, P_S_dBm=50.0)
env = ISTNEnv(cfg, seed=42, n_steps_ep=10)
K = cfg.K
RHO = 0.2
w_p = np.full(K, (1 - RHO) / K * cfg.P_S)
w_c = np.full(2, RHO / 2 * cfg.P_S)
pidx = np.zeros((cfg.M, cfg.N), dtype=int)

blk_girs, fre_girs, fre_gdir = [], [], []
for ep in range(600):
    env.reset()
    ch = env.channels
    Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
    blocked = ch['su_blocked']
    nearest = np.argmin(ch['d_IRS_U'], axis=0)          # 0-based; ==0 → IRS-1
    # Γ_irs (all on IRS-1) and Γ_dir (all direct) on the SAME channels
    out_i = env.rate_computer.compute_sum_rate(np.ones(K, int), Phi, ch, w_p, w_c, active_irs_ids=[1])
    g_irs = (np.abs(out_i['h_eff']) ** 2) * cfg.P_S / cfg.sigma2
    out_d = env.rate_computer.compute_sum_rate(np.zeros(K, int), Phi, ch, w_p, w_c[:1], active_irs_ids=[])
    g_dir = (np.abs(out_d['h_eff']) ** 2) * cfg.P_S / cfg.sigma2
    sel = nearest == 0
    blk_girs.extend(g_irs[sel & blocked].tolist())
    ff = sel & ~blocked
    fre_girs.extend(g_irs[ff].tolist())
    fre_gdir.extend(g_dir[ff].tolist())

blk_girs = np.array(blk_girs)
fre_girs = np.array(fre_girs); fre_gdir = np.array(fre_gdir)
benef = fre_girs > fre_gdir                              # IRS beats direct
P_free = fre_girs[benef]

np.savez(os.path.join(os.path.dirname(__file__), 'gbar_pools.npz'),
         P_blk=blk_girs, P_free=P_free, GDIR_MED=np.median(fre_gdir))
print(f"σ²={cfg.sigma2:.3e}  N={cfg.N}  P_S={cfg.P_S:.1f} W  (P50 anchor)")
print(f"BLOCKED   n={len(blk_girs):5d}  median={np.median(blk_girs):8.1f}  mean={blk_girs.mean():8.1f}")
print(f"FREE all  n={len(fre_girs):5d}  median Γ_irs={np.median(fre_girs):6.1f}  median Γ_dir={np.median(fre_gdir):6.1f}")
print(f"FREE benef(Γ_irs>Γ_dir) n={len(P_free):5d} ({100*benef.mean():.0f}% of free)  "
      f"median Γ_irs={np.median(P_free):8.1f}  mean={P_free.mean():8.1f}")
print(f"→ pools saved to scratchpad/gbar_pools.npz")
