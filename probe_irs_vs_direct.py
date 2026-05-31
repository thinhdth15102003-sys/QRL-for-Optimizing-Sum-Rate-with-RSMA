"""
probe_irs_vs_direct.py
----------------------
Decide whether HIGH IRS-usage (target IRS/K ≥ ~2/3) is physically OPTIMAL, by comparing,
per user, the OPTIMAL-PHASE IRS link gain vs the DIRECT link gain — for the CURRENT
SystemConfig (params.py ACTIVE CASE / R_LoS).

Why this matters
----------------
The agent only routes a user to IRS when IRS beats direct (in reward). The built-in
"All-IRS" feasibility baseline uses RANDOM phases (|Σφ|≈√N) → an unfair lower bound that
makes IRS look worse than it is. This probe uses the OPTIMAL phase instead.

Channel model (CSI/rate.py)
---------------------------
  direct:  h_dir[k]      = g_SU_hat[k]                         → |h_dir[k]|  = |g_SU[k]|
  IRS:     h_irs[m,k]    = beta[m]·conj(g_SR[m])·(Σ_n φ_n)·g_RU[m,k]
           |Σ_n φ_n| ≤ N, max = N when ALL elements aligned to one level (achievable
           exactly with discrete 2-bit phases). Since g_SR, g_RU are scalars per (m)/(m,k),
           magnitude is independent of the alignment angle → ALL users of an IRS get their
           max simultaneously (a property of this simplified model; no multi-user tradeoff).
  → |h_irs[m,k]|_opt = beta[m]·|g_SR[m]|·N·|g_RU[m,k]|

We compare channel GAIN |h|² (first-order proxy for achievable rate at fixed power/noise).

CPU-only (channels are numpy) → safe to run alongside GPU training.

Usage:  python probe_irs_vs_direct.py
"""
import numpy as np
from params import make_config
from CSI.env import ISTNEnv

N_SAMPLES = 300          # number of fresh spawns (resets) to average over
SEED      = 20260530


def main():
    cfg = make_config()
    N   = cfg.N
    env = ISTNEnv(cfg=cfg, seed=SEED, n_steps_ep=1,
                  reward_noise_avg=1)

    gain_dir_all, gain_irs_all, blocked_all = [], [], []
    for _ in range(N_SAMPLES):
        env.reset()
        ch = env.channels
        gd  = np.abs(ch['g_SU_hat']) ** 2                       # (K,) direct gain
        # IRS-optimal gain per (m,k): (beta·|g_SR|·N·|g_RU|)^2
        coeff = ch['beta'][:, None] * np.abs(ch['g_SR_hat'])[:, None] * N  # (M,1)
        gi_mk = (coeff * np.abs(ch['g_RU_hat'])) ** 2           # (M,K)
        gi    = gi_mk.max(axis=0)                                # (K,) best IRS per user
        gain_dir_all.append(gd)
        gain_irs_all.append(gi)
        blocked_all.append(np.asarray(ch['su_blocked'], dtype=bool))

    gd  = np.concatenate(gain_dir_all)        # (N_SAMPLES*K,)
    gi  = np.concatenate(gain_irs_all)
    blk = np.concatenate(blocked_all)

    ratio_db = 10.0 * np.log10((gi + 1e-30) / (gd + 1e-30))     # IRS-opt / direct, per user

    def report(mask, label):
        if mask.sum() == 0:
            print(f"  {label:<14}: (no users)")
            return
        win = float(np.mean(gi[mask] > gd[mask])) * 100.0
        med = float(np.median(ratio_db[mask]))
        p25 = float(np.percentile(ratio_db[mask], 25))
        p75 = float(np.percentile(ratio_db[mask], 75))
        print(f"  {label:<14}: IRS-opt > direct ở {win:5.1f}% user | "
              f"gain ratio (dB) median={med:+5.1f}  [p25 {p25:+.1f}, p75 {p75:+.1f}]")

    print("=" * 78)
    print(f"  PROBE IRS-optimal vs DIRECT   ·   K={cfg.K} M={cfg.M} N={N} "
          f"P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km}km")
    print(f"  Samples: {N_SAMPLES} spawns × {cfg.K} users = {len(gd)} user-instances")
    print("=" * 78)
    print(f"  Blocked fraction (Blk/K): {blk.mean()*100:.1f}%")
    print("-" * 78)
    report(np.ones_like(blk),  "TẤT CẢ user")
    report(blk,                "BỊ BLOCK")
    report(~blk,               "KHÔNG block")
    print("-" * 78)
    # Verdict
    win_nonblk = float(np.mean(gi[~blk] > gd[~blk])) * 100.0 if (~blk).sum() else 0.0
    print("  KẾT LUẬN cho mục tiêu IRS/K ≥ 2/3:")
    if win_nonblk >= 60:
        print(f"    ✓ IRS-aligned THẮNG direct ở {win_nonblk:.0f}% user non-block → routing")
        print(f"      cả user non-block qua IRS là OPTIMAL → mục tiêu 2/3 IRS KHẢ THI.")
    elif win_nonblk >= 25:
        print(f"    ~ IRS thắng ở {win_nonblk:.0f}% user non-block (một phần) → 2/3 IRS hợp lý")
        print(f"      cho subset đó; phần còn lại direct tốt hơn. Mục tiêu 2/3 hơi cao.")
    else:
        print(f"    ✗ IRS chỉ thắng {win_nonblk:.0f}% user non-block → direct tốt hơn cho đa số")
        print(f"      → ép 2/3 IRS là DƯỚI TỐI ƯU. Cân nhắc đổi hệ (↑N / ↓G_U) hoặc hạ mục tiêu.")
    print("=" * 78)


if __name__ == "__main__":
    main()
