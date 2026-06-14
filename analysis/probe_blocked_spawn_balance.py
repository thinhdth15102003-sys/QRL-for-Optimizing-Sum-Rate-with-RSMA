"""
probe_blocked_spawn_balance.py
------------------------------
Q: Blocked users co duoc chia deu cho cac IRS khi spawn khong?
A (code, CSI/env.py reset): KHONG — n_blocked = round(K*2/3), moi blocked user chon
building ~ Uniform{0..M-1} DOC LAP (rng.integers) → split ~ Binomial, co the lech nang.

Probe nay do EMPIRICAL tren ACTIVE CASE (params.py K, M):
 (1) Phan phoi split blocked-users-per-building qua nhieu reset (vs Binomial ly thuyet).
 (2) Voi moi confined user: IRS-gain own-building vs other-IRS (dB) → chi phi neu
     load-balance bat user sang IRS khac (quyet dinh "ep [3,4]" co re hay dat).
 (3) Sanity: confined user co su_blocked=True dung ty le khong.

CPU-only, nhanh. Usage:  python analysis/probe_blocked_spawn_balance.py [--resets 2000]
"""

# ── path bootstrap ───────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import argparse, collections, math
import numpy as np
from params import make_config
from CSI.env import ISTNEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resets", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260612)
    args = ap.parse_args()

    cfg = make_config()
    K, M = cfg.K, cfg.M
    n_blocked = int(round(K * 2 / 3))
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=1, reward_noise_avg=1)

    split_count = collections.Counter()
    max_load = []
    ratio_db_all = []          # own-IRS coeff / best-OTHER-IRS coeff per confined user (dB)
    blocked_flag_match = []

    for _ in range(args.resets):
        env.reset()
        bld = env._user_building[:n_blocked]
        counts = tuple(sorted(int(np.sum(bld == m)) for m in range(M)))
        split_count[counts] += 1
        max_load.append(max(counts))

        ch = env.channels
        # full IRS link coeff per (m,k): beta_m * |g_SR[m]| * |g_RU[m,k]|  (N chung, bo qua)
        coeff = ch["beta"][:, None] * np.abs(ch["g_SR_hat"])[:, None] * np.abs(ch["g_RU_hat"])  # (M,K)
        for i in range(n_blocked):
            own = int(bld[i])
            others = [m for m in range(M) if m != own]
            if not others:
                continue
            best_other = max(coeff[m, i] for m in others)
            ratio_db_all.append(10 * math.log10((coeff[own, i] + 1e-30) / (best_other + 1e-30)))
        blocked_flag_match.append(float(np.mean(np.asarray(ch["su_blocked"], bool)[:n_blocked])))

    print("=" * 74)
    print(f"  SPAWN BALANCE · K={K} M={M} → n_blocked={n_blocked} · {args.resets} resets")
    print("=" * 74)
    print(f"  (1) SPLIT distribution (sorted per-building counts) — vs Binomial:")
    tot = sum(split_count.values())
    for s, c in sorted(split_count.items(), key=lambda kv: -kv[1]):
        print(f"      {s}: {100*c/tot:5.1f}%")
    ml = np.array(max_load)
    fair_max = math.ceil(n_blocked / M)
    print(f"      max-load/IRS: mean {ml.mean():.2f} · P(max > {fair_max} [fair]) = "
          f"{100*np.mean(ml > fair_max):.1f}%")
    if ratio_db_all:
        r = np.array(ratio_db_all)
        print(f"  (2) OWN-IRS vs best-OTHER-IRS gain (confined users, dB):")
        print(f"      median {np.median(r):+.1f} dB · p25 {np.percentile(r,25):+.1f} · "
              f"p75 {np.percentile(r,75):+.1f} · own thang {100*np.mean(r>0):.1f}%")
        print(f"      → chi phi re-route sang IRS khac ≈ median {abs(np.median(r)):.1f} dB")
    print(f"  (3) sanity: confined → su_blocked rate = {100*np.mean(blocked_flag_match):.1f}%")
    print("=" * 74)


if __name__ == "__main__":
    main()
