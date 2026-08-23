"""
probe_link_budget.py
--------------------
Where does our sum-rate physically come from, and how far is it from Tan's?

Motivation: after normalising the noise to Tan (n_0 ~ N(-30 dBW, 10 dB^2), the
reading fixed in docs/Dk-Noise-Calibration.md), our AO still reports 1.84 at
Case 1 against Tan's 1.52. A 21% rate gap sounds large, but rate is logarithmic
-- the question is how large it is in dB of effective SINR, and which branch of
the channel carries it.

Three measurements:

  LINK BUDGET   median per-user SNR of the direct branch |g_SU|^2 P_S/sigma^2
                and of the optimal-phase IRS branch
                    |beta * conj(g_SR) * N * g_RU|^2 P_S/sigma^2
                both in dB. Tan's g_RU carries no path loss and no rain, ours
                carries both, so this quantifies exactly the term that differs.

  BRANCH SHARE  what an IRS assignment is worth over the direct link per user.
                If the IRS branch sits far below the direct branch, the reported
                sum-rate is set by g_SU -- which is modelled identically in both
                papers -- and the g_RU discrepancy cannot explain the gap.

  dB GAP        the effective per-user SINR implied by a given sum-rate,
                gamma = 2^(R_tot/K) - 1. Converts the rate gap into the dB gap
                that a reviewer would actually have to explain.

Loads the config from a trained Case-1 run dir so params.py (global, read at
import by every live training) is never touched.

Usage:
  python analysis/probe_link_budget.py --run results/result_234
  python analysis/probe_link_budget.py --run results/result_234 --tan-rate 1.52
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import load_training_cfg


def _db(x):
    x = np.asarray(x, dtype=float)
    return 10.0 * np.log10(np.maximum(x, 1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="a trained run dir, for its cfg")
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tan-rate", type=float, default=1.52,
                    help="Tan's reported sum-rate at the matching setting")
    ap.add_argument("--our-rate", type=float, default=None,
                    help="our AO sum-rate; default = read nothing, report ours only")
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    K, M = cfg.K, cfg.M

    dir_snr, irs_snr, s2 = [], [], []
    obs = env.reset(seed=a.seed)
    N = int(np.asarray(env.channels["g_RU"]).shape[1])
    for _ in range(a.states):
        ch = env.channels
        # sigma^2 is redrawn per step; one draw per state over --states states is
        # a fair sample of the same distribution the env scores on.
        sig2 = np.asarray(env.channel_model.sample_noise_sigma2(),
                          dtype=float).reshape(-1)
        if sig2.size == 1:
            sig2 = np.full(K, float(sig2[0]))
        s2.append(sig2)

        g_SU = np.asarray(ch["g_SU"])                      # (K,)
        g_SR = np.asarray(ch["g_SR"])                      # (M,)
        g_RU = np.asarray(ch["g_RU"])                      # (M, N, K)
        beta = np.asarray(ch["beta"], dtype=float).reshape(-1)   # (M,)

        # direct branch
        dir_snr.append(np.abs(g_SU) ** 2 * cfg.P_S / sig2)

        # IRS branch at the best achievable alignment: every element co-phased,
        # so the element sum is the sum of magnitudes rather than a random walk.
        coh = np.abs(g_RU).sum(axis=1)                     # (M, K)
        h_irs = beta[:, None] * np.abs(g_SR)[:, None] * coh   # (M, K)
        irs_snr.append((np.abs(h_irs) ** 2 * cfg.P_S / sig2[None, :]).max(axis=0))

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
        obs = env._get_obs()

    dir_snr = np.concatenate(dir_snr)
    irs_snr = np.concatenate(irs_snr)
    s2 = np.concatenate(s2)

    print("=" * 76)
    print(f"  LINK BUDGET · {a.run} · K={K} M={M} N={N} · {a.states} states")
    print(f"  P_S = {_db(cfg.P_S)[()]:.1f} dBW   median sigma^2 = "
          f"{np.median(_db(s2)):.1f} dBW")
    print("=" * 76)
    for lbl, v in (("direct  |g_SU|^2 P/s2", dir_snr),
                   ("IRS opt-phase (best m)", irs_snr)):
        q = np.percentile(_db(v), [10, 50, 90])
        print(f"  {lbl:<24} p10 {q[0]:>8.2f}   median {q[1]:>8.2f}   "
              f"p90 {q[2]:>8.2f}  dB")

    adv = _db(irs_snr) - _db(dir_snr)
    print(f"\n  IRS advantage over direct   median {np.median(adv):>8.2f} dB"
          f"   ({100 * np.mean(adv > 0):.1f}% of users have IRS > direct)")

    cap_dir = np.log2(1.0 + dir_snr)
    cap_both = np.log2(1.0 + dir_snr + irs_snr)
    print(f"  single-user capacity   direct only {np.median(cap_dir):.4f}"
          f"   direct+IRS {np.median(cap_both):.4f}  bps/Hz  (median)")

    print("\n" + "-" * 76)
    print("  RATE GAP EXPRESSED IN dB OF EFFECTIVE SINR   gamma = 2^(R/K) - 1")
    print("-" * 76)
    rates = [("Tan (reported)", a.tan_rate)]
    if a.our_rate:
        rates.append(("ours (AO)", a.our_rate))
    for lbl, R in rates:
        per = R / K
        gam = 2.0 ** per - 1.0
        print(f"  {lbl:<18} R_tot {R:.4f}   per-user {per:.4f} bps/Hz   "
              f"gamma {_db(gam):>7.2f} dB")
    if a.our_rate:
        g1 = 2.0 ** (a.tan_rate / K) - 1.0
        g2 = 2.0 ** (a.our_rate / K) - 1.0
        print(f"\n  rate gap  {100 * (a.our_rate / a.tan_rate - 1):+.1f}%"
              f"   ->  SINR gap  {_db(g2) - _db(g1):+.2f} dB")
    print("=" * 76)


if __name__ == "__main__":
    sys.exit(main())
