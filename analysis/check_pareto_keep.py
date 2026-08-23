"""Does the survivor cap in probe_pareto_exhaustive lose frontier points?

Builds the achievable set for a few states at two caps and compares the upper
envelope R(q) they induce, level by level. If the cap were biting, the smaller
cap would give a lower envelope somewhere.

    python analysis/check_pareto_keep.py --K 10 --M 2 --states 3
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P                                              # noqa: E402
from params import make_config                                  # noqa: E402
from CSI.env import ISTNEnv                                     # noqa: E402
from CSI.rate import RateComputer                               # noqa: E402
from analysis.probe_pareto_exhaustive import (                  # noqa: E402
    enumerate_assignments, group_stream_index, build_h2, eval_grid)


def envelope(pts, grid):
    """Best R_tot among points with QoS >= q, for each q in grid."""
    out = np.full(len(grid), np.nan)
    for i, q in enumerate(grid):
        ok = pts[pts[:, 1] >= q - 1e-12]
        if ok.size:
            out[i] = ok[:, 0].max()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--caps", default="32,256")
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M, P_S_dBm=50.0, R_LoS_km=0.5)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=42, n_steps_ep=200,
                  reward_noise_avg=P.reward_noise_avg)
    digits = enumerate_assignments(a.K, a.M)
    sidx, nstr = group_stream_index(digits, a.M)
    caps = [int(c) for c in a.caps.split(",")]
    qgrid = np.arange(a.K + 1) / a.K

    env.reset(seed=42)
    worst = 0.0
    for st in range(a.states):
        ch = env.channels
        s2s = [env.channel_model.sample_noise_sigma2()
               for _ in range(P.reward_noise_avg)]
        H2e, H2t, _ = build_h2(digits, ch, cfg, rate)
        envs = {}
        for c in caps:
            pts = eval_grid(H2e, H2t, digits, sidx, nstr, cfg, s2s, c)
            envs[c] = envelope(pts, qgrid)
        lo, hi = envs[caps[0]], envs[caps[-1]]
        d = np.nanmax(np.abs(np.nan_to_num(hi - lo)))
        worst = max(worst, d)
        print("state %d  cap%d vs cap%d  max |dR(q)| = %.3e"
              % (st, caps[0], caps[-1], d))
        for q, x, y in zip(qgrid, lo, hi):
            if not (np.isnan(x) and np.isnan(y)):
                flag = "" if abs(np.nan_to_num(x - y)) < 1e-12 else "   <-- DIFFERS"
                print("    QoS>=%.2f   %8.4f   %8.4f%s" % (q, x, y, flag))
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    print("\nworst envelope difference across states: %.3e" % worst)
    print("cap %d is %s" % (caps[0], "SAFE" if worst < 1e-9 else "TOO SMALL"))


if __name__ == "__main__":
    main()
