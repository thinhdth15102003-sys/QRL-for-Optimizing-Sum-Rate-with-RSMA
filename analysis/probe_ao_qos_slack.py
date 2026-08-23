"""
probe_ao_qos_slack.py
---------------------
AO allocates C_k to meet D_k EXACTLY, computed on the estimated channel, and is
then scored on the true one. Every user funded to precisely D_k(g-hat) should
therefore fall short whenever the realisation is worse than the estimate --
about half the time, since g_hat = g + kappa*|g|*eps with eps ~ CN(0,1) is
symmetric. Yet AO measures 100.0% QoS at all three cases.

Either the measurement is wrong or something absorbs the error. This probe finds
out which, by reporting for AO:

  QoS ON BOTH VIEWS   the same allocation scored on g-hat and on g. If they are
                      identical, the CSI error never crosses the threshold.

  MARGIN              the distribution of R_k / D_k on the true channel. A
                      minimum far above 1 means no user is anywhere near the
                      cliff, so a few-percent channel error cannot push one over.

  THE CUSHION         oracle_ck_met funds minimally, then AO's _ck spreads the
                      LEFTOVER common budget evenly over the group. That leftover
                      is the actual safety margin -- reported as a fraction of
                      the group budget, and as a multiple of D_k per user.

Usage:
  python analysis/probe_ao_qos_slack.py --run results/result_299/best --states 60
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import load_training_cfg
from analysis.oracle_alloc import phi_from_idx, oracle_ck_met
from analysis.probe_ao_rate import AORatePolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    rc, Dk = env.rate_computer, cfg.D_k_bps_hz
    sw = {5: 1, 10: 1, 15: 2}.get(cfg.K, 1)
    ao = AORatePolicy(cfg, RateComputer(cfg), rounds=1, n_split=11,
                      phase_sweeps=sw)

    ratio_true, ratio_est, cushions, funded = [], [], [], []
    env.reset(seed=a.seed)
    for _ in range(a.states):
        obs = env._get_obs()
        act = ao.act(obs, env)
        phi = act["assignment"]
        Phi = phi_from_idx(act["phase_idx"], cfg)
        ids = sorted({int(x) for x in phi if x > 0})
        wp, wc = act["w_p"], act["w_c_vec"]

        p = rc.compute_rates_partial(phi, Phi, env.channels, wp, wc,
                                     active_irs_ids=ids)
        _, Ck_min = oracle_ck_met(p["R_private"], p["R_c_group"], p["groups"], Dk)
        Ck = Ck_min.copy()
        for g, m in p["groups"].items():
            m = np.asarray(list(m), dtype=int)
            if m.size:
                budget = float(p["R_c_group"].get(int(g), 0.0))
                left = budget - float(Ck_min[m].sum())
                if left > 1e-12:
                    Ck[m] += left / m.size
                    cushions.append(left / max(budget, 1e-12))
                    funded.append((left / m.size) / Dk)

        for use_true, sink in ((True, ratio_true), (False, ratio_est)):
            o = rc.compute_sum_rate(phi, Phi, env.channels, wp, wc, C_k=Ck,
                                    active_irs_ids=ids, use_true=use_true)
            R = np.asarray(o["R_private"]) + np.asarray(o["C_k"])
            sink.append(R / Dk)

        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)

    rt = np.concatenate(ratio_true)
    re_ = np.concatenate(ratio_est)
    print("=" * 72)
    print(f"  AO QoS SLACK · K={cfg.K} M={cfg.M} · D_k={Dk} · kappa={cfg.kappa}"
          f" · {a.states} states")
    print("=" * 72)
    print(f"  QoS on g-hat (design view) : {100*np.mean(re_ >= 1.0):6.2f}%")
    print(f"  QoS on true g  (scored)    : {100*np.mean(rt >= 1.0):6.2f}%")
    print(f"  users flipped to FAIL by the CSI error: "
          f"{100*np.mean((re_ >= 1.0) & (rt < 1.0)):.2f}%")
    print(f"\n  MARGIN  R_k / D_k on the true channel")
    q = np.percentile(rt, [0, 1, 5, 25, 50])
    print(f"    min {q[0]:.2f}   p1 {q[1]:.2f}   p5 {q[2]:.2f}"
          f"   p25 {q[3]:.2f}   median {q[4]:.2f}")
    print(f"    fraction of users within 20% of the threshold "
          f"(1.0 <= R/D < 1.2): {100*np.mean((rt >= 1.0) & (rt < 1.2)):.2f}%")
    if cushions:
        c = np.asarray(cushions)
        fu = np.asarray(funded)
        print(f"\n  CUSHION  leftover common budget spread over the group")
        print(f"    leftover as a share of the group budget: "
              f"median {100*np.median(c):.1f}%   p10 {100*np.percentile(c,10):.1f}%")
        print(f"    extra handed to each user: median {np.median(fu):.1f} x D_k")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
