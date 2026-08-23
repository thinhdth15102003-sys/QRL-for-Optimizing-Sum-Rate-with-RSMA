"""
probe_flat_rate_spread.py
-------------------------
Where does PPO-flat's sum-rate actually go?

Written because the retrained flat rows at P_S=70 report R_tot ~10.2 in BOTH
cases (C1 10.2956, C2 10.1880) against ~2.1-2.3 for AO/DNN/VQC at the same
budget. The power-budget check already ruled out over-spending
(total/budget = 1.0000), so the remaining question is distributional: is every
user served better, or is nearly all of it landing on one user while the rest
merely clear a very low D_k floor?

Reports, over (state, user) pairs: R_tot, the share held by the top user and
top two, the share left to the bottom half, per-user rate quantiles, and the
fraction of users sitting at or just above D_k.

Usage:
  python analysis/probe_flat_rate_spread.py --run results/result_399/checkpoints/ep_20004
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P
from CSI.env import ISTNEnv
from infer import load_training_cfg, _compute_irs_favored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()

    from RL.flat_actor import FlatActor
    cfg = load_training_cfg(a.run)
    fa = FlatActor.from_dir(os.path.join(a.run, "agents"), seed=0)
    dem = np.full(cfg.K, cfg.D_k_bps_hz)
    print("env %s" % a.run)
    print("  K=%d M=%d N=%d P_S=%.1f dBm  D_k=%.4f"
          % (cfg.K, cfg.M, cfg.N, cfg.P_S_dBm, cfg.D_k_bps_hz))

    rows, wrows, crows = [], [], []
    for s in a.seeds:
        env = ISTNEnv(cfg=cfg, seed=s, n_steps_ep=a.steps,
                      reward_noise_avg=getattr(P, "reward_noise_avg", 1))
        obs = env.reset(seed=s)
        prev = None
        for _ in range(a.steps):
            h_now = env.rate_computer.effective_channels_all(
                env.assignment, env.Phi, env.channels)
            st = fa.extract_state(obs, dem, _compute_irs_favored(env),
                                  h_eff=h_now, prev_rates=prev, update_norm=False)
            act, _, fi = fa.forward(st, greedy=True)
            Phi = env.phase_model.build_phi(
                env.phase_model.index_to_phase(act["phase_idx"]))
            part = env.rate_computer.compute_rates_partial(
                act["phi"], Phi, env.channels, fi["w_p"], fi["w_c_vec"],
                active_irs_ids=act["active_ids"])
            C_k, _, _, _ = fa.sample_ck(fi["logits"], act["phi"],
                                        part["R_c_group"], greedy=True)
            obs, _, _, info = env.step({"assignment": act["phi"],
                                        "phase_idx": act["phase_idx"],
                                        "w_p": fi["w_p"],
                                        "w_c_vec": fi["w_c_vec"], "C_k": C_k})
            Rp = np.asarray(info["R_private"], dtype=float)
            rows.append(Rp + np.asarray(C_k, dtype=float))
            wrows.append(np.asarray(fi["w_p"], dtype=float))
            crows.append(float(np.sum(fi["w_c_vec"])))
            prev = (Rp, float(np.mean(C_k)))

    R = np.vstack(rows)                       # (states, K)
    tot = R.sum(1)
    srt = np.sort(R, axis=1)[:, ::-1]
    half = max(1, cfg.K // 2)

    print("\n  states=%d  K=%d" % (R.shape[0], cfg.K))
    print("  R_tot          mean %.4f" % tot.mean())
    print("  top-1 user     mean %.4f   %.1f%% of R_tot"
          % (srt[:, 0].mean(), 100 * srt[:, 0].mean() / tot.mean()))
    print("  top-2 users                  %.1f%% of R_tot"
          % (100 * srt[:, :2].sum(1).mean() / tot.mean()))
    print("  bottom half                  %.1f%% of R_tot"
          % (100 * srt[:, half:].sum(1).mean() / tot.mean()))
    W = np.vstack(wrows); C = np.asarray(crows)
    wsrt = np.sort(W, axis=1)[:, ::-1]
    wtot = W.sum(1)
    print("  PRIVATE power: top-1 user holds %.1f%% of sum(w_p); "
          "private/(private+common) = %.1f%%"
          % (100 * wsrt[:, 0].mean() / wtot.mean(),
             100 * wtot.mean() / (wtot.mean() + C.mean())))
    q = np.percentile(R.ravel(), [5, 25, 50, 75, 95])
    print("  per-user rate  p05 %.4f  p25 %.4f  med %.4f  p75 %.4f  p95 %.4f" % tuple(q))
    print("  D_k=%.4f   users at or below 1.5*D_k: %.1f%%   below D_k: %.1f%%"
          % (cfg.D_k_bps_hz,
             100 * np.mean(R.ravel() <= 1.5 * cfg.D_k_bps_hz),
             100 * np.mean(R.ravel() < cfg.D_k_bps_hz)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
