"""Assemble the Rate-QoS frontier from probe_pareto_exhaustive shards.

Table III reports the MEAN QoS over states, so the ceiling a policy is measured
against has to be defined the same way,

    F(q) = max E[R_tot]   subject to   E[QoS] >= q,

not the per-state maximum subject to a per-state QoS threshold. The two are
different objects and the second is strictly lower, because it forbids trading
service between states. Reporting one and comparing against the other is part of
why the three earlier probes disagreed.

F is traced by Lagrangian sweep: for each mu >= 0 pick, in every state, the
achievable point maximising R + mu*Q, then average both coordinates. Sweeping mu
walks the upper concave envelope of the average-constrained frontier, which is
the frontier itself once states can be mixed. Both definitions are printed so the
size of the difference is on the record.

    python analysis/aggregate_pareto.py --K 5  --M 1
    python analysis/aggregate_pareto.py --K 10 --M 2
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as P                                              # noqa: E402
from params import make_config                                  # noqa: E402
from CSI.env import ISTNEnv                                     # noqa: E402
from CSI.rate import RateComputer                               # noqa: E402
from analysis.phase_oracle import oracle_phase_idx, est_channels  # noqa: E402
from analysis.oracle_alloc import phi_from_idx                  # noqa: E402

# Table III, greedy inference at P_S = 50 dBm: (R_tot, QoS)
POLICIES = {
    (5, 1): {"HQC-HAC (VQC)": (2.482, 0.924), "DNN": (2.501, 0.946),
             "PPO-flat": (2.417, 0.960), "AO": (1.876, 1.000)},
    (10, 2): {"HQC-HAC (VQC)": (1.742, 0.944), "DNN": (1.747, 0.921),
              "PPO-flat": (0.557, 0.472), "AO": (1.651, 1.000)},
}


def load_states(K, M):
    pat = "analysis/data/pareto_exh_K%dM%d_shard*.npz" % (K, M)
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit("no shards matching " + pat)
    sets, meta = [], []
    for f in files:
        z = np.load(f)
        m = z["meta"]
        for i in range(m.shape[0]):
            sets.append(z["s%d" % i])
            meta.append(tuple(m[i]))
    print("loaded %d shard file(s), %d states" % (len(files), len(sets)))
    return sets, meta


def lagrangian_frontier(sets, n_mu=600):
    """Upper concave envelope of (E[QoS], E[R_tot]) over the achievable sets."""
    mus = np.r_[0.0, np.logspace(-3, 3, n_mu)]
    out = []
    for mu in mus:
        r = q = 0.0
        for pts in sets:
            j = np.argmax(pts[:, 0] + mu * pts[:, 1])
            r += pts[j, 0]; q += pts[j, 1]
        out.append((q / len(sets), r / len(sets)))
    out = np.array(out)
    # keep the Pareto-efficient sweep points, sorted by QoS
    out = out[np.argsort(out[:, 0])]
    keep, best = [], -np.inf
    for i in range(len(out) - 1, -1, -1):        # from high QoS downwards
        if out[i, 1] > best + 1e-12:
            keep.append(i); best = out[i, 1]
    return out[sorted(keep)]


def threshold_frontier(sets, K):
    """The per-state definition, for comparison only."""
    grid = np.arange(K + 1) / K
    rows = []
    for t in grid:
        vals = []
        for pts in sets:
            ok = pts[pts[:, 1] >= t - 1e-9]
            vals.append(ok[:, 0].max() if ok.size else np.nan)
        rows.append((t, float(np.nanmean(vals)),
                     float(np.mean(~np.isnan(vals)))))
    return np.array(rows)


def reward_optimum(sets, n_steps=200):
    """The operating point the training objective itself selects.

    Per state, take the achievable point of highest J; average over states.
    The policies cannot be expected to reach the frontier, since the frontier
    maximises rate at a fixed service level while they maximise J. This point
    is the best J the same action space admits, so the distance from a policy
    to it is an optimisation gap, and the distance from it to the frontier is
    the price of the objective itself.
    """
    r = q = j = 0.0
    for pts in sets:
        if pts.shape[1] < 3:
            return None
        i = int(np.argmax(pts[:, 2]))
        r += pts[i, 0]; q += pts[i, 1]; j += pts[i, 2]
    n = len(sets)
    return r / n, q / n, j / n * n_steps


def F_at(front, q):
    """Frontier rate at mean QoS q, by interpolation along the envelope."""
    Q, R = front[:, 0], front[:, 1]
    if q <= Q[0]:
        return R[0]
    if q >= Q[-1]:
        return R[-1]
    return float(np.interp(q, Q, R))


def v_star_rate(cfg, rate, env, seeds, steps):
    """E[log2(1+Gamma_max)]: all power on the single strongest user-link pair,
    the one configuration with no intra-system interference."""
    vals = []
    for s in seeds:
        env.reset(seed=s)
        for t in range(steps):
            ch = env.channels
            est = est_channels(ch)
            gmax = 0.0
            for k in range(cfg.K):
                a = np.zeros(cfg.K, dtype=int)
                Phi0 = phi_from_idx(np.zeros((cfg.M, cfg.N), dtype=int), cfg)
                h = rate.effective_channels_all(a, Phi0, ch, use_true=True)
                gmax = max(gmax, abs(h[k]) ** 2)
                for m in range(1, cfg.M + 1):
                    a2 = np.zeros(cfg.K, dtype=int); a2[k] = m
                    Phi = phi_from_idx(oracle_phase_idx(est, a2, cfg), cfg)
                    h2 = rate.effective_channels_all(a2, Phi, ch, use_true=True)
                    gmax = max(gmax, abs(h2[k]) ** 2)
            vals.append(np.log2(1.0 + gmax * cfg.P_S / cfg.sigma2))
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--vstar-steps", type=int, default=40,
                    help="states per seed for V*_rate (its own cheap pass)")
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M, P_S_dBm=50.0, R_LoS_km=0.5)
    sets, meta = load_states(a.K, a.M)

    front = lagrangian_frontier(sets)
    thr = threshold_frontier(sets, a.K)

    print("\n=== F(q) = max E[R_tot] s.t. E[QoS] >= q   (Lagrangian envelope) ===")
    print("  %-9s %-10s" % ("E[QoS]", "E[R_tot]"))
    for q, r in front[::max(1, len(front) // 18)]:
        print("  %8.4f  %8.4f" % (q, r))
    print("  %8.4f  %8.4f   <- highest QoS reached" % (front[-1, 0], front[-1, 1]))

    print("\n=== per-state threshold definition (for comparison only) ===")
    print("  %-9s %-10s %s" % ("QoS>=", "mean R_tot", "states attainable"))
    for t, v, frac in thr:
        print("  %8.2f  %8.4f   %5.1f%%" % (t, v, 100 * frac))

    ropt = reward_optimum(sets)
    if ropt is not None:
        print("")
        print("=== operating point the reward itself selects ===")
        print("  R_tot %.4f   QoS %.4f   episode return %.1f   (frontier here %.4f)"
              % (ropt[0], ropt[1], ropt[2], F_at(front, ropt[1])))
    print("\n=== attainment at matched QoS ===")
    print("  %-16s %-9s %-9s %-9s %s"
          % ("policy", "R_tot", "QoS", "F(QoS)", "attainment"))
    for name, (r, q) in POLICIES.get((a.K, a.M), {}).items():
        f = F_at(front, q)
        print("  %-16s %8.4f  %7.3f  %8.4f   %6.1f%%"
              % (name, r, q, f, 100.0 * r / f))

    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=42, n_steps_ep=200,
                  reward_noise_avg=P.reward_noise_avg)
    vs = v_star_rate(cfg, rate, env, [42, 43, 44], a.vstar_steps)
    R_serve_all = front[-1, 1]
    p_hat = (cfg.D_k_bps_hz / (cfg.D_k_bps_hz + P.epsilon_qp)) ** 2
    lam_crit = (vs - R_serve_all) / ((cfg.K - 1) * p_hat)
    print("\n=== closed-form quantities, re-measured ===")
    print("  V*_rate  = E[log2(1+Gamma_max)]      = %8.4f   (%d states)"
          % (vs, 3 * a.vstar_steps))
    print("  R*(K)    = frontier at max mean QoS  = %8.4f  at QoS %.3f"
          % (R_serve_all, front[-1, 0]))
    print("  p_hat    = (D/(D+eps_q))^2           = %8.4f" % p_hat)
    print("  lambda_crit = (V*_rate - R*(K))/((K-1)p_hat) = %8.4f" % lam_crit)
    print("  training lambda_D = %.2f  ->  %s"
          % (P.lambda_D,
             "below lambda_crit (single-user regime)" if P.lambda_D < lam_crit
             else "above lambda_crit (serve-all regime)"))

    m = a.K
    print("\n  equal-split identity R(m) = m log2(m/(m-1)):  R(%d) = %.4f"
          % (m, m * np.log2(m / (m - 1.0))))


if __name__ == "__main__":
    main()
