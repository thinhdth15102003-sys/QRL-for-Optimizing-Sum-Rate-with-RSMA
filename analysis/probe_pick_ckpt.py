"""
probe_pick_ckpt.py
------------------
Rank checkpoints of ONE run by the environment's own reward — the objective the
agent was actually trained against — rather than by the comparison metric.

Why this is not the same as probe_vs_ao's J, even though the formula is
identical (R_tot - lambda_D * sum (shortfall/(D_k+eps))^2):

  probe_vs_ao substitutes the ORACLE C_k for every arm, because C_k is
  sum-rate-neutral and AO is locked to that rule, so holding it fixed is what
  makes the arms comparable. That is right for a cross-method table and WRONG
  for choosing between checkpoints of one policy: it scores a C_k head the
  deployed model does not have.

Here every episode is run through `run_hqchac_episode` exactly as inference
does — greedy actions, the policy's OWN C_k head, the env's own noise draws and
its own reward accumulator. That is "the reward function as designed".

Reports per checkpoint: episode reward (mean +/- sd over env seeds), R_tot, QoS,
and the same figure at several lambda_D so a checkpoint that merely shaved its
QoS margin cannot hide behind the weak penalty at 1.5.

Usage:
  python analysis/probe_pick_ckpt.py --run results/result_261 \
      --ckpts 4000 6000 8000 10200 --seeds 42 7 21 --eps 6
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import load_training_cfg, load_agents, run_hqchac_episode

LAMBDAS = (1.5, 3.0, 5.0, 10.0, 20.0)


def eval_ckpt(path, seeds, n_eps, n_steps):
    cfg = load_training_cfg(path)
    actor, phase_net, power_net, ck_net = load_agents(path, seed=0)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    Dk = cfg.D_k_bps_hz
    eps_q = float(getattr(cfg, "epsilon_qp", 1e-3))

    per_seed_rew, rt, qs, short2 = [], [], [], []
    for sd in seeds:
        env = ISTNEnv(cfg, seed=sd, n_steps_ep=n_steps)
        acc = []
        for _ in range(n_eps):
            m = run_hqchac_episode(env, actor, phase_net, power_net, ck_net,
                                   cfg, demand, n_steps, greedy=True)
            acc.append(m["reward"])
            rt.append(m["sum_rate"])
            qs.append(m["qos_frac"])
            # rebuild the penalty term so J can be recomputed at other lambdas:
            # reward = n_steps * (R_tot - lambda*S)  =>  S = (R_tot - reward/n)/lambda
            lam = float(getattr(cfg, "lambda_D", 1.5))
            S = (m["sum_rate"] - m["reward"] / n_steps) / lam
            short2.append(S)
        per_seed_rew.append(float(np.mean(acc)))
    return (np.array(per_seed_rew), float(np.mean(rt)), float(np.mean(qs)),
            float(np.mean(short2)), n_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="episode numbers under checkpoints/, or 'best'/'final'")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 7, 21])
    ap.add_argument("--eps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=200)
    a = ap.parse_args()

    rows = []
    for c in a.ckpts:
        if c == "final":
            p = a.run
        elif c == "best":
            p = os.path.join(a.run, "best")
        else:
            p = os.path.join(a.run, "checkpoints", "ep_%05d" % int(c))
        if not os.path.isdir(p):
            print("  skip %s (not found)" % p)
            continue
        r, rt, q, S, nst = eval_ckpt(p, a.seeds, a.eps, a.steps)
        rows.append((c, r, rt, q, S, nst))
        print("    %-8s reward %8.2f  R_tot %.4f  QoS %5.1f%%"
              % (c, r.mean(), rt, 100 * q), flush=True)

    if not rows:
        return 1
    nst = rows[0][5]
    print("\n" + "=" * 84)
    print("  PICK CHECKPOINT · %s · env reward, greedy, policy's OWN C_k"
          % a.run)
    print("  %d episodes x %d env seeds x %d steps" % (a.eps, len(a.seeds), nst))
    print("=" * 84)
    print("  %-9s %10s %8s %8s %7s" % ("ckpt", "reward", "sd", "R_tot", "QoS"))
    best = max(rows, key=lambda r: r[1].mean())
    for c, r, rt, q, S, _ in rows:
        mark = "  <- best" if c == best[0] else ""
        print("  %-9s %10.2f %8.2f %8.4f %6.1f%%%s"
              % (c, r.mean(), r.std(ddof=1) if len(r) > 1 else 0.0,
                 rt, 100 * q, mark))

    print("\n  same checkpoints re-scored at other lambda_D")
    print("  %-9s" % "ckpt" + "".join("%11s" % ("l=%g" % l) for l in LAMBDAS))
    for c, r, rt, q, S, _ in rows:
        line = "  %-9s" % c
        for lam in LAMBDAS:
            line += "%11.2f" % (nst * (rt - lam * S))
        print(line)
    print("=" * 84)
    print("  A checkpoint that wins at 1.5 but loses at 3+ bought its reward by")
    print("  shaving QoS margin the weak penalty barely prices.")


if __name__ == "__main__":
    sys.exit(main())
