"""Where does AO stop improving?

The baseline has to be reported at its own best, or the comparison is against a
strawman -- and our own note already flags that our AO measured 3.6% below Tan's
SA. AO exposes three knobs that trade solution quality for time: outer `rounds`,
per-element phase `phase_sweeps`, and the power-split grid `n_split`. Defaults
(2, 1, 11) were never justified.

Sweeps them on a PAIRED state sequence -- every config replays the same seed, so
differences are not env noise -- and reports J (not R_tot: comparing sum-rate at
unequal QoS is meaningless) alongside the solve time each setting actually costs.
The setting to publish is where J saturates, and its latency is AO's honest cost.

J = sum_rate - lambda_D * sum( (max(0, D_k - R_k) / (D_k + eps))^2 )
"""
import argparse
import os
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.probe_ao_rate import AORatePolicy


def run_cfg(cfg, n_states, seed, rounds, sweeps, n_split):
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg, seed=seed, n_steps_ep=n_states)
    pol = AORatePolicy(cfg, rate, rounds=rounds, n_split=n_split,
                       phase_sweeps=sweeps)
    Dk, lam = cfg.D_k_bps_hz, float(getattr(cfg, "lambda_D", 1.5))
    eps = float(getattr(cfg, "epsilon_qp", 1e-3))

    obs = env.reset(seed=seed)
    Js, srs, qos, ms = [], [], [], []
    for _ in range(n_states):
        t0 = perf_counter()
        action = pol.act(obs, env)
        ms.append((perf_counter() - t0) * 1e3)
        obs, _r, _d, info = env.step(action)
        R = np.asarray(info["R_tot"])
        short = np.maximum(0.0, Dk - R) / (Dk + eps)
        srs.append(float(info["sum_rate"]))
        Js.append(float(info["sum_rate"]) - lam * float((short ** 2).sum()))
        qos.append(float(np.sum(R >= Dk)) / cfg.K)
    return (float(np.mean(Js)), float(np.std(Js) / np.sqrt(len(Js))),
            float(np.mean(srs)), 100 * float(np.mean(qos)), float(np.mean(ms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--M", type=int, default=2)
    ap.add_argument("--states", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = make_config(K=a.K, M=a.M)
    print("=" * 82)
    print(f"  AO saturation sweep · K={a.K} M={a.M} · {a.states} paired states "
          f"· seed {a.seed}")
    print(f"  D_k={cfg.D_k_bps_hz} lambda_D={getattr(cfg,'lambda_D',1.5)}")
    print("=" * 82)
    print(f"  {'rounds':>7}{'sweeps':>7}{'split':>6}"
          f"{'J':>10}{'+/-':>7}{'R_tot':>9}{'QoS%':>7}{'ms':>9}{'dJ vs base':>12}")

    base = None
    # sweeps up to 3: the published setting must be shown to SATURATE, and a
    # grid that stops at the chosen value cannot demonstrate that.
    grid = [(r, s, 11) for r in (1, 2, 3) for s in (0, 1, 2, 3)]
    grid += [(3, 1, 21), (3, 1, 41)]          # split resolution at a mid setting
    for r, s, ns in grid:
        J, se, sr, q, ms = run_cfg(cfg, a.states, a.seed, r, s, ns)
        if base is None:
            base = J
        tag = "" if (r, s, ns) != (2, 1, 11) else "  <- current default"
        print(f"  {r:>7}{s:>7}{ns:>6}{J:>10.4f}{se:>7.4f}{sr:>9.4f}"
              f"{q:>7.1f}{ms:>9.1f}{J - base:>+12.4f}{tag}", flush=True)

    print("=" * 82)
    print("  Publish AO at the first setting where J stops rising beyond its")
    print("  standard error, and quote THAT setting's ms as AO's latency.")
    print("=" * 82)


if __name__ == "__main__":
    main()
