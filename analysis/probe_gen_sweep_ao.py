"""
probe_gen_sweep_ao.py
---------------------
Run the per-state AO baseline across the SAME distribution shifts, seeds and episode
budget as probe_gen_sweep.py, so its zero-shot row is directly comparable to the VQC
and DNN columns of tab:abl_genshift.

The AO re-solves every channel realization, so it has no train/test mismatch by
construction. The interesting question is the opposite one: because it greedily
searches on the ESTIMATED channel, does it over-fit a corrupted estimate when the CSI
error grows (kappa 0.10/0.20, combined shifts) relative to a learned policy?

Usage:
  python analysis/probe_gen_sweep_ao.py --run results/result_95 \
      --seeds 42 0 1 2 3 --eps 10 --steps 200
"""
import sys, os, argparse
from time import perf_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from infer import load_training_cfg
from analysis.probe_gen_sweep import SHIFTS
from analysis.probe_ao_baseline import AOPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run dir, used only for its cfg/case')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3])
    ap.add_argument('--eps', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--rounds', type=int, default=1)
    # Re-running all 14 shifts costs ~1 h (Case 1) / ~2 h (Case 2) because AO
    # re-solves every state. When only one or two rows changed, name them.
    ap.add_argument('--only', nargs='*', default=None,
                    help='substrings; run only the SHIFTS whose label matches one')
    a = ap.parse_args()

    shifts = SHIFTS
    if a.only:
        shifts = [s for s in SHIFTS if any(k in s[0] for k in a.only)]
        if not shifts:
            print("no shift label matched %s" % a.only)
            return 1

    print(f"AO GEN-SWEEP · seeds={a.seeds} eps={a.eps} steps={a.steps} rounds={a.rounds}")
    print("=" * 82)
    rows = {}
    for label, _dom, ov in shifts:
        cfg = load_training_cfg(a.run, **ov)
        rate = RateComputer(cfg)
        pol = AOPolicy(cfg, rate, rounds=a.rounds)
        Rs, Qs, Ws, Ts = [], [], [], []
        for sd in a.seeds:
            env = ISTNEnv(cfg, seed=sd, n_steps_ep=a.steps)
            r_acc, q_acc, w_acc = [], [], []
            for _ in range(a.eps):
                obs = env.reset()
                ep_rew = 0.0
                for _ in range(a.steps):
                    t0 = perf_counter()
                    action = pol.act(obs, env)
                    Ts.append((perf_counter() - t0) * 1e3)
                    obs, _r, _d, info = env.step(action)
                    # Episode RETURN, the column the paper table carries. It is not
                    # R_tot*steps once QoS drops below 100% — the shortfall penalty
                    # is then non-zero, which is exactly what happens at low P_S.
                    ep_rew += float(_r)
                    r_acc.append(float(info['sum_rate']))
                    q_acc.append(float(np.sum(info['R_tot'] >= cfg.D_k_bps_hz)) / cfg.K)
                w_acc.append(ep_rew)
            Rs.append(np.mean(r_acc)); Qs.append(np.mean(q_acc)); Ws.append(np.mean(w_acc))
        Rs, Qs, Ws = np.array(Rs), np.array(Qs), np.array(Ws)
        rows[label] = (Rs, Qs, Ws, float(np.mean(Ts)))
        print(f"  [AO] {label:20s} R={Rs.mean():.4f}±{Rs.std():.4f} "
              f"QoS={100*Qs.mean():5.1f}±{100*Qs.std():.1f}%  Rew={Ws.mean():7.1f}"
              f"  solve={np.mean(Ts):.1f} ms", flush=True)

    print("\n" + "=" * 82 + "\n  LaTeX AO cells (R_tot & QoS & Reward):\n" + "=" * 82)
    for label, _dom, _ov in shifts:
        Rs, Qs, Ws, t = rows[label]
        print(f"{label} & ${Rs.mean():.4f}$ & ${100*Qs.mean():.1f}$\\% "
              f"& ${Ws.mean():.1f}$ \\\\")
    print("=" * 82)


if __name__ == '__main__':
    main()
