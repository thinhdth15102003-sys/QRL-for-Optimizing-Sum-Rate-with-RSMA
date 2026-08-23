"""
probe_gen_sweep_one.py
----------------------
The same zero-shot shift sweep as `probe_gen_sweep.py`, but for ONE run.

`probe_gen_sweep` insists on a VQC/DNN pair because it exists to emit the
Delta(VQC-DNN) columns. Filling a single method's column -- here PPO-flat, whose
row in tab:abl_genshift is still all `---` -- does not need a partner, and
forcing one only invites mislabelling the output.

Everything else is identical: same SHIFTS list, same `run_hqchac_episode` (which
takes the flat path automatically when load_agents returns no phase net), same
reward/R_tot/QoS triple that the table's three columns want.

Rows the paper deliberately leaves blank -- P_S=40 and combined-B, both marked
(retraining) because the power head is scale-free and cannot transfer across a
change of transmit budget -- are computed anyway and simply not pasted; skipping
them here would make the run non-reproducible for whoever revisits that decision.

Usage:
  python analysis/probe_gen_sweep_one.py --run results/result_310 --tag flatC1 \
      --seeds 42 0 1 2 3 --eps 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.probe_gen_sweep import SHIFTS, eval_actor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--tag', default='run')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3])
    ap.add_argument('--eps', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--only', nargs='*', default=None,
                    help='substrings; evaluate only the SHIFTS whose label matches one')
    a = ap.parse_args()

    shifts = SHIFTS
    if a.only:
        shifts = [s for s in SHIFTS if any(k in s[0] for k in a.only)]
        if not shifts:
            print("no shift label matched %s" % a.only)
            return 1

    print(f"GEN-SWEEP (single) · {a.run} · seeds={a.seeds} eps={a.eps} "
          f"steps={a.steps}")
    print("=" * 78)
    out = eval_actor(a.run, a.seeds, a.eps, a.steps, a.tag, shifts=shifts)

    print("\n" + "=" * 78)
    print("  LaTeX cells  (R_tot & QoS & Reward)")
    print("=" * 78)
    prev = None
    for label, dom, _ov in shifts:
        if prev is not None and dom != prev:
            print("\\midrule")
        prev = dom
        r, q, w = out[label]
        print(f"{label} & ${r.mean():.4f}$ & ${100*q.mean():.1f}$\\% "
              f"& ${w.mean():.1f}$ \\\\")
    print("=" * 78)


if __name__ == '__main__':
    sys.exit(main())
