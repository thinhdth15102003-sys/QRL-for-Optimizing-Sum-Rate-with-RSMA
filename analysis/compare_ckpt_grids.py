"""Did the coarse (1000-ep) checkpoint grid miss a better checkpoint?

The reported checkpoints were chosen by scanning every 1000 episodes, but runs
save every 100. This re-reads a full-resolution scan and reports, per run:

  * the pick the coarse grid would make
  * the pick the fine grid makes
  * the validation-reward gap between them

Both under the same rule: argmax episode return subject to QoS >= `--qos-floor`,
on the validation env seeds.

A gap here is NOT by itself a reason to change the table. The checkpoint
landscape on this problem is flat and noisy (sd 3-18 on a 1200-state eval), so
the max over ~200 candidates is largely picking noise, and it has to be re-scored
on the disjoint test seeds before it means anything. Precedent: switching Case-2
DNN from the final checkpoint to the coarse-grid "best" *lowered* the test-seed
score, 338.3 -> 337.0.

Usage:
  python analysis/compare_ckpt_grids.py --dir analysis/data/ckpt_scan_fine
"""
import argparse
import glob
import os
import re
import sys

# "  11000         472.92     1.17   2.3877   93.3%"  (also 'best' / 'final')
ROW = re.compile(r'^\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)%')


def parse(path):
    rows, inpick = [], False
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            if 'PICK CHECKPOINT' in ln:
                inpick = True
                continue
            if 'same checkpoints re-scored' in ln:
                break
            if inpick:
                m = ROW.match(ln)
                if m:
                    rows.append((m.group(1), float(m.group(2)),
                                 float(m.group(4)), float(m.group(5))))
    return rows                                   # (ckpt, reward, R_tot, QoS%)


def pick(rows, floor, coarse_step=None):
    cand = [r for r in rows if r[3] >= floor]
    if coarse_step is not None:
        def on_grid(name):
            if not name.isdigit():
                return name == 'best'             # 'best' was in the coarse list too
            return int(name) % coarse_step == 0
        cand = [r for r in cand if on_grid(r[0])]
    return max(cand, key=lambda r: r[1]) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--qos-floor', type=float, default=90.0)
    ap.add_argument('--coarse-step', type=int, default=1000)
    a = ap.parse_args()

    print("=" * 104)
    print("  coarse (%d-ep grid) vs fine (all saved ckpts) · QoS floor %.0f%% · validation seeds"
          % (a.coarse_step, a.qos_floor))
    print("=" * 104)
    print("  %-14s %6s | %-9s %8s %6s | %-9s %8s %6s | %s"
          % ("run", "n_ckpt", "coarse", "reward", "QoS", "fine", "reward", "QoS", "gap"))
    print("  " + "-" * 100)
    changed = 0
    for f in sorted(glob.glob(os.path.join(a.dir, '*.txt'))):
        tag = os.path.basename(f)[:-4]
        if tag.startswith('_'):
            continue
        rows = parse(f)
        if not rows:
            print("  %-14s  (no parsable rows)" % tag)
            continue
        c = pick(rows, a.qos_floor, a.coarse_step)
        fi = pick(rows, a.qos_floor, None)
        if c is None or fi is None:
            best_q = max(r[3] for r in rows)
            print("  %-14s %6d | no ckpt clears the %.0f%% floor (best QoS %.1f%%)"
                  % (tag, len(rows), a.qos_floor, best_q))
            continue
        gap = fi[1] - c[1]
        mark = "" if fi[0] == c[0] else "  <-- differs"
        if fi[0] != c[0]:
            changed += 1
        print("  %-14s %6d | %-9s %8.2f %5.1f%% | %-9s %8.2f %5.1f%% | %+7.2f%s"
              % (tag, len(rows), c[0], c[1], c[3], fi[0], fi[1], fi[3], gap, mark))
    print("  " + "-" * 100)
    print("  %d run(s) would change checkpoint. Re-score those on the TEST seeds"
          % changed)
    print("  before touching the table — a fine-grid win on validation is often noise.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
