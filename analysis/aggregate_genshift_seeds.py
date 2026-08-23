"""Average `probe_gen_sweep_one` outputs across TRAINING seeds, per shift row.

tab:abl_genshift used to carry one checkpoint per method, so every cell inherited
whichever seed happened to be picked. The seed spread on this problem is large
(Case-1 episode return ranges 455-517 across five seeds, ~13%), which is bigger
than the differences the table is trying to show — so a single-seed column could
reorder the methods on its own.

This reads the per-seed sweep files and reports mean +/- sd across seeds for each
shift, plus the LaTeX cells.

Usage:
  python analysis/aggregate_genshift_seeds.py --dir analysis/data/genshift_c1_5seed \
      --group VQC=vqc_ --group DNN=dnn_ --group PPO-flat=flat_
"""
import argparse
import glob
import os
import re
import sys

import numpy as np

# "  [tag] LABEL   R=1.7018±0.0270 QoS= 81.0±0.5% Rew=  309.1"
LINE = re.compile(
    r'^\s*\[[^\]]*\]\s+(.*?)\s+R=(-?\d+\.\d+)±[\d.]+\s+QoS=\s*(-?\d+\.\d+)±[\d.]+%\s+Rew=\s*(-?\d+\.\d+)')


def parse(path):
    """-> {shift_label: (R_tot, QoS_frac, reward)}"""
    out = {}
    with open(path, encoding='utf-8', errors='replace') as fh:
        for ln in fh:
            m = LINE.match(ln)
            if m:
                out[m.group(1).strip()] = (float(m.group(2)),
                                           float(m.group(3)) / 100.0,
                                           float(m.group(4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--group', action='append', required=True,
                    help='NAME=filename_prefix, e.g. VQC=vqc_')
    a = ap.parse_args()

    groups = []
    for g in a.group:
        name, prefix = g.split('=', 1)
        files = sorted(glob.glob(os.path.join(a.dir, prefix + '*.txt')))
        runs = [(os.path.basename(f), parse(f)) for f in files]
        runs = [(n, d) for n, d in runs if d]
        if not runs:
            print("  !! no parsable files for %s (%s*)" % (name, prefix))
            continue
        groups.append((name, runs))

    if not groups:
        return 1

    # shift order: take it from the first group's first file, which preserves the
    # order probe_gen_sweep_one printed (i.e. the SHIFTS order).
    order = list(groups[0][1][0][1].keys())

    for name, runs in groups:
        print("=" * 96)
        print("  %s — %d training seeds: %s"
              % (name, len(runs), ", ".join(n.replace('.txt', '') for n, _ in runs)))
        print("=" * 96)
        print("  %-22s %20s %18s %16s" % ("shift", "R_tot", "QoS", "Reward"))
        cells = []
        for lbl in order:
            vals = [d[lbl] for _, d in runs if lbl in d]
            if not vals:
                continue
            arr = np.array(vals)                     # (n_seed, 3)
            m, s = arr.mean(axis=0), arr.std(axis=0, ddof=1) if len(vals) > 1 else np.zeros(3)
            print("  %-22s %9.4f±%-9.4f %7.1f±%-9.1f %8.1f±%.1f"
                  % (lbl, m[0], s[0], 100 * m[1], 100 * s[1], m[2], s[2]))
            cells.append((lbl, m, s))
        print("\n  LaTeX cells (mean over seeds):")
        for lbl, m, s in cells:
            print("    %-22s & $%.4f$ & $%.1f$\\%% & $%.1f$ \\\\" % (lbl, m[0], 100 * m[1], m[2]))
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
