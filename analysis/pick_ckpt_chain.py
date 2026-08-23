"""Pick one checkpoint per SEED when the seed is a multi-part resume chain.

`probe_pick_ckpt.py` ranks the checkpoints of ONE directory. A seed that was
resumed twice lives in three directories whose episode counters each restart at
zero, so ranking them separately answers the wrong question — it picks the best of
each segment instead of the best of the seed.

This stitches the segments onto one global episode axis (using the episode the next
segment actually resumed FROM, not the last episode the previous one logged — the
episodes past the resume point were discarded) and applies the reporting rule:

    argmax episode return  subject to  QoS >= --qos-floor

on whatever validation seeds the underlying scan used.

Usage:
  python analysis/pick_ckpt_chain.py --scan-dir analysis/data/ckpt_scan_c2_vqc \
      --chain "seed0=s0_a:261:0:10200,s0_b:278:10200:2900,s0_c:289:13100:99999" \
      --chain "seed1=s1_a:285:0:1600,s1_b:290:1600:14200,s1_c:320:15800:99999"

  chain spec: LABEL=tag:runid:global_offset:cap[,...]
    tag    — basename of the scan file (without .txt)
    runid  — results/result_<runid>, used to emit the checkpoint path
    offset — global episode at which this segment starts
    cap    — highest LOCAL episode still on the chain (the resume point); segments
             ran past it and those checkpoints are NOT part of the lineage
"""
import argparse
import os
import re
import sys

ROW = re.compile(r'^\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)%')


def parse(path):
    """-> [(ckpt_name, reward, R_tot, QoS%)] from a probe_pick_ckpt summary table."""
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
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan-dir', required=True)
    ap.add_argument('--chain', action='append', required=True)
    ap.add_argument('--qos-floor', type=float, default=90.0)
    a = ap.parse_args()

    print("=" * 100)
    print("  chain-aware checkpoint pick · argmax reward s.t. QoS >= %.0f%%" % a.qos_floor)
    print("=" * 100)
    print("  %-8s %8s %-34s %9s %8s %7s" % ("seed", "global ep", "checkpoint", "reward", "R_tot", "QoS"))
    print("  " + "-" * 94)
    picks = []
    for spec in a.chain:
        label, parts = spec.split('=', 1)
        cands = []
        for seg in parts.split(','):
            tag, runid, off, cap = seg.split(':')
            off, cap = int(off), int(cap)
            f = os.path.join(a.scan_dir, tag + '.txt')
            if not os.path.isfile(f):
                print("  %-8s  !! missing %s" % (label, f))
                continue
            for name, rew, rt, q in parse(f):
                if not name.isdigit():
                    continue                      # 'best'/'final' have no global ep
                loc = int(name)
                if loc > cap:
                    continue                      # past the resume point → discarded
                cands.append((loc + off, runid, loc, rew, rt, q))
        ok = [c for c in cands if c[5] >= a.qos_floor]
        if not ok:
            best_q = max((c[5] for c in cands), default=float('nan'))
            print("  %-8s  no checkpoint clears the floor (best QoS %.1f%%, %d candidates)"
                  % (label, best_q, len(cands)))
            continue
        g, runid, loc, rew, rt, q = max(ok, key=lambda c: c[3])
        path = "results/result_%s/checkpoints/ep_%05d" % (runid, loc)
        print("  %-8s %8d %-34s %9.2f %8.4f %6.1f%%   (%d of %d cand clear the floor)"
              % (label, g, path, rew, rt, q, len(ok), len(cands)))
        picks.append(path)

    if picks:
        print("\n  paste into probe_main_table --hier:")
        print("  " + ",".join(picks))
    return 0


if __name__ == '__main__':
    sys.exit(main())
