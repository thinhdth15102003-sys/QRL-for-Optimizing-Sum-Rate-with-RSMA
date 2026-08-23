"""
probe_ab_band.py
----------------
Is a single-seed scout actually different from the recipe, or is it inside the
noise the control seeds already show among themselves?

Replaces two one-off scripts (scratchpad/pwfeat_ab.py, scratchpad/final_ab.py)
that hard-coded their run lists.

Every scout in this project is one seed and every control set is three or four,
so scout-vs-one-control is not a test: the controls span a band, and a delta
smaller than that band is not evidence. Reports z against the band, and — more
importantly — the TREND of z across windows.

⚠ One z is not a verdict. Two runs made that concrete:
    r270 (Case 2, --power-scale-feats) read z=-1.64 at ep 4400 and was called
    harmful; at its full 10000 it came in at z=-0.20, inside the band. The early
    negative was amortisation lag.
    r273 (Case 1, same flag) sat inside the band too (z=+0.77) but was positive
    in 6 of 6 windows, which is what a real effect looks like at this sample size.
So read the trend row, not the summary number.

Usage:
  python analysis/probe_ab_band.py \
      --scout r275:results/result_275 \
      --ctrl  r231:results/result_231 r232:results/result_232 \
              r233:results/result_233 r236:results/result_236
  # several comparisons at once: repeat --scout/--ctrl as a group with --also
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows(run_dir):
    """(episode, reward, R_tot, 'q/K') from a training log."""
    p = os.path.join(ROOT, run_dir, "training_log.txt")
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding="utf-8", errors="ignore"):
        f = ln.split()
        if len(f) >= 15 and f[0].isdigit():
            try:
                out.append((int(f[0]), float(f[1]), float(f[9]), f[11]))
            except ValueError:
                pass
    return out


def win(rs, lo, hi, col=1):
    v = [r[col] for r in rs if lo < r[0] <= hi]
    return float(np.mean(v)) if v else None


def qos(rs, lo, hi):
    v = []
    for r in rs:
        if lo < r[0] <= hi and "/" in r[3]:
            a, b = r[3].split("/")
            try:
                v.append(float(a) / float(b))
            except ValueError:
                pass
    return float(np.mean(v)) if v else None


def compare(scout, ctrl, window, n_win, title=""):
    slbl, sdir = scout
    srs = rows(sdir)
    if not srs:
        print(f"  {slbl}: no training_log.txt under {sdir}\n")
        return
    ep = srs[-1][0]
    lo, hi = ep - window, ep

    print("=" * 84)
    print(f"  {title or slbl}   [scout at ep {ep}, window {window}]")
    print("=" * 84)
    print(f"  {'run':<8}{'reward':>10}{'R_tot':>9}{'QoS':>8}")
    print("  " + "-" * 36)
    s_rw, s_rt, s_q = win(srs, lo, hi), win(srs, lo, hi, 2), qos(srs, lo, hi)
    print(f"  {slbl:<8}{s_rw:>10.1f}{s_rt:>9.3f}{100*s_q:>7.1f}%   <- scout")

    cr, ct, cq = [], [], []
    for lbl, d in ctrl:
        rs = rows(d)
        v = win(rs, lo, hi)
        if v is None:
            print(f"  {lbl:<8}{'(no data in window)':>27}")
            continue
        cr.append(v)
        ct.append(win(rs, lo, hi, 2))
        cq.append(qos(rs, lo, hi))
        print(f"  {lbl:<8}{v:>10.1f}{ct[-1]:>9.3f}{100*cq[-1]:>7.1f}%")
    if len(cr) < 2:
        print("  fewer than two controls in window — no band, no test\n")
        return

    mu, sd = float(np.mean(cr)), float(np.std(cr, ddof=1))
    z = (s_rw - mu) / sd if sd > 0 else float("nan")
    print("  " + "-" * 36)
    print(f"  {'band':<8}{mu:>10.1f}{np.mean(ct):>9.3f}{100*np.mean(cq):>7.1f}%"
          f"   sd {sd:.1f}, n={len(cr)}")
    print(f"\n  delta {s_rw-mu:+.1f} reward ({100*(s_rw-mu)/mu:+.1f}%)   z = {z:+.2f}"
          f"   R_tot {s_rt-float(np.mean(ct)):+.3f}"
          f"   QoS {100*(s_q-float(np.mean(cq))):+.1f} pt")

    zs = []
    for i in range(n_win, 0, -1):
        h2 = ep - (i - 1) * window
        l2 = h2 - window
        sv = win(srs, l2, h2)
        cv = [win(rows(d), l2, h2) for _, d in ctrl]
        cv = [x for x in cv if x is not None]
        if sv is None or len(cv) < 2:
            zs.append(None)
            continue
        m2, s2 = float(np.mean(cv)), float(np.std(cv, ddof=1))
        zs.append((sv - m2) / s2 if s2 > 0 else None)
    print(f"\n  trend (z per {window}-ep window, oldest -> newest):")
    print("   " + " ".join(f"{v:+6.2f}" if v is not None else "    --"
                           for v in zs))
    got = [v for v in zs if v is not None]
    n_neg = sum(1 for v in got if v < 0)
    print(f"   ({n_neg}/{len(got)} windows negative)")
    if got and all(b > a for a, b in zip(got, got[1:])):
        print("   monotone rising — a slow start being paid back, not harm")
    print(f"  verdict: {'INSIDE band' if abs(z) < 1.0 else ('HELPS' if z > 0 else 'HURTS')}"
          f"  (one seed: read the trend, not this line)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scout", required=True, help="LABEL:run_dir")
    ap.add_argument("--ctrl", nargs="+", required=True, help="LABEL:run_dir ...")
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--n-win", dest="n_win", type=int, default=6)
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    def split(s):
        lbl, d = s.split(":", 1)
        return lbl, d

    compare(split(a.scout), [split(c) for c in a.ctrl], a.window, a.n_win,
            a.title)


if __name__ == "__main__":
    sys.exit(main())
