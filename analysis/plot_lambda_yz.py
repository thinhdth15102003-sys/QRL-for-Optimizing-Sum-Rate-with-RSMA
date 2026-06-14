"""
plot_lambda_yz.py  —  "draw it yourself" step
---------------------------------------------
Draw the VQC encoding-scale trajectory figure from a CSV produced by
analysis/extract_lambda_yz.py. Only needs numpy + matplotlib (no repo access), so
you can run it anywhere the CSV is and tweak the styling freely.

CSV columns (header row required): cum_ep, lam_y_inf, lam_z_inf [, ramp, ...]

Usage:
  python analysis/plot_lambda_yz.py
  python analysis/plot_lambda_yz.py --csv analysis/data/lambda_yz_case1.csv \
      --out "Research Paper/figures/case1_lambda_yz_T1_1.png" --smooth-window 9 --ymax 1.0
"""

import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless save; remove this line to show interactively
import matplotlib.pyplot as plt


def smooth(y, w):
    """Centered moving average (edge-padded so length is preserved)."""
    y = np.asarray(y, float)
    if w <= 1 or len(y) < w:
        return y
    pad = w // 2
    return np.convolve(np.pad(y, pad, mode="edge"), np.ones(w) / w, mode="valid")


def load(path):
    ep, ly, lz = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ep.append(float(row["cum_ep"]))
            ly.append(float(row["lam_y_inf"]))
            lz.append(float(row["lam_z_inf"]))
    order = np.argsort(ep)
    return np.array(ep)[order], np.array(ly)[order], np.array(lz)[order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="analysis/data/lambda_yz_case1.csv")
    ap.add_argument("--out", default="Research Paper/figures/case1_lambda_yz_T1_1.png")
    ap.add_argument("--smooth-window", type=int, default=9, help="moving-avg window (1 = raw)")
    ap.add_argument("--ymax", type=float, default=1.0)
    ap.add_argument("--xlabel", default="VQC encoding $\lambda$ trajectory")
    args = ap.parse_args()

    ep, ly, lz = load(args.csv)
    w = args.smooth_window

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ep, smooth(ly, w), "-", lw=2.0, color="#c0392b", label=r"$\|\lambda_y\|_\infty$")
    ax.plot(ep, smooth(lz, w), "-", lw=2.0, color="#2471a3", label=r"$\|\lambda_z\|_\infty$")
    ax.set_xlim(ep.min(), ep.max())
    ax.set_ylim(0.0, args.ymax)
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(r"VQC encoding scale $\|\lambda\|_\infty$")
    ax.set_title(args.title)
    ax.legend(loc="center right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    base, ext = os.path.splitext(args.out)
    exts = {ext.lstrip(".").lower(), "pdf"} if ext else {"png", "pdf"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for e in sorted(exts):
        p = f"{base}.{e}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
