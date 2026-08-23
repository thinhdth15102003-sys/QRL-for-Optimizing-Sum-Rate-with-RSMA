"""Figure: training curves (episode reward only) for VQC, DNN and PPO-flat.

One panel per case, written to Research Paper/figures/reward_log_case{1,2}.pdf.

Seeds whose training was resumed are stitched from their segment chain, the
chains being those recorded in analysis/data/TABLE-PROVENANCE.md for
tab:case1_main / tab:case2_main, so the curves cover exactly the runs the two
performance tables are built from.

Segment spec is (run_id, global_offset, n_local); n_local=None means "to the
end of that segment". Reward is the episode return; the flat baseline logs a
per-step J, which is multiplied by the episode length to put all three methods
on one axis.

    python analysis/plot_learning_curves.py [--check]
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "Research Paper", "figures")

CHAINS = {
    ("c1", "vqc"): [
        [(262, 0, 13600), (328, 13600, None)],
        [(260, 0, None)],
        [(265, 0, 8400), (282, 8400, 1600), (327, 10000, 6600), (331, 16600, None)],
        [(266, 0, 8400), (283, 8400, 1600), (334, 10000, None)],
        [(267, 0, 8300), (284, 8300, None)],
    ],
    ("c1", "dnn"):  [[(r, 0, None)] for r in (234, 238, 239, 240, 294)],
    ("c1", "flat"): [[(r, 0, None)] for r in (301, 307, 308, 347, 356)],
    ("c2", "vqc"): [
        [(261, 0, 10200), (278, 10200, 2900), (289, 13100, None)],
        [(285, 0, 1600), (290, 1600, 14200), (320, 15800, None)],
        [(286, 0, 1700), (291, 1700, 14100), (321, 15800, None)],
        [(315, 0, 3000), (323, 3000, None)],
        [(316, 0, 3000), (324, 3000, None)],
    ],
    ("c2", "dnn"):  [[(r, 0, None)] for r in (231, 232, 233, 236, 309)],
    ("c2", "flat"): [[(r, 0, None)] for r in (304, 305)],
}

STYLE = {
    "flat": dict(color="#c0392b", label="PPO-flat"),
    "dnn":  dict(color="#2471a3", label="DNN"),
    "vqc":  dict(color="#1e8449", label="HQC-HAC (VQC)"),
}
ORDER = ("flat", "dnn", "vqc")
TITLE = {"c1": "Case 1  ($K{=}5$, $M{=}1$)", "c2": "Case 2  ($K{=}10$, $M{=}2$)"}

EP_LINE = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def run_dir(rid):
    for d in (os.path.join(ROOT, "results", "result_%d" % rid),
              os.path.join(ROOT, "results", "Training-Case 1", "result_%d" % rid),
              os.path.join(ROOT, "results", "Training-Case 2", "result_%d" % rid)):
        if os.path.isdir(d):
            return d
    raise IOError("no dir for r%d" % rid)


def load_reward(rid):
    """Return (ep_index_1based, reward) for one run, from whichever log it has."""
    d = run_dir(rid)

    npz = os.path.join(d, "metrics.npz")
    if os.path.isfile(npz):
        r = np.load(npz)["episode_reward"]
        return np.arange(1, len(r) + 1), np.asarray(r, float), "npz"

    js = os.path.join(d, "training_log.json")
    if os.path.isfile(js):
        with open(js) as f:
            g = json.load(f)
        n_steps = 200
        hp = os.path.join(d, "hyperparameters.json")
        if os.path.isfile(hp):
            with open(hp) as f:
                h = json.load(f)
            n_steps = h.get("training", {}).get("n_steps_per_ep", n_steps)
        return (np.asarray(g["ep"], float),
                np.asarray(g["J"], float) * n_steps, "json x%d" % n_steps)

    txt = os.path.join(d, "training_log.txt")
    if os.path.isfile(txt):
        ep, rw = [], []
        with open(txt, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = EP_LINE.match(line)
                if m:
                    ep.append(int(m.group(1)))
                    rw.append(float(m.group(2)))
        if ep:
            return np.array(ep, float), np.array(rw, float), "txt"

    raise IOError("no usable log in %s" % d)


def stitch(chain):
    """Concatenate one seed's segments onto a global episode axis."""
    E, R = [], []
    for rid, off, ncap in chain:
        ep, rw, _ = load_reward(rid)
        if ncap is not None:
            keep = ep <= ncap
            ep, rw = ep[keep], rw[keep]
        E.append(ep + off)
        R.append(rw)
    E, R = np.concatenate(E), np.concatenate(R)
    o = np.argsort(E, kind="stable")
    return E[o], R[o]


def smooth_to_grid(E, R, grid, win):
    """Rolling mean of R over `win` episodes, sampled on `grid`."""
    y = np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        m = (E > g - win / 2.0) & (E <= g + win / 2.0)
        if m.sum():
            y[i] = R[m].mean()
    return y


def check():
    for (case, meth), chains in CHAINS.items():
        print("\n== %s / %s" % (case, meth))
        for si, ch in enumerate(chains):
            parts = []
            for rid, off, ncap in ch:
                try:
                    ep, rw, src = load_reward(rid)
                except IOError as e:
                    parts.append("r%d ERR(%s)" % (rid, e)); continue
                parts.append("r%d[%s] ep %d..%d n=%d rw %.0f..%.0f off=%d cap=%s"
                             % (rid, src, ep.min(), ep.max(), len(ep),
                                rw.min(), rw.max(), off, ncap))
            print("  seed %d: %s" % (si, "\n           ".join(parts)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9,
                         "legend.fontsize": 7.5, "axes.titlesize": 9})
    os.makedirs(OUT, exist_ok=True)

    for case in ("c1", "c2"):
        fig, ax = plt.subplots(figsize=(4.1, 2.9))
        top = 0
        for meth in ORDER:
            curves = []
            for ch in CHAINS[(case, meth)]:
                curves.append(stitch(ch))
            if not curves:
                continue
            hi = 20000.0
            grid = np.linspace(0, hi, 400)
            win = 200.0
            with np.errstate(invalid="ignore"):
                Y = np.vstack([smooth_to_grid(E, R, grid, win) for E, R in curves])
                alive = np.isfinite(Y).sum(0)
                mu = np.where(alive > 0, np.nanmean(np.where(np.isfinite(Y), Y, np.nan), 0), np.nan)
                sd = np.where(alive > 1, np.nanstd(np.where(np.isfinite(Y), Y, np.nan), 0), 0.0)
            ax.fill_between(grid, mu - sd, mu + sd, alpha=0.16,
                            color=STYLE[meth]["color"], lw=0)
            ax.plot(grid, mu, lw=1.4, **STYLE[meth])
            top = max(top, np.nanmax(mu + sd))

        ax.set_xlabel("training episode")
        ax.set_ylabel("episode reward")
        ax.set_title(TITLE[case])
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_ylim(bottom=min(-100, ax.get_ylim()[0]), top=top * 1.12)
        ax.legend(loc="lower right", framealpha=0.9)
        fig.tight_layout(pad=0.4)
        for ext in ("pdf", "png"):
            p = os.path.join(OUT, "reward_log_case%s.%s" % (case[-1], ext))
            fig.savefig(p, dpi=200)
        print("wrote reward_log_case%s.{pdf,png}" % case[-1])
        plt.close(fig)


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
