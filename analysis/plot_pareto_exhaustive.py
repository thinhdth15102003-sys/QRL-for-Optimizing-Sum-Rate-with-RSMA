"""Rate-QoS frontier with each policy placed on it, both cases.

The curve is the average-constrained frontier of aggregate_pareto: the best mean
sum-rate attainable at a given mean QoS, assembled from the exhaustive per-state
achievable sets. Each policy is drawn at its Table III operating point and
labelled with its episode return, because proximity to this frontier and quality
under the training objective are not the same thing: the frontier maximises rate
at a fixed service level, whereas the policies maximise the QoS-penalised reward.
At (10,2) the per-state solver is the closest point to the frontier and the
lowest-scoring one on that reward.

    python analysis/plot_pareto_exhaustive.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.aggregate_pareto import (load_states, lagrangian_frontier,  # noqa: E402
                                       reward_optimum)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "Research Paper", "figures")

# Table III: (R_tot, QoS, episode return)
POLICIES = {
    (5, 1): [("AO", 1.876, 1.000, 375.1),
             ("PPO-flat", 2.417, 0.960, 481.2),
             ("DNN", 2.501, 0.946, 494.8),
             ("HQC-HAC (VQC)", 2.482, 0.924, 490.3)],
    (10, 2): [("AO", 1.651, 1.000, 330.1),
              ("PPO-flat", 0.557, 0.472, -367.1),
              ("DNN", 1.747, 0.921, 337.5),
              ("HQC-HAC (VQC)", 1.742, 0.944, 337.3)],
}
STYLE = {"AO": ("#444444", "s"), "PPO-flat": ("#c0392b", "o"),
         "DNN": ("#2471a3", "^"), "HQC-HAC (VQC)": ("#1e8449", "D")}
TITLE = {(5, 1): "Case 1  ($K{=}5$, $M{=}1$)",
         (10, 2): "Case 2  ($K{=}10$, $M{=}2$)"}
ZOOM = {(5, 1): (88.0, 100.6), (10, 2): (88.0, 100.6)}
# (dx, dy, ha) in points, chosen so neighbouring labels do not collide
LABEL_OFFSET = {
    (5, 1): {"AO": (-9, -4, "right"), "PPO-flat": (9, -3, "left"),
             "DNN": (7, 4, "left"), "HQC-HAC (VQC)": (-8, -3, "right")},
    (10, 2): {"AO": (-9, -4, "right"), "PPO-flat": (0, 0, "center"),
              "DNN": (-8, -3, "right"), "HQC-HAC (VQC)": (7, 3, "left")},
}


def main():
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 9,
                         "axes.titlesize": 9, "legend.fontsize": 8})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4))

    for ax, case in zip(axes, [(5, 1), (10, 2)]):
        K, M = case
        sets, _ = load_states(K, M)
        fr = lagrangian_frontier(sets)
        q, r = fr[:, 0] * 100.0, fr[:, 1]

        lo, hi = ZOOM[case]
        ax.plot(q, r, color="#222222", lw=1.5, zorder=3)
        ax.fill_between(q, r, r.max() * 3, color="#000000", alpha=0.06, lw=0)
        vis = (q >= lo)
        ax.text(lo + 0.4, np.interp(lo, q, r) * 1.03, "unattainable",
                fontsize=7.5, style="italic", color="#666666", va="bottom")

        # the point the training objective itself selects, not 100% QoS
        ropt = reward_optimum(sets)
        if ropt is not None:
            ax.plot(ropt[1] * 100.0, ropt[0], "*", color="#8e44ad", ms=13,
                    mec="white", mew=0.8, zorder=7)
            ax.annotate("%.1f" % ropt[2], (ropt[1] * 100.0, ropt[0]),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7, color="#8e44ad", clip_on=False)

        for name, rt, qos, rew in POLICIES[case]:
            c, mk = STYLE[name]
            x = qos * 100.0
            if x < lo:
                continue                       # shown in the inset instead
            ax.plot(x, rt, mk, color=c, ms=6.5, mec="white", mew=0.8, zorder=6)
            dx, dy, ha = LABEL_OFFSET[case][name]
            ax.annotate("%.1f" % rew, (x, rt), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, fontsize=7, color=c,
                        clip_on=False, zorder=6)

        ax.set_xlim(lo, hi)
        vis_r = [p[1] for p in POLICIES[case] if p[2] * 100.0 >= lo]
        # headroom so the inset sits clear of the curve in the unattainable corner
        ax.set_ylim(min(vis_r) - 0.20, float(np.interp(lo, q, r)) * 1.26)
        ax.set_xlabel("mean QoS satisfaction (%)")
        ax.set_ylabel("mean $R_{\\mathrm{tot}}$ (bps/Hz)")
        ax.set_title(TITLE[case])
        ax.grid(alpha=0.25, lw=0.6)

        # the inset sits in the unattainable corner, where nothing is plotted
        ins = ax.inset_axes([0.640, 0.640, 0.340, 0.310], zorder=4)
        ins.set_facecolor("white")
        for sp in ins.spines.values():
            sp.set_linewidth(0.6)
        ins.plot(q, r, color="#222222", lw=1.0)
        ins.fill_between(q, r, r.max() * 3, color="#000000", alpha=0.06, lw=0)
        for name, rt, qos, rew in POLICIES[case]:
            c, mk = STYLE[name]
            ins.plot(qos * 100.0, rt, mk, color=c, ms=3.4, mec="white", mew=0.4)
        ins.set_xlim(0, 103); ins.set_ylim(0, r.max() * 1.08)
        ins.tick_params(labelsize=5.5, length=2, pad=1)
        ins.set_title("full range", fontsize=6, pad=2)

    handles = [plt.Line2D([], [], color=STYLE[n][0], marker=STYLE[n][1], ls="",
                          ms=6, mec="white", mew=0.8, label=n)
               for n in ("HQC-HAC (VQC)", "DNN", "PPO-flat", "AO")]
    handles.append(plt.Line2D([], [], color="#8e44ad", marker="*", ls="", ms=10,
                              mec="white", mew=0.8, label="reward optimum"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "pareto_frontier." + ext), dpi=200,
                    bbox_inches="tight")
    print("wrote pareto_frontier.{pdf,png}")


if __name__ == "__main__":
    main()
