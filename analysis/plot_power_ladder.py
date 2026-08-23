"""Lorenz curves of the private-power allocation across the power ladder.

Reads analysis/data/power_ladder.jsonl (written by analysis/data/_run_power_ladder.sh)
and writes Research Paper/figures/power_ladder.{pdf,png}.

Each panel is one (case, P_S) cell. The x axis is the fraction of users sorted
by the private power they receive, the y axis the cumulative share of that
power, so the diagonal is an equal split and the sag below it is concentration.
Only PRIVATE power is shown: the common stream is a per-group quantity, so
attributing it to individual users would need an arbitrary convention.
"""
import io
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "power_ladder.jsonl")
OUT = os.path.join(os.path.dirname(HERE), "Research Paper", "figures")

STYLE = {"AO":   ("#444444", "s", "AO"),
         "flat": ("#c0392b", "o", "PPO-flat"),
         "DNN":  ("#2471a3", "^", "DNN"),
         "VQC":  ("#1e8449", "D", "HQC-HAC (VQC)")}
ORDER = ("AO", "flat", "DNN", "VQC")
PS = [40, 50, 60, 70]


def main():
    rows = [json.loads(l) for l in io.open(SRC) if l.strip()]
    by = {(r["K"], int(r["P_S"])): r for r in rows}

    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 8,
                         "axes.titlesize": 8, "legend.fontsize": 8})
    fig, axes = plt.subplots(2, 4, figsize=(7.15, 3.55),
                             sharex=True, sharey=True)

    for i, (K, tag) in enumerate([(5, "Case 1  ($K{=}5$)"),
                                  (10, "Case 2  ($K{=}10$)")]):
        for j, ps in enumerate(PS):
            ax = axes[i][j]
            r = by.get((K, ps))
            ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls=":", zorder=1)
            if r:
                x = np.r_[0.0, np.arange(1, K + 1) / K]
                for m in ORDER:
                    d = r["rows"].get(m)
                    if not d:
                        continue
                    c, mk, _ = STYLE[m]
                    ax.plot(x, np.r_[0.0, d["w_lorenz"]], color=c, lw=1.3,
                            marker=mk, ms=2.8, mec="none", zorder=3)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            ax.grid(alpha=0.22, lw=0.5)
            if i == 0:
                ax.set_title("$P_S=%d$ dBm" % ps)

    handles = [plt.Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls="-",
                          lw=1.3, ms=4, mec="none", label=STYLE[m][2])
               for m in ("VQC", "DNN", "flat", "AO")]
    handles.append(plt.Line2D([], [], color="#999999", ls=":", lw=0.8,
                              label="Equal split"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.015))
    fig.supxlabel("Cumulative user share", fontsize=8.5, y=0.125)
    fig.supylabel("Cumulative private-power share", fontsize=8.5, x=0.012)
    fig.tight_layout(rect=[0.035, 0.155, 0.975, 1])
    for i, tag in enumerate(("$K{=}5$, $M{=}1$", "$K{=}10$, $M{=}2$")):
        fig.text(0.988, 0.735 - 0.415 * i, tag, rotation=270,
                 va="center", ha="right", fontsize=8)
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "power_ladder." + ext), dpi=200,
                    bbox_inches="tight")
    print("wrote power_ladder.{pdf,png}")


if __name__ == "__main__":
    main()
