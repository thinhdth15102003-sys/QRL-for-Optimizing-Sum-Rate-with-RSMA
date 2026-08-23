"""Figure: how concentrated each method's allocation is, versus the power budget.

Left column  = share of sum(w_p) held by the top-ranked user (the DECISION).
Right column = share of R_tot held by the top-ranked user (the CONSEQUENCE).
Dashed line  = 1/K, a perfectly even split.

Reads analysis/data/power_concentration.jsonl written by
analysis/probe_power_concentration.py.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "data", "power_concentration.jsonl")
OUT = os.path.join(os.path.dirname(HERE), "Research Paper", "figures")

STYLE = {"AO":   dict(color="#444444", marker="s", ls="--", label="AO"),
         "flat": dict(color="#c0392b", marker="o", ls="-",  label="PPO-flat"),
         "DNN":  dict(color="#2471a3", marker="^", ls="-",  label="DNN"),
         "VQC":  dict(color="#1e8449", marker="D", ls="-",  label="HQC-HAC (VQC)")}

rows = [json.loads(l) for l in open(SRC) if l.strip()]
cases = [(5, 1, "Case 1 ($K{=}5$, $M{=}1$)"), (10, 2, "Case 2 ($K{=}10$, $M{=}2$)")]

fig, ax = plt.subplots(2, 2, figsize=(8.6, 6.0), sharex=True)
for i, (K, M, title) in enumerate(cases):
    sub = sorted([r for r in rows if r["K"] == K], key=lambda r: r["P_S"])
    if not sub:
        continue
    ps = [r["P_S"] for r in sub]
    for j, key in enumerate(("w_top", "r_top")):
        A = ax[i][j]
        for m in ("AO", "flat", "DNN", "VQC"):
            y = [r["rows"][m][key] for r in sub if m in r["rows"]]
            x = [r["P_S"] for r in sub if m in r["rows"]]
            A.plot(x, y, ms=5, lw=1.6, **STYLE[m])
        A.axhline(100.0 / K, color="#999999", lw=1.0, ls=":",
                  label="even split $1/K$" if (i == 0 and j == 1) else None)
        A.set_ylim(0, 108)
        A.grid(alpha=0.25, lw=0.6)
        if i == 1:
            A.set_xlabel("$P_S$ (dBm)")
        A.set_xticks(ps)
    ax[i][0].set_ylabel(title + "\n\ntop-1 share of $\\sum_k w_{p,k}$ (%)", fontsize=9)
    ax[i][1].set_ylabel("top-1 share of $R_{\\mathrm{tot}}$ (%)", fontsize=9)

ax[0][0].set_title("Allocation: private power", fontsize=10)
ax[0][1].set_title("Outcome: achieved rate", fontsize=10)
h0, l0 = ax[0][1].get_legend_handles_labels()
fig.legend(h0, l0, fontsize=8.5, framealpha=0.95, ncol=5,
           loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.045, 1, 1))
os.makedirs(OUT, exist_ok=True)
for ext in ("pdf", "png"):
    p = os.path.join(OUT, "power_concentration." + ext)
    fig.savefig(p, dpi=170, bbox_inches="tight")
    print("wrote", p)
