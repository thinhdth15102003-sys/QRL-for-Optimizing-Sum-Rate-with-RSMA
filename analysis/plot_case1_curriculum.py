"""
plot_case1_curriculum.py
------------------------
Stitch the Case 1 curriculum (R_LoS 0.2→0.5) from its resume chain into one
continuous training curve and plot QoS + R_tot vs cumulative curriculum episode.

Chain (resume lineage):
  result_5 (0.2, fresh) --ep900--> result_8 (0.3) --ep300--> result_12 (0.4)
  --ep500--> result_18 (0.5, LOCK ep400)

Placeholder figure — refresh when the final uniform-config Case 1 run is available.
"""
import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (disk run, ramp R_LoS, milestone ep used by the curriculum, lambda label)
CHAIN = [
    ("result_5",  0.2, 900, r"$\lambda_D{=}1.5$"),
    ("result_8",  0.3, 300, r"$\lambda_D{=}2.5$"),
    ("result_12", 0.4, 500, r"$\lambda_D{=}1.5$"),
    ("result_18", 0.5, 400, r"$\lambda_D{=}1.5$"),
]
RAMP_COLORS = ["#e8f0fe", "#e6f4ea", "#fef7e0", "#fce8e6"]

RE_DIAG = re.compile(r"diag\[(\d+)\]")
RE_ROLL = re.compile(r"rolling-50:.*QoS μ=\s*(\d+)%\s+Rtot μ=([\d.]+)")


def parse(run, ep_max):
    """Return (eps, qos, rtot) from rolling-50 lines up to ep_max."""
    path = os.path.join(ROOT, "results", run, "training_log.txt")
    eps, qos, rtot = [], [], []
    cur = 0
    for ln in open(path, errors="ignore"):
        m = RE_DIAG.search(ln)
        if m:
            cur = int(m.group(1)); continue
        r = RE_ROLL.search(ln)
        if r and cur <= ep_max:
            eps.append(cur); qos.append(int(r.group(1))); rtot.append(float(r.group(2)))
    return np.array(eps), np.array(qos), np.array(rtot)


fig, ax1 = plt.subplots(figsize=(7.2, 3.6))
ax2 = ax1.twinx()

offset = 0
bounds = [0]
X, Q, RT = [], [], []          # accumulate → ONE continuous line (no boundary gaps)
for i, (run, rlos, epm, lam) in enumerate(CHAIN):
    e, q, rt = parse(run, epm)
    x = e + offset
    ax1.axvspan(offset, offset + epm, color=RAMP_COLORS[i], zorder=0)
    X.append(x); Q.append(q); RT.append(rt)
    # ramp label
    ax1.text(offset + epm / 2, 99, rf"$R_{{\rm LoS}}{{=}}{rlos}$",
             ha="center", va="top", fontsize=8.5, color="#333")
    ax1.text(offset + epm / 2, 95, lam, ha="center", va="top", fontsize=7.5, color="#555")
    offset += epm
    bounds.append(offset)

X  = np.concatenate(X); Q = np.concatenate(Q); RT = np.concatenate(RT)
order = np.argsort(X)                       # ensure monotone x
X, Q, RT = X[order], Q[order], RT[order]
ax1.plot(X, Q,  color="#1a73e8", lw=1.8, zorder=3)
ax2.plot(X, RT, color="#d93025", lw=1.4, ls="--", zorder=2, alpha=0.9)

for b in bounds[1:-1]:
    ax1.axvline(b, color="#888", ls=":", lw=0.9, zorder=1)

# LOCK marker (ramp 0.5 ep400 = end)
ax1.scatter([offset], [84], color="#188038", s=45, zorder=5, marker="*")
ax1.annotate("LOCK\nQoS 84%\n$R_{\\rm tot}$ 2.54", (offset, 84),
             textcoords="offset points", xytext=(-46, -2), fontsize=7.5,
             color="#188038", ha="right", va="center")

ax1.set_xlabel("Cumulative curriculum episode")
ax1.set_ylabel("QoS satisfaction (%)", color="#1a73e8")
ax2.set_ylabel(r"Sum-rate $R_{\rm tot}$", color="#d93025")
ax1.set_ylim(45, 103); ax2.set_ylim(0.8, 3.4)
ax1.tick_params(axis="y", labelcolor="#1a73e8")
ax2.tick_params(axis="y", labelcolor="#d93025")
ax1.set_xlim(0, offset)
ax1.set_title("Case 1 (K=5, M=1) curriculum: QoS held across LoS-radius ramp 0.2→0.5",
              fontsize=9.5)
# legend proxies
from matplotlib.lines import Line2D
ax1.legend([Line2D([0],[0],color="#1a73e8",lw=1.8),
            Line2D([0],[0],color="#d93025",lw=1.4,ls="--")],
           ["QoS (rolling-50)", r"$R_{\rm tot}$ (rolling-50)"],
           loc="lower right", fontsize=7.5, framealpha=0.9)
fig.tight_layout()

outdir = os.path.join(ROOT, "Research Paper", "figures")
os.makedirs(outdir, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(outdir, f"case1_curriculum.{ext}"), dpi=160, bbox_inches="tight")
print("saved:", os.path.join(outdir, "case1_curriculum.{pdf,png}"))
print("boundaries (cum ep):", bounds, " total:", offset)
