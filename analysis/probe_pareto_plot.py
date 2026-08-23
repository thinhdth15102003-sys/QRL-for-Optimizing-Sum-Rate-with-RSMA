"""Draw the rate--QoS Pareto frontier with each policy as a point.

The curve is the non-dominated envelope of EVERY action measured on the state
pool: both constrained oracle families (#met >= n and robust margin >= delta)
and the evaluated policies themselves. Including the policies is not circular —
it is what an empirical Pareto frontier is. It is also necessary here: the oracle
searches power on a fixed grid while AO runs a coordinate ascent, so at the
QoS -> 100% corner AO finds an allocation the grid misses (Case 2: AO reaches
100.000% QoS at 1.654 where the grid saturates at 99.976% / 1.563). Drawing only
the grid result would place a measured, attainable point above the "unattainable"
region. The curve therefore remains a LOWER BOUND on the true frontier.

Layout: the main axes zoom to QoS >= 80%, where every policy lies and where the
paper's claims are made; the inset carries the full curve so the hyperbolic shape
— steep only once QoS is allowed to collapse — stays visible.
"""
import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STYLE = {
    'AO':     ('#d62728', 'D', 'AO (per-state solver)'),
    'DNN':    ('#ff7f0e', 's', 'DNN'),
    'VQC':    ('#1f77b4', 'o', 'HQC-HAC (VQC)'),
    'Greedy': ('#2ca02c', '^', 'Greedy'),
    'Random': ('#7f7f7f', 'v', 'Random'),
}


def envelope(pts):
    """(qos, rate) -> non-dominated set, ascending in QoS."""
    out, best = [], -np.inf
    for q, r in sorted(pts, key=lambda p: (-p[0], -p[1])):   # QoS high -> low
        if r > best + 1e-12:
            out.append((q, r))
            best = r
    return sorted(out)


def load(path):
    d = np.load(path, allow_pickle=True)
    K = int(d['K'])
    oracle = []
    for n in range(1, K + 1):
        v = d[f'frn_{n}']
        if len(v):
            oracle.append((v[:, 1].mean(), v[:, 0].mean()))
    for i in range(len(d['delta_grid'])):
        v = d[f'frd_{i}']
        if len(v):
            oracle.append((v[:, 1].mean(), v[:, 0].mean()))
    mods = {k[4:]: (d[k][:, 1].mean(), d[k][:, 0].mean())
            for k in d.files if k.startswith('mod_')}
    return K, int(d['n_states']), oracle, mods


def group(mods):
    """DNN-s0..s3 -> one mean point with its seed spread."""
    seeds = {k: v for k, v in mods.items() if k.startswith('DNN-s')}
    out = {k: (v[0], v[1], 0.0, 0.0) for k, v in mods.items()
           if not k.startswith('DNN-s')}
    if seeds:
        q = np.array([v[0] for v in seeds.values()])
        r = np.array([v[1] for v in seeds.values()])
        out['DNN'] = (q.mean(), r.mean(), q.std(ddof=1), r.std(ddof=1))
    return out


def panel(ax, path, title, qlo):
    K, ns, oracle, mods = load(path)
    pts = group(mods)
    env = envelope(oracle + [(v[0], v[1]) for v in pts.values()])
    q = np.array([p[0] for p in env]) * 100
    r = np.array([p[1] for p in env])

    top = r.max() * 1.06
    ax.fill_between(q, r, top, color='0.90', zorder=0, step=None)
    ax.plot(q, r, '-', color='0.2', lw=1.7, zorder=2)
    ax.plot(q, r, '.', color='0.2', ms=4.5, zorder=3)

    handles = []
    for lbl in ('AO', 'DNN', 'VQC', 'Greedy', 'Random'):
        if lbl not in pts:
            continue
        qq, rr, qs, rs = pts[lbl]
        col, mk, name = STYLE[lbl]
        h = ax.errorbar(qq * 100, rr, xerr=qs * 100, yerr=rs, fmt=mk, color=col,
                        ms=8, mec='white', mew=0.9, capsize=3, lw=1.1,
                        zorder=5, label=name)
        handles.append(h)

    sel = q >= qlo
    ax.set_xlim(qlo, 100.7)
    ax.set_ylim(min(r[sel].min(), min(v[1] for v in pts.values())) * 0.93,
                max(r[sel].max(), max(v[1] for v in pts.values())) * 1.07)
    ax.set_xlabel('QoS satisfaction (%)')
    ax.set_ylabel(r'$R_{\mathrm{tot}}$ (bps/Hz)')
    ax.set_title(f'{title}    ({ns} states)', fontsize=10.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(0.035, 0.955, 'unattainable', transform=ax.transAxes,
            fontsize=8, color='0.4', va='top', style='italic')

    # inset: the whole curve, so the shape is not hidden by the zoom
    ins = ax.inset_axes([0.09, 0.10, 0.36, 0.40])
    ins.fill_between(q, r, r.max() * 1.05, color='0.90', zorder=0)
    ins.plot(q, r, '-', color='0.2', lw=1.2)
    for lbl in ('AO', 'DNN', 'VQC', 'Greedy', 'Random'):
        if lbl in pts:
            col, mk, _ = STYLE[lbl]
            ins.plot(pts[lbl][0] * 100, pts[lbl][1], mk, color=col, ms=3.4,
                     mec='white', mew=0.4)
    ins.set_xlim(0, 103)
    ins.tick_params(labelsize=6, length=2, pad=1)
    ins.set_title('full range', fontsize=6.5, pad=2)
    ins.grid(alpha=0.2, lw=0.4)
    return handles, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case1', required=True)
    ap.add_argument('--case2', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qlo', type=float, default=84.0)
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    h1, n1 = panel(axes[0], a.case1, r'Case 1  ($K{=}5$, $M{=}1$)', a.qlo)
    h2, n2 = panel(axes[1], a.case2, r'Case 2  ($K{=}10$, $M{=}2$)', a.qlo)
    fig.legend(handles=h1, loc='lower center', ncol=5, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(a.out, dpi=230, bbox_inches='tight')
    fig.savefig(a.out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"  states: case1 {n1}, case2 {n2}   -> {a.out}")


if __name__ == '__main__':
    sys.exit(main())
