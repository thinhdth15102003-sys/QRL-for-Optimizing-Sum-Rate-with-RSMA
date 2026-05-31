"""
analyze_critic_run.py
---------------------
Tier-3 diagnostic: post-hoc analysis of a finished (or in-progress) training
run.  Consumes:

    results/result_N/critic_diag.jsonl     ← written by RL/critic_diag.py
    results/result_N/training_log.txt      ← human log (fallback for old runs)
    results/result_N/agents/critic_*       ← final critic weights (optional)

Outputs (written next to the input run):
    results/result_N/analysis_critic/
        trends.png          time series: explVar, corr, bias, clipfrac, ‖∇‖V, ‖ψ‖
        residual_hist.png   histogram of (V − ret) at last K updates
        per_epoch_grad.png  grad/epoch heatmap across the run
        state_coverage.png  σ-per-dim evolution
        stratified_ev.png   EV by Blk / σ² bucket over time
        baselines.csv       linear V / mean V / zero V comparison
        sensitivity.csv     ∂V/∂s_i per state dim (if critic weights loaded)
        summary.txt         numeric digest + verdict

Usage
-----
    python analyze_critic_run.py results/result_5
    python analyze_critic_run.py results/result_5 --no-plots
    python analyze_critic_run.py results/result_5 --baselines
"""
from __future__ import annotations

import os
import json
import argparse
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


# ───────────────────────────────────────────────────────────────────────────────
# Loader
# ───────────────────────────────────────────────────────────────────────────────

def load_diag_jsonl(path: str) -> list:
    """Read critic_diag.jsonl → list of {'ep': int, 'diag': dict}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _series(recs: list, *keys, default=np.nan):
    """Extract a time-series from records, navigating nested keys.
    keys = ('critic_fit', 'expl_var')  →  rec['diag']['critic_fit']['expl_var']."""
    out = []
    for r in recs:
        v = r['diag']
        try:
            for k in keys:
                v = v[k]
        except (KeyError, TypeError):
            v = default
        out.append(v)
    return np.asarray(out, dtype=float)


# ───────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ───────────────────────────────────────────────────────────────────────────────

def plot_trends(recs: list, out_path: str) -> None:
    if not HAVE_MPL:
        return
    eps = np.array([r['ep'] for r in recs])
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    # explVar + corr
    axes[0, 0].plot(eps, _series(recs, 'critic_fit', 'expl_var'),
                    label='explVar', color='tab:blue')
    axes[0, 0].plot(eps, _series(recs, 'critic_fit', 'corr_V_ret'),
                    label='corr(V,ret)', color='tab:orange', alpha=0.7)
    axes[0, 0].axhline(0.5, ls='--', c='gray', alpha=0.3)
    axes[0, 0].set_title('Critic fit')
    axes[0, 0].set_ylim(-0.5, 1.05)
    axes[0, 0].legend(loc='best'); axes[0, 0].grid(alpha=0.3)
    # bias + resid_std
    axes[0, 1].plot(eps, _series(recs, 'critic_fit', 'bias'),
                    label='bias = μ(V-ret)', color='tab:red')
    axes[0, 1].plot(eps, _series(recs, 'critic_fit', 'resid_std'),
                    label='resid σ', color='tab:purple', alpha=0.7)
    axes[0, 1].axhline(0, ls='--', c='gray', alpha=0.3)
    axes[0, 1].set_title('Residuals')
    axes[0, 1].legend(loc='best'); axes[0, 1].grid(alpha=0.3)
    # grad norm (preclip) + clipfrac
    ax = axes[1, 0]
    ax.plot(eps, _series(recs, 'critic_grad', 'preclip_mean'),
            label='preclip ‖∇‖V mean', color='tab:green')
    ax.plot(eps, _series(recs, 'critic_grad', 'preclip_max'),
            label='preclip max', color='tab:olive', alpha=0.5)
    ax.set_ylabel('grad norm'); ax.legend(loc='upper left')
    ax.grid(alpha=0.3); ax.set_title('Critic gradient (pre-clip)')
    ax.set_yscale('log')
    ax2 = ax.twinx()
    ax2.plot(eps, _series(recs, 'critic_grad', 'clip_frac') * 100,
             label='clipfrac %', color='tab:red', linestyle=':')
    ax2.set_ylabel('clipfrac (%)', color='tab:red'); ax2.set_ylim(-5, 105)
    # V mean + return mean
    axes[1, 1].plot(eps, _series(recs, 'critic_fit', 'V_mean'),
                    label='V mean', color='tab:blue')
    axes[1, 1].plot(eps, _series(recs, 'return_stats', 'ret_mean'),
                    label='return mean', color='tab:orange', alpha=0.7)
    axes[1, 1].fill_between(eps,
                            _series(recs, 'return_stats', 'ret_p25'),
                            _series(recs, 'return_stats', 'ret_p75'),
                            color='tab:orange', alpha=0.1, label='ret IQR')
    axes[1, 1].set_title('V mean vs return mean (scale drift)')
    axes[1, 1].legend(loc='best'); axes[1, 1].grid(alpha=0.3)
    # ‖ψ‖ critic weight + per-epoch overtrain ratio
    axes[2, 0].plot(eps, _series(recs, 'critic_weights', 'psi_global_norm'),
                    label='‖ψ‖', color='tab:cyan')
    axes[2, 0].set_title('Critic weight L2 norm'); axes[2, 0].grid(alpha=0.3)
    axes[2, 0].legend(loc='best')
    axes[2, 1].plot(eps,
                    _series(recs, 'critic_grad', 'overtrain_ratio_e6_e1'),
                    label='grad_e6/grad_e1', color='tab:brown')
    axes[2, 1].axhline(1.0, ls='--', c='gray', alpha=0.5)
    axes[2, 1].set_title('PPO over-train ratio (epoch-6 grad / epoch-1)')
    axes[2, 1].grid(alpha=0.3); axes[2, 1].legend(loc='best')
    axes[2, 0].set_xlabel('episode'); axes[2, 1].set_xlabel('episode')
    fig.suptitle('Critic diagnostic trends', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_per_epoch_grad_heatmap(recs: list, out_path: str) -> None:
    if not HAVE_MPL:
        return
    pe = [r['diag']['critic_grad'].get('per_epoch_mean', []) for r in recs]
    n_ep = max((len(p) for p in pe), default=0)
    if n_ep == 0:
        return
    M = np.full((len(pe), n_ep), np.nan)
    for i, p in enumerate(pe):
        M[i, :len(p)] = p
    eps = np.array([r['ep'] for r in recs])
    fig, ax = plt.subplots(figsize=(11, 5))
    # log-scale color for visibility
    im = ax.imshow(np.log10(M.T + 1e-3), aspect='auto', origin='lower',
                   extent=[eps[0], eps[-1], 0.5, n_ep + 0.5],
                   cmap='viridis')
    ax.set_xlabel('episode'); ax.set_ylabel('PPO epoch')
    ax.set_title('Per-PPO-epoch critic grad norm (log10)')
    fig.colorbar(im, ax=ax, label='log10 ‖∇‖V')
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_stratified_ev(recs: list, out_path: str) -> None:
    if not HAVE_MPL:
        return
    eps = np.array([r['ep'] for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    # Blk buckets
    blk_names = ['blk_0', 'blk_1_3', 'blk_4_6', 'blk_7p']
    for name in blk_names:
        ys = [r['diag'].get('stratified_ev', {}).get('by_blk', {})
              .get(name, {}).get('ev', np.nan) for r in recs]
        axes[0].plot(eps, ys, label=name)
    axes[0].set_title('explVar by Blk bucket')
    axes[0].axhline(0.5, ls='--', c='gray', alpha=0.3)
    axes[0].legend(loc='best'); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel('episode'); axes[0].set_ylabel('explVar')
    # σ² buckets
    s2_names = ['s2_q1', 's2_q2', 's2_q3', 's2_q4']
    for name in s2_names:
        ys = [r['diag'].get('stratified_ev', {}).get('by_sigma2', {})
              .get(name, {}).get('ev', np.nan) for r in recs]
        axes[1].plot(eps, ys, label=name)
    axes[1].set_title('explVar by σ² quartile')
    axes[1].axhline(0.5, ls='--', c='gray', alpha=0.3)
    axes[1].legend(loc='best'); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel('episode')
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def plot_layer_health(recs: list, out_path: str) -> None:
    if not HAVE_MPL:
        return
    eps = np.array([r['ep'] for r in recs])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    # Dead fraction per layer over time
    layer_keys = sorted({k for r in recs
                         for k in r['diag'].get('critic_activations', {}).keys()
                         if k.startswith('layer') and not k.endswith('_output')})
    for k in layer_keys:
        dead = [r['diag'].get('critic_activations', {}).get(k, {})
                .get('dead_frac', np.nan) for r in recs]
        axes[0].plot(eps, np.array(dead) * 100, label=k)
    axes[0].set_title('ReLU dead fraction per layer')
    axes[0].set_ylabel('% dead'); axes[0].set_ylim(0, 100)
    axes[0].legend(loc='best'); axes[0].grid(alpha=0.3)
    # State coverage: collapsed (pathological) vs constant-by-design over time
    n_coll  = _series(recs, 'state_coverage', 'n_collapsed_dims')
    n_const = _series(recs, 'state_coverage', 'n_const_dims')
    n_dims  = _series(recs, 'state_coverage', 'n_dims')
    axes[1].plot(eps, n_coll,  label='collapsed dims (pathological)', color='tab:red')
    axes[1].plot(eps, n_const, label='constant dims (by-design)',
                 color='tab:gray', linestyle=':')
    axes[1].plot(eps, n_dims, label='total dims', linestyle='--', alpha=0.4)
    axes[1].set_title('State coverage (scale-aware)')
    axes[1].legend(loc='best'); axes[1].grid(alpha=0.3)
    axes[1].set_xlabel('episode'); axes[0].set_xlabel('episode')
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


# ───────────────────────────────────────────────────────────────────────────────
# Baseline comparisons — what would a TRIVIAL V give?
# ───────────────────────────────────────────────────────────────────────────────

def baselines_csv(recs: list, out_path: str) -> dict:
    """For each rollout, compute explVar of trivial baselines:
        zero  V(s) = 0
        mean  V(s) = mean(ret over this rollout)
        For 'linear' we'd need s_t which is not in JSONL → skipped here.
    Compare with the recorded online critic.
    """
    eps  = []
    on   = []
    zero = []
    mean = []
    for r in recs:
        d = r['diag']
        ret_mean = d['return_stats']['ret_mean']
        ret_std  = d['return_stats']['ret_std']
        ev_online = d['critic_fit']['expl_var']
        # EV(zero baseline) = 1 - mean(ret²) / Var(ret) → we approximate
        # since we don't have per-sample rets here, only μ/σ.
        # EV(zero) = 1 - (μ² + σ²) / σ² = -μ²/σ² (≤ 0 if μ≠0)
        ev_zero = -(ret_mean ** 2) / max(ret_std ** 2, 1e-12)
        # EV(mean) = 0 by construction
        ev_mean = 0.0
        eps.append(r['ep']); on.append(ev_online)
        zero.append(ev_zero); mean.append(ev_mean)
    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ep', 'online_EV', 'EV_zero_baseline',
                    'EV_mean_baseline', 'online_minus_mean',
                    'online_minus_zero'])
        for i, e in enumerate(eps):
            w.writerow([e, f"{on[i]:.4f}", f"{zero[i]:.4f}",
                        f"{mean[i]:.4f}",
                        f"{on[i] - mean[i]:.4f}",
                        f"{on[i] - zero[i]:.4f}"])
    return {
        'mean_online_EV' : float(np.nanmean(on)),
        'mean_zero_EV'   : float(np.nanmean(zero)),
        'mean_mean_EV'   : float(np.nanmean(mean)),
        'online_beats_mean_frac':
            float(np.mean(np.array(on) > np.array(mean))),
        'online_beats_zero_frac':
            float(np.mean(np.array(on) > np.array(zero))),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Critic input sensitivity ∂V/∂s_i (requires loaded weights)
# ───────────────────────────────────────────────────────────────────────────────

def critic_input_sensitivity(critic_dir: str, sample_states: np.ndarray) -> dict:
    """
    Load ClassicalCritic, then for each state in sample compute |∂V/∂s_i|
    via finite differences (cheap; d_state ≤ ~50).  Returns mean abs over
    samples per dim.
    """
    from RL import ClassicalCritic
    critic = ClassicalCritic.from_dir(critic_dir, seed=0)
    eps_fd = 1e-3
    N, d = sample_states.shape
    grads = np.zeros(d)
    for i in range(N):
        s = sample_states[i]
        V0 = critic.forward(s)
        for j in range(d):
            s_p = s.copy(); s_p[j] += eps_fd
            s_m = s.copy(); s_m[j] -= eps_fd
            g = (critic.forward(s_p) - critic.forward(s_m)) / (2 * eps_fd)
            grads[j] += abs(g)
    grads /= N
    return {
        'mean_abs_grad': grads.tolist(),
        'top5_dims'   : np.argsort(grads)[-5:][::-1].tolist(),
        'bot5_dims'   : np.argsort(grads)[:5].tolist(),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Summary writer
# ───────────────────────────────────────────────────────────────────────────────

def write_summary(recs: list, baselines: dict, out_path: str) -> None:
    eps = np.array([r['ep'] for r in recs])
    n   = len(recs)
    last_k = recs[-min(10, n):]
    def _avg(xs):
        xs = [x for x in xs if x is not None and not np.isnan(x)]
        return float(np.mean(xs)) if xs else float('nan')
    def _std(xs):
        xs = [x for x in xs if x is not None and not np.isnan(x)]
        return float(np.std(xs)) if xs else float('nan')
    def _minv(xs):
        xs = [x for x in xs if x is not None and not np.isnan(x)]
        return float(np.min(xs)) if xs else float('nan')
    # ── T3 explVar STABILITY (the key PopArt success metric) ──
    # Judge post-β-anneal only (ep>400) so we measure the SETTLED regime, not warm-up.
    ev_all   = [(r['ep'], r['diag']['critic_fit']['expl_var']) for r in recs]
    ev_post  = [v for (e, v) in ev_all if e >= 400 and v is not None and not np.isnan(v)]
    ev_post  = ev_post if ev_post else [v for (_e, v) in ev_all if v is not None and not np.isnan(v)]
    ev_post_mean = float(np.mean(ev_post)) if ev_post else float('nan')
    ev_post_std  = float(np.std(ev_post))  if ev_post else float('nan')
    ev_post_min  = float(np.min(ev_post))  if ev_post else float('nan')
    ev_frac_gt05 = float(np.mean([v > 0.5 for v in ev_post])) if ev_post else float('nan')
    ev_stable    = (ev_post_mean > 0.5 and ev_post_std < 0.15 and ev_post_min > 0.3)
    ev_last   = _avg([r['diag']['critic_fit']['expl_var']  for r in last_k])
    ev_std10  = _std([r['diag']['critic_fit']['expl_var']  for r in last_k])
    # PopArt-specific (if present)
    pop_dv_max = _avg([r['diag'].get('popart', {}).get('pop_dv_max')
                       for r in last_k if 'popart' in r['diag']])
    pa_track_mu = _avg([r['diag'].get('popart', {}).get('track_mu_err')
                        for r in last_k if 'popart' in r['diag']])
    corr_last = _avg([r['diag']['critic_fit']['corr_V_ret'] for r in last_k])
    bias_last = _avg([r['diag']['critic_fit']['bias'] for r in last_k])
    clip_last = _avg([r['diag']['critic_grad']['clip_frac'] for r in last_k])
    ot_last   = _avg([r['diag']['critic_grad']
                      .get('overtrain_ratio_e6_e1') for r in last_k])
    coll_last  = _avg([r['diag']['state_coverage'].get('n_collapsed_dims', 0)
                       for r in last_k])
    const_last = _avg([r['diag']['state_coverage'].get('n_const_dims', 0)
                       for r in last_k])
    L = []
    L.append("=" * 78)
    L.append("  POST-HOC CRITIC ANALYSIS  ·  Tier-3 diagnostic")
    L.append("=" * 78)
    L.append(f"  Run window: ep {int(eps[0])} .. {int(eps[-1])}  "
             f"(n_updates = {n})")
    L.append("")
    L.append("HEADLINE (last 10 updates)")
    L.append("-" * 78)
    L.append(f"  explVar mean = {ev_last:+.3f}  (std {ev_std10:.3f})  ← compare with Tier-1 ceiling")
    L.append(f"  corr(V,ret)  = {corr_last:+.3f}")
    L.append(f"  bias         = {bias_last:+.3f}")
    L.append("")
    L.append("★ T3 explVar STABILITY (post-β-anneal ep>400 — KEY PopArt success metric)")
    L.append(f"  mean={ev_post_mean:+.3f}  std={ev_post_std:.3f}  min={ev_post_min:+.3f}  "
             f"frac>0.5={ev_frac_gt05*100:.0f}%")
    L.append(f"  → {'✅ STABLE (mean>0.5, std<0.15, min>0.3) — PopArt WORKED' if ev_stable else '❌ WOBBLE — PopArt insufficient → non-stationarity root (see VERDICT)'}")
    if not np.isnan(pop_dv_max):
        _pop_ok = pop_dv_max < 1e-3
        L.append(f"  T5 POP |ΔV|max={pop_dv_max:.1e} [{'OK' if _pop_ok else '⚠IMPLEMENTATION BUG'}]  "
                 f"· T4 track Δμ={pa_track_mu:.2f}")
    L.append(f"  clipfrac     = {clip_last*100:.1f}%   "
             f"({'⚠ clipping ≥50% — grad clip too tight' if clip_last >= 0.5 else 'OK'})")
    L.append(f"  e6/e1 grad   = {ot_last:.2f}   "
             f"({'⚠ critic over-trains per rollout' if ot_last > 2.0 else 'OK'})")
    L.append(f"  collapsed state dims = {coll_last:.0f}   "
             f"(constant-by-design = {const_last:.0f}, e.g. demand block)")
    L.append("")
    L.append("BASELINE COMPARISON")
    L.append("-" * 78)
    L.append(f"  mean online critic EV     = {baselines['mean_online_EV']:+.3f}")
    L.append(f"  mean zero-V baseline EV   = {baselines['mean_zero_EV']:+.3f}")
    L.append(f"  mean mean-V baseline EV   = {baselines['mean_mean_EV']:+.3f}")
    L.append(f"  online beats mean-V on    "
             f"{baselines['online_beats_mean_frac']*100:.0f}% of updates")
    L.append(f"  online beats zero-V on    "
             f"{baselines['online_beats_zero_frac']*100:.0f}% of updates")
    L.append("")
    L.append("VERDICT GUIDE (cross-reference with Tier-1 ceiling / Tier-0 oracle)")
    L.append("-" * 78)
    if clip_last >= 0.5:
        L.append("  → clipfrac high → first action: raise grad_clip_critic (1.0 → 5.0).")
    if ot_last > 2.0:
        L.append("  → e6/e1 high → critic over-trains per rollout. "
                 "Reduce critic PPO epochs (split from actor's 6 to ~2-3).")
    if coll_last > 3:
        L.append("  → collapsed state dims (vary ≪ peers) → feature collapse, "
                 "possibly from frozen sub-actors (Phase/quantum idle).")
    if abs(bias_last) > 1.0:
        L.append("  → large |bias| → return scale drift. Add reward "
                 "normalisation or PopArt (Nhóm A in survey).")
    L.append("")
    L.append("FILES")
    L.append("-" * 78)
    L.append("  trends.png · per_epoch_grad.png · stratified_ev.png · "
             "layer_health.png · baselines.csv")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))


# ───────────────────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir', type=str,
                    help="path to a results/result_N/ directory")
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--baselines', action='store_true',
                    help="(already always on — kept for backwards compat)")
    args = ap.parse_args()

    jsonl_path = os.path.join(args.run_dir, 'critic_diag.jsonl')
    out_dir    = os.path.join(args.run_dir, 'analysis_critic')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {jsonl_path} ...")
    recs = load_diag_jsonl(jsonl_path)
    print(f"  {len(recs)} diagnostic records")

    if not args.no_plots and HAVE_MPL:
        plot_trends(recs, os.path.join(out_dir, 'trends.png'))
        plot_per_epoch_grad_heatmap(recs,
                                    os.path.join(out_dir, 'per_epoch_grad.png'))
        plot_stratified_ev(recs, os.path.join(out_dir, 'stratified_ev.png'))
        plot_layer_health(recs, os.path.join(out_dir, 'layer_health.png'))
        print(f"  plots → {out_dir}/*.png")
    elif args.no_plots:
        print("  plots skipped (--no-plots)")
    else:
        print("  matplotlib not available — plots skipped")

    bl = baselines_csv(recs, os.path.join(out_dir, 'baselines.csv'))
    print(f"  baselines → {out_dir}/baselines.csv")

    summary_path = os.path.join(out_dir, 'summary.txt')
    write_summary(recs, bl, summary_path)
    print(f"  summary  → {summary_path}")


if __name__ == '__main__':
    main()
