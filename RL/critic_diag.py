"""
critic_diag.py
--------------
Tier-2 diagnostic helpers for the critic.  Called once per PPO update from
train.py.  Outputs:

    (i)  A multi-line `crit-diag[ep]` block written to the training log
         (visible during/after training, parseable by analyze_critic_run.py).
    (ii) One structured JSON record appended to results/result_N/critic_diag.jsonl,
         indexed by episode.  Easier downstream parsing than reading the log.

Diagnostics in each record (target the categories from Common-Knowledge.txt
Part 3 [B-1/B-2/B-2b]):

  * preclip / postclip critic grad norm  (per minibatch),  clipfrac
  * per-PPO-epoch critic grad-V mean      (detect overtrain / drift across epochs)
  * Pearson corr(V, ret),  bias = mean(V-ret),  resid σ = std(V-ret)
  * return stats: μ, σ, skew, kurtosis  (heavy-tail detector)
  * critic ‖ψ‖ weight norm (per-layer + global; blow-up detector)
  * per-layer activation stats on a rollout-sample forward pass
  * V_next-last bootstrap quality (MC return from episode-end vs critic V_next)
  * state coverage σ per-dim  (dead-dim detector)
  * stratified explVar by Blk/K bucket and σ² (n0) quartile
  * CSI gap stats |g - g_hat| / |g|  (state→reward mismatch via κ)

These are READ-ONLY diagnostics: no critic parameters are touched here.
"""
from __future__ import annotations
import json
import os
from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np


# ───────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ───────────────────────────────────────────────────────────────────────────────

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return float('nan')
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return float('nan')
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _moment_skew(x: np.ndarray) -> float:
    if x.size < 3:
        return float('nan')
    m = x.mean(); s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def _moment_kurt(x: np.ndarray) -> float:
    """Excess kurtosis (0 = Gaussian)."""
    if x.size < 4:
        return float('nan')
    m = x.mean(); s = x.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _percentile_clipped(x: np.ndarray, p: float) -> float:
    if x.size == 0:
        return float('nan')
    return float(np.percentile(x, p))


# ───────────────────────────────────────────────────────────────────────────────
# Critic introspection — forward pass with per-layer activations captured
# ───────────────────────────────────────────────────────────────────────────────

def critic_weight_norms(critic) -> dict:
    """Per-layer L2 weight norms + global L2 norm of ψ."""
    out: dict = {}
    total_sq = 0.0
    for k, v in critic._params.items():
        n = float(np.linalg.norm(v))
        out[f'{k}_norm'] = n
        total_sq += n ** 2
    out['psi_global_norm'] = float(np.sqrt(total_sq))
    return out


def critic_layer_activation_stats(critic, states: np.ndarray) -> dict:
    """Forward `states` (B, d_state) through critic capturing per-layer
    pre-activation (Wx+b) stats: mean / std / fraction-zero (for ReLU dead).

    Returns dict keyed by layer index (0..n_layers-1).
    """
    B = states.shape[0]
    # Replicate the layer-norm done in critic._sa for d_action=0
    mu  = states.mean(axis=1, keepdims=True)
    sd  = states.std(axis=1,  keepdims=True) + 1e-6
    x   = (states - mu) / sd
    out: dict = {}
    for i, (W, b) in enumerate(zip(critic.Ws, critic.bs)):
        pre = x @ W + b
        if i < critic.n_layers - 1:
            post = np.maximum(0.0, pre)
            dead_frac = float(np.mean(post == 0.0))
            out[f'layer{i}'] = {
                'pre_mean'  : float(pre.mean()),
                'pre_std'   : float(pre.std()),
                'post_std'  : float(post.std()),
                'dead_frac' : dead_frac,
                'width'     : int(W.shape[1]),
            }
            x = post
        else:
            out[f'layer{i}_output'] = {
                'pred_mean': float(pre.mean()),
                'pred_std' : float(pre.std()),
                'width'    : 1,
            }
            x = pre
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Bootstrap quality — V_next at episode boundaries vs MC return from end
# ───────────────────────────────────────────────────────────────────────────────

def bootstrap_quality(rollout_buf: list, critic, gamma: float) -> dict:
    """
    For each episode in rollout_buf, compare V_next bootstrapped at the final
    transition to the actual MC return-from-end (sum of remaining rewards).

    Returns
    -------
    dict with mean, std, p25, p75 of (V_next − MC_return).
    """
    # Re-group rollout_buf into episodes by walking 'is_terminal'
    episodes: list = []
    cur: list = []
    for t in rollout_buf:
        cur.append(t)
        if t.get('is_terminal', False):
            episodes.append(cur); cur = []
    if cur:
        episodes.append(cur)

    errs = []
    for ep_buf in episodes:
        # MC return from each step to end
        H = len(ep_buf)
        mc_from = np.zeros(H)
        G = 0.0; disc = 1.0
        for i in reversed(range(H)):
            G    = ep_buf[i]['reward'] + gamma * G
            mc_from[i] = G
        # Bootstrap quality at the LAST step: V_next_last vs 0 (after last)
        # The 'V_next_last' used in train.py is critic.forward(ep_buf[-1]['s_t_next']).
        # Here we reproduce it.
        try:
            v_next_last = float(critic.forward(ep_buf[-1]['s_t_next']))
        except Exception:
            v_next_last = 0.0
        # The "true" bootstrap target is the discounted return BEYOND the episode
        # window, which we don't observe.  As a proxy: error = V_next_last − 0
        # since we're treating as truncated.  More meaningful is the per-step
        # comparison V_t vs MC_from_t over the episode (already in V_t / ret).
        for i, t in enumerate(ep_buf):
            errs.append(float(t['V_t']) - mc_from[i])
    errs = np.array(errs) if errs else np.array([0.0])
    return {
        'V_minus_MCret_mean'  : float(errs.mean()),
        'V_minus_MCret_std'   : float(errs.std()),
        'V_minus_MCret_p25'   : _percentile_clipped(errs, 25),
        'V_minus_MCret_p75'   : _percentile_clipped(errs, 75),
        'n_transitions'       : int(errs.size),
        'n_episodes'          : len(episodes),
    }


# ───────────────────────────────────────────────────────────────────────────────
# State coverage — find dead dimensions
# ───────────────────────────────────────────────────────────────────────────────

def state_coverage(rollout_buf: list,
                   const_thresh: float = 1e-12,
                   collapse_frac: float = 0.01) -> dict:
    """Per-dim σ of s_t over the rollout, with SCALE-AWARE dead detection.

    The state here lives at a tiny absolute scale (satellite channels ~1e-7,
    affinities ~1e-3), and the critic applies a per-sample LayerNorm, so an
    ABSOLUTE σ threshold is meaningless.  Instead we distinguish:

      * constant-by-design : σ < const_thresh  (e.g. the demand block D_k —
        legitimately fixed, carries no info but is NOT pathological).
      * collapsed          : 0 < σ < collapse_frac · median(σ over non-const
        dims) — varies far less than its peers (a genuine red flag if a dim
        that should carry signal has collapsed).

    This avoids the false-positive where half the state is flagged just
    because the whole vector is small in magnitude.
    """
    S = np.stack([t['s_t'] for t in rollout_buf])
    sigs = S.std(axis=0)
    mus  = np.abs(S.mean(axis=0)) + 1e-30
    cov  = sigs / mus                                  # coefficient of variation

    const_mask = sigs < const_thresh
    nonconst   = sigs[~const_mask]
    med_nc     = float(np.median(nonconst)) if nonconst.size else 0.0
    collapse_mask = (~const_mask) & (sigs < collapse_frac * med_nc)

    top_idx = np.argsort(sigs)[-5:][::-1]
    return {
        'n_dims'          : int(sigs.size),
        'sigma_mean'      : float(sigs.mean()),
        'sigma_min'       : float(sigs.min()),
        'sigma_max'       : float(sigs.max()),
        'sigma_median_nonconst': med_nc,
        'cov_mean'        : float(np.mean(cov[~const_mask])) if (~const_mask).any() else 0.0,
        # constant-by-design (e.g. demand) — expected, reported for transparency
        'n_const_dims'    : int(const_mask.sum()),
        'const_dims'      : np.where(const_mask)[0].tolist(),
        # collapsed = pathological (varies ≪ peers)
        'n_collapsed_dims': int(collapse_mask.sum()),
        'collapsed_dims'  : np.where(collapse_mask)[0].tolist(),
        'top5_dims'       : top_idx.tolist(),
        'top5_sigma'      : sigs[top_idx].tolist(),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Stratified explVar — split rollout by Blk/K bucket and σ² quartile
# ───────────────────────────────────────────────────────────────────────────────

def stratified_expl_var(rollout_buf: list) -> dict:
    """Compute explVar PER STRATUM defined by (Blk count, σ² quartile).
    Identifies subsets where the critic is systematically worse."""
    rets = np.array([t['ret']    for t in rollout_buf])
    Vs   = np.array([t['V_t']    for t in rollout_buf])
    blk  = np.array([t.get('blocked_n', 0) for t in rollout_buf], dtype=int)
    s2   = np.array([t.get('sigma2',    np.nan) for t in rollout_buf])

    def _ev(y, v):
        var_y = y.var()
        if var_y < 1e-12 or y.size < 5:
            return float('nan'), int(y.size)
        return 1.0 - (y - v).var() / var_y, int(y.size)

    # Bucket by Blk count: 0, 1-3, 4-6, 7+
    blk_buckets = OrderedDict([
        ('blk_0',   blk == 0),
        ('blk_1_3', (blk >= 1) & (blk <= 3)),
        ('blk_4_6', (blk >= 4) & (blk <= 6)),
        ('blk_7p',  blk >= 7),
    ])
    blk_ev = OrderedDict()
    for name, mask in blk_buckets.items():
        if mask.sum() > 0:
            ev, n = _ev(rets[mask], Vs[mask])
            blk_ev[name] = {'ev': ev, 'n': n}

    # Bucket by σ² quartile
    s2_finite = s2[np.isfinite(s2)]
    s2_ev = OrderedDict()
    if s2_finite.size >= 4:
        q = np.quantile(s2_finite, [0.25, 0.5, 0.75])
        s2_buckets = OrderedDict([
            ('s2_q1', s2 <= q[0]),
            ('s2_q2', (s2 > q[0]) & (s2 <= q[1])),
            ('s2_q3', (s2 > q[1]) & (s2 <= q[2])),
            ('s2_q4', s2 > q[2]),
        ])
        for name, mask in s2_buckets.items():
            if mask.sum() > 0:
                ev, n = _ev(rets[mask], Vs[mask])
                s2_ev[name] = {'ev': ev, 'n': n}
    return {'by_blk': blk_ev, 'by_sigma2': s2_ev}


# ───────────────────────────────────────────────────────────────────────────────
# CSI gap (multiplicative κ error) — state → reward mismatch magnitude
# ───────────────────────────────────────────────────────────────────────────────

def csi_gap_stats(env) -> dict:
    """|g - g_hat| / |g|  averaged over each channel.  Returns dict of means."""
    ch = env.channels
    def _rel_err(g_true, g_hat):
        denom = np.abs(g_true) + 1e-12
        return float(np.mean(np.abs(g_true - g_hat) / denom))
    return {
        'rel_err_g_SR': _rel_err(ch['g_SR'], ch['g_SR_hat']),
        'rel_err_g_SU': _rel_err(ch['g_SU'], ch['g_SU_hat']),
        'rel_err_g_RU': _rel_err(ch['g_RU'], ch['g_RU_hat']),
    }


# ───────────────────────────────────────────────────────────────────────────────
# Main entry — called once per PPO update
# ───────────────────────────────────────────────────────────────────────────────

def compute_critic_diag(rollout_buf: list,
                        critic,
                        ep_critic_gn_preclip_by_epoch: list,
                        grad_clip_critic: float,
                        gamma: float,
                        env=None,
                        pop_dv: float = float('nan')) -> dict:
    """
    Assemble the full critic-diagnostic dict for one PPO update.

    Parameters
    ----------
    rollout_buf                : the full PPO rollout buffer (list of dicts
                                  with V_t, ret, s_t, s_t_next, reward,
                                  is_terminal, [blocked_n, sigma2])
    critic                     : ClassicalCritic
    ep_critic_gn_preclip_by_epoch :
                                  list of length P.ppo_epochs.  Each element
                                  is the list of per-minibatch pre-clip critic
                                  grad norms for that epoch.
    grad_clip_critic           : params.grad_clip_critic (cutoff)
    gamma                      : discount γ
    env                        : ISTNEnv (optional) for CSI gap

    Returns
    -------
    dict ready to be (a) pretty-printed and (b) JSON-serialised.
    """
    out: dict = OrderedDict()

    Vs   = np.array([t['V_t'] for t in rollout_buf])
    rets = np.array([t['ret']  for t in rollout_buf])

    # ── critic fit quality ──
    res = Vs - rets
    out['critic_fit'] = {
        'corr_V_ret' : _safe_corr(Vs, rets),
        'bias'       : float(res.mean()),
        'resid_std'  : float(res.std()),
        'expl_var'   : 1.0 - (res.var() / max(rets.var(), 1e-12)),
        'V_mean'     : float(Vs.mean()),
        'V_std'      : float(Vs.std()),
        'V_p25'      : _percentile_clipped(Vs, 25),
        'V_p75'      : _percentile_clipped(Vs, 75),
    }

    # ── return distribution ──
    out['return_stats'] = {
        'ret_mean'   : float(rets.mean()),
        'ret_std'    : float(rets.std()),
        'ret_p25'    : _percentile_clipped(rets, 25),
        'ret_p75'    : _percentile_clipped(rets, 75),
        'ret_skew'   : _moment_skew(rets),
        'ret_kurt'   : _moment_kurt(rets),
    }

    # ── grad V per epoch + clipfrac ──
    grad_per_epoch = []
    clip_counts    = 0
    total_mb       = 0
    preclip_all    = []
    for ep_grads in ep_critic_gn_preclip_by_epoch:
        if not ep_grads:
            grad_per_epoch.append(float('nan'))
            continue
        arr = np.asarray(ep_grads)
        grad_per_epoch.append(float(arr.mean()))
        preclip_all.extend(ep_grads)
        clip_counts += int((arr > grad_clip_critic).sum())
        total_mb    += int(arr.size)
    preclip_all = np.asarray(preclip_all) if preclip_all else np.array([0.0])
    out['critic_grad'] = {
        'preclip_mean'  : float(preclip_all.mean()),
        'preclip_max'   : float(preclip_all.max()),
        'preclip_p75'   : _percentile_clipped(preclip_all, 75),
        'clip_frac'     : float(clip_counts / max(total_mb, 1)),
        'grad_clip_thresh': float(grad_clip_critic),
        'per_epoch_mean': grad_per_epoch,
        'n_minibatches' : int(total_mb),
        'overtrain_ratio_e6_e1':
            float(grad_per_epoch[-1] / max(grad_per_epoch[0], 1e-12))
            if (len(grad_per_epoch) > 1 and not np.isnan(grad_per_epoch[0])
                and not np.isnan(grad_per_epoch[-1])) else float('nan'),
    }

    # ── critic weights ──
    out['critic_weights'] = critic_weight_norms(critic)

    # ── PopArt running stats (value scale) + POP correctness (ΔV across update) ──
    if getattr(critic, 'popart', False):
        ret_mean = float(rets.mean()); ret_std = float(rets.std())
        pa_mu = float(getattr(critic, 'pa_mu', 0.0))
        pa_sg = float(getattr(critic, 'pa_sigma', 1.0))
        out['popart'] = {
            'mu'   : pa_mu,
            'sigma': pa_sg,
            'initialized': bool(getattr(critic, 'pa_initialized', False)),
            # T4 tracking: PopArt stats should ≈ this rollout's return stats (EMA lag)
            'track_mu_err' : abs(pa_mu - ret_mean),
            'track_sig_err': abs(pa_sg - ret_std),
            # T5 POP correctness: max |ΔV| for fixed states across the stats update.
            # POP is algebraically exact → should be ~1e-6 (numerical). Large = BUG.
            'pop_dv_max'   : float(pop_dv),
        }

    # ── per-layer activation stats — sample ≤256 states ──
    n_samp = min(256, len(rollout_buf))
    if n_samp > 0:
        idx = np.random.default_rng(0).choice(len(rollout_buf), n_samp, False)
        S = np.stack([rollout_buf[i]['s_t'] for i in idx])
        out['critic_activations'] = critic_layer_activation_stats(critic, S)

    # ── state coverage ──
    out['state_coverage'] = state_coverage(rollout_buf)

    # ── bootstrap quality ──
    out['bootstrap'] = bootstrap_quality(rollout_buf, critic, gamma)

    # ── stratified explVar ──
    out['stratified_ev'] = stratified_expl_var(rollout_buf)

    # ── CSI gap (env-side info) ──
    if env is not None:
        out['csi_gap'] = csi_gap_stats(env)

    return out


# ───────────────────────────────────────────────────────────────────────────────
# Pretty-printer for log file
# ───────────────────────────────────────────────────────────────────────────────

def format_critic_diag_lines(ep: int, diag: dict) -> list:
    """Render the diagnostic dict as a list of log lines starting with
    `  ┄ crit-diag[ep]  ...` mirroring the existing diag panel style."""
    L = []
    fit = diag['critic_fit']; grad = diag['critic_grad']
    L.append(f"  ┄ crit-diag[{ep}]  preclip‖∇‖V μ={grad['preclip_mean']:.1f} "
             f"max={grad['preclip_max']:.1f} p75={grad['preclip_p75']:.1f}  "
             f"clipfrac={grad['clip_frac']*100:.1f}%  thresh={grad['grad_clip_thresh']:.1f}")
    pe = grad['per_epoch_mean']
    pe_str = " ".join(f"{x:6.1f}" for x in pe)
    L.append(f"            grad/epoch: [{pe_str}]  e6/e1={grad['overtrain_ratio_e6_e1']:.2f}")
    L.append(f"            corr(V,ret)={fit['corr_V_ret']:+.3f}  bias={fit['bias']:+.3f}  "
             f"resid_σ={fit['resid_std']:.3f}  explVar={fit['expl_var']:+.3f}")
    rs = diag['return_stats']
    L.append(f"            return: μ={rs['ret_mean']:+.2f}  σ={rs['ret_std']:.2f}  "
             f"p25={rs['ret_p25']:+.2f}  p75={rs['ret_p75']:+.2f}  "
             f"skew={rs['ret_skew']:+.2f}  kurt={rs['ret_kurt']:+.2f}")
    wn = diag['critic_weights']
    L.append(f"            ‖ψ‖={wn['psi_global_norm']:.2f}  "
             + "  ".join(f"{k.replace('_norm',''):s}={v:.1f}" for k, v
                         in wn.items() if k != 'psi_global_norm' and 'W' in k))
    if 'popart' in diag:
        pa = diag['popart']
        _pop_ok = "OK" if (pa.get('pop_dv_max', 1) < 1e-3) else "⚠BUG"
        L.append(f"            PopArt: μ={pa['mu']:+.2f} σ={pa['sigma']:.2f}  "
                 f"track(Δμ={pa.get('track_mu_err',0):.2f} Δσ={pa.get('track_sig_err',0):.2f})  "
                 f"POP|ΔV|max={pa.get('pop_dv_max',float('nan')):.1e} [{_pop_ok}]")
    if 'critic_activations' in diag:
        acts = diag['critic_activations']
        layer_lines = []
        for k, v in acts.items():
            if 'dead_frac' in v:
                layer_lines.append(f"{k}(w={v['width']}: σ={v['post_std']:.2f} "
                                   f"dead={v['dead_frac']*100:.0f}%)")
        L.append(f"            acts: " + "  ".join(layer_lines))
    sc = diag['state_coverage']
    L.append(f"            state σ: med_nonconst={sc['sigma_median_nonconst']:.2e}  "
             f"CoV={sc['cov_mean']:.2f}  "
             f"const={sc['n_const_dims']}/{sc['n_dims']}(by-design) "
             f"collapsed={sc['n_collapsed_dims']} {sc['collapsed_dims']}")
    bs = diag['bootstrap']
    L.append(f"            bootstrap V-MC: μ={bs['V_minus_MCret_mean']:+.3f}  "
             f"σ={bs['V_minus_MCret_std']:.3f}  "
             f"n_ep={bs['n_episodes']} n_tr={bs['n_transitions']}")
    se = diag['stratified_ev']
    if se.get('by_blk'):
        blk_str = " ".join(f"{k}:{v['ev']:+.2f}(n={v['n']})"
                           for k, v in se['by_blk'].items())
        L.append(f"            EV by Blk: {blk_str}")
    if se.get('by_sigma2'):
        s2_str = " ".join(f"{k}:{v['ev']:+.2f}(n={v['n']})"
                          for k, v in se['by_sigma2'].items())
        L.append(f"            EV by σ²: {s2_str}")
    if 'csi_gap' in diag:
        cg = diag['csi_gap']
        L.append(f"            CSI gap (mean |g-ĝ|/|g|): "
                 f"SR={cg['rel_err_g_SR']:.3f}  SU={cg['rel_err_g_SU']:.3f}  "
                 f"RU={cg['rel_err_g_RU']:.3f}")
    return L


def write_critic_diag_jsonl(jsonl_path: str, ep: int, diag: dict) -> None:
    """Append one JSON record (with episode tag) to the jsonl file."""
    rec = {'ep': int(ep), 'diag': _to_native(diag)}
    os.makedirs(os.path.dirname(jsonl_path) or '.', exist_ok=True)
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec, separators=(',', ':')) + '\n')


def _to_native(obj):
    """Convert numpy scalars/arrays to plain Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj
