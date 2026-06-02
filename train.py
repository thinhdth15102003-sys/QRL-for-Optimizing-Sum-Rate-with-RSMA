"""
train.py
--------
Training script: Quantum Actor full pipeline for IRS-assisted RSMA.

Actor pipeline (4 stages per step)
------------------------------------
  1. Quantum IRS selection  : state s_t  →  φ ∈ {0,..,M}^K
  2. PhaseMLP               : per-IRS state (M, 2K)  →  phase_idx ∈ {0..L-1}^{G×N}  (G active IRS only)
  3. PowerMLP               : h_eff state (2K) [Re‖Im]  →  [w_c_vec (G+1), w_p (K)] × P_S
  4. CkMLP                  : [D_k, R_p_k, R_c_g_k] state (3·K) → C_k per user

All hyperparameters live in params.py — edit there, not here.
Results are saved to results/result_<N>/ after training completes.

Usage
-----
    python train.py                           # defaults from params.py
    python train.py --episodes 50 --steps 50  # quick test run
    python train.py --seed 42
    python train.py --no-plots                # skip matplotlib
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

import params as P
from params  import make_config
from CSI.env import ISTNEnv
from CSI.baselines import DirectOnlyPolicy, AllIRSPolicy
from RL        import QuantumActor, ClassicalCritic, PhaseMLP, PowerMLP, CkMLP
from RL.critic_diag import (compute_critic_diag, format_critic_diag_lines,
                            write_critic_diag_jsonl)
from RL.quantum_circuit import GPU_BACKEND


# ══════════════════════════════════════════════════════════════════════════════
# Logging helpers
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    """Mirrors all writes to both the terminal and a log file."""
    def __init__(self, fh):
        self._fh     = fh
        self._stdout = sys.stdout
    def write(self, data: str) -> None:
        self._stdout.write(data)
        self._fh.write(data)
    def log_only(self, data: str) -> None:
        """Write to the log file ONLY (not the terminal)."""
        self._fh.write(data)
    def flush(self) -> None:
        self._stdout.flush()
        self._fh.flush()
    def isatty(self) -> bool:
        return False


def _flog(msg: str = "") -> None:
    """Write a line ONLY to the training_log file (skips the terminal).
    Used for verbose diagnostics we want recorded but not cluttering the screen."""
    out = sys.stdout
    if hasattr(out, 'log_only'):
        out.log_only(msg + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline state helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_phase_state(channels: dict, phi: np.ndarray, cfg,
                       z_t: np.ndarray) -> np.ndarray:
    """
    Build per-IRS cascade channel states for PhaseMLP.

    c^SRU_{m,k} = conj(g_SR_hat[m]) · g_RU_hat[m,k]  if phi[k] == m+1, else 0.
    Uses ESTIMATED channels (g_hat) — same CSI the system uses for precoding.

    Appends z_t (system spatial latent) and a per-IRS binary user-mask to each row.
      phi_mask_m[k] = 1  iff user k is assigned to IRS m  — tells PhaseMLP which
      users are its "clients" without having to unmix the global z_t.

    Returns (M, 2K + n_latent + K) float:
      row m = [Re(c^SRU_m) (K), Im(c^SRU_m) (K), z_t (n_latent), phi_mask_m (K)]
      inactive IRS rows are all-zero in channel/mask cols; z_t is broadcast to all rows.
    """
    M, K     = cfg.M, cfg.K
    n_latent = len(z_t)
    s        = np.zeros((M, 2 * K + n_latent + K))
    for k in range(K):
        m = int(phi[k]) - 1          # 0-based IRS index; -1 for direct users
        if m >= 0:
            c_mk = channels['g_SR_hat'][m].conj() * channels['g_RU_hat'][m, k]
            s[m, k]                    += c_mk.real
            s[m, K + k]                += c_mk.imag
            s[m, 2 * K + n_latent + k]  = 1.0   # phi_mask: user k belongs to IRS m
    # Broadcast z_t into every row (same system spatial state for all IRS panels)
    s[:, 2 * K : 2 * K + n_latent] = z_t[None, :]
    return s


def _get_active_irs(phi: np.ndarray) -> np.ndarray:
    """
    Sorted 0-based indices of IRS panels with ≥1 assigned user.
    phi: (K,) int — 0=direct, 1..M=IRS (1-based).
    Used by PhaseMLP which expects 0-based indices.
    """
    return np.array(
        sorted({int(phi[k]) - 1 for k in range(len(phi)) if phi[k] > 0}),
        dtype=int,
    )


def _get_active_irs_ids(phi: np.ndarray) -> list:
    """
    Sorted 1-based physical IRS gids with ≥1 assigned user.
    Used by PowerMLP and RateComputer (which use 1-based gid convention).
    """
    return sorted(set(int(phi[k]) for k in range(len(phi)) if phi[k] > 0))


def _build_ck_state(demand: np.ndarray, R_private: np.ndarray,
                    R_c_group: dict, phi: np.ndarray, cfg) -> np.ndarray:
    """
    Build the [D_k, R_p_k, R_c_g_k, shortfall_k, phi_float_k] state for CkMLP.

    shortfall_k = max(0, D_k - R_p_k)  — minimum C_k still needed to hit QoS.
    phi_float_k = phi[k].astype(float) — group-ID context (0=direct, 1..M=IRS).

    Returns (5·K,) float.
    """
    K     = cfg.K
    R_c_g = np.zeros(K)
    for k in range(K):
        gid      = int(phi[k])
        R_c_g[k] = float(R_c_group.get(gid, 0.0))
    shortfall = np.maximum(0.0, demand - R_private)
    phi_float = phi[:K].astype(float)
    return np.concatenate([demand, R_private, R_c_g, shortfall, phi_float])


# ══════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_action_vec(phi: np.ndarray, phase_idx: np.ndarray,
                      w_c_vec: np.ndarray, active_irs_ids: list,
                      w_p: np.ndarray, C_k: np.ndarray,
                      cfg, n_levels: int) -> np.ndarray:
    """
    Fixed-size (d_action,) float encoding of the joint action for the
    action-conditioned critic Q(s, a).

    Encoding (each sub-vector normalised to ~[0, 1]):
      phi_f      : (K,)     assignment index / M
      phase_f    : (M*N,)   phase level     / (n_levels - 1)
      w_c_padded : (M+1,)   common power per group padded to full size / P_S
      w_p_n      : (K,)     private power / P_S
      C_k        : (K,)     common-rate share (bps/Hz)

    d_action = K + M*N + (M+1) + K + K
    """
    phi_f   = phi.astype(float) / max(cfg.M, 1)
    phase_f = phase_idx.flatten().astype(float) / max(n_levels - 1, 1)

    # Expand (G+1,) w_c_vec back to fixed (M+1,) — slot 0 = direct, slot gid = IRS gid
    w_c_pad = np.zeros(cfg.M + 1)
    w_c_pad[0] = float(w_c_vec[0])
    for j, gid in enumerate(active_irs_ids):
        if 1 <= gid <= cfg.M:
            w_c_pad[gid] = float(w_c_vec[j + 1])
    w_c_pad /= cfg.P_S

    w_p_n = w_p / cfg.P_S
    return np.concatenate([phi_f, phase_f, w_c_pad, w_p_n, C_k])


def _compute_blocked(env: ISTNEnv) -> np.ndarray:
    """
    Per-user indicator: True when the best IRS path (via estimated CSI)
    gives a stronger effective channel than the direct satellite link.
    """
    ch       = env.channels
    h_direct = np.abs(ch['g_SU_hat'])
    h_irs    = np.array([
        ch['beta'][m] * np.abs(np.sum(
            ch['g_SR_hat'][m].conj() * np.diag(env.Phi[m])[:, np.newaxis] * ch['g_RU_hat'][m],
            axis=0))
        for m in range(env.cfg.M)
    ])
    return h_irs.max(axis=0) > h_direct


def _save_agents(run_dir: str, actor, phase_net, power_net, ck_net, cfg,
                 critic=None) -> str:
    """Save all four actor networks (+ critic) + topology snapshot to run_dir/agents/."""
    agents_dir = os.path.join(run_dir, 'agents')
    os.makedirs(agents_dir, exist_ok=True)
    actor.save(agents_dir)
    phase_net.save(agents_dir)
    power_net.save(agents_dir)
    ck_net.save(agents_dir)
    # Persist the critic too so a --resume run can warm-start V(s) instead of
    # cold-starting it (a blind critic feeds noisy advantages to a good actor).
    if critic is not None:
        critic.save(agents_dir)
    # Topology snapshot so infer.py can rebuild a consistent cfg regardless of
    # what the user has set in params.py at inference time.
    topo = {
        'K':                cfg.K,
        'M':                cfg.M,
        'N':                cfg.N,
        'quantization_bits': cfg.quantization_bits,
        'P_S_dBm':          cfg.P_S_dBm,
        'D_k_bps_hz':       cfg.D_k_bps_hz,
    }
    with open(os.path.join(agents_dir, 'training_config.json'), 'w') as f:
        json.dump(topo, f, indent=2)
    return agents_dir


def _accum_grads(acc: dict, new: dict) -> dict:
    """In-place add a grad-dict into the accumulator (or initialise it).
    Handles empty dicts gracefully (e.g. when no IRS is active)."""
    if not new:                   # new is None or {} — nothing to add
        return acc
    if not acc:                   # acc is None or {} — start fresh
        return {k: v.copy() for k, v in new.items()}
    for k, v in new.items():
        acc[k] += v
    return acc


def _scale_grads(g: dict, factor: float) -> None:
    """In-place scale every entry in a grad-dict."""
    if g is None:
        return
    for k in g:
        g[k] *= factor


def _ppo_eff_adv(log_curr: float, log_old: float,
                 advantage: float, epsilon: float) -> tuple:
    """
    PPO-clip effective advantage for one sample.

    r_t = π_current(a|s) / π_old(a|s) = exp(log_curr − log_old)

    Returns
    -------
    eff_adv   : float  pass as `advantage` to compute_grads
                       = r_t × Â  (gradient flows) or 0 (clipped)
    ratio     : float  r_t (for monitoring)
    ppo_loss  : float  −min(r_t·Â, clip(r_t)·Â)  (for logging)
    """
    ratio = float(np.exp(np.clip(log_curr - log_old, -10.0, 10.0)))
    surr1 = ratio * advantage
    surr2 = float(np.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)) * advantage
    if surr1 <= surr2:          # surr1 is the min → gradient: r_t × ∇log π
        return ratio * advantage, ratio, float(-surr1)
    else:                       # surr2 is min (clip active) → gradient zeroed
        return 0.0, ratio, float(-surr2)


def _ppo_eff_adv_batch(lp_new_b: np.ndarray, lp_old_b: np.ndarray,
                       adv_b: np.ndarray, epsilon: float) -> tuple:
    """
    Vectorised PPO-clip effective advantage for a mini-batch.

    Mirrors the scalar _ppo_eff_adv logic element-wise:
      eff_adv[b] = ratio[b] × adv[b]  when surr1 ≤ surr2  (not clipped)
                 = 0                   otherwise            (clipped)

    Returns
    -------
    eff_adv_b  : (B,) float  pass to compute_grads_batch as effective advantage
    ppo_loss_b : (B,) float  −min(surr1, surr2) per sample (for logging)
    clip_frac  : float       fraction of samples whose gradient was zeroed (clipped)
    """
    ratio_b = np.exp(np.clip(lp_new_b - lp_old_b, -10.0, 10.0))
    surr1_b = ratio_b * adv_b
    surr2_b = np.clip(ratio_b, 1.0 - epsilon, 1.0 + epsilon) * adv_b
    eff_adv_b  = np.where(surr1_b <= surr2_b, surr1_b, 0.0)
    ppo_loss_b = -np.minimum(surr1_b, surr2_b)
    clip_frac  = float(np.mean(surr1_b > surr2_b))
    return eff_adv_b, ppo_loss_b, clip_frac


def _clip_grad_norm(g: dict, max_norm: float) -> float:
    """In-place global gradient norm clipping. Returns the PRE-clip global norm
    (a key divergence diagnostic — a spike here precedes blow-ups)."""
    if not g:
        return 0.0
    norm = np.sqrt(sum(float(np.sum(v ** 2)) for v in g.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        for k in g:
            g[k] *= scale
    return float(norm)


def _explained_variance(returns: np.ndarray, values: np.ndarray) -> float:
    """1 − Var(returns − V) / Var(returns).  ≈1 critic predicts well, ≤0 = useless."""
    var_y = float(np.var(returns))
    if var_y < 1e-12:
        return float('nan')
    return 1.0 - float(np.var(returns - values)) / var_y


def _warmup_critic(env, actor, critic, phase_net, power_net, ck_net,
                   cfg, args, demand: np.ndarray,
                   n_episodes: int, n_epochs: int) -> dict:
    """
    Calibrate V(s) to the (resumed) policy BEFORE PPO begins.

    A freshly-built critic is blind (explVar≈0) for the first hundreds of
    episodes; with a small GAE-λ its noisy 1-step advantages then erode a good
    warm-started actor. Here we roll the *current* policy for `n_episodes`,
    compute Monte-Carlo discounted returns (no bootstrap → unbiased value
    scale, matched to the current env/reward params), and fit the critic by
    supervised regression. The actors are NOT updated. Finally the target net
    is synced to the fitted weights so TD targets are calibrated from step one.
    """
    buf_s: list  = []
    buf_ret: list = []
    rng = np.random.default_rng(args.seed + 31337)

    for _ep in range(n_episodes):
        obs  = env.reset()
        ep_s: list = []
        ep_r: list = []
        for _step in range(args.steps):
            blocked = _compute_blocked(env)
            s_t     = actor.extract_state(obs, demand, blocked)
            phi, _, actor_info = actor.forward(s_t)
            z_t = actor_info['z_t']
            active_irs     = _get_active_irs(phi)
            active_irs_ids = _get_active_irs_ids(phi)
            s_phase  = _build_phase_state(env.channels, phi, cfg, z_t)
            phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
            phases_rad   = env.phase_model.index_to_phase(phase_idx)
            proposed_Phi = env.phase_model.build_phi(phases_rad)
            h_eff   = env.rate_computer.effective_channels_all(
                          phi, proposed_Phi, env.channels)
            s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
            w_c_vec, w_p, _, _ = power_net.forward(s_power, active_irs_ids)
            partial = env.rate_computer.compute_rates_partial(
                phi, proposed_Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=active_irs_ids)
            s_ck = _build_ck_state(
                demand, partial['R_private'], partial['R_c_group'], phi, cfg)
            C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
            action = {'assignment': phi, 'phase_idx': phase_idx,
                      'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k}
            obs, reward, _, _ = env.step(action)
            ep_s.append(s_t.copy())
            ep_r.append(float(reward))
        # Truncated Monte-Carlo return (γ^steps ≈ 0 → ≈ infinite-horizon).
        G = 0.0
        ep_ret = [0.0] * len(ep_r)
        for t in reversed(range(len(ep_r))):
            G = ep_r[t] + args.gamma * G
            ep_ret[t] = G
        buf_s.extend(ep_s)
        buf_ret.extend(ep_ret)

    rets = np.asarray(buf_ret, dtype=float)
    n    = len(buf_s)
    bs   = P.batch_size
    idx  = np.arange(n)
    # PopArt: initialise (μ,σ) from the warm-up returns so the normalised head
    # starts calibrated (no-op when popart disabled).
    critic.update_popart_stats(rets)
    for _epoch in range(n_epochs):
        rng.shuffle(idx)
        for start in range(0, n, bs):
            mb = idx[start:start + bs]
            _, g = critic.compute_grads_batch(
                [buf_s[i] for i in mb], [None] * len(mb), rets[mb])
            critic.apply_grads(g)

    V_pred = np.array([critic.forward(s) for s in buf_s])
    return {
        'n_samples': n,
        'ret_mean':  float(rets.mean()),
        'ret_std':   float(rets.std()),
        'v_mean':    float(V_pred.mean()),
        'expl_var':  _explained_variance(rets, V_pred),
    }


def _divergence_signals(d: dict, gnorm_ema: dict) -> list:
    """Triggered divergence signals. Gradients are clipped before apply, so a
    large pre-clip norm is normal — we flag a sudden SPIKE vs its EMA, not the
    absolute value. KL and NaN/Inf are scale-independent red flags."""
    sig = []
    for k, v in d.get('scalars', {}).items():            # numerical blow-up
        if not np.isfinite(v):
            sig.append(f"{k}=NaN/Inf")
    for net, gn in d.get('gnorm', {}).items():           # sudden grad-norm spike
        ema = gnorm_ema.get(net)
        if ema is not None and ema > 1e-6 and np.isfinite(gn) and gn > 8.0 * ema and gn > 1.0:
            sig.append(f"‖∇{net}‖={gn:.1f} (8×↑ vs {ema:.1f})")
    for act, kl in d.get('kl', {}).items():              # policy moving too fast
        if np.isfinite(kl) and kl > 0.5:
            sig.append(f"KL_{act}={kl:.2f}>0.5")
    return sig


def _print_diag_panel(ep: int, d: dict, signals: list) -> None:
    """Compact multi-line health panel, printed each PPO update."""
    gn, kl, ent = d['gnorm'], d['kl'], d['ent']
    print(f"  ┄ diag[{ep+1}]  ‖∇‖ ae={gn['ae']:.2f} qc={gn['qc']:.2f} xi={gn['xi']:.2f} "
          f"ph={gn['phase']:.2f} pw={gn['power']:.2f} ck={gn['ck']:.2f} V={gn['critic']:.2f}")
    print(f"            KL q={kl['q']:.3f} ph={kl['ph']:.3f} pw={kl['pw']:.3f} ck={kl['ck']:.3f}"
          f"  │ ent q={ent['q']:.3f} ph={ent['ph']:.3f} pw={ent['pw']:.3f} ck={ent['ck']:.3f}")
    print(f"            V̄={d['v_mean']:+.2f} σV={d['v_std']:.2f} explVar={d['expl_var']:+.2f}"
          f"  │ λ|max| y={d['lam_y']:.2f} z={d['lam_z']:.2f}"
          f"  │ R̄={d['sum_rate']:.3f} − qp {d['qp']:.3f}")
    if signals:
        print(f"  ⚠ DIVERGENCE SIGNAL[{ep+1}]: " + " ; ".join(signals))


def _env_feasibility_report(cfg, seed: int, n_ep: int = 15, n_steps: int = 12) -> dict:
    """
    Probe the CURRENT environment with fixed reference policies to separate
    'geometry difficulty' from 'agent skill'. Reports what fraction of users
    CAN meet the QoS demand D_k under Direct-only / All-IRS / best-of-both.
    Use this BEFORE a run to calibrate λ_D vs R_LoS:
      servable ≫ agent-QoS  → agent's fault → raise λ_D
      servable low          → geometry too hard → lower R_LoS or D_k, don't crank λ_D
    """
    probe = ISTNEnv(cfg=cfg, seed=seed + 12345, n_steps_ep=n_steps,
                    reward_noise_avg=1)
    pol_direct, pol_irs = DirectOnlyPolicy(cfg), AllIRSPolicy(cfg)
    qos_direct, qos_irs, qos_best = [], [], []
    rt_direct,  rt_irs  = [], []
    blocked = []
    Dk = cfg.D_k_bps_hz
    for _ in range(n_ep):
        obs = probe.reset()
        for _ in range(n_steps):
            # per-user rate under each reference link choice (current Phi)
            info_d = probe.rate_computer.compute_sum_rate(**_baseline_kwargs(pol_direct.act(obs), probe))
            info_i = probe.rate_computer.compute_sum_rate(**_baseline_kwargs(pol_irs.act(obs), probe))
            Rd = info_d['R_private'] + info_d['C_k']
            Ri = info_i['R_private'] + info_i['C_k']
            Rbest = np.maximum(Rd, Ri)
            qos_direct.append(np.mean(Rd >= Dk)); qos_irs.append(np.mean(Ri >= Dk))
            qos_best.append(np.mean(Rbest >= Dk))
            rt_direct.append(float(np.sum(Rd))); rt_irs.append(float(np.sum(Ri)))
            blocked.append(float(np.mean(probe.channels['su_blocked'])))
            obs, _, _, _ = probe.step(pol_irs.act(obs))
    rep = {
        'servable_frac':  float(np.mean(qos_best)),
        'qos_direct':     float(np.mean(qos_direct)),
        'qos_irs':        float(np.mean(qos_irs)),
        'rtot_direct':    float(np.mean(rt_direct)),
        'rtot_irs':       float(np.mean(rt_irs)),
        'blocked_frac':   float(np.mean(blocked)),
    }
    print(f"\n  Environment feasibility probe  (R_LoS={cfg.R_LoS_km} km, D_k={Dk}, λ_D={cfg.lambda_D})")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"    Servable users (best link ≥ D_k) : {rep['servable_frac']*100:5.1f}%   "
          f"← QoS ceiling for ANY policy")
    print(f"    QoS under Direct-only            : {rep['qos_direct']*100:5.1f}%   "
          f"(Σ R_tot {rep['rtot_direct']:.2f})")
    print(f"    QoS under All-IRS                : {rep['qos_irs']*100:5.1f}%   "
          f"(Σ R_tot {rep['rtot_irs']:.2f})")
    print(f"    Direct link blocked              : {rep['blocked_frac']*100:5.1f}% of users")
    if rep['servable_frac'] < 0.6:
        print(f"    ⚠ Only {rep['servable_frac']*100:.0f}% servable → λ_D high will punish "
              f"UNAVOIDABLE failures. Lower R_LoS / D_k before raising λ_D.")
    else:
        print(f"    ✓ {rep['servable_frac']*100:.0f}% servable → if agent QoS is below this, "
              f"raising λ_D is safe & should help.")
    return rep


def _baseline_kwargs(action: dict, env) -> dict:
    """Map a baseline action dict to compute_sum_rate kwargs (uses current Phi)."""
    assignment = action['assignment']
    active = sorted(set(int(a) for a in assignment if a > 0))
    return dict(assignment=assignment, Phi=env.Phi, channels=env.channels,
                w_p=action['w_p'], w_c_vec=action['w_c_vec'],
                active_irs_ids=active, sigma2=env.cfg.sigma2)


def _analysis_summary(hist: dict, n_diverge: int, args) -> None:
    """End-of-run analysis + heuristic recommendations for the next run."""
    W = 110
    rew = hist['episode_reward']; n = len(rew)
    if n < 10:
        return
    w = max(10, n // 20)                       # tail window ≈ last 5%
    tail   = lambda k: float(np.mean(hist[k][-w:])) if hist.get(k) else float('nan')
    best   = float(np.max(rew))
    best_ep = int(np.argmax(rew)) + 1
    # plateau check: did best improve in the last 20% of episodes?
    cut = max(1, int(0.8 * n))
    plateaued = (np.max(rew[cut:]) <= np.max(rew[:cut]) + 1e-6) if cut < n else False
    ev   = tail('mean_expl_var')   if 'mean_expl_var'   in hist else float('nan')
    clipq= tail('mean_clip_q')     if 'mean_clip_q'     in hist else float('nan')
    qos  = tail('mean_qos_rate')
    qp   = tail('mean_qp_penalty') if 'mean_qp_penalty' in hist else float('nan')
    sr   = tail('mean_sum_rate')

    print(f"\n  {'═'*W}")
    print(f"  ANALYSIS SUMMARY  (tail window = last {w} ep)")
    print(f"  {'─'*W}")
    print(f"    reward  tail={tail('episode_reward'):8.1f}   best={best:8.1f} @ep {best_ep}"
          f"   {'⚠ PLATEAUED (no new best in last 20%)' if plateaued else '↗ still improving'}")
    print(f"    rates   ΣR_tot={sr:6.3f}   QoS={qos*100:4.1f}%   qp_penalty={qp:6.3f}")
    print(f"    health  explVar={ev:+.2f}   clip_q={clipq:.2f}   divergence-signals={n_diverge}")
    print(f"  {'─'*W}")
    print(f"  Recommendations for next run:")
    recs = []
    if not np.isnan(qos) and qos < 0.6 and not np.isnan(qp) and qp > 1.0:
        recs.append(f"QoS low ({qos*100:.0f}%) + penalty high → check feasibility probe; "
                    f"if servable≫QoS raise λ_D, else lower R_LoS/D_k")
    if not np.isnan(ev) and ev < 0.2:
        recs.append(f"explVar low ({ev:+.2f}) → critic struggling: ↑lr_critic or ↓reward variance "
                    f"(↑reward_noise_avg)")
    if not np.isnan(clipq) and clipq > 0.3:
        recs.append(f"clip_q high ({clipq:.2f}) → policy moving fast: ↓ppo_epochs or ↑reward_noise_avg")
    if n_diverge > n // 50:
        recs.append(f"{n_diverge} divergence signals → consider ↓lr or tighter grad-clip")
    if plateaued:
        recs.append("plateaued → raise difficulty (R_LoS) via --resume, or stop")
    if not recs:
        recs.append("no red flags — safe to raise difficulty (R_LoS↑ / case↑) via --resume")
    for r in recs:
        print(f"    • {r}")
    print(f"  {'═'*W}\n")


def _compute_beta(ep: int, n_episodes: int) -> float:
    """Linear entropy annealing: β₀ → β_min by beta_entropy_anneal_end of training."""
    end_ep = max(1, int(P.beta_entropy_anneal_end * n_episodes))
    prog   = min(1.0, ep / end_ep)
    return max(P.beta_entropy_min, P.beta_entropy * (1.0 - prog))


def _compute_lr_frac(ep: int, n_episodes: int) -> float:
    """Linear LR decay after lr_decay_start warm-up: 1.0 → lr_min_frac."""
    start_ep = int(P.lr_decay_start * n_episodes)
    if ep < start_ep:
        return 1.0
    prog = (ep - start_ep) / max(1, n_episodes - start_ep)
    return max(P.lr_min_frac, 1.0 - prog)


def _next_run_dir(base: str = "results") -> tuple:
    os.makedirs(base, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(base, f"result_{i}")):
        i += 1
    path = os.path.join(base, f"result_{i}")
    os.makedirs(path)
    return path, i


def _moving_avg(arr: list, w: int) -> np.ndarray:
    a = np.array(arr, dtype=float)
    if len(a) < w:
        return a
    return np.convolve(a, np.ones(w) / w, mode='valid')


# ══════════════════════════════════════════════════════════════════════════════
# Results persistence
# ══════════════════════════════════════════════════════════════════════════════

def _save_hyperparameters(run_dir: str, cfg, actor: QuantumActor,
                           critic: ClassicalCritic,
                           phase_net: PhaseMLP, power_net: PowerMLP,
                           ck_net: CkMLP,
                           args, run_id: int) -> None:
    hp = {
        "run_id":    run_id,
        "timestamp": datetime.now().isoformat(timespec='seconds'),
        "system": {
            "K":              cfg.K,
            "M":              cfg.M,
            "N":              cfg.N,
            "P_S_dBm":        cfg.P_S_dBm,
            "f_GHz":          cfg.f_GHz,
            "h_SR_km":        cfg.h_SR_km,
            "G_S_dBi":        cfg.G_S_dBi,
            "G_U_dBi":        cfg.G_U_dBi,
            "noise_mean_dBW": cfg.noise_mean_dBW,
            "noise_var_dBW":  cfg.noise_var_dBW,
            "kappa":          cfg.kappa,
            "R_LoS_km":       cfg.R_LoS_km,
            "irs_spawn_radius_frac": getattr(cfg, 'irs_spawn_radius_frac', 1.0),
            "user_free_radius_frac": getattr(cfg, 'user_free_radius_frac', 1.0),
            "h_IRS_km":       cfg.h_IRS_km,
            "beta_IRS":       cfg.beta_IRS,
            "beta_blocking":  cfg.beta_blocking,
            "d_block_km":     cfg.d_block_km,
            "path_loss_exp":  cfg.path_loss_exp,
            "user_speed_mps": cfg.user_speed_mps,
            "D_k_bps_hz":     cfg.D_k_bps_hz,
            "lambda_D":       cfg.lambda_D,
            "epsilon_qp":     cfg.epsilon_qp,
        },
        "training": {
            "n_episodes":      args.episodes,
            "n_steps_per_ep":  args.steps,
            "gamma":           args.gamma,
            "ae_weight":                P.ae_weight,
            "beta_entropy":             P.beta_entropy,
            "beta_entropy_min":         P.beta_entropy_min,
            "beta_entropy_anneal_end":  P.beta_entropy_anneal_end,
            "lr_decay_start":           P.lr_decay_start,
            "lr_min_frac":              P.lr_min_frac,
            "batch_size":               P.batch_size,
            "ppo_epsilon":   P.ppo_epsilon,
            "ppo_epochs":    P.ppo_epochs,
            "lr_actor_ae":   P.lr_actor_ae,
            "lr_actor_qc":     P.lr_actor_qc,
            "lr_actor_xi":     P.lr_actor_xi,
            "lr_critic":       P.lr_critic,
            "n_shots_train":      P.n_shots_train,
            "n_shots_eval":       P.n_shots_eval,
            "eval_interval":      P.eval_interval,
            "gae_lambda":         P.gae_lambda,
            "reward_noise_avg":   getattr(P, 'reward_noise_avg', 1),
            "ae_pretrain_epochs": P.ae_pretrain_epochs,
            "ae_pretrain_lr":     P.ae_pretrain_lr,
            "seed":               args.seed,
        },
        "actor": {
            "d_s":           actor.d_s,
            "n_latent":      actor.N_LATENT,
            "n_qubits":      actor.N_QUBITS,
            "n_var_layers":  actor.N_VAR_LAYERS,
            "n_shots":       actor.n_shots,
            "hidden_ae":     actor.N_HIDDEN_AE,
            "hidden_post":   actor.N_HIDDEN_POST,
            "n_choices":        actor.n_choices,
            "data_reuploading": actor.DATA_REUPLOADING,
            "architecture":  (f"{actor.d_s} → LN → "
                              f"{'→'.join(str(h) for h in actor.N_HIDDEN_AE)} → "
                              f"{actor.N_LATENT}(z) | QC({actor.N_QUBITS}q,L="
                              f"{actor.N_VAR_LAYERS}) → {actor.N_QUANTUM}(o) → "
                              f"{actor.N_LATENT + actor.N_QUANTUM} → "
                              f"{'→'.join(str(h) for h in actor.N_HIDDEN_POST)} → "
                              f"{cfg.K}×{actor.n_choices}"),
        },
        "critic": {
            "d_state":      critic.d_state,
            "d_action":     critic.d_action,
            "d_in":         critic.d_in,
            "hidden":       critic.hidden,
            "gamma":        critic.gamma,
            "architecture": critic.architecture_str,
            "lr":           P.lr_critic,
            "grad_clip":    P.grad_clip_critic,
            "popart":       getattr(critic, 'popart', False),
            "popart_beta":  getattr(critic, 'pa_beta', None),
        },
        "phase_net": {
            "d_s":      2 * cfg.K,
            "M":        cfg.M,
            "N":        cfg.N,
            "n_levels": phase_net.n_levels,
            "hidden":   P.n_hidden_phase,
            "lr":       P.lr_phase,
        },
        "power_net": {
            "d_s":    cfg.K,
            "K":      cfg.K,
            "M":      cfg.M,
            "P_S":    cfg.P_S,
            "hidden": P.n_hidden_power,
            "lr":     P.lr_power,
        },
        "ck_net": {
            "d_s":    3 * cfg.K,
            "K":      cfg.K,
            "hidden": P.n_hidden_ck,
            "lr":     P.lr_ck,
        },
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), 'w') as f:
        json.dump(hp, f, indent=2)


def _save_plots(run_dir: str, hist: dict, args, window: int = 10) -> list:
    if not HAVE_MPL or args.no_plots:
        return []

    saved = []
    eps   = np.arange(1, len(hist['episode_reward']) + 1)
    w     = min(window, max(1, len(eps) // 5))

    C = dict(reward='steelblue', lpg='tomato', lae='darkorange',
             td='mediumseagreen', feas='mediumpurple',
             ohm='royalblue', znm='darkcyan')

    def _plot_single(ax, key, label, color, pct=False):
        vals = [v * 100 for v in hist[key]] if pct else hist[key]
        ax.plot(eps, vals, alpha=0.30, color=color, lw=1)
        if len(eps) >= w:
            ma = _moving_avg(vals, w)
            ax.plot(np.arange(w, len(eps) + 1), ma,
                    color=color, lw=2.0, label=f'{w}-ep avg')
        ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.legend(fontsize=8)

    # 1. Reward curve
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_single(ax, 'episode_reward', 'Total episode reward', C['reward'])
    ax.set_title('Episode Reward', fontsize=11)
    fig.tight_layout()
    p = os.path.join(run_dir, 'reward_curve.png')
    fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    # 2. Losses (2 panels: combined actor + critic TD)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_L_actor', 'L_actor  (combined actor loss)', C['lpg'])
    _plot_single(axes[1], 'mean_td_loss', 'TD loss  (critic)',               C['td'])
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Training Losses', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'losses.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 3. Feasibility
    fig, ax = plt.subplots(figsize=(8, 4))
    _plot_single(ax, 'feasibility_rate', 'Feasible steps (%)', C['feas'], pct=True)
    ax.set_ylim(-2, 105)
    ax.axhline(100, color='gray', ls='--', lw=0.8, label='100 %')
    ax.set_title('Feasibility Rate', fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(run_dir, 'feasibility.png')
    fig.savefig(p, dpi=130); plt.close(fig); saved.append(p)

    # 4. Quantum features (2 panels)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_o_hat_norm', '‖o_hat‖  (quantum output)', C['ohm'])
    _plot_single(axes[1], 'mean_z_norm',     '‖z_t‖  (latent vector)',    C['znm'])
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Quantum Feature Statistics', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'quantum_stats.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 5. Sub-actor losses (3 panels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_single(axes[0], 'mean_L_phase', 'L_phase (phase-shift MLP)',   'saddlebrown')
    _plot_single(axes[1], 'mean_L_power', 'L_power (power alloc. MLP)',  'steelblue')
    _plot_single(axes[2], 'mean_L_ck',    'L_ck    (C_k split MLP)',     'darkgreen')
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    fig.suptitle('Sub-actor Losses', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'sub_actor_losses.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 6. Rates (sum-rate & QoS fraction)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(axes[0], 'mean_sum_rate', 'Σ R_tot  (bps/Hz)',            'teal')
    _plot_single(axes[1], 'mean_qos_rate', 'QoS fraction  (users / K)',   'darkorchid')
    for ax in axes:
        ax.set_title(ax.get_ylabel(), fontsize=10)
    axes[1].set_ylim(-0.05, 1.05)
    fig.suptitle('Rate & QoS Statistics', fontsize=12, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'rates.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    # 7. Summary dashboard (2×2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    _plot_single(axes[0, 0], 'episode_reward',   'Total reward',            C['reward'])
    _plot_single(axes[0, 1], 'feasibility_rate', 'Feasible steps (%)',      C['feas'], pct=True)
    _plot_single(axes[1, 0], 'mean_L_actor',     'L_actor  (combined)',     C['lpg'])
    _plot_single(axes[1, 1], 'mean_td_loss',     'TD loss  (critic)',        C['td'])
    fig.suptitle(f'Training Summary  —  result_{run_dir.split("_")[-1]}',
                 fontsize=13, y=1.01)
    fig.tight_layout()
    p = os.path.join(run_dir, 'summary.png')
    fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); saved.append(p)

    return saved


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def _build_components(args):
    """
    Build all environment and agent objects without printing anything.
    Called before the log file is opened so that hyperparameters.json
    can be written first.
    """
    cfg = make_config()
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps,
                  reward_noise_avg=getattr(P, 'reward_noise_avg', 1))

    actor = QuantumActor(
        cfg,
        n_qubits         = P.n_qubits,
        n_latent         = P.n_latent,
        n_hidden_ae      = P.n_hidden_ae,
        n_hidden_post    = P.n_hidden_post,
        n_var_layers     = P.n_var_layers,
        n_shots          = P.n_shots_train,
        lr_ae            = P.lr_actor_ae,
        lr_qc            = P.lr_actor_qc,
        lr_xi            = P.lr_actor_xi,
        data_reuploading = P.data_reuploading,
        ae_pretrain_lr   = P.ae_pretrain_lr,
        spsa_n_reps      = P.spsa_n_reps,
        spsa_epsilon     = P.spsa_epsilon,
        extra_cz_pairs   = P.extra_cz_pairs,
        extra_zz_pairs   = P.extra_zz_pairs,
        full_zz_pairs    = getattr(P, 'full_zz_pairs', ()),
        seed             = args.seed,
    )
    # State-value critic V(s) — action-independent baseline (d_action=0).
    # A Q(s,a) baseline cancels the action's own value, giving E[advantage]≈0
    # and a vanishing policy gradient; V(s) is the correct PPO/GAE baseline.
    critic = ClassicalCritic(
        actor.d_s,
        d_action = 0,
        hidden   = P.critic_hidden,
        lr       = P.lr_critic,
        gamma    = args.gamma,
        seed     = args.seed,
        popart       = getattr(P, 'popart_enabled', False),
        popart_beta  = getattr(P, 'popart_beta', 0.1),
        popart_sigma_floor = getattr(P, 'popart_sigma_floor', 1e-2),
    )
    phase_net = PhaseMLP(
        d_s      = 3 * cfg.K + P.n_latent,   # c^SRU (2K) + z_t (n_latent) + phi_mask (K)
        M        = cfg.M,
        N        = cfg.N,
        n_levels = env.n_phase_levels,
        hidden   = P.n_hidden_phase,
        lr       = P.lr_phase,
        seed     = args.seed,
    )
    power_net = PowerMLP(
        d_s    = 2 * cfg.K + P.n_latent,     # h_eff Re+Im (2K) + z_t (n_latent)
        K      = cfg.K,
        M      = cfg.M,
        P_S    = cfg.P_S,
        hidden = P.n_hidden_power,
        lr     = P.lr_power,
        seed   = args.seed,
    )
    ck_net = CkMLP(
        d_s    = 5 * cfg.K,                  # D_k + R_p + R_c + shortfall + phi_float
        K      = cfg.K,
        hidden = P.n_hidden_ck,
        lr     = P.lr_ck,
        seed   = args.seed,
    )

    # ── Resume / curriculum transfer ──────────────────────────────────────────
    # Load the 4 trained actors (same architecture → same K/M/nq) to continue
    # training under new env/reward params (e.g. higher λ_D, larger R_LoS).
    # The critic is ALSO warm-started when a saved critic exists in the resume
    # dir: a freshly-initialised V(s) is blind for the first hundreds of
    # episodes (explVar≈0), and with gae_lambda small its noisy advantages
    # erode the good warm-started actor. Loading V(s) gives calibrated targets
    # from step one. (Minor return-scale shifts from changing λ_D / R_LoS are
    # re-fit far faster than a cold start.) If no critic file is present
    # (e.g. runs saved before critic persistence) we fall back to a fresh one.
    args._resumed_critic = False
    resume_dir = getattr(args, 'resume', None)
    if resume_dir:
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"--resume dir not found: {resume_dir}")
        actor     = QuantumActor.from_dir(resume_dir, seed=args.seed)
        phase_net = PhaseMLP.from_dir(resume_dir, seed=args.seed)
        power_net = PowerMLP.from_dir(resume_dir, seed=args.seed)
        ck_net    = CkMLP.from_dir(resume_dir, seed=args.seed)
        # Keep training-time shot count / spsa settings from params.py
        actor.n_shots     = P.n_shots_train
        actor.spsa_n_reps = P.spsa_n_reps
        # Warm-start the critic if it was persisted alongside the actors.
        if os.path.isfile(os.path.join(resume_dir, 'critic_config.json')):
            critic = ClassicalCritic.from_dir(resume_dir, seed=args.seed)
            args._resumed_critic = True
            # [Case2 PopArt bug-fix] On a curriculum-ramp resume (λ_D and R_LoS
            # change → return SCALE shifts), the loaded PopArt stats (μ,σ) are
            # STALE for the new ramp. With pa_initialized=True the warm-up only
            # EMA-adapts them (β=0.1 → ~10 rollouts of mis-scaled critic → noisy
            # advantages right when the ramp transition is most fragile). Force a
            # re-snap: reset pa_initialized so the critic warm-up (which rolls the
            # NEW policy in the NEW env) re-initialises μ,σ to the new return scale
            # and re-fits the normalised head before any actor update.
            if getattr(critic, 'popart', False):
                critic.pa_initialized = False

    return cfg, env, actor, critic, phase_net, power_net, ck_net


def train(args) -> None:
    # Ensure the console accepts UTF-8 box-drawing characters on Windows.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    run_dir, run_id = _next_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    # 1. Build all components silently (no prints yet).
    cfg, env, actor, critic, phase_net, power_net, ck_net = \
        _build_components(args)

    # 2. hyperparameters.json — written first, before the log is opened.
    _save_hyperparameters(run_dir, cfg, actor, critic,
                          phase_net, power_net, ck_net, args, run_id)

    # 3. training_log.txt — opened second; captures all subsequent stdout.
    log_path   = os.path.join(run_dir, 'training_log.txt')
    log_f      = open(log_path, 'w', encoding='utf-8', buffering=1)
    tee        = _Tee(log_f)
    sys.stdout = tee
    try:
        _train_body(args, run_dir, run_id,
                    cfg, env, actor, critic, phase_net, power_net, ck_net)
    finally:
        sys.stdout = tee._stdout
        log_f.close()


def _train_body(args, run_dir: str, run_id: int,
                cfg, env, actor, critic,
                phase_net, power_net, ck_net) -> None:
    # ── Derived values ────────────────────────────────────────────────────────
    demand = np.full(cfg.K, cfg.D_k_bps_hz)   # (K,) — demand per user

    # ── Print environment summary ─────────────────────────────────────────────
    W = 130
    _gpu_str = f"GPU (CuPy)" if GPU_BACKEND else "CPU (NumPy)"
    print(f"\n{'═'*W}")
    print(f"  Quantum AC Training  ·  IRS-assisted RSMA Satellite Comm.  ·  QC backend: {_gpu_str}")
    print(f"{'═'*W}")
    print(f"\n{cfg.summary()}\n")

    d_s = actor.d_s
    n_phase_params = cfg.M * cfg.N * env.n_phase_levels

    print(f"  Actor Pipeline  (4 stages per step)")
    print(f"  {'─'*64}")
    print(f"    1. Quantum IRS selection")
    print(f"       State  : a_t ∈ R^{d_s}  =  {cfg.K}×({cfg.M}+2)+2×{cfg.M}  (affinity ‖ D_k ‖ |g_SU|)")
    _dec_arch = ' → '.join(str(h) for h in reversed(actor.N_HIDDEN_AE))
    print(f"       Encoder: {d_s} → B2-DualAE → {actor.N_LATENT}  (z_t)"
          f"  [IRS:{2*cfg.M}D + User:2×{cfg.K}D→P]")
    print(f"       Decoder: {actor.N_LATENT} → {_dec_arch} → {d_s}  (AE regulariser)")
    _cz_desc = (f"+{len(actor.EXTRA_CZ_PAIRS)}×CZ-bridge" if actor.EXTRA_CZ_PAIRS else "")
    if actor.FULL_ZZ_PAIRS:
        _zz_desc = f"+{len(actor.FULL_ZZ_PAIRS)}×ZZ-full(B1)"
    elif actor.EXTRA_ZZ_PAIRS:
        _zz_desc = f"+{len(actor.EXTRA_ZZ_PAIRS)}×ZZ-cross(B4)"
    else:
        _zz_desc = ""
    print(f"       QC     : {actor.N_LATENT}-dim → {actor.N_QUBITS} qubits "
          f"(H+U_enc+U_var×{actor.N_VAR_LAYERS}{_cz_desc}) → {actor.N_QUANTUM}-dim{_zz_desc}  (o_hat)")
    _post_arch = ' → '.join(str(h) for h in actor.N_HIDDEN_POST)
    print(f"       Post-NN: {actor.N_LATENT+actor.N_QUANTUM} → {_post_arch}"
          f" → {cfg.K}×{actor.n_choices} logits")
    print(f"       Output : φ ∈ {{0,…,{cfg.M}}}^{cfg.K}  "
          f"({actor.n_choices}^{cfg.K} combinations)")
    print(f"    2. PhaseMLP  (IRS phase shifts — active IRS only)")
    _d_phase = 3 * cfg.K + P.n_latent
    print(f"       State  : c^SRU_m ∈ R^{_d_phase} per active IRS  "
          f"([Re+Im c^SRU({2*cfg.K}) ‖ z_t({P.n_latent}) ‖ phi_mask({cfg.K})])")
    print(f"       Net    : {phase_net.architecture}  "
          f"(shared weights, N={cfg.N} elements/IRS, categorical)")
    print(f"       Output : phase_idx ∈ {{0..{env.n_phase_levels-1}}}^{{G×{cfg.N}}}  "
          f"(G ≤ {cfg.M} active IRS, expanded to {cfg.M}×{cfg.N})")
    print(f"    3. PowerMLP  (power allocation)")
    _d_power = 2 * cfg.K + P.n_latent
    print(f"       State  : h_eff+z_t ∈ R^{_d_power}  "
          f"([Re+Im h_eff({2*cfg.K}) ‖ z_t({P.n_latent})])")
    print(f"       Net    : {power_net.architecture}  softmax → ×P_S")
    print(f"       Output : w_c_vec ∈ R^{{G+1}}  (per-group common, G≤{cfg.M}),  "
          f"w_p ∈ R^{cfg.K}  (private)  [Σ=P_S]")
    print(f"    4. CkMLP  (common-rate split)")
    _d_ck = 5 * cfg.K
    print(f"       State  : [D_k, R_p_k, R_c_g_k, shortfall_k, phi_k] ∈ R^{_d_ck}  (per user)")
    print(f"       Net    : {ck_net.architecture}  within-group softmax → C_k")
    print(f"       Output : C_k ∈ R^{cfg.K}  (Σ_{{k∈g}} C_k = R_c_g per group)")

    print(f"\n  Critic  (State-Value MLP  V(s))")
    print(f"    Input        : s_t ({critic.d_state})  =  {critic.d_in} total")
    print(f"    Architecture : {critic.architecture_str}  (V)")

    print(f"\n  Training plan")
    print(f"    Episodes       : {args.episodes}")
    print(f"    Steps/episode  : {args.steps}")
    print(f"    PPO epochs     : {P.ppo_epochs}  (ε={P.ppo_epsilon},  batch={P.batch_size})")
    print(f"    Total steps    : {args.episodes * args.steps:,}")
    print(f"    γ (discount)   : {args.gamma}")
    print(f"    AE loss weight : {P.ae_weight}")
    print(f"    β entropy      : {P.beta_entropy} → {P.beta_entropy_min}  "
          f"(anneal to {int(P.beta_entropy_anneal_end*100)}% of training)")
    print(f"    LR schedule    : warm-up {int(P.lr_decay_start*100)}% eps, "
          f"then linear decay to ×{P.lr_min_frac}")
    print(f"    GAE lambda (λ) : {P.gae_lambda}")
    print(f"    AE pretrain    : {P.ae_pretrain_epochs} steps @ lr={P.ae_pretrain_lr}")
    print(f"    Data reupload  : {P.data_reuploading}")
    print(f"    lr  ω/λθ/ξ/ψ  : "
          f"{P.lr_actor_ae}/{P.lr_actor_qc}/{P.lr_actor_xi}/{P.lr_critic}")
    print(f"    Shots (train)  : {P.n_shots_train}")
    print(f"    Seed           : {args.seed}")
    n_batches   = -(-args.steps // P.batch_size)          # ceil div
    n_updates   = P.ppo_epochs * n_batches * P.n_rollout_episodes
    if P.spsa_n_reps > 0:
        # SPSA: 2 * n_reps circuits per update (independent of n_params)
        n_qc_per_update = 2 * P.spsa_n_reps * P.batch_size
        n_qc_per_ep     = n_qc_per_update * P.ppo_epochs * n_batches
        speedup_vs_ps   = (4 * (1 + actor.N_VAR_LAYERS) * actor.N_QUBITS) / (2 * P.spsa_n_reps)
        grad_desc = (f"SPSA  n_reps={P.spsa_n_reps}  eps={P.spsa_epsilon}"
                     f"  ({n_qc_per_update} circuits/update,  {speedup_vs_ps:.0f}x vs param-shift)")
    else:
        n_qc_evals  = 4 * (1 + actor.N_VAR_LAYERS) * actor.N_QUBITS
        n_qc_per_ep = n_qc_evals * P.batch_size * P.ppo_epochs * n_batches
        grad_desc   = (f"parameter-shift  ({n_qc_evals} circuits/sample,  "
                       f"{n_qc_per_ep:,} per episode)")
    print(f"\n{'─'*W}")
    print(f"  QC gradient : {grad_desc}")
    print(f"  Rollout buf : {P.n_rollout_episodes} ep x {args.steps} steps = "
          f"{P.n_rollout_episodes * args.steps} transitions/PPO call")
    print(f"{'─'*W}\n")

    # ── Column header strings (used after optional pre-training) ─────────────
    # NOTE: ClpQ/ClpP/ClpW/ClpC columns removed from the on-screen table for a
    # cleaner display (clip data still saved to metrics.npz). Rollout episodes
    # (no PPO update) show "—" for all loss columns, like L_ae.
    hdr = (f"  {'Ep':>5}  {'Reward':>9}  {'BestReward':>10}  "
           f"{'L_pg':>8}  {'L_ae':>9}  {'L_phs':>7}  {'L_pw':>7}  {'L_ck':>7}  "
           f"{'TD':>8}  "
           f"{'R_tot':>8}  {'R/user':>7}  {'QoS/K':>7}  {'IRS/K':>7}  {'Blk/K':>7}  {'s/ep':>6}")
    sep = (f"  {'─'*5}  {'─'*9}  {'─'*10}  "
           f"{'─'*8}  {'─'*9}  {'─'*7}  {'─'*7}  {'─'*7}  "
           f"{'─'*8}  "
           f"{'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*6}")

    # ════════════════════════════════════════════════════════════════════════
    # Environment feasibility probe (calibrates λ_D vs R_LoS before training)
    # ════════════════════════════════════════════════════════════════════════
    try:
        _env_feasibility_report(cfg, seed=args.seed)
    except Exception as _e:
        print(f"  (feasibility probe skipped: {_e})")

    # ════════════════════════════════════════════════════════════════════════
    # AE Hot-start pre-training (reconstruction only, no RL signal)
    # ════════════════════════════════════════════════════════════════════════
    if getattr(args, 'resume', None):
        print(f"  ⟳ RESUMED actors from: {args.resume}")
        if getattr(args, '_resumed_critic', False):
            print(f"    (AE pre-training skipped; critic V(s) warm-started from resume dir)")
        else:
            print(f"    (AE pre-training skipped; no saved critic found → critic re-initialised)")
    if (P.ae_pretrain_epochs > 0 and not getattr(args, 'no_pretrain', False)
            and not getattr(args, 'resume', None)):
        print(f"  Pre-training AE encoder/decoder for {P.ae_pretrain_epochs} steps …")
        rng_pre = np.random.default_rng(args.seed + 9999)
        pre_loss_log = []
        for pre_step in range(P.ae_pretrain_epochs):
            s_rand = rng_pre.standard_normal(actor.d_s)
            # Fix: demand slice is not normalised by _group_norm — inject real value
            s_rand[cfg.K * cfg.M : cfg.K * (cfg.M + 1)] = cfg.D_k_bps_hz
            loss   = actor.pretrain_ae_step(s_rand)
            pre_loss_log.append(loss)
            if (pre_step + 1) % 100 == 0:
                avg = float(np.mean(pre_loss_log[-100:]))
                print(f"    [{pre_step+1:>4}/{P.ae_pretrain_epochs}]  L_ae = {avg:.4f}")
        print(f"  AE pre-training complete  "
              f"(final L_ae ≈ {float(np.mean(pre_loss_log[-20:])):.4f})")

    # ── Critic warm-up (resume only): fit V(s) to the resumed policy ──────────
    # Calibrates the value scale before PPO so advantages are meaningful from
    # episode 1 — essential when the resume dir has no saved critic (e.g.
    # result_6) and the critic would otherwise cold-start blind.
    # Critic warm-up: fit V(s) to MC returns of the (frozen) current policy before PPO.
    # Run for BOTH resume AND fresh runs [Case 2 result_2 fix H2]: on fresh, the RL heads
    # are random (post AE-pretrain) but the rolled returns still give V a calibrated SCALE
    # → better early advantages → bootstrap out of the blind-critic stuck equilibrium.
    # (One-shot on fresh is less durable than on resume since the policy then changes, but
    #  combined with reward_noise_avg↑ it gives the critic a fighting start.)
    wu_eps = getattr(P, 'critic_warmup_episodes', 0)
    if wu_eps > 0:
        wu_epochs = getattr(P, 'critic_warmup_epochs', 30)
        _wu_src = 'resumed' if getattr(args, 'resume', None) else 'fresh (random) '
        print(f"  Warming up critic V(s) [{_wu_src} policy]: {wu_eps} rollout ep × "
              f"{wu_epochs} fit epochs (actors frozen) …")
        _wu = _warmup_critic(env, actor, critic, phase_net, power_net, ck_net,
                             cfg, args, demand, wu_eps, wu_epochs)
        print(f"    critic warm-up done: N={_wu['n_samples']}  "
              f"return μ={_wu['ret_mean']:+.2f} σ={_wu['ret_std']:.2f}  "
              f"V̄={_wu['v_mean']:+.2f}  explVar={_wu['expl_var']:+.2f}")

    print(f"\n{'─'*W}")
    print(hdr)
    print(sep)

    # ── History buffers ───────────────────────────────────────────────────────
    hist: dict = {k: [] for k in (
        'episode_reward', 'mean_L_actor', 'mean_L_pg', 'mean_L_ae', 'mean_td_loss',
        'feasibility_rate', 'mean_o_hat_norm', 'mean_z_norm',
        'mean_sum_rate', 'mean_per_user_rate', 'mean_qos_rate',
        'mean_irs_rate', 'mean_blocked_rate',
        'mean_L_phase', 'mean_L_power', 'mean_L_ck',
        'mean_clip_q', 'mean_clip_phase', 'mean_clip_power', 'mean_clip_ck',
        # divergence diagnostics (carry-forward between PPO updates)
        'mean_kl_q', 'mean_kl_ph', 'mean_kl_pw', 'mean_kl_ck',
        'mean_qp_penalty', 'mean_expl_var', 'mean_v', 'mean_v_std',
        'mean_lam_y', 'mean_lam_z',
        'gnorm_ae', 'gnorm_qc', 'gnorm_xi', 'gnorm_phase',
        'gnorm_power', 'gnorm_ck', 'gnorm_critic',
    )}

    best_reward   = None
    best_ep       = 0     # episode index of the current best (for eps-since-best)
    n_diverge     = 0     # cumulative divergence-signal count (for end summary)
    # carry-forward of the latest PPO-update diagnostics (so npz arrays stay
    # per-episode aligned even though diagnostics are computed every N episodes)
    last_diag = {k: 0.0 for k in (
        'kl_q', 'kl_ph', 'kl_pw', 'kl_ck', 'qp_penalty', 'expl_var', 'v', 'v_std',
        'lam_y', 'lam_z', 'gn_ae', 'gn_qc', 'gn_xi', 'gn_phase', 'gn_power',
        'gn_ck', 'gn_critic')}
    t_train_start = time.perf_counter()
    rng_ppo       = np.random.default_rng(args.seed + 77)   # for ep_buf shuffling
    rollout_buf   = []   # accumulates ep_buf across n_rollout_episodes episodes
    gnorm_ema     = {}   # EMA of per-net grad norms (for spike-based divergence detection)

    # ════════════════════════════════════════════════════════════════════════
    # Episode loop
    # ════════════════════════════════════════════════════════════════════════
    for ep in range(args.episodes):
        obs   = env.reset()
        K     = cfg.K          # fixed K throughout
        t0_ep = time.perf_counter()

        # ── Per-episode schedule ──────────────────────────────────────────────
        beta    = _compute_beta(ep, args.episodes)
        lr_frac = _compute_lr_frac(ep, args.episodes)
        actor.opt_ae.lr      = P.lr_actor_ae * lr_frac
        actor.opt_qc.lr      = P.lr_actor_qc * lr_frac
        actor.opt_xi.lr      = P.lr_actor_xi * lr_frac
        phase_net.opt.lr     = P.lr_phase    * lr_frac
        power_net.opt.lr     = P.lr_power    * lr_frac
        ck_net.opt.lr        = P.lr_ck       * lr_frac
        critic.opt.lr        = P.lr_critic   * lr_frac

        ep_reward = 0.0
        ep_L_actor = []
        ep_L_pg, ep_L_ae, ep_td = [], [], []
        ep_L_phase, ep_L_power, ep_L_ck = [], [], []
        ep_clip_q, ep_clip_ph, ep_clip_pw, ep_clip_ck = [], [], [], []
        # Divergence diagnostics (populated during PPO update)
        ep_gnorm = {'ae': [], 'qc': [], 'xi': [], 'phase': [], 'power': [], 'ck': [], 'critic': []}
        ep_glam  = []   # [Bước 0] per-mini-batch λ (encoding) grad vectors → frozen-λ diag
        ep_kl    = {'q': [], 'ph': [], 'pw': [], 'ck': []}
        ep_ent   = {'q': [], 'ph': [], 'pw': [], 'ck': []}
        ep_qp    = []   # QoS penalty per step (reward breakdown)
        # Behaviour diagnostics: grouping, power allocation, common-rate split
        ep_n_dir = []   # # users on direct link
        ep_irs_cnt = []  # (M,) # users per IRS
        ep_wc_frac = []; ep_wp_frac = []   # common / private power fraction of P_S
        ep_wp_top  = []  # fraction of private power held by the top-2 users (concentration)
        ep_ck_tot  = []; ep_ck_top = []; ep_ck_active = []  # common-rate total / top-share / #served
        ep_buf = []
        ep_feas      = 0
        ep_o_norm    = []
        ep_z_norm    = []
        ep_sum_rate  = []
        ep_qos_count     = []
        ep_irs_count     = []
        ep_blocked_count = []

        # ── Step loop ────────────────────────────────────────────────────────
        for step_idx in range(args.steps):
            is_last_step = (step_idx == args.steps - 1)

            # ════════════════════════════════════════════════════════════════
            # 1) Forward on the LIVE trajectory
            # ════════════════════════════════════════════════════════════════
            blocked = _compute_blocked(env)
            ep_blocked_count.append(int(np.sum(env.channels['su_blocked'])))
            s_t     = actor.extract_state(obs, demand, blocked)
            phi, _, actor_info = actor.forward(s_t)
            z_t = actor_info['z_t']                              # (n_latent,) spatial latent

            # 0-based active IRS indices for PhaseMLP; 1-based ids for PowerMLP/rate
            active_irs     = _get_active_irs(phi)        # (G,) 0-based
            active_irs_ids = _get_active_irs_ids(phi)    # [gid, …] 1-based

            s_phase  = _build_phase_state(env.channels, phi, cfg, z_t)  # (M, 3K+n_latent)
            phase_idx, lp_old_ph, _ = phase_net.forward(s_phase, active_irs)
            phases_rad   = env.phase_model.index_to_phase(phase_idx)
            proposed_Phi = env.phase_model.build_phi(phases_rad)

            h_eff   = env.rate_computer.effective_channels_all(
                          phi, proposed_Phi, env.channels)
            s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])   # (2K + n_latent,)
            w_c_vec, w_p, _, power_action = power_net.forward(s_power, active_irs_ids)

            partial = env.rate_computer.compute_rates_partial(
                phi, proposed_Phi, env.channels, w_p, w_c_vec,
                active_irs_ids=active_irs_ids)
            s_ck = _build_ck_state(
                demand, partial['R_private'], partial['R_c_group'], phi, cfg)
            C_k, _, ck_group_sel = ck_net.forward(
                s_ck, phi, partial['R_c_group'])

            action = {
                'assignment': phi, 'phase_idx': phase_idx,
                'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k,
            }
            obs2, reward, _, env_info = env.step(action)

            blocked2 = _compute_blocked(env)
            s_t_next = actor.extract_state(obs2, demand, blocked2)

            # ════════════════════════════════════════════════════════════════
            # 2) Collect transition for GAE + PPO
            # ════════════════════════════════════════════════════════════════
            a_t = _build_action_vec(phi, phase_idx, w_c_vec, active_irs_ids,
                                    w_p, C_k, cfg, env.n_phase_levels)
            V_t_collected = float(critic.forward(s_t))
            lp_old_q  = actor.compute_log_prob(s_t, phi, K)
            lp_old_pw = power_net.compute_log_prob(s_power, active_irs_ids, power_action)
            lp_old_ck = ck_net.compute_log_prob(s_ck, phi, K, ck_group_sel)
            ep_buf.append({
                's_t':             s_t.copy(),
                'a_t':             a_t.copy(),
                's_t_next':        s_t_next.copy(),
                'phi':             phi.copy(),
                's_phase':         s_phase.copy(),
                'phase_idx':       phase_idx.copy(),
                's_power':         s_power.copy(),
                'active_irs_ids':  list(active_irs_ids),
                's_ck':            s_ck.copy(),
                'power_a_split':   int(power_action['split']),
                'power_a_common':  int(power_action['common']),
                'power_a_private': int(power_action['private']),
                'ck_group_sel':    dict(ck_group_sel),
                'lp_old_q':        float(lp_old_q),
                'lp_old_ph':       float(lp_old_ph),
                'lp_old_pw':       float(lp_old_pw),
                'lp_old_ck':       float(lp_old_ck),
                'reward':          float(reward),
                'is_terminal':     bool(is_last_step),
                'V_t':             V_t_collected,
                # ── Tier-2 diag fields (read by RL/critic_diag.py) ────────
                'blocked_n':       int(np.sum(env.channels['su_blocked'])),
                'sigma2':          float(env_info.get('sigma2', np.nan)),
            })

            ep_reward += reward
            ep_o_norm.append(float(np.linalg.norm(actor_info['o_hat'])))
            ep_z_norm.append(float(np.linalg.norm(actor_info['z_t'])))
            ep_sum_rate.append(float(env_info['sum_rate']))
            ep_qp.append(float(env_info['qp_penalty']))
            ep_qos_count.append(int(np.sum(env_info['R_tot'] >= cfg.D_k_bps_hz)))
            ep_irs_count.append(int(np.sum(phi > 0)))
            if env_info['feasible']:
                ep_feas += 1

            # ── Behaviour diagnostics (per step) ──────────────────────────────
            ep_n_dir.append(int(np.sum(phi == 0)))
            ep_irs_cnt.append(np.array([int(np.sum(phi == m + 1)) for m in range(cfg.M)]))
            _wc_t = float(np.sum(w_c_vec)); _wp_t = float(np.sum(w_p)); _tot = _wc_t + _wp_t + 1e-9
            ep_wc_frac.append(_wc_t / _tot); ep_wp_frac.append(_wp_t / _tot)
            _wp_sorted = np.sort(w_p)[::-1]
            ep_wp_top.append(float(_wp_sorted[:2].sum() / (_wp_t + 1e-9)))
            _ck = env_info.get('C_k')
            if _ck is not None:
                _ck = np.asarray(_ck, float); _ckt = float(_ck.sum())
                ep_ck_tot.append(_ckt)
                ep_ck_top.append(float(_ck.max() / (_ckt + 1e-9)) if _ckt > 1e-9 else 0.0)
                ep_ck_active.append(int(np.sum(_ck > 1e-6)))

            obs = obs2

        # ════════════════════════════════════════════════════════════════════
        # 3) GAE backward — Â_t = Σ (γλ)^l δ_{t+l},  δ_t = r_t + γV(s_{t+1}) − V(s_t)
        # ════════════════════════════════════════════════════════════════════
        # V(s_{i+1}) is the value stored at the next buffer entry. Episodes are
        # time-truncated (done is always False), so the final step bootstraps the
        # next-state value V(s_T') rather than treating it as a true terminal.
        # Advantages are normalised once per rollout (see PPO block), not here.
        gae = 0.0
        V_next_last = float(critic.forward(ep_buf[-1]['s_t_next'])) if ep_buf else 0.0
        for i in reversed(range(len(ep_buf))):
            t      = ep_buf[i]
            V_next = ep_buf[i + 1]['V_t'] if i + 1 < len(ep_buf) else V_next_last
            delta  = t['reward'] + args.gamma * V_next - t['V_t']
            gae    = delta + args.gamma * P.gae_lambda * gae
            t['gae_adv'] = float(gae)
            t['ret']     = float(gae + t['V_t'])   # GAE return target (for explained-variance)

        rollout_buf.extend(ep_buf)

        # ════════════════════════════════════════════════════════════════════
        # 5) PPO-clip — runs every n_rollout_episodes episodes
        # ════════════════════════════════════════════════════════════════════
        if (ep + 1) % P.n_rollout_episodes == 0 or ep == args.episodes - 1:
          # Per-rollout advantage normalisation (zero-mean, unit-std over all
          # collected transitions), then CLIP to ±adv_clip. Clipping bounds the
          # gradient from extreme-advantage outliers (e.g. high spawn-variance at
          # large R_LoS → reward swings → huge AE/QC grad spikes). At ±5 it only
          # touches ~5σ outliers, so stable runs are unaffected.
          advs     = np.array([t['gae_adv'] for t in rollout_buf])
          adv_mean = float(advs.mean())
          adv_std  = float(advs.std()) + 1e-8
          _norm    = (advs - adv_mean) / adv_std
          _aclip   = getattr(P, 'adv_clip', 0.0) or 0.0
          adv_clip_frac = float(np.mean(np.abs(_norm) > _aclip)) if _aclip > 0 else 0.0
          if _aclip > 0:
              _norm = np.clip(_norm, -_aclip, _aclip)
          for t, _a in zip(rollout_buf, _norm):
              t['gae_adv'] = float(_a)

          ep_indices = np.arange(len(rollout_buf))
          # PopArt: update value-scale stats (μ,σ) ONCE per rollout from this
          # batch's GAE-returns, and PRESERVE the critic's outputs (rescale last
          # layer). Must run BEFORE the PPO epochs so all minibatch regressions
          # use the same normalised target. (no-op when popart disabled.)
          # [T5 diag] POP correctness: V(s) for fixed states must be UNCHANGED across
          # the stats update (POP is algebraically exact → ΔV≈0). A large ΔV ⇒ bug.
          _pa_on = getattr(P, 'popart_enabled', False)
          _pa_s  = [t['s_t'] for t in rollout_buf[:64]]
          _v_pre = np.array([critic.forward(s) for s in _pa_s]) if _pa_on else None
          critic.update_popart_stats(np.array([t['ret'] for t in rollout_buf]))
          pop_dv = (float(np.max(np.abs(
                       np.array([critic.forward(s) for s in _pa_s]) - _v_pre)))
                    if _v_pre is not None else float('nan'))
          # [Tier-2 diag] per-PPO-epoch critic pre-clip grad-norm tracker.
          # Detects whether the critic over-trains across the 6 PPO epochs
          # (epoch-6 grad ≫ epoch-1 grad ⇒ critic is "chasing" its own targets).
          ep_critic_gn_preclip_by_epoch = [[] for _ in range(P.ppo_epochs)]
          # [factored PowerMLP diag] per-axis entropy & split policy bias —
          # detects collapsed axes (low H/Hmax) or runaway uniformisation (high H/Hmax)
          # in the new split/common/private factorisation (Case2 result_10 fix).
          ep_pw_axis_stats: dict = {}
          for _epoch in range(P.ppo_epochs):
            rng_ppo.shuffle(ep_indices)
            for start in range(0, len(rollout_buf), P.batch_size):
                mini = [rollout_buf[i] for i in ep_indices[start : start + P.batch_size]]
                B = len(mini)
                if B == 0:
                    continue

                # Loss accumulators (for logging)
                L_pg_b = L_ae_b = L_ent_b = L_phase_b = L_power_b = L_ck_b = 0.0
                td_b   = 0.0

                # Gradient outputs (assigned by batch methods below)
                g_phase_acc = g_power_acc = g_ck_acc = None

                # ── Phase 1: batch MLP actor gradients (vectorised — no Python loop) ─
                adv_arr     = np.array([t['gae_adv']   for t in mini])
                lp_old_ph_b = np.array([t['lp_old_ph'] for t in mini])
                lp_old_pw_b = np.array([t['lp_old_pw'] for t in mini])
                lp_old_ck_b = np.array([t['lp_old_ck'] for t in mini])

                lp_ph_b = phase_net.compute_log_prob_batch(mini)
                lp_pw_b = power_net.compute_log_prob_batch(mini)
                lp_ck_b = ck_net.compute_log_prob_batch(mini, K)

                # Approx KL(old‖new) per sub-actor: mean(exp(Δ) − 1 − Δ), Δ=lp_new−lp_old
                _d_ph = lp_ph_b - lp_old_ph_b
                _d_pw = lp_pw_b - lp_old_pw_b
                _d_ck = lp_ck_b - lp_old_ck_b
                kl_ph_b = float(np.mean(np.exp(_d_ph) - 1.0 - _d_ph))
                kl_pw_b = float(np.mean(np.exp(_d_pw) - 1.0 - _d_pw))
                kl_ck_b = float(np.mean(np.exp(_d_ck) - 1.0 - _d_ck))

                eff_ph_b, l_ph_arr, clip_ph_b = _ppo_eff_adv_batch(
                    lp_ph_b, lp_old_ph_b, adv_arr, P.ppo_epsilon)
                eff_pw_b, l_pw_arr, clip_pw_b = _ppo_eff_adv_batch(
                    lp_pw_b, lp_old_pw_b, adv_arr, P.ppo_epsilon)
                eff_ck_b, l_ck_arr, clip_ck_b = _ppo_eff_adv_batch(
                    lp_ck_b, lp_old_ck_b, adv_arr, P.ppo_epsilon)

                L_phase_b = float(l_ph_arr.mean())
                L_power_b = float(l_pw_arr.mean())
                L_ck_b    = float(l_ck_arr.mean())

                _, L_ent_ph, g_phase_acc = phase_net.compute_grads_batch(
                    mini, eff_ph_b, beta)
                _, L_ent_pw, g_power_acc, pw_axis = power_net.compute_grads_batch(
                    mini, eff_pw_b, beta,
                    beta_entropy_private_extra=P.beta_entropy_pwr_private)
                for _k, _v in pw_axis.items():
                    ep_pw_axis_stats.setdefault(_k, []).append(float(_v))
                _, L_ent_ck, g_ck_acc    = ck_net.compute_grads_batch(
                    mini, K, eff_ck_b, beta)
                L_ent_b += L_ent_ph + L_ent_pw + L_ent_ck

                # ── Critic batch gradient (vectorised — no Python loop) ────
                # V(s) critic: action list is unused (d_action=0).
                # Target = GAE λ-return t['ret'] (= Â_t + V_t), KHÔNG phải TD(0) target.
                # [Case2 result_3 fix B-2]: TD(0) (r+γV_target(s')) là target 1-bước phương
                # sai CAO → grad V kẹt lớn (700-1000), explVar dao động 0.14-0.64, QoS kẹt.
                # GAE-return (λ=0.9, ~7-step avg) ít nhiễu hơn + align với explVar (đo vs ret)
                # + là value-target PPO CHUẨN → grad giảm/settle, explVar ổn.
                td_b, g_critic_acc = critic.compute_grads_batch(
                    [t['s_t']       for t in mini],
                    [None           for _ in mini],
                    np.array([t['ret'] for t in mini]))

                # ── Phase 2: fused quantum log-prob + gradient (1 QC forward) ─
                l_q_arr, L_ae_b, L_ent_q, g_ae_acc, g_qc_acc, g_xi_acc, clip_q_b, kl_q_b = \
                    actor.compute_logprobs_grads_batch(
                        [t['s_t']      for t in mini],
                        [t['phi']      for t in mini],
                        np.array([t['lp_old_q'] for t in mini]),
                        np.array([t['gae_adv']  for t in mini]),
                        P.ppo_epsilon, P.ae_weight, beta)
                L_pg_b  = float(l_q_arr.mean())
                L_ent_b += L_ent_q

                # [Bước 0 diag] capture UNCLIPPED λ (encoding-scale) gradient per
                # mini-batch to diagnose the frozen-λ issue [E]. We want to know if
                # the λ gradient is zero-mean sign-cancelling noise (→ Adam stalls →
                # λ frozen) vs a tiny-but-consistent signal. Store the concatenated
                # lam_y/lam_z grad vector; stats computed in the diag panel below.
                ep_glam.append(np.concatenate([
                    np.ravel(g_qc_acc['lam_y']), np.ravel(g_qc_acc['lam_z'])]))

                # Clip non-quantum gradients
                # All batch gradient methods already return B-averaged grads internally

                gn_ae = _clip_grad_norm(g_ae_acc,     P.grad_clip_ae)
                gn_qc = _clip_grad_norm(g_qc_acc,     P.grad_clip_actor)
                gn_xi = _clip_grad_norm(g_xi_acc,     P.grad_clip_actor)
                gn_ph = _clip_grad_norm(g_phase_acc,  P.grad_clip_actor)
                gn_pw = _clip_grad_norm(g_power_acc,  P.grad_clip_actor)
                gn_ck = _clip_grad_norm(g_ck_acc,     P.grad_clip_actor)
                gn_cr = _clip_grad_norm(g_critic_acc, P.grad_clip_critic)
                # [Tier-2 diag] track pre-clip critic grad per PPO epoch
                ep_critic_gn_preclip_by_epoch[_epoch].append(gn_cr)

                # Apply
                actor.apply_grads(g_ae_acc, g_qc_acc, g_xi_acc)
                phase_net.apply_grads(g_phase_acc)
                power_net.apply_grads(g_power_acc)
                ck_net.apply_grads(g_ck_acc)
                critic.apply_grads(g_critic_acc)

                L_actor_b = (L_pg_b + P.ae_weight * L_ae_b
                             + L_phase_b + L_power_b + L_ck_b + L_ent_b)
                ep_L_actor.append(L_actor_b)
                ep_L_pg.append(L_pg_b)
                ep_L_ae.append(L_ae_b)
                ep_td.append(td_b)
                ep_L_phase.append(L_phase_b)
                ep_L_power.append(L_power_b)
                ep_L_ck.append(L_ck_b)
                ep_clip_q.append(clip_q_b)
                ep_clip_ph.append(clip_ph_b)
                ep_clip_pw.append(clip_pw_b)
                ep_clip_ck.append(clip_ck_b)
                # Divergence diagnostics
                ep_gnorm['ae'].append(gn_ae);  ep_gnorm['qc'].append(gn_qc)
                ep_gnorm['xi'].append(gn_xi);  ep_gnorm['phase'].append(gn_ph)
                ep_gnorm['power'].append(gn_pw);  ep_gnorm['ck'].append(gn_ck)
                ep_gnorm['critic'].append(gn_cr)
                ep_kl['q'].append(kl_q_b);   ep_kl['ph'].append(kl_ph_b)
                ep_kl['pw'].append(kl_pw_b); ep_kl['ck'].append(kl_ck_b)
                _bi = max(beta, 1e-8)
                ep_ent['q'].append(-L_ent_q / _bi);   ep_ent['ph'].append(-L_ent_ph / _bi)
                ep_ent['pw'].append(-L_ent_pw / _bi); ep_ent['ck'].append(-L_ent_ck / _bi)

          # ── Divergence diagnostics panel (once per PPO update) ────────────────
          _retb = np.array([t['ret'] for t in rollout_buf])
          _Vb   = np.array([t['V_t'] for t in rollout_buf])
          _gx   = lambda x: float(np.max(x)) if x else float('nan')   # max → spike detector
          _gm   = lambda x: float(np.mean(x)) if x else float('nan')
          diag = {
              'gnorm':    {k: _gx(v) for k, v in ep_gnorm.items()},
              'kl':       {k: _gm(v) for k, v in ep_kl.items()},
              'ent':      {k: _gm(v) for k, v in ep_ent.items()},
              'v_mean':   float(_Vb.mean()), 'v_std': float(_Vb.std()),
              'expl_var': _explained_variance(_retb, _Vb),
              'lam_y':    float(np.max(np.abs(actor.lam_y))),
              'lam_z':    float(np.max(np.abs(actor.lam_z))),
              'sum_rate': float(np.mean(ep_sum_rate)), 'qp': float(np.mean(ep_qp)),
          }
          diag['scalars'] = {'L_pg': _gm(ep_L_pg), 'TD': _gm(ep_td),
                             'V_mean': diag['v_mean'], 'expl_var': diag['expl_var'],
                             **{f"gn_{k}": v for k, v in diag['gnorm'].items()}}
          _signals = _divergence_signals(diag, gnorm_ema)
          _print_diag_panel(ep, diag, _signals)      # screen + file (concise)
          if _signals:
              n_diverge += 1

          # ── Verbose diagnostics → log FILE ONLY (screen stays concise) ───────
          # (1) entropy-domination ratio β·H/|L_pg| per actor — the direct
          #     predictor of the result_3 collapse (entropy overwhelming the
          #     policy gradient → policy drifts to random → QoS collapse).
          _pg  = lambda xs: max(abs(_gm(xs)), 1e-6)
          _ed  = {a: beta * diag['ent'][a] / _pg(xs) for a, xs in
                  (('q', ep_L_pg), ('ph', ep_L_phase), ('pw', ep_L_power), ('ck', ep_L_ck))}
          _edf = '  ⚠ ENTROPY-DOMINATED (↓β)' if max(_ed.values()) > 1.0 else ''
          _acf = '  ⚠ HIGH-VARIANCE' if adv_clip_frac > 0.02 else ''
          _flog(f"  · ent/pg (β·H/|L_pg|): q={_ed['q']:.2f} ph={_ed['ph']:.2f} "
                f"pw={_ed['pw']:.2f} ck={_ed['ck']:.2f}{_edf}  │ adv-clip {adv_clip_frac*100:.1f}%{_acf}")
          # [factored PowerMLP] per-axis health: H/Hmax for each axis (1.0 = uniform,
          # 0 = collapsed); π_split(common) = policy's mean common-share decision
          # (independent of sampled w_c; bias-init starts at ~73%, PG drifts it).
          if ep_pw_axis_stats:
              _ax = {k: float(np.mean(v)) for k, v in ep_pw_axis_stats.items()}
              _flog(f"  · pw axis (H/Hmax): split={_ax['H_split_frac']*100:.0f}% "
                    f"common={_ax['H_common_frac']*100:.0f}% "
                    f"private={_ax['H_private_frac']*100:.0f}%  │ "
                    f"π_split(common)={_ax['pi_split_common']*100:.0f}%")
          # [Bước 0] frozen-λ diagnostic: across this update's mini-batch λ grads,
          #   mag = mean |g| (typical push size);  net = mean_c |mean_s g| (directional
          #   component, per-component signed-mean then abs-averaged);  r = net/mag.
          #   r≈1 → consistent direction (λ would move);  r≈0 → zero-mean sign-cancelling
          #   noise → Adam's 1st moment ≈0 → λ FROZEN [E]. Compare mag vs gn_qc too.
          if ep_glam:
              _G = np.stack(ep_glam)                       # (n_mb, 2*nq)
              _mag = float(np.mean(np.abs(_G)))
              _net = float(np.mean(np.abs(_G.mean(axis=0))))
              _r   = _net / (_mag + 1e-12)
              _flog(f"  · λgrad: mag={_mag:.2e} net={_net:.2e} r={_r:.2f}"
                    f"  ({'consistent→should move' if _r > 0.5 else 'zero-mean noise→frozen [E]'})")
          # (2) rolling-mean trend (reward/QoS/R_tot) + episodes-since-best —
          #     reveals slow drift hidden by per-episode noise.
          _rw = hist['episode_reward']; _w = min(50, len(_rw))
          if _w > 0:
              _flog(f"  · rolling-{_w}: reward μ={float(np.mean(_rw[-_w:])):8.1f}  "
                    f"QoS μ={float(np.mean(hist['mean_qos_rate'][-_w:]))*100:4.0f}%  "
                    f"Rtot μ={float(np.mean(hist['mean_sum_rate'][-_w:])):.2f}  │ "
                    f"best={best_reward:.1f} ({(ep+1)-best_ep} ep ago)")
          # (3) Behaviour: grouping / power allocation / common-rate split (this episode).
          #     Reveals HOW the agent acts — e.g. power concentrated on few strong users
          #     (wp-top2 high) = abuse; how common rate is shared; group balance.
          _gm2     = lambda xs: float(np.mean(xs)) if xs else 0.0
          _irs_mean = np.mean(np.array(ep_irs_cnt), axis=0) if ep_irs_cnt else np.zeros(cfg.M)
          _irs_str  = "[" + " ".join(f"{x:.1f}" for x in _irs_mean) + "]"
          _flog(f"  · behaviour: grp dir={_gm2(ep_n_dir):.1f} irs={_irs_str}  │ "
                f"pwr wc={_gm2(ep_wc_frac)*100:.0f}% wp={_gm2(ep_wp_frac)*100:.0f}% "
                f"(wp-top2={_gm2(ep_wp_top)*100:.0f}%)  │ "
                f"Ck tot={_gm2(ep_ck_tot):.2f} top={_gm2(ep_ck_top)*100:.0f}% on {_gm2(ep_ck_active):.1f}/{cfg.K}")
          # Update grad-norm EMA AFTER signalling (so a spike is judged vs history)
          for _net, _gn in diag['gnorm'].items():
              if np.isfinite(_gn):
                  gnorm_ema[_net] = _gn if _net not in gnorm_ema else 0.8 * gnorm_ema[_net] + 0.2 * _gn
          # Carry-forward diagnostics for per-episode npz persistence
          last_diag.update(
              kl_q=diag['kl']['q'], kl_ph=diag['kl']['ph'],
              kl_pw=diag['kl']['pw'], kl_ck=diag['kl']['ck'],
              qp_penalty=diag['qp'], expl_var=diag['expl_var'],
              v=diag['v_mean'], v_std=diag['v_std'],
              lam_y=diag['lam_y'], lam_z=diag['lam_z'],
              gn_ae=diag['gnorm']['ae'], gn_qc=diag['gnorm']['qc'],
              gn_xi=diag['gnorm']['xi'], gn_phase=diag['gnorm']['phase'],
              gn_power=diag['gnorm']['power'], gn_ck=diag['gnorm']['ck'],
              gn_critic=diag['gnorm']['critic'])

          # ── Tier-2 critic diagnostic (read-only, ~50ms / PPO update) ─────────
          try:
              crit_diag = compute_critic_diag(
                  rollout_buf, critic,
                  ep_critic_gn_preclip_by_epoch,
                  P.grad_clip_critic, args.gamma, env=env,
                  pop_dv=pop_dv)
              for line in format_critic_diag_lines(ep, crit_diag):
                  _flog(line)
              write_critic_diag_jsonl(
                  os.path.join(run_dir, 'critic_diag.jsonl'),
                  ep, crit_diag)
          except Exception as _exc:  # noqa — never let diag kill training
              _flog(f"  ┄ crit-diag[{ep}]  ERROR: {_exc}")

          rollout_buf = []   # clear after PPO update

        # ── Episode stats ─────────────────────────────────────────────────────
        ep_time      = time.perf_counter() - t0_ep
        feas_rate     = ep_feas / args.steps
        mean_sum_rate = float(np.mean(ep_sum_rate))
        mean_qos_n    = float(np.mean(ep_qos_count))          # avg # users meeting D_k
        mean_qos_rate = mean_qos_n / K                        # fraction (for hist/plots)
        mean_irs_n        = float(np.mean(ep_irs_count))      # avg # users on IRS
        mean_irs_rate     = mean_irs_n / K                   # fraction (for hist/plots)
        mean_blocked_n    = float(np.mean(ep_blocked_count)) # avg # users blocked by buildings
        mean_per_user_rate = mean_sum_rate / K               # avg rate per user

        # Loss means — guard against warmup episodes where no update happened
        _mean = lambda xs: float(np.mean(xs)) if len(xs) > 0 else 0.0

        hist['episode_reward'].append(ep_reward)
        hist['mean_L_actor'].append(_mean(ep_L_actor))
        hist['mean_L_pg'].append(_mean(ep_L_pg))
        hist['mean_L_ae'].append(_mean(ep_L_ae))
        hist['mean_td_loss'].append(_mean(ep_td))
        hist['feasibility_rate'].append(feas_rate)
        hist['mean_o_hat_norm'].append(float(np.mean(ep_o_norm)))
        hist['mean_z_norm'].append(float(np.mean(ep_z_norm)))
        hist['mean_sum_rate'].append(mean_sum_rate)
        hist['mean_per_user_rate'].append(mean_per_user_rate)
        hist['mean_qos_rate'].append(mean_qos_rate)
        hist['mean_irs_rate'].append(mean_irs_rate)
        hist['mean_blocked_rate'].append(mean_blocked_n / K)
        hist['mean_L_phase'].append(_mean(ep_L_phase))
        hist['mean_L_power'].append(_mean(ep_L_power))
        hist['mean_L_ck'].append(_mean(ep_L_ck))
        hist['mean_clip_q'].append(_mean(ep_clip_q))
        hist['mean_clip_phase'].append(_mean(ep_clip_ph))
        hist['mean_clip_power'].append(_mean(ep_clip_pw))
        hist['mean_clip_ck'].append(_mean(ep_clip_ck))
        # Carry-forward PPO-update diagnostics (per-episode aligned for npz)
        hist['mean_kl_q'].append(last_diag['kl_q']);   hist['mean_kl_ph'].append(last_diag['kl_ph'])
        hist['mean_kl_pw'].append(last_diag['kl_pw']); hist['mean_kl_ck'].append(last_diag['kl_ck'])
        hist['mean_qp_penalty'].append(last_diag['qp_penalty'])
        hist['mean_expl_var'].append(last_diag['expl_var'])
        hist['mean_v'].append(last_diag['v']);         hist['mean_v_std'].append(last_diag['v_std'])
        hist['mean_lam_y'].append(last_diag['lam_y']); hist['mean_lam_z'].append(last_diag['lam_z'])
        hist['gnorm_ae'].append(last_diag['gn_ae']);   hist['gnorm_qc'].append(last_diag['gn_qc'])
        hist['gnorm_xi'].append(last_diag['gn_xi']);   hist['gnorm_phase'].append(last_diag['gn_phase'])
        hist['gnorm_power'].append(last_diag['gn_power']); hist['gnorm_ck'].append(last_diag['gn_ck'])
        hist['gnorm_critic'].append(last_diag['gn_critic'])

        is_new_best = (best_reward is None) or (ep_reward > best_reward)
        best_reward = ep_reward if best_reward is None else max(best_reward, ep_reward)
        if is_new_best:
            best_ep = ep + 1
        best_str    = f"{best_reward:>10.3f}"

        qos_str = f"{mean_qos_n:.1f}/{K}"
        irs_str = f"{mean_irs_n:.1f}/{K}"
        blk_str = f"{mean_blocked_n:.1f}/{K}"
        flag    = "*" if is_new_best else " "
        # A PPO update ran this episode iff loss lists were populated. On rollout
        # episodes (no update) show "—" for every loss column (L_pg/L_ae/L_phs/
        # L_pw/L_ck/TD), consistent with how L_ae already displays.
        _upd = len(ep_L_pg) > 0
        _lf  = lambda xs, w, p: (f"{_mean(xs):>{w}.{p}f}" if _upd else f"{'—':>{w}}")
        _L_ae_str = f"{_mean(ep_L_ae):9.2e}" if _upd else f"{'—':>9}"
        print(f"{flag} {ep+1:>5}  {ep_reward:>9.3f}  {best_str}  "
              f"{_lf(ep_L_pg,8,3)}  {_L_ae_str}  "
              f"{_lf(ep_L_phase,7,3)}  {_lf(ep_L_power,7,3)}  {_lf(ep_L_ck,7,3)}  "
              f"{_lf(ep_td,8,4)}  "
              f"{mean_sum_rate:>8.3f}  {mean_per_user_rate:>7.3f}  "
              f"{qos_str:>7}  {irs_str:>7}  {blk_str:>7}  "
              f"{ep_time:>5.1f}s")

        # ── Save best agent (overwrite) whenever reward improves ──────────────
        if is_new_best:
            best_dir = os.path.join(run_dir, 'best')
            _save_agents(best_dir, actor, phase_net, power_net, ck_net, cfg, critic)

        # ── Periodic checkpoint every checkpoint_interval episodes ────────────
        ci = getattr(P, 'checkpoint_interval', 500)
        if ci > 0 and (ep + 1) % ci == 0:
            ckpt_dir = os.path.join(run_dir, 'checkpoints', f'ep_{ep+1:05d}')
            _save_agents(ckpt_dir, actor, phase_net, power_net, ck_net, cfg, critic)
            print(f"  [ckpt ep {ep+1}] → {os.path.relpath(ckpt_dir)}"
                  f"  (best reward so far: {best_reward:.3f})")

    # ── Total training time ───────────────────────────────────────────────────
    t_total  = time.perf_counter() - t_train_start
    final_ma = float(np.mean(hist['episode_reward'][-10:]))
    print(f"\n  {'─'*W}")
    print(f"  Training complete in {t_total/60:.1f} min  "
          f"({t_total/args.episodes:.1f} s/ep)")
    print(f"  Final 10-ep avg reward   : {final_ma:.4f}")
    print(f"  Final feasibility rate   : "
          f"{np.mean(hist['feasibility_rate'][-10:])*100:.1f}%")
    print(f"  Final avg Σ R_tot        : "
          f"{np.mean(hist['mean_sum_rate'][-10:]):.4f} bps/Hz"
          f"  ({np.mean(hist['mean_per_user_rate'][-10:]):.4f} bps/Hz/user)")
    print(f"  Final avg QoS fraction   : "
          f"{np.mean(hist['mean_qos_rate'][-10:])*100:.1f}%")

    # End-of-run analysis + recommendations for the next run
    _analysis_summary(hist, n_diverge, args)

    # ════════════════════════════════════════════════════════════════════════
    # Save results  (hyperparameters.json already written before training)
    # ════════════════════════════════════════════════════════════════════════
    np.savez(os.path.join(run_dir, 'metrics.npz'), **hist)
    _save_agents(run_dir, actor, phase_net, power_net, ck_net, cfg, critic)
    saved_plots = _save_plots(run_dir, hist, args)

    print(f"\n{'═'*W}")
    print(f"  Results  →  {os.path.abspath(run_dir)}")
    print(f"{'─'*W}")
    print(f"  hyperparameters.json   — all params for this run")
    print(f"  metrics.npz            — raw episode arrays (reload with np.load)")
    print(f"  training_log.txt       — full CLI output from this run")
    print(f"  agents/                — final actor weights (load with infer.py)")
    print(f"  best/                  — best-reward snapshot (overwrites on improvement)")
    ci = getattr(P, 'checkpoint_interval', 500)
    if ci > 0:
        n_ckpts = args.episodes // ci
        print(f"  checkpoints/ep_XXXXX/  — periodic snapshots every {ci} ep "
              f"({n_ckpts} total)")
    if saved_plots:
        for p in saved_plots:
            print(f"  {os.path.basename(p):<26} — performance plot")
    elif not HAVE_MPL or args.no_plots:
        print(f"  (plots skipped — pass --no-plots=false or install matplotlib)")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Quantum AC agent for IRS-assisted RSMA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--episodes', type=int,   default=P.n_episodes,
                        help='Number of training episodes')
    parser.add_argument('--steps',    type=int,   default=P.n_steps_per_ep,
                        help='Environment steps per episode')
    parser.add_argument('--gamma',    type=float, default=P.gamma,
                        help='TD discount factor γ')
    parser.add_argument('--seed',     type=int,   default=P.seed_default,
                        help='Global random seed')
    parser.add_argument('--no-plots',    action='store_true',
                        help='Skip matplotlib plot generation')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Skip AE hot-start pre-training (fast pipeline inspection)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to an agents/ dir to load the 4 actors from and continue '
                             'training (curriculum transfer; critic re-init). E.g. '
                             'results/result_2/best/agents')
    return parser.parse_args()


if __name__ == '__main__':
    train(_parse_args())
