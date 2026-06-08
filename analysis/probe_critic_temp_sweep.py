"""
probe_critic_temp_sweep.py
--------------------------
STEP A (2-curve version) of the "is the critic at ceiling?" test.

For a FIXED trained checkpoint, sweep the policy sampling-temperature τ and measure,
at each τ, two curves:

  EV_ceiling(τ) = 1 − E[Var(R | s, π_τ)] / Var(R)      (Bayes-optimal explained var)
  EV_critic(τ)  = 1 − Var(R − V(s)) / Var(R)            (the LIVE checkpoint critic)

τ scales the categorical heads via  softmax(logits/τ) = π^(1/τ)/Σ  (we re-sample from the
returned probs — no logit surgery). τ=1 = the policy as trained; τ→0 = greedy (lowest
entropy); τ>1 = flatter (higher entropy). Lower τ ⇒ less action-variance ⇒ E[Var(R|s)]↓
⇒ ceiling↑ (the T3 mechanism: explVar_max = f(π)).

VERDICT
  EV_critic TRACKS EV_ceiling as τ↓  → critic healthy across entropy → bottleneck is POLICY
                                       → close critic + dead-units, pivot to actor/head.
  EV_critic FALLS BELOW rising ceiling → underfit candidate → run Step B (capacity scaling,
                                       input-norm, LeakyReLU/GELU) — but first rule out the
                                       distribution-shift confound with a fresh-refit control.

CAVEATS (by design, this 2-curve version):
  • EV_critic on the LIVE critic mixes capacity-limit with OFF-DISTRIBUTION eval (V trained
    at τ=1). A "falls off" here is a TRIGGER to investigate, not proof of underfit.
  • τ applied to the two CATEGORICAL heads (assignment + phase = dominant entropy: ph≫q);
    power/Ck left at sampled default → ceiling shift is a LOWER bound.
  • temp-sweep is a MECHANISM proxy, not an ep600 prediction (greedy-of-half-trained ≠
    trained-to-be-sharp).
  • moderate n_shots for speed → shot noise slightly depresses absolute ceiling; the τ-TREND
    (the gated quantity) is preserved.

Usage:
  python analysis/probe_critic_temp_sweep.py --ckpt results/result_11/checkpoints/ep_00300
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from CSI.env_probe import (snapshot_env_state, restore_env_state,
                           step_with_source_control)


TAU_GREEDY = 0.05  # τ ≤ this → use greedy argmax path (avoid 0**inf)


def _temper(probs_2d: np.ndarray, tau: float) -> np.ndarray:
    """softmax(logits/τ) given the softmax PROBS: π^(1/τ) row-renormalised."""
    p = np.power(np.clip(probs_2d, 1e-12, 1.0), 1.0 / tau)
    return p / p.sum(axis=1, keepdims=True)


class TempPolicy:
    """Loads a checkpoint; .act(env, tau) applies sampling-temperature τ to the
    assignment + phase heads (re-sampled from their softmax probs), propagating
    the tempered phase through power/Ck. Stashes last critic-state + entropy."""

    def __init__(self, ckpt_dir: str, cfg, n_shots: int, seed: int = 0):
        if os.path.isdir(os.path.join(ckpt_dir, 'agents')) and \
           not os.path.isfile(os.path.join(ckpt_dir, 'actor_config.json')):
            ckpt_dir = os.path.join(ckpt_dir, 'agents')
        from RL import QuantumActor, PhaseMLP, PowerMLP, CkMLP, ClassicalCritic
        from train import (_build_phase_state, _build_ck_state, _get_active_irs)
        self._bps = _build_phase_state
        self._bcs = _build_ck_state
        self._gai = _get_active_irs

        self.actor  = QuantumActor.from_dir(ckpt_dir, seed=seed)
        self.phase  = PhaseMLP.from_dir(ckpt_dir, seed=seed)
        self.power  = PowerMLP.from_dir(ckpt_dir, seed=seed)
        self.ck     = CkMLP.from_dir(ckpt_dir, seed=seed)
        self.critic = ClassicalCritic.from_dir(ckpt_dir, seed=seed)
        self.actor.n_shots = int(n_shots)
        self.cfg = cfg
        self.rng = np.random.default_rng(seed + 777)
        self.last_s_t = None
        self.last_ent = 0.0   # assignment entropy of π_τ (nats), this call

    def act(self, env, tau: float) -> dict:
        cfg = self.cfg; K = cfg.K
        greedy = (tau <= TAU_GREEDY)
        unit_tau = abs(tau - 1.0) < 1e-6

        obs     = env._get_obs()
        demand  = np.full(K, cfg.D_k_bps_hz)
        blocked = env.channels['su_blocked'].astype(int)
        s_t = self.actor.extract_state(obs, demand, blocked)
        self.last_s_t = s_t

        phi, _, info = self.actor.forward(s_t, greedy=greedy)
        z_t = info['z_t']
        if not greedy and not unit_tau:
            pit = _temper(info['pi'], tau)                       # (K, n_choices)
            phi = np.array([self.rng.choice(pit.shape[1], p=pit[k])
                            for k in range(K)], dtype=int)
        # assignment entropy of the distribution actually sampled from (diag).
        # greedy = point mass → 0; else entropy of (tempered) π.
        if greedy:
            self.last_ent = 0.0
        else:
            pe = info['pi'] if unit_tau else pit
            self.last_ent = float(-(pe * np.log(pe + 1e-12)).sum(axis=1).mean())

        active_irs     = self._gai(phi)
        active_irs_ids = [int(a) + 1 for a in active_irs]

        s_phase = self._bps(env.channels, phi, cfg, z_t)
        phase_idx, _, probs_per_irs = self.phase.forward(s_phase, active_irs,
                                                         greedy=greedy)
        if not greedy and not unit_tau:
            for j, m in enumerate(active_irs):
                pm = _temper(probs_per_irs[j], tau)              # (N, n_levels)
                phase_idx[m] = np.array([self.rng.choice(pm.shape[1], p=pm[n])
                                         for n in range(pm.shape[0])], dtype=int)

        phases_rad   = env.phase_model.index_to_phase(phase_idx)
        proposed_Phi = env.phase_model.build_phi(phases_rad)
        h_eff   = env.rate_computer.effective_channels_all(phi, proposed_Phi, env.channels)
        s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
        w_c_vec, w_p, _, _ = self.power.forward(s_power, active_irs_ids)

        partial = env.rate_computer.compute_rates_partial(
            phi, proposed_Phi, env.channels, w_p, w_c_vec,
            active_irs_ids=active_irs_ids)
        s_ck = self._bcs(demand, partial['R_private'], partial['R_c_group'], phi, cfg)
        C_k, _, _ = self.ck.forward(s_ck, phi, partial['R_c_group'])
        return {'assignment': phi, 'phase_idx': phase_idx,
                'w_p': w_p, 'w_c_vec': w_c_vec, 'C_k': C_k}


def _boot_returns(recs, H, gamma, v_fn):
    """Bootstrapped H-step returns  R_i = Σ_{h<H} γ^h r_{i+h} + γ^H V(s_{i+H}).
    Matches the critic's training-target scale (full-horizon via bootstrap), so
    EV vs V(s_i) is not corrupted by the V(full)-vs-truncated-MC horizon mismatch.
    Returns (R array, record indices, V(s_i) array)."""
    rets, idxs, vpred = [], [], []
    n = len(recs)
    for i in range(n):
        ep = recs[i]['ep']; G = 0.0; disc = 1.0; cnt = 0
        for h in range(H):
            j = i + h
            if j >= n or recs[j]['ep'] != ep:
                break
            G += disc * recs[j]['r']; disc *= gamma; cnt += 1
        if cnt == H and (i + H) < n and recs[i + H]['ep'] == ep:
            G += disc * v_fn(recs[i + H]['s'])          # disc == γ^H
            rets.append(G); idxs.append(i); vpred.append(v_fn(recs[i]['s']))
    return np.array(rets), idxs, np.array(vpred)


def _boot_intra_std(env, tp, tau, cfg, H, R, gamma, base_seed):
    """σ of the BOOTSTRAPPED H-step return from a FIXED start state, over R
    replays (env-noise + policy-sampling under π_τ). Bootstrap with the same
    critic so numerator (intra) and denominator (cross-state) share R's scale."""
    snap = snapshot_env_state(env)
    rets = np.empty(R)
    for i in range(R):
        restore_env_state(env, snap)
        env.rng = np.random.default_rng(base_seed + i)
        env.channel_model.rng = env.rng
        G, disc = 0.0, 1.0
        for _ in range(H):
            a = tp.act(env, tau)
            _, r, _, _ = step_with_source_control(env, a, disable=())
            G += disc * float(r); disc *= gamma
        obs = env._get_obs()
        demand = np.full(cfg.K, cfg.D_k_bps_hz)
        blocked = env.channels['su_blocked'].astype(int)
        s_end = tp.actor.extract_state(obs, demand, blocked)
        G += disc * float(tp.critic.forward(s_end))     # disc == γ^H
        rets[i] = G
    restore_env_state(env, snap)
    return float(rets.std())


def _explained_var(target, pred):
    t = np.asarray(target); p = np.asarray(pred)
    vt = t.var()
    if vt < 1e-12:
        return float('nan')
    return float(1.0 - ((t - p).var() / vt))


def sweep_one_tau(tp: TempPolicy, cfg, tau, args, gamma):
    """Return dict with ceiling(τ), EV_critic(τ), realized entropy, scales."""
    act_fn = lambda env: tp.act(env, tau)

    # ── cross-state rollout: collect (s_t, reward) under π_τ ──
    env = ISTNEnv(cfg=cfg, seed=args.seed + 1, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=args.reward_noise_avg)
    recs, ents = [], []
    for ep in range(args.n_ep):
        env.reset(seed=(args.seed * 131 + ep) & 0xFFFF)
        for _ in range(args.warmup):
            env.step(act_fn(env))
        for _ in range(args.steps):
            a = act_fn(env)
            s_t = tp.last_s_t; ents.append(tp.last_ent)
            _, r, _, _ = env.step(a)
            recs.append({'s': s_t, 'r': float(r), 'ep': ep})
    v_fn = lambda s: float(tp.critic.forward(s))
    rets, idxs, V = _boot_returns(recs, args.H, gamma, v_fn)
    sigma_total = float(rets.std())
    ev_critic = _explained_var(rets, V)
    corr = float(np.corrcoef(rets, V)[0, 1]) if len(rets) > 2 else float('nan')

    # ── ceiling: BOOTSTRAPPED intra-state std under π_τ over several states ──
    cenv = ISTNEnv(cfg=cfg, seed=args.seed + 9, n_steps_ep=512,
                   reward_noise_avg=args.reward_noise_avg)
    cenv.reset(seed=(args.seed * 977) & 0xFFFF)
    for _ in range(args.warmup):
        cenv.step(act_fn(cenv))
    intra = []
    for k in range(args.n_states):
        for _ in range(3):
            cenv.step(act_fn(cenv))
        intra.append(_boot_intra_std(cenv, tp, tau, cfg, args.H, args.R, gamma,
                                     base_seed=args.seed * 7 + k))
    sigma_intra = float(np.mean(intra))
    frac = float(np.clip((sigma_intra ** 2) / (sigma_total ** 2 + 1e-12), 0.0, 1.0))
    ceiling = 1.0 - frac

    return {
        'tau': tau, 'ent_assign': float(np.mean(ents)),
        'ceiling': ceiling, 'ev_critic': ev_critic, 'gap': ceiling - ev_critic,
        'sigma_total': sigma_total, 'sigma_intra': sigma_intra,
        'corr_V_ret': corr, 'Vbar': float(V.mean()), 'ret_bar': float(rets.mean()),
        'n_ret': len(rets),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_11/checkpoints/ep_00300')
    ap.add_argument('--taus', type=float, nargs='+', default=[1.5, 1.0, 0.6, 0.3, 0.0])
    ap.add_argument('--n_shots', type=int, default=512)
    ap.add_argument('--reward_noise_avg', type=int,
                    default=getattr(P, 'reward_noise_avg', 16))
    ap.add_argument('--n_ep', type=int, default=8)
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--n_states', type=int, default=18)
    ap.add_argument('--R', type=int, default=18)
    ap.add_argument('--H', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260603)
    args = ap.parse_args()

    cfg = make_config()
    gamma = getattr(P, 'gamma', 0.95)
    print("=" * 76)
    print(f"  CRITIC TEMPERATURE-SWEEP (Step A, 2-curve)  ·  ckpt={args.ckpt}")
    print(f"  K={cfg.K} M={cfg.M} N={cfg.N} · n_shots={args.n_shots} · H={args.H} "
          f"γ={gamma} · noise_avg={args.reward_noise_avg}")
    print(f"  collect {args.n_ep}×{args.steps} · ceiling {args.n_states} states×{args.R} "
          f"repeats · τ applied to assignment+phase (power/Ck sampled)")
    print("=" * 76)

    tp = TempPolicy(args.ckpt, cfg, n_shots=args.n_shots, seed=args.seed)

    rows = []
    for tau in args.taus:
        t0 = time.time()
        r = sweep_one_tau(tp, cfg, tau, args, gamma)
        r['secs'] = time.time() - t0
        rows.append(r)
        tag = 'greedy' if tau <= TAU_GREEDY else f'{tau:.2f}'
        print(f"  τ={tag:>6} | H(assign)={r['ent_assign']:.3f} | "
              f"ceiling={r['ceiling']:+.3f} | EV_critic={r['ev_critic']:+.3f} | "
              f"gap={r['gap']:+.3f} | σtot={r['sigma_total']:.2f} "
              f"σintra={r['sigma_intra']:.2f} | corr={r['corr_V_ret']:+.2f} "
              f"({r['secs']:.0f}s)")

    # ── verdict ──
    print("-" * 76)
    rs = sorted(rows, key=lambda x: x['tau'])           # low τ (sharp) → high τ
    sharp = rs[0]; base = next((x for x in rs if abs(x['tau'] - 1.0) < 1e-6), rs[-1])
    d_ceiling = sharp['ceiling'] - base['ceiling']
    d_critic  = sharp['ev_critic'] - base['ev_critic']
    gap_sharp = sharp['gap']; gap_base = base['gap']
    print(f"  Δceiling (sharp−base) = {d_ceiling:+.3f} | "
          f"ΔEV_critic = {d_critic:+.3f} | gap base={gap_base:+.3f} → sharp={gap_sharp:+.3f}")
    if d_ceiling < 0.05:
        print("  → CEILING ~FLAT vs entropy: T3 weak here (power/Ck/env dominate variance,"
              " or τ-shift too small). EV_max is genuinely capped → close direction.")
    elif (gap_sharp - gap_base) < 0.08:
        print("  → critic TRACKS the rising ceiling (gap stays small) → critic HEALTHY across"
              " entropy → bottleneck is POLICY → close critic + dead-units, pivot actor/head.")
    else:
        print("  → critic FALLS OFF rising ceiling (gap widens) → underfit CANDIDATE → run"
              " Step B (capacity / input-norm / LeakyReLU). ⚠ first rule out dist-shift via"
              " fresh-refit control before trusting this.")
    print("=" * 76)


if __name__ == '__main__':
    main()
