"""
probe_assignment_oracle.py
--------------------------
Augmented oracle-assignment probe — is the assignment a real bottleneck, and on the
RIGHT objective? Discriminates the F1 family (gap is artifact) from F2/F3 (real gap).

Levels per state (downstream phase/power/Ck RE-RUN each time → power adapts, fixes the
blocked_direct fixed-w_c limitation):
  • SAMPLED   : policy's sampled assignment
  • GREEDY    : policy's argmax assignment            [A5: sampled→greedy = exploration]
  • ORACLE-rew: coordinate-ascent maximizing reward=R_tot−λ_D·qp  (policy's TRUE objective)
  • ORACLE-Rt : coordinate-ascent maximizing R_tot     [A6: compare IRS-share of the two
                oracles — if oracle-rew routes LESS IRS than oracle-Rt, λ_D objective itself
                wants less IRS ⇒ policy is right, R_tot-swap was misleading]
  • +oracle-phase: on ORACLE-rew assignment, brute 4^M uniform phase  [A8: externality-aware]

A7 (risk-aversion): during ascent, per user record (chose_IRS?, Δmean, Δstd) of direct vs
  best-IRS option → does policy avoid high-variance IRS at equal mean?
A8 (externality): coordinate ascent is MULTI-user (stops adding to IRS when group phase
  compromise outweighs gain) → captures cumulative phase tax single-swap missed.
A10 (non-stationarity): run on ep600 AND ep1200 → did the oracle-optimal assignment MOVE?

Usage:
  python analysis/probe_assignment_oracle.py --ckpts results/result_11/checkpoints/ep_00600 results/result_11/checkpoints/ep_01200
"""

# ── path bootstrap ──────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import time
import itertools
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_critic_ceiling import make_checkpoint_policy


def _load_nets(ckpt_dir):
    if os.path.isdir(os.path.join(ckpt_dir, 'agents')) and \
       not os.path.isfile(os.path.join(ckpt_dir, 'actor_config.json')):
        ckpt_dir = os.path.join(ckpt_dir, 'agents')
    from RL import QuantumActor, PhaseMLP, PowerMLP, CkMLP
    actor    = QuantumActor.from_dir(ckpt_dir, seed=0)
    phase_net = PhaseMLP.from_dir(ckpt_dir, seed=0)
    power_net = PowerMLP.from_dir(ckpt_dir, seed=0)
    ck_net    = CkMLP.from_dir(ckpt_dir, seed=0)
    actor.n_shots = getattr(P, 'n_shots_train', 1500)
    return actor, (phase_net, phase_net, power_net, ck_net)  # phase twice placeholder


def _downstream(env, nets, phi, cfg, z_t):
    """Re-run phase/power/Ck for assignment phi. Returns (phase_idx, w_p, w_c_vec, C_k, active_ids)."""
    _, phase_net, power_net, ck_net = nets
    from train import _build_phase_state, _build_ck_state, _get_active_irs
    active_irs = _get_active_irs(phi)
    active_ids = [int(a) + 1 for a in active_irs]
    s_phase = _build_phase_state(env.channels, phi, cfg, z_t)
    phase_idx, _, _ = phase_net.forward(s_phase, active_irs)
    Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(phase_idx))
    h_eff = env.rate_computer.effective_channels_all(phi, Phi, env.channels)
    s_power = np.concatenate([h_eff.real, h_eff.imag, z_t])
    w_c_vec, w_p, _, _ = power_net.forward(s_power, active_ids)
    partial = env.rate_computer.compute_rates_partial(
        phi, Phi, env.channels, w_p, w_c_vec, active_irs_ids=active_ids)
    s_ck = _build_ck_state(np.full(cfg.K, cfg.D_k_bps_hz),
                           partial['R_private'], partial['R_c_group'], phi, cfg)
    C_k, _, _ = ck_net.forward(s_ck, phi, partial['R_c_group'])
    return phase_idx, w_p, w_c_vec, C_k, active_ids


def _eval(env, nets, phi, cfg, z_t, sigmas, D_k, lamD, phase_override=None):
    """Re-run downstream + eval over noise. Returns dict(R_tot, qp, reward, QoS, R_user, R_user_std, IRS_share, phi, phase_idx)."""
    try:
        phase_idx, w_p, w_c_vec, C_k, active_ids = _downstream(env, nets, phi, cfg, z_t)
    except Exception:
        return None
    pidx = phase_override if phase_override is not None else phase_idx
    Phi = env.phase_model.build_phi(env.phase_model.index_to_phase(pidx))
    draws = []
    for s2 in sigmas:
        r = env.rate_computer.compute_sum_rate(phi, Phi, env.channels, w_p, w_c_vec,
                                               C_k=C_k, active_irs_ids=active_ids, sigma2=s2)
        draws.append(np.asarray(r['R_private']) + np.asarray(r['C_k']))
    draws = np.array(draws)
    R_user = draws.mean(0); R_user_std = draws.std(0)
    R_tot = float(R_user.sum()); qp = float(np.maximum(0.0, D_k - R_user).sum())
    return dict(R_tot=R_tot, qp=qp, reward=R_tot - lamD * qp,
                QoS=float(100*(R_user >= D_k).mean()), R_user=R_user, R_user_std=R_user_std,
                IRS_share=float((phi > 0).mean()), phi=phi.copy(), phase_idx=phase_idx)


def _coord_ascent(env, nets, phi_init, cfg, z_t, sigmas, D_k, lamD, M, objective,
                  max_passes=2, risk_rec=None, phi_pol=None):
    """Greedy coordinate ascent over per-user assignment maximizing `objective`."""
    phi = phi_init.copy()
    cur = _eval(env, nets, phi, cfg, z_t, sigmas, D_k, lamD)
    if cur is None:
        return None
    K = cfg.K
    for p in range(max_passes):
        changed = False
        for u in range(K):
            opt_metrics = {}
            for a in range(0, M + 1):
                phi_try = phi.copy(); phi_try[u] = a
                m = _eval(env, nets, phi_try, cfg, z_t, sigmas, D_k, lamD)
                if m is not None:
                    opt_metrics[a] = m
            if not opt_metrics:
                continue
            # A7 risk recording (first pass only): direct(a=0) vs best-IRS by mean-R_user[u]
            if risk_rec is not None and p == 0 and 0 in opt_metrics:
                irs_opts = {a: mm for a, mm in opt_metrics.items() if a >= 1}
                if irs_opts:
                    a_irs = max(irs_opts, key=lambda a: irs_opts[a]['R_user'][u])
                    md = opt_metrics[0]['R_user'][u];   sd = opt_metrics[0]['R_user_std'][u]
                    mi = irs_opts[a_irs]['R_user'][u];  si = irs_opts[a_irs]['R_user_std'][u]
                    risk_rec['chose_irs'].append(int(phi_pol[u] > 0))
                    risk_rec['dmean'].append(float(mi - md))
                    risk_rec['dstd'].append(float(si - sd))
            best_a = max(opt_metrics, key=lambda a: opt_metrics[a][objective])
            if opt_metrics[best_a][objective] > cur[objective] + 1e-9 and best_a != phi[u]:
                phi[u] = best_a; cur = opt_metrics[best_a]; changed = True
        if not changed:
            break
    return cur


def run_ckpt(ckpt, cfg, args, lamD, D_k):
    M, K = cfg.M, cfg.K
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps + args.warmup + 2,
                  reward_noise_avg=1)
    policy = make_checkpoint_policy(ckpt, cfg)
    actor, nets = _load_nets(ckpt)

    rows = {k: [] for k in ['samp', 'greedy', 'orw', 'ort', 'orw_oph']}
    risk = {'chose_irs': [], 'dmean': [], 'dstd': []}
    t0 = time.time()
    for ep in range(args.episodes):
        env.reset(seed=args.seed * 19 + ep)
        for _ in range(args.warmup):
            env.step(policy(env))
        for _ in range(args.steps):
            obs = env._get_obs()
            blocked = env.channels['su_blocked'].astype(int)
            s_t = actor.extract_state(obs, np.full(K, D_k), blocked)
            phi_pol, _, info = actor.forward(s_t)
            z_t = info['z_t']
            phi_greedy, _, _ = actor.forward(s_t, greedy=True)
            sigmas = [env.channel_model.sample_noise_sigma2() for _ in range(args.noise_avg)]

            m_samp = _eval(env, nets, phi_pol, cfg, z_t, sigmas, D_k, lamD)
            m_greedy = _eval(env, nets, phi_greedy, cfg, z_t, sigmas, D_k, lamD)
            m_orw = _coord_ascent(env, nets, phi_greedy, cfg, z_t, sigmas, D_k, lamD, M,
                                  'reward', args.passes, risk, phi_pol)
            m_ort = _coord_ascent(env, nets, phi_greedy, cfg, z_t, sigmas, D_k, lamD, M,
                                  'R_tot', args.passes)
            if any(x is None for x in [m_samp, m_greedy, m_orw, m_ort]):
                continue
            # oracle-rew assignment + oracle 4^M uniform phase
            best_ph = m_orw['phase_idx']; best_R = m_orw['R_tot']
            phi_orw = m_orw['phi']
            for combo in itertools.product(range(env.n_phase_levels), repeat=M):
                ph = np.zeros((M, cfg.N), dtype=int)
                for mm, lev in enumerate(combo):
                    ph[mm, :] = lev
                mm_ = _eval(env, nets, phi_orw, cfg, z_t, sigmas, D_k, lamD, phase_override=ph)
                if mm_ and mm_['R_tot'] > best_R:
                    best_R = mm_['R_tot']; m_orw_oph = mm_
                    best_ph = ph
            m_orw_oph = _eval(env, nets, phi_orw, cfg, z_t, sigmas, D_k, lamD, phase_override=best_ph)

            for key, m in [('samp', m_samp), ('greedy', m_greedy), ('orw', m_orw),
                           ('ort', m_ort), ('orw_oph', m_orw_oph)]:
                rows[key].append((m['R_tot'], m['QoS'], m['reward'], m['IRS_share']))
        if (ep + 1) % max(1, args.episodes // 4) == 0:
            print(f"    {os.path.basename(ckpt)}: {ep+1}/{args.episodes} ep ({time.time()-t0:.0f}s)")

    out = {}
    for k, v in rows.items():
        a = np.array(v)
        out[k] = dict(R_tot=a[:, 0].mean(), QoS=a[:, 1].mean(),
                      reward=a[:, 2].mean(), IRS=a[:, 3].mean())
    out['N'] = len(rows['samp'])
    out['risk'] = {k: np.array(v) for k, v in risk.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+',
                    default=['results/result_11/checkpoints/ep_00600',
                             'results/result_11/checkpoints/ep_01200'])
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--noise_avg', type=int, default=5)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20260603)
    ap.add_argument('--out', default='results/result_11/assignment_oracle.txt')
    args = ap.parse_args()

    cfg = make_config(); D_k = cfg.D_k_bps_hz; lamD = float(getattr(P, 'lambda_D', 1.5))
    print("=" * 80)
    print(f"  ASSIGNMENT-ORACLE (augmented A5/A6/A7/A8/A10)  ·  K={cfg.K} M={cfg.M} N={cfg.N}")
    print(f"  λ_D={lamD} D_k={D_k:.2f} · {args.episodes}ep×{args.steps}st noise={args.noise_avg} passes={args.passes}")
    print("=" * 80)

    L = []
    results = {}
    for ckpt in args.ckpts:
        print(f"  running {ckpt} ...")
        results[ckpt] = run_ckpt(ckpt, cfg, args, lamD, D_k)

    L.append("=" * 80)
    L.append("  ASSIGNMENT-ORACLE RESULTS")
    L.append("=" * 80)
    for ckpt, r in results.items():
        tag = os.path.basename(ckpt)
        L.append(f"\n  ── {tag}  (N={r['N']} states) ──")
        L.append(f"  {'level':14s} {'R_tot':>7s} {'QoS%':>6s} {'reward':>8s} {'IRS%':>6s}")
        for key, nm in [('samp', 'sampled'), ('greedy', 'greedy-policy'),
                        ('orw', 'oracle-reward'), ('ort', 'oracle-Rtot'),
                        ('orw_oph', 'orw+oracle-φ')]:
            d = r[key]
            L.append(f"  {nm:14s} {d['R_tot']:7.3f} {d['QoS']:6.1f} {d['reward']:+8.3f} {100*d['IRS']:6.1f}")
        # gaps
        g_sg = r['greedy']['reward'] - r['samp']['reward']
        g_go = r['orw']['reward'] - r['greedy']['reward']
        g_a6 = r['ort']['IRS'] - r['orw']['IRS']          # IRS-share: Rtot-oracle minus reward-oracle
        g_a8 = r['orw_oph']['R_tot'] - r['orw']['R_tot']
        L.append(f"    Δ(sampled→greedy) reward = {g_sg:+.3f}   [A5 exploration]")
        L.append(f"    Δ(greedy→oracle-rew) rew = {g_go:+.3f}   [real head gap on TRUE objective]")
        L.append(f"    A6: IRS%(oracle-Rtot {100*r['ort']['IRS']:.0f}) − IRS%(oracle-rew {100*r['orw']['IRS']:.0f}) = {100*g_a6:+.1f}pt")
        L.append(f"        (>0 ⇒ R_tot wants MORE IRS than reward does ⇒ λ_D objective justifies policy's lower IRS)")
        L.append(f"    A8: oracle-φ lifts R_tot by {g_a8:+.3f} on the oracle assignment [phase still the cap]")
        L.append(f"    policy IRS% {100*r['samp']['IRS']:.0f}  vs  oracle-rew IRS% {100*r['orw']['IRS']:.0f}  "
                 f"(Δ={100*(r['orw']['IRS']-r['samp']['IRS']):+.0f} = under/over-use)")
        # A7 risk
        rr = r['risk']
        if len(rr['chose_irs']) >= 30:
            ci = rr['chose_irs']; dm = rr['dmean']; ds = rr['dstd']
            both = (dm > 0)                                   # IRS higher mean
            L.append(f"    A7 risk: among users where IRS has higher MEAN (n={both.sum()}):")
            if both.sum() >= 10:
                hi_std = both & (ds > np.median(ds[both]))
                lo_std = both & (ds <= np.median(ds[both]))
                L.append(f"        chose-IRS frac | low Δstd: {100*ci[lo_std].mean():.0f}%  ·  "
                         f"high Δstd: {100*ci[hi_std].mean():.0f}%")
                rho = float(np.corrcoef(ci, ds)[0, 1])
                L.append(f"        corr(chose_IRS, Δstd) = {rho:+.3f}  "
                         f"(<0 ⇒ policy avoids high-variance IRS = RISK-AVERSION [A7])")
    # A10 cross-ckpt
    if len(args.ckpts) >= 2:
        c0, c1 = args.ckpts[0], args.ckpts[1]
        L.append(f"\n  ── A10 non-stationarity: oracle-rew IRS% across ckpts ──")
        L.append(f"    {os.path.basename(c0)}: oracle-rew IRS% = {100*results[c0]['orw']['IRS']:.1f}  "
                 f"(policy {100*results[c0]['samp']['IRS']:.1f})")
        L.append(f"    {os.path.basename(c1)}: oracle-rew IRS% = {100*results[c1]['orw']['IRS']:.1f}  "
                 f"(policy {100*results[c1]['samp']['IRS']:.1f})")
        d_or = results[c1]['orw']['IRS'] - results[c0]['orw']['IRS']
        d_pol = results[c1]['samp']['IRS'] - results[c0]['samp']['IRS']
        L.append(f"    Δoracle-IRS = {100*d_or:+.1f}pt · Δpolicy-IRS = {100*d_pol:+.1f}pt")
        L.append(f"    → oracle moved {'UP' if d_or>0 else 'DOWN'} while policy moved {'UP' if d_pol>0 else 'DOWN'}:")
        if d_or > 0.02 and d_pol < 0:
            L.append(f"      ⚠ oracle wants MORE IRS over time but policy went LESS = NON-STATIONARITY LAG [A10] supported.")
        elif abs(d_or) < 0.03:
            L.append(f"      oracle target ~stable → not a moving-target story; lag (if any) is credit/repr.")
    L.append("=" * 80)

    report = "\n".join(L)
    print("\n" + report)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print(f"\n  report → {args.out}")


if __name__ == '__main__':
    main()
