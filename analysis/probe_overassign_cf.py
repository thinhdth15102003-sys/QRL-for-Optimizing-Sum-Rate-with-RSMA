"""
probe_overassign_cf.py
----------------------
Policy-conditioned OVER-ASSIGNMENT counterfactual + reward NON-STATIONARITY
decomposition, for a trained checkpoint (built for the ckpt's OWN case via its
hyperparameters.json, so safe to run while params.py is set to another case).

PART A — OVER-ASSIGNMENT COUNTERFACTUAL  (answers the user's question)
  The policy sometimes routes NON-BLOCKED (free) users onto an IRS. For every
  step we take the policy's assignment and build a counterfactual:
      a_cf = a_policy, but every FREE user that was on an IRS → moved to DIRECT.
  Then we score policy / counterfactual / oracle assignments under the SAME
  oracle-phase + reference-power recipe (the apples-to-apples assignment test
  used by probe_assignment_quality), so the ONLY thing that differs is where the
  free users sit. We report:
    · QoS & reward(=ΣR − λ·#missed) for policy vs cf vs oracle
    · how often cf ≥ policy (does sending free users to direct help?)
    · the flipped users' own fate (met on IRS vs met on direct)
    · collateral: do the BLOCKED users sharing that IRS improve once the free
      users leave (common-stream un-dilution)?

PART B — NON-STATIONARITY / REWARD-VARIANCE decomposition
  Why do per-episode rewards swing so much? We separate:
    · TASK churn   : oracle reward varies episode-to-episode because each reset
                     hands a different spawn (positions + which-building blocked
                     users land in → per-IRS load 3-vs-4). std(oracle) = the
                     irreducible difficulty churn the agent cannot control.
    · NOISE        : σ² ~ N(-30,10 dBW) per step; we re-score one fixed step with
                     many fresh noise draws → reward std from noise (raw and after
                     reward_noise_avg averaging).
    · per-IRS load imbalance distribution (the over-load driver).

Usage:
  python analysis/probe_overassign_cf.py --ckpt results/result_35/checkpoints/ep_00000
  python analysis/probe_overassign_cf.py --ckpt <dir> --episodes 80 --steps 12
"""
import sys, os, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx
from analysis.probe_assignment_quality import _oracle_assignment


def _eval_full(assignment, ch, cfg, rate, sigma2=None):
    """Per-user (met_vec, Rtot_vec, ΣR) under oracle phase + reference power.
    Same recipe as probe_assignment_quality._qos_of but returns per-user arrays."""
    K, M, N = cfg.K, cfg.M, cfg.N
    assignment = np.asarray(assignment, dtype=int)
    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    pidx = oracle_phase_idx(est, assignment, cfg)
    Phi = np.zeros((M, N, N), dtype=complex)
    di = np.arange(N)
    Phi[:, di, di] = np.exp(1j * cfg.phase_levels[pidx])
    active = sorted(set(int(a) for a in assignment if a > 0))
    G = len(active)
    w_p = np.full(K, cfg.P_S * 0.8 / K)
    w_c_vec = np.full(G + 1, cfg.P_S * 0.2 / (G + 1))
    out = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                w_p=w_p, w_c_vec=w_c_vec,
                                active_irs_ids=active,
                                sigma2=cfg.sigma2 if sigma2 is None else sigma2)
    Rtot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
    return (Rtot >= cfg.D_k_bps_hz), Rtot, float(np.sum(Rtot))


def _reward(met_vec, sumR, lam):
    return sumR - lam * int(np.sum(~met_vec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', type=int, default=80)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--lambda-D', dest='lam', type=float, default=1.5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise-draws', type=int, default=200)
    args = ap.parse_args()

    # ── build cfg for the CKPT's own case ───────────────────────────────────
    d = os.path.abspath(args.ckpt); hp = None
    for _ in range(4):
        d = os.path.dirname(d)
        cand = os.path.join(d, 'hyperparameters.json')
        if os.path.isfile(cand):
            hp = json.load(open(cand)); break
    if hp is None:
        raise SystemExit(f"hyperparameters.json not found near {args.ckpt}")
    s = hp['system']
    cfg = make_config(K=s['K'], M=s['M'], N=s['N'], P_S_dBm=s['P_S_dBm'],
                      R_LoS_km=s['R_LoS_km'], D_k_bps_hz=s['D_k_bps_hz'])
    rate = RateComputer(cfg); K, M, Dk, lam = cfg.K, cfg.M, cfg.D_k_bps_hz, args.lam

    from probe_critic_ceiling import make_checkpoint_policy
    policy = make_checkpoint_policy(args.ckpt, cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps,
                  reward_noise_avg=16)

    # accumulators ----------------------------------------------------------
    q_pol, q_cf, q_ora = [], [], []          # per-step QoS (eval recipe)
    r_pol, r_cf, r_ora = [], [], []          # per-step reward
    n_over_steps = []                        # #free-on-IRS per step
    flip_on_irs_met = flip_on_dir_met = flip_n = 0   # flipped users' fate
    blk_share_before = blk_share_after = blk_share_n = 0  # collateral on blocked
    cf_ge_pol_over = cf_better_over = n_over_with = 0
    ep_oracle_R, ep_agent_R = [], []         # per-episode reward (variance)
    load_max = []                            # per-step max per-IRS load
    one_ch = None                            # snapshot for noise probe

    ep_oracle_q = []                         # per-episode oracle QoS (separate)
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        ep_or = ep_ag = ep_orq = 0.0
        for _ in range(args.steps):
            ch = {k: (v.copy() if hasattr(v, 'copy') else v)
                  for k, v in env.channels.items()}
            blk = ch['su_blocked'].astype(bool)
            a = policy(env)
            a_pol = np.asarray(a['assignment'], dtype=int)
            if one_ch is None:
                one_ch = (ch, a_pol.copy())

            # counterfactual: free users on IRS → direct
            a_cf = a_pol.copy()
            free_on_irs = (~blk) & (a_pol > 0)
            a_cf[free_on_irs] = 0
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)

            mp, _, sp = _eval_full(a_pol, ch, cfg, rate)
            mc, _, sc = _eval_full(a_cf, ch, cfg, rate)
            mo, _, so = _eval_full(a_or, ch, cfg, rate)
            rp, rc, ro = _reward(mp, sp, lam), _reward(mc, sc, lam), _reward(mo, so, lam)

            q_pol.append(mp.mean()); q_cf.append(mc.mean()); q_ora.append(mo.mean())
            r_pol.append(rp); r_cf.append(rc); r_ora.append(ro)
            n_ov = int(free_on_irs.sum()); n_over_steps.append(n_ov)
            load_max.append(max((int(np.sum(a_pol == (m + 1))) for m in range(M)), default=0))

            if n_ov:
                n_over_with += 1
                cf_ge_pol_over += int(rc >= rp - 1e-9)
                cf_better_over += int(rc > rp + 1e-9)
                # flipped users' own fate (IRS vs direct)
                idx = np.where(free_on_irs)[0]
                flip_n += len(idx)
                flip_on_irs_met += int(mp[idx].sum())
                flip_on_dir_met += int(mc[idx].sum())
                # collateral: blocked users sharing an over-loaded IRS
                over_irs = set(int(a_pol[k]) for k in idx)
                share = np.array([blk[k] and (a_pol[k] in over_irs) for k in range(K)])
                if share.any():
                    blk_share_n += int(share.sum())
                    blk_share_before += int(mp[share].sum())
                    blk_share_after += int(mc[share].sum())

            ep_or += ro; ep_orq += mo.mean()
            _, rew, _, info = env.step(a)
            ep_ag += float(np.mean(np.asarray(info['R_tot']) >= Dk))
        ep_oracle_R.append(ep_or / args.steps)
        ep_oracle_q.append(ep_orq / args.steps)
        ep_agent_R.append(ep_ag / args.steps)

    # ── PART A report ──────────────────────────────────────────────────────
    pct = lambda x: 100.0 * np.mean(x)
    print("=" * 80)
    print(f"  OVER-ASSIGNMENT COUNTERFACTUAL · {args.ckpt}")
    print(f"  Case K={K} M={M} N={cfg.N} P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} "
          f"D_k={Dk} λ={lam} · {args.episodes}ep×{args.steps}step")
    print("=" * 80)
    print("  [eval recipe = oracle-phase + reference-power; only ASSIGNMENT differs]")
    print(f"  QoS    policy(as-is)        : {pct(q_pol):5.1f}%   reward {np.mean(r_pol):+7.3f}")
    print(f"  QoS    counterfactual(→dir) : {pct(q_cf):5.1f}%   reward {np.mean(r_cf):+7.3f}"
          f"   Δ {np.mean(r_cf)-np.mean(r_pol):+.3f}")
    print(f"  QoS    oracle               : {pct(q_ora):5.1f}%   reward {np.mean(r_ora):+7.3f}")
    print("  " + "-" * 76)
    print(f"  over-assign: {np.mean(n_over_steps):.2f} free users on IRS / step "
          f"({pct([x > 0 for x in n_over_steps]):.0f}% of steps have ≥1)")
    if n_over_with:
        print(f"  on steps WITH over-assign ({n_over_with}):")
        print(f"     cf ≥ policy reward : {100*cf_ge_pol_over/n_over_with:5.1f}%   "
              f"cf > policy : {100*cf_better_over/n_over_with:5.1f}%")
    if flip_n:
        print(f"  flipped (free) users' fate ({flip_n}):  "
              f"met ON IRS {100*flip_on_irs_met/flip_n:5.1f}%  →  "
              f"met ON DIRECT {100*flip_on_dir_met/flip_n:5.1f}%")
    if blk_share_n:
        print(f"  collateral on BLOCKED sharing that IRS ({blk_share_n}):  "
              f"met before {100*blk_share_before/blk_share_n:5.1f}%  →  "
              f"after free leave {100*blk_share_after/blk_share_n:5.1f}%")
    # ── PART B report ──────────────────────────────────────────────────────
    print("=" * 80)
    print("  NON-STATIONARITY / REWARD-VARIANCE decomposition")
    print("-" * 80)
    oR, oQ, aR = np.array(ep_oracle_R), np.array(ep_oracle_q), np.array(ep_agent_R)
    print(f"  per-episode oracle reward: {oR.mean():+.3f} ± {oR.std():.3f} "
          f"(min {oR.min():+.2f} / max {oR.max():+.2f})  ← TASK-difficulty churn (best-achievable)")
    print(f"  per-episode oracle QoS   : {oQ.mean()*100:5.1f}% ± {oQ.std()*100:4.1f}pp "
          f"(min {oQ.min()*100:.0f} / max {oQ.max()*100:.0f})")
    print(f"  per-episode agent  QoS   : {aR.mean()*100:5.1f}% ± {aR.std()*100:4.1f}pp "
          f"(min {aR.min()*100:.0f} / max {aR.max()*100:.0f})")
    lm = np.array(load_max)
    print(f"  per-IRS max load       : {lm.mean():.2f} users ± {lm.std():.2f} "
          f"(max seen {lm.max()})  ← over-load churn (oracle target ≈ {int(round(K*2/3/M))}/IRS)")
    # noise contribution on one fixed (spawn, oracle-assignment)
    if one_ch is not None:
        ch0, _ = one_ch
        a0 = _oracle_assignment(ch0, env.user_pos, env.irs_pos)  # any fixed assign
        draws = []
        for _ in range(args.noise_draws):
            s2 = env.channel_model.sample_noise_sigma2()
            m, _, sR = _eval_full(a0, ch0, cfg, rate, sigma2=s2)
            draws.append(_reward(m, sR, lam))
        draws = np.array(draws)
        print(f"  NOISE on fixed step    : reward {draws.mean():+.3f} ± {draws.std():.3f} "
              f"(raw 1-draw)  →  ÷√16 ≈ ±{draws.std()/4:.3f} (after reward_noise_avg)")
    print("=" * 80)


if __name__ == "__main__":
    main()
