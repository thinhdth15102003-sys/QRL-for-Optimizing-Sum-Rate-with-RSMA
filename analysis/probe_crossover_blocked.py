"""
probe_crossover_blocked.py
--------------------------
CONTROLLED crossover test (sanity-check the IRS oracle probe).

Scenario (fully controlled, not random sampling):
  • K=5, M=1, N=24 (Case 1).
  • ALL users forced INSIDE the single building's footprint → ALL blocked
    (direct link ×beta_blocking) AND co-located in the SAME building shadow
    (so g_RU to that building's IRS is similarly strong for everyone).

Sweep the assignment crossover:
  (5 direct, 0 IRS) → (4 direct, 1 IRS) → ... → (0 direct, 5 IRS)
and report R_tot under ORACLE sub-actors:
  • phase   : closed-form oracle (max coherent IRS gain, analysis/phase_oracle).
  • routing : oracle picks the n users with the largest IRS-over-direct margin.
  • power   : reference equal split (0.8/K private, 0.2/(G+1) common); C_k = R_c/n.

Reports per split: Σ R_tot, mean R_tot of the IRS group, of the (blocked) direct
users still on direct, and QoS-met% — so the crossover (when moving a blocked
user onto the IRS stops paying off) is visible. CPU-only.

Usage:
  python analysis/probe_crossover_blocked.py --K 5 --M 1 --N 24 --P 50
"""

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx


def _force_all_in_building0(env):
    """Place every user inside building/IRS 0's footprint (same shadow), regen
    channels. Returns channels dict; su_blocked should be all-True."""
    K = env.cfg.K
    env.user_pos = env._sample_in_footprints(np.zeros(K, dtype=int))
    env.channels = env.channel_model.generate(env.user_pos, env.irs_pos)
    return env.channels


def _eval_split(ch, cfg, rate, n_irs, select, rng):
    """Assign n_irs users to IRS#1 (oracle phase), rest direct; reference power.
    Returns dict of metrics for this split."""
    K, M, N = cfg.K, cfg.M, cfg.N
    g_dir = np.abs(ch['g_SU_hat']) ** 2
    coeff = ch['beta'][0] * np.abs(ch['g_SR_hat'][0]) * N
    g_irs = (coeff * np.abs(ch['g_RU_hat'][0])) ** 2
    margin = g_irs / (g_dir + 1e-30)
    order = np.argsort(-margin) if select == 'best' else rng.permutation(K)
    irs_users = np.sort(order[:n_irs])
    dir_users = np.sort(order[n_irs:])

    assignment = np.zeros(K, dtype=int)
    assignment[irs_users] = 1

    est = {'g_SR': ch['g_SR_hat'], 'g_RU': ch['g_RU_hat'],
           'g_SU': ch['g_SU_hat'], 'beta': ch['beta']}
    phase_idx = oracle_phase_idx(est, assignment, cfg)
    Phi = np.zeros((M, N, N), dtype=complex)
    di = np.arange(N)
    Phi[:, di, di] = np.exp(1j * cfg.phase_levels[phase_idx])

    G = 1 if n_irs > 0 else 0
    w_p = np.full(K, cfg.P_S * 0.8 / K)
    w_c_vec = np.full(G + 1, cfg.P_S * 0.2 / (G + 1))
    active = [1] if n_irs > 0 else []

    out = rate.compute_sum_rate(assignment=assignment, Phi=Phi, channels=ch,
                                w_p=w_p, w_c_vec=w_c_vec,
                                active_irs_ids=active, sigma2=cfg.sigma2)
    Rtot = out['R_private'] + out['C_k']
    Dk = cfg.D_k_bps_hz
    return {
        'sumR':   float(np.sum(Rtot)),
        'irs_mu': float(np.mean(Rtot[irs_users])) if n_irs > 0 else float('nan'),
        'dir_mu': float(np.mean(Rtot[dir_users])) if n_irs < K else float('nan'),
        'qos':    float(np.mean(Rtot >= Dk)),
        'Rtot':   Rtot.copy(), 'irs_users': irs_users, 'dir_users': dir_users,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--M', type=int, default=1)
    ap.add_argument('--N', type=int, default=24)
    ap.add_argument('--P', type=float, default=50.0)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--realizations', type=int, default=300)
    ap.add_argument('--select', choices=['best', 'random'], default='best')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    cfg = make_config(K=args.K, M=args.M, N=args.N, P_S_dBm=args.P, R_LoS_km=args.rlos)
    rate = RateComputer(cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=1, reward_noise_avg=1)
    rng = np.random.default_rng(args.seed)
    K = cfg.K

    splits = list(range(0, K + 1))
    agg = {n: {'sumR': [], 'irs_mu': [], 'dir_mu': [], 'qos': []} for n in splits}
    n_ok = 0
    sample = None
    for r in range(args.realizations):
        env.reset(seed=args.seed + r)
        ch = _force_all_in_building0(env)
        if not bool(np.all(ch['su_blocked'])):
            continue                                  # keep only all-blocked
        n_ok += 1
        per = {}
        for n in splits:
            m = _eval_split(ch, cfg, rate, n, args.select, rng)
            per[n] = m
            for k in ('sumR', 'irs_mu', 'dir_mu', 'qos'):
                if not np.isnan(m[k]):
                    agg[n][k].append(m[k])
        if sample is None:
            sample = per

    Dk = cfg.D_k_bps_hz
    print("=" * 84)
    print(f"  CONTROLLED CROSSOVER · K={K} M={cfg.M} N={cfg.N} P_S={cfg.P_S_dBm}dBm "
          f"R_LoS={cfg.R_LoS_km}  D_k={Dk}")
    print(f"  ALL users blocked + same building shadow · oracle phase + ref power · "
          f"{n_ok}/{args.realizations} all-blocked realizations")
    print("=" * 84)
    print(f"  {'split (dir/IRS)':>15} | {'Σ R_tot':>8} | {'IRS-grp R/u':>11} | "
          f"{'direct R/u':>10} | {'QoS-met%':>8}")
    print("  " + "-" * 78)
    sumR_by_n = {}
    for n in splits:
        sR = np.mean(agg[n]['sumR'])
        sumR_by_n[n] = sR
        im = np.mean(agg[n]['irs_mu']) if agg[n]['irs_mu'] else float('nan')
        dm = np.mean(agg[n]['dir_mu']) if agg[n]['dir_mu'] else float('nan')
        qo = np.mean(agg[n]['qos']) * 100
        tag = f"{K-n}d / {n}i"
        print(f"  {tag:>15} | {sR:>8.4f} | "
              f"{(f'{im:.4f}' if not np.isnan(im) else '   —   '):>11} | "
              f"{(f'{dm:.4f}' if not np.isnan(dm) else '   —   '):>10} | {qo:>7.1f}%")
    best_n = max(sumR_by_n, key=sumR_by_n.get)
    print("  " + "-" * 78)
    print(f"  Σ R_tot maximised at  {K-best_n} direct / {best_n} IRS  "
          f"(Σ R_tot = {sumR_by_n[best_n]:.4f})")
    # marginal gain of moving the n-th user onto the IRS
    print("  marginal ΔΣR_tot of moving one more user direct→IRS:")
    for n in range(1, K + 1):
        d = sumR_by_n[n] - sumR_by_n[n - 1]
        print(f"     {K-n+1}d/{n-1}i → {K-n}d/{n}i :  ΔΣR = {d:+.4f}  "
              f"{'(still pays off)' if d > 0 else '(crossover — stop)'}")
    print("=" * 84)

    if sample is not None:
        print(f"  SAMPLE realization — per-user R_tot (rows=split, cols=user 0..{K-1}):")
        for n in splits:
            rt = sample[n]['Rtot']
            cells = "  ".join(
                (f"[{v:.3f}]" if i in sample[n]['irs_users'] else f" {v:.3f} ")
                for i, v in enumerate(rt))
            print(f"     {K-n}d/{n}i:  {cells}   ΣR={np.sum(rt):.3f}")
        print("     ([x] = user routed to IRS;  D_k = "
              f"{Dk}  → R_tot below that = QoS fail)")
    print("=" * 84)


if __name__ == '__main__':
    main()
