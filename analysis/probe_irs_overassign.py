"""
probe_irs_overassign.py
-----------------------
Does the policy OVER-ASSIGN users to the IRS at R_LoS=0.5 (Case 1), degrading
overall QoS — and would lowering / dropping IRS-routing priority help?

Two parts (CPU; loads ckpt for part 2):
 (1) PHYSICAL  — per-user IRS-OPTIMAL gain vs DIRECT gain (blocked / non-blocked).
      Answers: at this R_LoS, do NON-BLOCKED users physically benefit from IRS?
      If direct >= IRS-opt for non-blocked users, routing them to IRS is WASTEFUL.
      NOTE (UPDATED post per-element refactor): g_RU is now (M,N,K), so each
      element has its own channel and one phase setting can NOT co-phase every
      IRS user at once — a real multi-user phase tradeoff now exists. Group
      degradation may therefore come from (a) non-blocked users on a worse link,
      (b) RSMA power-sharing, AND (c) phase conflict. The old claim that (c) was
      impossible held only under the scalar g_RU model.
 (2) POLICY    — run the trained ckpt; decompose QoS-satisfaction by assignment
      (IRS/DIR × blk/non), count non-blocked users routed to IRS, and bin per-step
      by n_irs (#users on the single IRS) to see if the IRS GROUP's per-user rate
      and QoS-met fall as more users pile onto it.

Env is built for the CHECKPOINT's case via explicit overrides (NOT the params
ACTIVE CASE), so it is safe to run while params.py is set to another case.

Usage:
  python analysis/probe_irs_overassign.py \
      --ckpt results/result_18/checkpoints/ep_00400 --episodes 12
  python analysis/probe_irs_overassign.py --skip-policy        # physics only (fast)
"""

# ── path bootstrap: make project root importable when run as script ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import argparse, collections
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from analysis.phase_oracle import irs_optimal_gain_mag

# Case-1 R_LoS=0.5 env — matches result_18 / result_20 hyperparameters.json
# (Case 1 here used P_S=70 dBm, NOT the stale auto-derived 50).
CASE1_R05 = dict(K=5, M=1, N=24, P_S_dBm=70.0, R_LoS_km=0.5,
                 kappa=0.05, D_k_bps_hz=0.1)


def physical(cfg, n_samples=400, seed=20260530):
    N = cfg.N
    env = ISTNEnv(cfg=cfg, seed=seed, n_steps_ep=1, reward_noise_avg=1)
    gd_all, gi_all, blk_all = [], [], []
    for _ in range(n_samples):
        env.reset()
        ch = env.channels
        gd = np.abs(ch['g_SU_hat']) ** 2                          # (K,) direct gain
        gi = irs_optimal_gain_mag(ch) ** 2                         # (M,K) IRS-opt gain
        gi = gi.max(axis=0)                                        # best IRS per user
        gd_all.append(gd); gi_all.append(gi)
        blk_all.append(np.asarray(ch['su_blocked'], dtype=bool))
    gd  = np.concatenate(gd_all); gi = np.concatenate(gi_all)
    blk = np.concatenate(blk_all)
    ratio_db = 10.0 * np.log10((gi + 1e-30) / (gd + 1e-30))

    def rep(mask, label):
        if mask.sum() == 0:
            print(f"  {label:<16}: (none)"); return
        win = float(np.mean(gi[mask] > gd[mask])) * 100.0
        print(f"  {label:<16}: IRS-opt>direct {win:5.1f}% | ratio dB "
              f"med={np.median(ratio_db[mask]):+5.1f} "
              f"[p25 {np.percentile(ratio_db[mask],25):+.1f}, "
              f"p75 {np.percentile(ratio_db[mask],75):+.1f}]")

    print("=" * 76)
    print(f"  (1) PHYSICAL IRS-opt vs DIRECT · K={cfg.K} M={cfg.M} N={N} "
          f"P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km}km")
    print(f"      Blocked fraction: {blk.mean()*100:.1f}%   ({len(gd)} user-instances)")
    print("-" * 76)
    rep(np.ones_like(blk), "ALL"); rep(blk, "BLOCKED"); rep(~blk, "NON-BLOCKED")
    win_nb = float(np.mean(gi[~blk] > gd[~blk])) * 100.0 if (~blk).sum() else 0.0
    print("-" * 76)
    print(f"  → NON-BLOCKED users better on IRS: {win_nb:.0f}%  "
          + ("(IRS helps them too)" if win_nb >= 50 else
             "(DIRECT better for most → routing them to IRS is WASTEFUL over-assignment)"))
    print("=" * 76)


def policy_probe(cfg, ckpt, episodes, steps, seed=20260601):
    from probe_critic_ceiling import make_checkpoint_policy
    K = cfg.K; D_k = cfg.D_k_bps_hz
    env = ISTNEnv(cfg=cfg, seed=seed, n_steps_ep=steps,
                  reward_noise_avg=getattr(P, 'reward_noise_avg', 16))
    print(f"\nLoading policy {ckpt} ...")
    policy = make_checkpoint_policy(ckpt, cfg)

    tot = collections.Counter(); met = collections.Counter()
    bin_rate = collections.defaultdict(list)             # n_irs → [per-IRS-user R_tot]
    bin_met  = collections.Counter(); bin_tot = collections.Counter()
    irs_counts = []
    nonblk_to_irs = 0; total_irs = 0

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for _ in range(steps):
            blocked = env.channels['su_blocked'].copy()
            a = policy(env)
            _, _, _, info = env.step(a)
            assign = np.asarray(info['assignment']); Rtot = np.asarray(info['R_tot'])
            n_irs = int(np.sum(assign > 0)); irs_counts.append(n_irs)
            for k in range(K):
                grp = 'IRS' if assign[k] > 0 else 'DIR'
                blk = 'blk' if blocked[k] else 'non'
                tot[(grp, blk)] += 1; tot[(grp, 'all')] += 1; tot[('ALL', 'all')] += 1
                if Rtot[k] >= D_k:
                    met[(grp, blk)] += 1; met[(grp, 'all')] += 1; met[('ALL', 'all')] += 1
                if grp == 'IRS':
                    total_irs += 1
                    if blk == 'non':
                        nonblk_to_irs += 1
                    bin_rate[n_irs].append(float(Rtot[k]))
                    bin_tot[n_irs] += 1
                    if Rtot[k] >= D_k:
                        bin_met[n_irs] += 1

    def rate(key):
        return 100.0 * met[key] / tot[key] if tot[key] else float('nan')

    step_count = collections.Counter(irs_counts)
    print("=" * 76)
    print(f"  (2) POLICY QoS by ASSIGNMENT · ckpt={ckpt} · {episodes}ep×{steps}step")
    print(f"      D_k={D_k} · IRS/K mean={np.mean(irs_counts):.2f}/{K}")
    print("-" * 76)
    print(f"  OVERALL QoS         : {rate(('ALL','all')):5.1f}%  (n={tot[('ALL','all')]})")
    print(f"  IRS-assigned QoS    : {rate(('IRS','all')):5.1f}%  (n={tot[('IRS','all')]})")
    print(f"     IRS & blocked    : {rate(('IRS','blk')):5.1f}%  (n={tot[('IRS','blk')]})")
    print(f"     IRS & non-block  : {rate(('IRS','non')):5.1f}%  (n={tot[('IRS','non')]})")
    print(f"  Direct-assigned QoS : {rate(('DIR','all')):5.1f}%  (n={tot[('DIR','all')]})")
    print(f"     DIR & blocked    : {rate(('DIR','blk')):5.1f}%  (n={tot[('DIR','blk')]})")
    print(f"     DIR & non-block  : {rate(('DIR','non')):5.1f}%  (n={tot[('DIR','non')]})")
    print("-" * 76)
    frac_waste = 100.0 * nonblk_to_irs / total_irs if total_irs else 0.0
    print(f"  OVER-ASSIGNMENT: {frac_waste:.0f}% of IRS-routed users are NON-BLOCKED "
          f"({nonblk_to_irs}/{total_irs})")
    print("-" * 76)
    print("  IRS-GROUP quality vs n_irs (#users on the single IRS):")
    print("    n_irs |  steps  | mean R/IRS-user | IRS-user QoS-met%")
    for n in sorted(bin_rate):
        r = float(np.mean(bin_rate[n]))
        q = 100.0 * bin_met[n] / bin_tot[n] if bin_tot[n] else float('nan')
        print(f"    {n:>5d} | {step_count[n]:>6d}  | {r:>13.3f}   | {q:>6.1f}%  (n={bin_tot[n]})")
    print("=" * 76)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_18/checkpoints/ep_00400')
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--samples', type=int, default=400)
    ap.add_argument('--skip-policy', action='store_true')
    args = ap.parse_args()

    cfg = make_config(**CASE1_R05)
    physical(cfg, n_samples=args.samples)
    if not args.skip_policy:
        policy_probe(cfg, args.ckpt, args.episodes, args.steps)


if __name__ == "__main__":
    main()
