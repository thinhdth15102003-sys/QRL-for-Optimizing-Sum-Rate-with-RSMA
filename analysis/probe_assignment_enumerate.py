"""
probe_assignment_enumerate.py
-----------------------------
Single-step ASSIGNMENT-QUALITY enumeration for a Case-2 checkpoint.

Question (user, 2026-06-18): pick a stable Case-2 ckpt, find a step where the
agent OVER-routes (IRS-routed count > blocked count, e.g. 7/10 > 6/10), and ask:
how many assignments are STRICTLY BETTER than the agent's, holding the other
sub-actors at their optimum?  Is pulling FREE users onto an IRS ever worth it,
or is it hit-or-miss?

Method: at the chosen step we BRUTE-FORCE all (M+1)^K assignments (Case 2: 3^10
= 59049) and score each under the canonical assignment-eval recipe `_eval_full`
(oracle dominant-user phase FOR that assignment + equal 80/20 power + equal Ck;
equal-power ≈ oracle-power per probe_oracle_alloc, so this is the near-optimal
downstream — i.e. "optimal other agents, assignment-only"). Score per assignment:
    reward = ΣR_tot − λ_D · (#users below D_k)   (the training objective)
We then rank the AGENT's assignment, count how many beat it, and dissect the
free-user-on-IRS question with a direct counterfactual.

Builds cfg from the ckpt's hyperparameters.json (safe vs params.py active case).

Usage:
  python analysis/probe_assignment_enumerate.py --ckpt results/result_47/checkpoints/ep_00200
  python analysis/probe_assignment_enumerate.py --ckpt <dir> --ep 3 --step 7   # force a step
"""
import sys, os, json, argparse, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.probe_overassign_cf import _eval_full, _reward
from analysis.probe_assignment_quality import _oracle_assignment


def _score(assignment, ch, cfg, rate, lam):
    """(n_met, sumR, reward, met_vec) under the _eval_full recipe at nominal σ²."""
    met, _, sumR = _eval_full(assignment, ch, cfg, rate)
    n_met = int(met.sum())
    reward = float(sumR) - lam * (cfg.K - n_met)
    return n_met, float(sumR), reward, met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', type=int, default=8, help='episodes to scan for an over-assign step')
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--ep', type=int, default=None, help='force a specific episode index')
    ap.add_argument('--step', type=int, default=None, help='force a specific step index')
    ap.add_argument('--seed', type=int, default=20260601)
    args = ap.parse_args()

    # ── cfg from the ckpt's own case ────────────────────────────────────────
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
    rate = RateComputer(cfg)
    K, M, Dk = cfg.K, cfg.M, cfg.D_k_bps_hz
    lam = float(hp.get('rl', {}).get('lambda_D', s.get('lambda_D', 1.5)))

    n_combos = (M + 1) ** K
    if n_combos > 600000:
        raise SystemExit(f"(M+1)^K = {n_combos} too large to enumerate; this probe is for Case 2.")

    from probe_critic_ceiling import make_checkpoint_policy
    print(f"Loading policy {args.ckpt} ...")
    policy = make_checkpoint_policy(args.ckpt, cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)

    # ── locate the target step (agent over-routes: n_irs > n_blk) ────────────
    target = None
    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for st in range(args.steps):
            ch = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in env.channels.items()}
            a = policy(env)
            a_pol = np.asarray(a['assignment'], dtype=int)
            blk = ch['su_blocked'].astype(bool)
            n_irs = int((a_pol > 0).sum()); n_blk = int(blk.sum())
            forced = (args.ep is not None and args.step is not None)
            hit = (ep == args.ep and st == args.step) if forced else (n_irs > n_blk)
            if hit:
                target = dict(ep=ep, step=st, ch=ch, a_pol=a_pol, blk=blk,
                              n_irs=n_irs, n_blk=n_blk, upos=env.user_pos.copy(),
                              ipos=env.irs_pos.copy())
                break
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)
        if target is not None:
            break
    if target is None:
        raise SystemExit("No step with n_irs > n_blk found; try more --episodes or force --ep/--step.")

    ch, a_pol, blk = target['ch'], target['a_pol'], target['blk']
    free = ~blk

    # ── reference assignments ───────────────────────────────────────────────
    a_or = _oracle_assignment(ch, target['upos'], target['ipos'])
    nm_ag, sr_ag, rw_ag, met_ag = _score(a_pol, ch, cfg, rate, lam)
    nm_or, sr_or, rw_or, _      = _score(a_or,  ch, cfg, rate, lam)

    # ── brute-force enumeration ─────────────────────────────────────────────
    t0 = time.time()
    rewards = np.empty(n_combos); qoss = np.empty(n_combos, dtype=int)
    best_rw, best_a = -1e18, None
    bestq, bestq_a = -1, None
    for i, combo in enumerate(itertools.product(range(M + 1), repeat=K)):
        a = np.asarray(combo, dtype=int)
        nm, sr, rw, _ = _score(a, ch, cfg, rate, lam)
        rewards[i] = rw; qoss[i] = nm
        if rw > best_rw:
            best_rw, best_a, best_rw_nm, best_rw_sr = rw, a.copy(), nm, sr
        if nm > bestq:
            bestq, bestq_a = nm, a.copy()
    dt = time.time() - t0

    n_better_rw  = int(np.sum(rewards > rw_ag + 1e-9))
    n_equal_rw   = int(np.sum(np.abs(rewards - rw_ag) <= 1e-9))
    n_better_qos = int(np.sum(qoss > nm_ag))
    pct_better_rw = 100.0 * n_better_rw / n_combos

    # ── free-on-IRS counterfactual (the "hên xui" test) ─────────────────────
    free_on_irs = free & (a_pol > 0)
    a_cf = a_pol.copy(); a_cf[free_on_irs] = 0   # move those free users to direct
    nm_cf, sr_cf, rw_cf, _ = _score(a_cf, ch, cfg, rate, lam)
    # how many free users does the reward-BEST assignment put on an IRS?
    best_free_on_irs = int((free & (best_a > 0)).sum())

    def fmt(a):
        return "[" + " ".join(str(int(x)) for x in a) + "]"

    print("=" * 82)
    print(f"  ASSIGNMENT ENUMERATION · {args.ckpt}")
    print(f"  Case K={K} M={M} P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} D_k={Dk} λ_D={lam}")
    print(f"  step = ep{target['ep']}/step{target['step']} · enumerated {n_combos} assignments in {dt:.1f}s")
    print(f"  recipe = oracle dominant-phase + equal 80/20 power + equal Ck (assignment-only)")
    print("=" * 82)
    print(f"  blocked users : {int(blk.sum())}/{K}   {fmt(blk.astype(int))}")
    print(f"  agent assign  : {fmt(a_pol)}   (n_IRS={target['n_irs']}, n_blk={target['n_blk']}, "
          f"free-on-IRS={int(free_on_irs.sum())})")
    print(f"  oracle blk→IRS: {fmt(a_or)}")
    print("  " + "-" * 78)
    print(f"  AGENT      : QoS {nm_ag}/{K} ({100*nm_ag/K:.0f}%)  ΣR {sr_ag:.3f}  reward {rw_ag:+.3f}")
    print(f"  oracle blk→IRS: QoS {nm_or}/{K} ({100*nm_or/K:.0f}%)  ΣR {sr_or:.3f}  reward {rw_or:+.3f}")
    print(f"  BEST reward : QoS {best_rw_nm}/{K} ({100*best_rw_nm/K:.0f}%)  ΣR {best_rw_sr:.3f}  reward {best_rw:+.3f}")
    print(f"                {fmt(best_a)}   (free-on-IRS={best_free_on_irs})")
    print(f"  BEST QoS    : {bestq}/{K} ({100*bestq/K:.0f}%)   {fmt(bestq_a)}")
    print("  " + "-" * 78)
    print(f"  ⇒ assignments STRICTLY BETTER than agent (by reward): {n_better_rw} / {n_combos} "
          f"({pct_better_rw:.2f}%)   [ties: {n_equal_rw}]")
    print(f"     assignments with HIGHER QoS than agent           : {n_better_qos} / {n_combos}")
    print(f"     agent reward percentile: top {pct_better_rw:.2f}%  "
          f"(0% = agent is the unique best)")
    print("  " + "-" * 78)
    print("  FREE-ON-IRS test (the 'hên xui'):")
    if int(free_on_irs.sum()) > 0:
        print(f"     agent put {int(free_on_irs.sum())} free user(s) on IRS. Move them → direct:")
        print(f"       reward {rw_ag:+.3f} → {rw_cf:+.3f}  (Δ {rw_cf-rw_ag:+.3f})   "
              f"QoS {nm_ag} → {nm_cf}")
        print(f"     verdict: {'HELPS (free→direct better)' if rw_cf > rw_ag + 1e-9 else ('HURTS (IRS was right)' if rw_cf < rw_ag - 1e-9 else 'neutral')}")
    else:
        print("     agent put NO free users on IRS this step.")
    print(f"     reward-best assignment routes {best_free_on_irs} free user(s) to IRS "
          f"→ over-assign is {'sometimes optimal' if best_free_on_irs>0 else 'NOT optimal here (free→direct)'}.")
    print("=" * 82)


if __name__ == "__main__":
    main()
