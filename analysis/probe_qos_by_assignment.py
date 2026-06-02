"""
probe_qos_by_assignment.py
--------------------------
For the CURRENT trained policy: of the users ASSIGNED TO IRS, what fraction meet
their QoS demand (R_tot >= D_k)?  Split IRS vs direct, blocked vs non-blocked.

Answers "users gán vào IRS thì tỉ lệ đạt QoS là bao nhiêu" — the per-assignment
QoS-satisfaction rate that the training log (aggregate QoS/K) does NOT expose.

Usage:
  python probe_qos_by_assignment.py --ckpt results/result_8/checkpoints/ep_00400 --episodes 20
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
from probe_critic_ceiling import make_checkpoint_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_8/checkpoints/ep_00400')
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260601)
    args = ap.parse_args()

    cfg = make_config()
    K   = cfg.K
    D_k = cfg.D_k_bps_hz
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps,
                  reward_noise_avg=getattr(P, 'reward_noise_avg', 16))
    print(f"Loading policy from {args.ckpt} ...")
    policy = make_checkpoint_policy(args.ckpt, cfg)

    tot = collections.Counter()
    met = collections.Counter()
    irs_count_per_step = []

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _step in range(args.steps):
            blocked = env.channels['su_blocked'].copy()   # channels used THIS step
            a = policy(env)
            _, _r, _, info = env.step(a)
            assign = np.asarray(info['assignment'])
            Rtot   = np.asarray(info['R_tot'])
            irs_count_per_step.append(int(np.sum(assign > 0)))
            for k in range(K):
                grp = 'IRS' if assign[k] > 0 else 'DIR'
                blk = 'blk' if blocked[k] else 'non'
                tot[(grp, blk)] += 1; tot[(grp, 'all')] += 1; tot[('ALL', 'all')] += 1
                if Rtot[k] >= D_k:
                    met[(grp, blk)] += 1; met[(grp, 'all')] += 1; met[('ALL', 'all')] += 1

    def rate(key):
        return 100.0 * met[key] / tot[key] if tot[key] else float('nan')

    print(f"\n=== QoS-satisfaction by ASSIGNMENT  ·  ckpt={args.ckpt} ·"
          f" {args.episodes}ep × {args.steps}step ===")
    print(f"  D_k = {D_k} bps/Hz   ·   IRS/K mean = {np.mean(irs_count_per_step):.2f}/{K}")
    print(f"  OVERALL QoS         : {rate(('ALL','all')):5.1f}%   (n={tot[('ALL','all')]})")
    print(f"  ⭐ IRS-assigned QoS : {rate(('IRS','all')):5.1f}%   (n={tot[('IRS','all')]})")
    print(f"       IRS & blocked  : {rate(('IRS','blk')):5.1f}%   (n={tot[('IRS','blk')]})")
    print(f"       IRS & non-block: {rate(('IRS','non')):5.1f}%   (n={tot[('IRS','non')]})")
    print(f"  Direct-assigned QoS : {rate(('DIR','all')):5.1f}%   (n={tot[('DIR','all')]})")
    print(f"       DIR & blocked  : {rate(('DIR','blk')):5.1f}%   (n={tot[('DIR','blk')]})")
    print(f"       DIR & non-block: {rate(('DIR','non')):5.1f}%   (n={tot[('DIR','non')]})")
    # Interpretation hint
    irs_q = rate(('IRS', 'all')); dir_q = rate(('DIR', 'all'))
    print(f"\n  → IRS-routed users served {irs_q:.0f}% vs direct-routed {dir_q:.0f}%. "
          + ("IRS-routing đang serve TỐT." if irs_q >= dir_q else
             "IRS-routing đang serve KÉM hơn direct (pha ngẫu nhiên kéo xuống)."))


if __name__ == '__main__':
    main()
