"""
probe_power_qos.py
------------------
VERIFY the PowerMLP-concentration theory for the QoS ceiling.

Hypothesis (from probe_qos_by_assignment): ~40% of users miss D_k on BOTH IRS and
direct links → the QoS bottleneck is the agent's PRIVATE-POWER ALLOCATION
(concentrated on a few users, wp-top2~50%), NOT assignment/phase.

Test = COUNTERFACTUAL.  For each step, run the agent's policy, then recompute the
rates twice on the SAME state + SAME noise draw:
   (A) agent's learned w_p   (concentrated)
   (E) EQUAL split of the same total private power
keeping assignment / phase / w_c / C_k identical.  Compare QoS-satisfaction.

  Δ = QoS(equal) − QoS(agent).
   Δ >> 0  AND  unmet users power-starved  →  power-concentration CONFIRMED.
   Δ ≈ 0                                   →  QoS ceiling is NOT power (noise/link).

CPU-light per step (2 rate computations); loads the quantum policy (GPU).

Usage:
  python probe_power_qos.py --ckpt results/result_8/checkpoints/ep_00400 --episodes 15
"""

# ── path bootstrap: make project root importable when run as script ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import argparse
import numpy as np

import params as P
from params import make_config
from CSI.env import ISTNEnv
from probe_critic_ceiling import make_checkpoint_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/result_8/checkpoints/ep_00400')
    ap.add_argument('--episodes', type=int, default=15)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260601)
    ap.add_argument('--noise_avg', type=int, default=8,
                    help="average QoS over this many noise draws per step (fair A/E)")
    args = ap.parse_args()

    cfg = make_config(); K = cfg.K; D_k = cfg.D_k_bps_hz
    env = ISTNEnv(cfg=cfg, seed=args.seed, n_steps_ep=args.steps, reward_noise_avg=1)
    print(f"Loading policy {args.ckpt} ...")
    policy = make_checkpoint_policy(args.ckpt, cfg)
    rc = env.rate_computer

    met_a = met_e = tot = 0
    wp_top2 = []
    wp_met, wp_unmet = [], []         # agent per-user power split by met/unmet
    rtot_a_all, rtot_e_all = [], []

    def rtot(wp, sigma2):
        r = rc.compute_sum_rate(env.assignment, env.Phi, env.channels, wp,
                                env.w_c_vec, C_k=env.C_k,
                                active_irs_ids=env.active_irs_ids, sigma2=sigma2)
        return np.asarray(r['R_private']) + np.asarray(r['C_k'])

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            action = policy(env)
            env._apply_action(action)              # set assignment/Phi/w_p/w_c/C_k
            wp_a = env.w_p.copy()
            wp_e = np.full(K, wp_a.sum() / K)
            # average QoS over a few noise draws (same draws for A and E → fair)
            Ra = np.zeros(K); Re = np.zeros(K)
            for _r in range(args.noise_avg):
                s2 = env.channel_model.sample_noise_sigma2()
                Ra += rtot(wp_a, s2); Re += rtot(wp_e, s2)
            Ra /= args.noise_avg; Re /= args.noise_avg
            ma = Ra >= D_k; me = Re >= D_k
            met_a += int(ma.sum()); met_e += int(me.sum()); tot += K
            wp_top2.append(float(np.sort(wp_a)[-2:].sum() / (wp_a.sum() + 1e-12)))
            wp_met.extend(wp_a[ma].tolist()); wp_unmet.extend(wp_a[~ma].tolist())
            rtot_a_all.append(float(Ra.sum())); rtot_e_all.append(float(Re.sum()))
            # advance mobility (replicate env.step WITHOUT re-applying the action)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    qa = 100.0 * met_a / tot
    qe = 100.0 * met_e / tot
    mean_met   = float(np.mean(wp_met))   if wp_met   else float('nan')
    mean_unmet = float(np.mean(wp_unmet)) if wp_unmet else float('nan')
    print(f"\n=== PowerMLP-concentration theory · ckpt={args.ckpt} ·"
          f" {args.episodes}ep×{args.steps}step ===")
    print(f"  Private-power concentration: wp-top2 = {100*np.mean(wp_top2):.1f}%"
          f"  (equal-split would be 2/{K} = {200.0/K:.0f}%)")
    print(f"  Sum-rate ΣR_tot: agent {np.mean(rtot_a_all):.3f}  vs  equal {np.mean(rtot_e_all):.3f}")
    print(f"  ── QoS satisfaction (same states + same noise draws) ──")
    print(f"     Agent power (learned, concentrated): {qa:5.1f}%")
    print(f"     Equal-split power (counterfactual) : {qe:5.1f}%")
    print(f"     Δ (equal − agent) = {qe-qa:+.1f} points")
    print(f"  ── Power-starvation check (agent's w_p) ──")
    print(f"     mean w_p of MET users  : {mean_met:.4g}")
    print(f"     mean w_p of UNMET users: {mean_unmet:.4g}"
          f"   (ratio met/unmet = {mean_met/max(mean_unmet,1e-12):.2f}×)")
    print(f"\n  VERDICT:")
    if qe - qa > 3 and mean_unmet < mean_met:
        print(f"     ✅ POWER-CONCENTRATION CONFIRMED là thủ phạm QoS: chia đều power"
              f" tăng QoS +{qe-qa:.0f}pts, unmet users bị đói power ({mean_met/max(mean_unmet,1e-12):.1f}× ít hơn).")
        print(f"     → lever QoS = PowerMLP (dàn power / QoS-aware), KHÔNG phải phase/assignment.")
    elif qe - qa <= 3:
        print(f"     ❌ BÁC: chia đều power KHÔNG cải thiện QoS (Δ={qe-qa:+.0f}). QoS ceiling KHÔNG"
              f" do power-concentration → gốc khác (noise per-step / link / common-rate split).")
    else:
        print(f"     🟡 MƠ HỒ: Δ={qe-qa:+.0f}pts nhưng unmet không rõ đói power. Cần xem thêm.")


if __name__ == '__main__':
    main()
