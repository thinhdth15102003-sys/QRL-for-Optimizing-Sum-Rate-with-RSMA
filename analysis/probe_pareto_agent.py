"""
probe_pareto_agent.py
---------------------
Is the TRAINED agent on the (QoS, R_tot) Pareto frontier, or DOMINATED (can we
raise BOTH)?  Unlike probe_pareto_frontier (oracle routing+phase), this holds the
agent's OWN routing+phase from a checkpoint and sweeps the private-power
concentration β under it, then places the agent's ACTUAL operating point on that
local frontier.

  agent ACTUAL point : (QoS, R_tot) from the agent's full policy (its own power+Ck).
  β-curve            : same assignment+phase, but w_p ∝ |h_eff|^(2β) (equal common,
                       equal Ck) — the achievable frontier by re-allocating power.

Verdict:
  · DOMINATED  → some β gives QoS ≥ agent AND R_tot ≥ agent (both up) → free headroom,
                 the plateau is an OPTIMIZATION gap (not a tradeoff).
  · ON FRONTIER→ every β that raises one lowers the other → pure tradeoff (need λ_D
                 to move along it).

Usage:
  python analysis/probe_pareto_agent.py --ckpt results/result_52/checkpoints/ep_01000
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer


def _point(assignment, Phi, ch, cfg, rate, w_p, w_c_vec, C_k, active):
    """(QoS, ΣR_tot, reward) at nominal σ² for a given power/ck recipe."""
    out = rate.compute_sum_rate(np.asarray(assignment), Phi, ch, w_p, w_c_vec,
                                C_k=C_k, active_irs_ids=active, sigma2=cfg.sigma2)
    Rtot = np.asarray(out['R_private']) + np.asarray(out['C_k'])
    qos  = float(np.mean(Rtot >= cfg.D_k_bps_hz))
    sumR = float(np.sum(Rtot))
    short = np.maximum(0.0, cfg.D_k_bps_hz - Rtot) / cfg.D_k_bps_hz
    reward = sumR - cfg.lambda_D * float(np.sum(short ** 2))
    return qos, sumR, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', type=int, default=25)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20260601)
    args = ap.parse_args()

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
                      R_LoS_km=s['R_LoS_km'], D_k_bps_hz=s['D_k_bps_hz'],
                      lambda_D=float(hp.get('rl', {}).get('lambda_D', 1.5)))
    rate = RateComputer(cfg); K = cfg.K

    from probe_critic_ceiling import make_checkpoint_policy
    print(f"Loading policy {args.ckpt} ...")
    policy = make_checkpoint_policy(args.ckpt, cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps, reward_noise_avg=1)

    betas = [-1.0, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    ag = {'q': [], 'r': [], 'rew': []}
    bsweep = {b: {'q': [], 'r': [], 'rew': []} for b in betas}

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            ch = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in env.channels.items()}
            action = policy(env); env._apply_action(action)
            a_ag, Phi_ag = env.assignment.copy(), env.Phi.copy()
            active = list(env.active_irs_ids)
            # agent ACTUAL point (its own power + ck)
            q, r, rw = _point(a_ag, Phi_ag, ch, cfg, rate, env.w_p.copy(), env.w_c_vec.copy(),
                               None if env.C_k is None else env.C_k.copy(), active)
            ag['q'].append(q); ag['r'].append(r); ag['rew'].append(rw)
            # β-sweep under the SAME assignment+phase (equal common, equal Ck)
            h_eff = rate.effective_channels_all(a_ag, Phi_ag, ch)
            g = np.abs(h_eff) ** 2 + 1e-30
            G = len(active)
            wc = np.full(G + 1, cfg.P_S * 0.2 / (G + 1))
            for b in betas:
                w = g ** b
                wp = w / w.sum() * (cfg.P_S * 0.8)
                q2, r2, rw2 = _point(a_ag, Phi_ag, ch, cfg, rate, wp, wc, None, active)
                bsweep[b]['q'].append(q2); bsweep[b]['r'].append(r2); bsweep[b]['rew'].append(rw2)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)

    Qa, Ra, RWa = 100*np.mean(ag['q']), np.mean(ag['r']), np.mean(ag['rew'])
    print("=" * 76)
    print(f"  PARETO-AGENT · {args.ckpt}")
    print(f"  K={cfg.K} M={cfg.M} P_S={cfg.P_S_dBm} λ_D={cfg.lambda_D} · {args.episodes}ep×{args.steps}step · nominal σ²")
    print(f"  [hold AGENT routing+phase; sweep power β; equal-Ck on the curve]")
    print("=" * 76)
    print(f"  AGENT actual (full policy)  :  QoS {Qa:5.1f}%   R_tot {Ra:6.3f}   reward {RWa:+7.3f}")
    print("  " + "-" * 70)
    print(f"  {'β':>6} | {'QoS%':>6} | {'R_tot':>7} | {'reward':>8} | vs agent")
    print("  " + "-" * 70)
    dominated_by = []
    best_rew, best_b = -1e9, None
    for b in betas:
        q = 100*np.mean(bsweep[b]['q']); r = np.mean(bsweep[b]['r']); rw = np.mean(bsweep[b]['rew'])
        both_up = (q >= Qa - 1e-6) and (r >= Ra - 1e-6) and (q > Qa + 0.3 or r > Ra + 0.005)
        tag = "  ★ BOTH ↑ (dominates agent)" if both_up else \
              ("  QoS↑/R↓" if q > Qa else ("  R↑/QoS↓" if r > Ra else ""))
        if both_up: dominated_by.append((b, q - Qa, r - Ra))
        if rw > best_rew: best_rew, best_b = rw, b
        print(f"  {b:>6.1f} | {q:>5.1f}% | {r:>7.3f} | {rw:>+8.3f} |{tag}")
    print("  " + "-" * 70)
    if dominated_by:
        b0, dq, dr = max(dominated_by, key=lambda x: x[1] + 20*x[2])
        print(f"  ⟹ DOMINATED: β={b0} raises BOTH (QoS +{dq:.1f}pp AND R_tot +{dr:.3f}) → optimization headroom,")
        print(f"     the plateau is NOT a frontier tradeoff (agent sits BELOW its own achievable frontier).")
    else:
        print(f"  ⟹ ON FRONTIER: no β raises both → pure QoS↔R_tot tradeoff. To move, change λ_D.")
    print(f"  reward-optimal β on curve = {best_b} (rew {best_rew:+.3f}) vs agent reward {RWa:+.3f}")
    print("=" * 76)


if __name__ == "__main__":
    main()
