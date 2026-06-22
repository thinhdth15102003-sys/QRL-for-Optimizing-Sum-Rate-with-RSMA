"""
probe_oracle_alloc.py
---------------------
Measure the QoS CEILING of the candidate WARM-UP TARGETS (oracle Ck, oracle power,
multi-user phase) BEFORE committing a fresh Case-2 warm-up run.

Decisive question for the fresh-run design:
  Under the agent's LIVE routing+phase (the realistic constraint), how high can QoS
  go if we fix the POWER / Ck allocation to a smart target instead of equal-split?
  · R1 vs R2  → does oracle-Ck (greedy fill weak users) lift QoS over equal-Ck?
  · R2 vs R3  → does oracle-power add on top?
  · R3 vs C2  → how much of the remaining gap is routing+phase (not power/Ck)?

Recipes (QoS = mean fraction of users with R_private+C_k ≥ D_k, noise-averaged):

  under AGENT routing + AGENT phase (env after _apply_action):
    R0  agent power + agent Ck            (= what the policy actually achieves)
    R1  equal power + equal Ck            (= probe_power_qos equal baseline ~76%)
    R2  equal power + ORACLE Ck           (isolates the Ck lever)
    R3  ORACLE power + ORACLE Ck          (full power+Ck oracle, agent route/phase)
  under ORACLE routing (blocked→its IRS):
    C1  dominant-phase + equal power + ORACLE Ck
    C2  MULTI-USER phase + ORACLE power + ORACLE Ck   (≈ true ceiling)

Builds cfg from the ckpt's hyperparameters.json → safe while params.py is another case.

Usage:
  python analysis/probe_oracle_alloc.py --ckpt results/result_47/checkpoints/ep_00100
  python analysis/probe_oracle_alloc.py --ckpt <dir> --episodes 12 --steps 15 --quick
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.phase_oracle import oracle_phase_idx
from analysis.probe_assignment_quality import _oracle_assignment
from analysis.oracle_alloc import (phi_from_idx, _est_channels, met_under_recipe,
                                    multiuser_phase_idx, greedy_power_ck)


def _agent_met(assignment, Phi, ch, cfg, rate, w_p, w_c_vec, C_k, active, Dk, s2):
    """R0: faithful agent recipe (its own w_p/w_c/C_k) via compute_sum_rate."""
    out = rate.compute_sum_rate(np.asarray(assignment), Phi, ch, w_p, w_c_vec,
                                C_k=C_k, active_irs_ids=active, sigma2=s2)
    return (np.asarray(out['R_private']) + np.asarray(out['C_k'])) >= Dk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--noise-avg', dest='navg', type=int, default=8)
    ap.add_argument('--seed', type=int, default=20260601)
    ap.add_argument('--quick', action='store_true', help='tiny sample for a smoke test')
    args = ap.parse_args()
    if args.quick:
        args.episodes, args.steps, args.navg = 2, 4, 2

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
    K, M, Dk, P_S = cfg.K, cfg.M, cfg.D_k_bps_hz, cfg.P_S

    from probe_critic_ceiling import make_checkpoint_policy
    print(f"Loading policy {args.ckpt} ...")
    policy = make_checkpoint_policy(args.ckpt, cfg)
    env = ISTNEnv(cfg=cfg, seed=args.seed + 777, n_steps_ep=args.steps,
                  reward_noise_avg=1)

    keys = ['R0_agent', 'R0b_agpw_orck', 'R0c_eqpw_agck', 'R0d_agsplit_unif',
            'R1_eq_eqck', 'R2_eq_orck', 'R3_op_orck',
            'C1_oroute_dom', 'C2_oroute_mu']
    split_priv = []   # agent's private fraction sum(wp)/P_S, per step
    acc = {k: [] for k in keys}

    for ep in range(args.episodes):
        env.reset(seed=args.seed + ep)
        for _ in range(args.steps):
            ch = {k: (v.copy() if hasattr(v, 'copy') else v)
                  for k, v in env.channels.items()}
            action = policy(env)
            env._apply_action(action)
            a_ag, Phi_ag = env.assignment.copy(), env.Phi.copy()
            act_ag = list(env.active_irs_ids)
            wp_ag, wc_ag = env.w_p.copy(), env.w_c_vec.copy()
            Ck_ag = None if env.C_k is None else env.C_k.copy()

            # equal-power references (agent routing)
            wp_eq = np.full(K, P_S * 0.8 / K)
            wc_eq = np.full(len(act_ag) + 1, P_S * 0.2 / (len(act_ag) + 1))
            # agent's OWN split (private fraction) but uniform within private+common
            priv_tot_ag = float(wp_ag.sum()); comm_tot_ag = float(wc_ag.sum())
            split_priv.append(priv_tot_ag / max(priv_tot_ag + comm_tot_ag, 1e-12))
            wp_d = np.full(K, priv_tot_ag / K)
            wc_d = np.full(len(act_ag) + 1, comm_tot_ag / (len(act_ag) + 1))
            # oracle power (search on nominal σ², agent routing+phase)
            _, wp_op, wc_op = greedy_power_ck(a_ag, Phi_ag, ch, cfg, rate, act_ag, Dk)

            # ── oracle routing reference ────────────────────────────────────
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
            act_or = sorted(set(int(a) for a in a_or if a > 0))
            wc_eq_or = np.full(len(act_or) + 1, P_S * 0.2 / (len(act_or) + 1))
            wp_eq_or = np.full(K, P_S * 0.8 / K)
            Phi_dom = phi_from_idx(oracle_phase_idx(_est_channels(ch), a_or, cfg), cfg)
            # multi-user phase: alternate (power → phase → power) once on nominal
            _, wp_a, wc_a = greedy_power_ck(a_or, Phi_dom, ch, cfg, rate, act_or, Dk)
            pidx_mu = multiuser_phase_idx(a_or, ch, cfg, rate, act_or, Dk, wp_a, wc_a)
            Phi_mu = phi_from_idx(pidx_mu, cfg)
            _, wp_mu, wc_mu = greedy_power_ck(a_or, Phi_mu, ch, cfg, rate, act_or, Dk)

            # ── noise-averaged QoS (same draws across recipes) ──────────────
            mt = {k: 0.0 for k in keys}
            for _r in range(args.navg):
                s2 = env.channel_model.sample_noise_sigma2()
                mt['R0_agent']    += _agent_met(a_ag, Phi_ag, ch, cfg, rate, wp_ag, wc_ag, Ck_ag, act_ag, Dk, s2).mean()
                mt['R0b_agpw_orck']+= met_under_recipe(a_ag, Phi_ag, ch, cfg, rate, wp_ag, wc_ag, act_ag, Dk, s2, True)[0].mean()
                mt['R0c_eqpw_agck']+= _agent_met(a_ag, Phi_ag, ch, cfg, rate, wp_eq, wc_eq, Ck_ag, act_ag, Dk, s2).mean()
                mt['R0d_agsplit_unif']+= met_under_recipe(a_ag, Phi_ag, ch, cfg, rate, wp_d, wc_d, act_ag, Dk, s2, False)[0].mean()
                mt['R1_eq_eqck']  += met_under_recipe(a_ag, Phi_ag, ch, cfg, rate, wp_eq, wc_eq, act_ag, Dk, s2, False)[0].mean()
                mt['R2_eq_orck']  += met_under_recipe(a_ag, Phi_ag, ch, cfg, rate, wp_eq, wc_eq, act_ag, Dk, s2, True)[0].mean()
                mt['R3_op_orck']  += met_under_recipe(a_ag, Phi_ag, ch, cfg, rate, wp_op, wc_op, act_ag, Dk, s2, True)[0].mean()
                mt['C1_oroute_dom']+= met_under_recipe(a_or, Phi_dom, ch, cfg, rate, wp_eq_or, wc_eq_or, act_or, Dk, s2, True)[0].mean()
                mt['C2_oroute_mu'] += met_under_recipe(a_or, Phi_mu, ch, cfg, rate, wp_mu, wc_mu, act_or, Dk, s2, True)[0].mean()
            for k in keys:
                acc[k].append(mt[k] / args.navg)

            # advance mobility (replicate env.step WITHOUT re-applying action)
            env.user_pos = env._walk_users(env.user_pos)
            env.channels = env.channel_model.update_user_channels(
                env.user_pos, env.irs_pos, env.channels)

    q = {k: 100.0 * float(np.mean(v)) for k, v in acc.items()}
    print("=" * 80)
    print(f"  ORACLE-ALLOC QoS CEILING · {args.ckpt}")
    print(f"  Case K={K} M={M} N={cfg.N} P_S={cfg.P_S_dBm}dBm R_LoS={cfg.R_LoS_km} "
          f"D_k={Dk} · {args.episodes}ep×{args.steps}step×{args.navg}noise")
    print("=" * 80)
    print("  AGENT routing + AGENT phase (the realistic constraint):")
    print(f"    R0  agent power + agent Ck          : {q['R0_agent']:5.1f}%   ← what the policy achieves")
    print(f"    R0b agent power + ORACLE Ck         : {q['R0b_agpw_orck']:5.1f}%   Δ vs R0 {q['R0b_agpw_orck']-q['R0_agent']:+.1f}  ← fix Ck only")
    print(f"    R0c equal power + agent Ck          : {q['R0c_eqpw_agck']:5.1f}%   Δ vs R0 {q['R0c_eqpw_agck']-q['R0_agent']:+.1f}  ← fix power only")
    print(f"    R0d agent SPLIT + uniform within     : {q['R0d_agsplit_unif']:5.1f}%   (agent priv-frac {100*float(np.mean(split_priv)):.0f}% vs target 80%)")
    print(f"        → R0d≈R0 ⇒ the SPLIT (common-vs-private) is the lever, not within-group concentration")
    print(f"    R1  equal power + equal Ck          : {q['R1_eq_eqck']:5.1f}%   (probe_power_qos baseline)")
    print(f"    R2  equal power + ORACLE Ck         : {q['R2_eq_orck']:5.1f}%   Δ vs R1 {q['R2_eq_orck']-q['R1_eq_eqck']:+.1f}  ← Ck lever")
    print(f"    R3  ORACLE power + ORACLE Ck        : {q['R3_op_orck']:5.1f}%   Δ vs R2 {q['R3_op_orck']-q['R2_eq_orck']:+.1f}  ← power lever")
    print("  ORACLE routing (blocked→its IRS) reference ceiling:")
    print(f"    C1  dom-phase + equal pw + ORACLE Ck: {q['C1_oroute_dom']:5.1f}%")
    print(f"    C2  MULTI-USER phase + ORACLE pw+Ck : {q['C2_oroute_mu']:5.1f}%   Δ vs R3 {q['C2_oroute_mu']-q['R3_op_orck']:+.1f}  ← routing+phase gap")
    print("=" * 80)
    print("  READ: R2−R1 = Ck-warmup headroom · R3−R2 = power-warmup headroom ·")
    print("        C2−R3 = remaining routing+phase gap (assignment/phase warmup territory).")
    print("        If R2≫R1 → bake ORACLE-Ck into warm-up (confirms Ck = binding lever).")


if __name__ == "__main__":
    main()
