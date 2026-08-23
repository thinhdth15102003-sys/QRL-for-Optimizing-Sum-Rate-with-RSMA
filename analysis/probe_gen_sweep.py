"""
probe_gen_sweep.py
------------------
Multi-seed zero-shot generalization sweep (Case-2), VQC vs. DNN. Reruns ALL rows
(including the base P_S=50 point) under ONE consistent protocol --- the same seed set
and episode budget for every cell --- so numbers are directly comparable and the
VQC-DNN delta is meaningful. Emits per-shift mean±std R_tot & QoS for both actors,
plus a LaTeX body with Delta(VQC-DNN) columns and domain dividers.

Usage:
  python analysis/probe_gen_sweep.py --vqc results/result_110 --dnn results/result_101 \
     --seeds 42 0 1 2 3 --eps 10 --steps 200
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from CSI.env import ISTNEnv
from infer import load_training_cfg, load_agents, run_hqchac_episode

# (label, domain, load_training_cfg overrides). Domain groups the LaTeX dividers.
SHIFTS = [
    ("base ($P_S{=}50$)", "base",  {}),
    # 2026-08-14: the P_S=30 rung was dropped from the paper table, so both joint
    # shifts were re-pinned onto rungs that still have a row of their own —
    # combined-A on 40 and combined-B on 60. Every component of either joint shift
    # is now traceable to a single-shift row the reader can look up.
    ("$P_S=30$",  "ps",    {"p_s_dbm": 30}),
    ("$P_S=40$",  "ps",    {"p_s_dbm": 40}),
    ("$P_S=60$",  "ps",    {"p_s_dbm": 60}),
    ("$P_S=70$",  "ps",    {"p_s_dbm": 70}),
    ("speed $3$", "speed", {"user_speed": 3.0}),
    ("speed $6$", "speed", {"user_speed": 6.0}),
    ("$\\kappa=0.10$", "csi", {"kappa": 0.10}),
    ("$\\kappa=0.15$", "csi", {"kappa": 0.15}),
    ("$\\kappa=0.20$", "csi", {"kappa": 0.20}),
    ("$\\sigma^2=13$", "noise", {"noise_var_dBW": 13.0}),
    ("$\\sigma^2=16$", "noise", {"noise_var_dBW": 16.0}),
    ("combined-A", "comb", {"p_s_dbm": 40, "kappa": 0.10, "user_speed": 3.0}),
    ("combined-B", "comb", {"p_s_dbm": 60, "kappa": 0.20, "user_speed": 6.0, "noise_var_dBW": 16.0}),
]


def eval_actor(run_dir, seeds, n_eps, n_steps, tag, shifts=None):
    """shifts: subset of SHIFTS to evaluate (default: all of them). Rows the paper
    leaves blank cost the same as rows it prints, so restricting the list is the
    difference between a 35-minute sweep and a 15-minute one per policy."""
    actor, phase_net, power_net, ck_net = load_agents(run_dir, seed=0)
    out = {}
    for label, _dom, ov in (shifts if shifts is not None else SHIFTS):
        cfg = load_training_cfg(run_dir, **ov)
        demand = np.full(cfg.K, cfg.D_k_bps_hz)
        # Reward is the episode return, i.e. J summed over the n_steps steps.
        # The paper's genshift table carries it alongside R_tot and QoS, and a
        # row whose R_tot is remeasured but whose Reward is not is internally
        # inconsistent — Reward is derived from the same rates.
        Rs, Qs, Ws = [], [], []
        for sd in seeds:
            env = ISTNEnv(cfg, seed=sd, n_steps_ep=n_steps)
            r_acc, q_acc, w_acc = [], [], []
            for _ in range(n_eps):
                m = run_hqchac_episode(env, actor, phase_net, power_net,
                                       ck_net, cfg, demand, n_steps, greedy=True)
                r_acc.append(m['sum_rate']); q_acc.append(m['qos_frac'])
                w_acc.append(m.get('reward', float('nan')))
            Rs.append(np.mean(r_acc)); Qs.append(np.mean(q_acc))
            Ws.append(np.mean(w_acc))
        out[label] = (np.array(Rs), np.array(Qs), np.array(Ws))
        print(f"  [{tag}] {label:18s} R={np.mean(Rs):.4f}±{np.std(Rs):.4f} "
              f"QoS={100*np.mean(Qs):5.1f}±{100*np.std(Qs):.1f}% "
              f"Rew={np.mean(Ws):7.1f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vqc', required=True)
    ap.add_argument('--dnn', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 0, 1, 2, 3])
    ap.add_argument('--eps', type=int, default=10)
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()
    print(f"GEN-SWEEP · seeds={args.seeds} eps={args.eps} steps={args.steps} "
          f"({len(args.seeds)}x{args.eps}={len(args.seeds)*args.eps} realizations/cell)")
    print("=" * 78)
    V = eval_actor(args.vqc, args.seeds, args.eps, args.steps, "VQC")
    D = eval_actor(args.dnn, args.seeds, args.eps, args.steps, "DNN")

    print("\n" + "=" * 78 + "\n  LaTeX body (VQC | DNN | Delta = VQC-DNN):\n" + "=" * 78)
    prev = None
    for label, dom, _ov in SHIFTS:
        if prev is not None and dom != prev:
            print("\\midrule")
        prev = dom
        vr, vq, vw = V[label]; dr, dq, dw = D[label]
        dR = vr.mean() - dr.mean(); dQ = 100*(vq.mean() - dq.mean())
        # DNN triple then VQC triple, matching the column order of
        # tab:abl_genshift (AO | PPO-flat | DNN | HQC-HAC), so the row can be
        # pasted straight in once the AO and flat cells are prefixed.
        print(f"{label} & ${dr.mean():.4f}$ & ${100*dq.mean():.1f}$\\% "
              f"& ${dw.mean():.1f}$ "
              f"& ${vr.mean():.4f}$ & ${100*vq.mean():.1f}$\\% "
              f"& ${vw.mean():.1f}$ "
              f"  % dR {dR:+.4f}  dQoS {dQ:+.1f}pp")
    print("=" * 78)


if __name__ == '__main__':
    main()
