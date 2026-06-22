"""
probe_realistic_ceiling.py
--------------------------
The "oracle ceiling" from probe_assignment_quality (99.9% @P50) is computed at
NOMINAL noise. But the env's noise is σ²~N(-30, var_dB) (lognormal, heavy), and the
logged QoS uses noise-AVERAGED rates (reward_noise_avg draws). So the REALISTIC
achievable ceiling — oracle routing (blocked→IRS) + oracle phase + reference power,
evaluated under the actual noise — can be well below the nominal oracle.

Reports, per P_S, the oracle QoS under:
  · nominal    σ² (= the idealized probe_assignment_quality number)
  · 1-draw     σ² (single realistic noise sample → most pessimistic)
  · avg-R      σ² (rates averaged over R draws → matches the training QoS metric)
so you can see how much of the "gap to 99.9%" is irreducible noise vs optimization.

Usage:
  python analysis/probe_realistic_ceiling.py --P 50 70 --K 10 --M 2 --R-avg 16
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from params import make_config
from CSI.env import ISTNEnv
from CSI.rate import RateComputer
from analysis.probe_assignment_quality import _oracle_assignment
from analysis.probe_overassign_cf import _eval_full


def ceiling_at(P, K, M, rlos, episodes, steps, R_avg, seed):
    cfg  = make_config(K=K, M=M, P_S_dBm=P, R_LoS_km=rlos)
    rate = RateComputer(cfg); Dk = cfg.D_k_bps_hz
    env  = ISTNEnv(cfg=cfg, seed=seed + 777, n_steps_ep=steps, reward_noise_avg=1)
    q_nom, q_avg, q_1 = [], [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        for _ in range(steps):
            ch   = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in env.channels.items()}
            a_or = _oracle_assignment(ch, env.user_pos, env.irs_pos)
            met_n, _, _ = _eval_full(a_or, ch, cfg, rate, sigma2=None)          # nominal σ²
            q_nom.append(float(met_n.mean()))
            Racc = None
            for r in range(R_avg):
                s2 = env.channel_model.sample_noise_sigma2()
                _, Rtot, _ = _eval_full(a_or, ch, cfg, rate, sigma2=s2)
                Racc = Rtot if Racc is None else Racc + Rtot
                if r == 0:
                    q_1.append(float(np.mean(Rtot >= Dk)))                       # single noisy draw
            q_avg.append(float(np.mean((Racc / R_avg) >= Dk)))                   # avg-R rates (training metric)
            env.user_pos  = env._walk_users(env.user_pos)
            env.channels  = env.channel_model.update_user_channels(env.user_pos, env.irs_pos, env.channels)
    serv = None
    return (np.mean(q_nom) * 100, np.mean(q_1) * 100, np.mean(q_avg) * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--P', type=float, nargs='+', default=[50.0, 70.0])
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=2)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=12)
    ap.add_argument('--R-avg', dest='ravg', type=int, default=16)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    print("=" * 78)
    print(f"  REALISTIC ORACLE CEILING (blocked→IRS + oracle-φ + ref-power) · K={args.K} M={args.M} "
          f"R_LoS={args.rlos} · R_avg={args.ravg}")
    print("=" * 78)
    print(f"  {'P_S(dBm)':>9} | {'QoS nominal-σ²':>15} | {'QoS 1-draw':>12} | {'QoS avg-R (train metric)':>26}")
    print("  " + "-" * 72)
    for P in args.P:
        n, d1, davg = ceiling_at(P, args.K, args.M, args.rlos, args.episodes, args.steps, args.ravg, args.seed)
        print(f"  {P:>9.0f} | {n:>14.1f}% | {d1:>11.1f}% | {davg:>25.1f}%")
    print("=" * 78)
    print("  nominal = idealized (probe_assignment_quality). avg-R = what training QoS actually sees.")
    print("  gap(nominal − avg-R) = IRREDUCIBLE noise penalty; agent QoS should be compared to avg-R, not nominal.")


if __name__ == "__main__":
    main()
