"""
probe_feasibility_at_P.py  (quick check)
----------------------------------------
Physical-ceiling check: sweep P_S_dBm for a given Case (K,M) at fixed R_LoS,
reporting Servable% / Direct-only QoS% / All-IRS QoS% / R_tot — to decide
whether lowering P (to encourage IRS) keeps QoS feasible (servable ~>=95%).

CPU-only (channel math) — safe alongside GPU training.

Usage:  python analysis/probe_feasibility_at_P.py --K 10 --M 2 --R-LoS 0.2 --P 70 60 50
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from params import make_config
from phase_b_feasibility import _env_feasibility_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--M', type=int, default=2)
    ap.add_argument('--R-LoS', dest='rlos', type=float, default=0.2)
    ap.add_argument('--P', type=float, nargs='+', default=[70.0, 60.0, 50.0])
    ap.add_argument('--episodes', type=int, default=30)
    ap.add_argument('--steps', type=int, default=15)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print(f"  FEASIBILITY vs P_S   ·   K={args.K} M={args.M} R_LoS={args.rlos}")
    print("=" * 78)
    print(f"  {'P_S(dBm)':>9} {'Servable%':>10} {'Direct%':>9} {'All-IRS%':>9} "
          f"{'blocked%':>9} {'Rtot(Dir)':>10} {'Rtot(IRS)':>10}")
    print("  " + "-" * 74)
    for P in args.P:
        cfg = make_config(K=args.K, M=args.M, P_S_dBm=P, R_LoS_km=args.rlos)
        r = _env_feasibility_report(cfg, seed=args.seed,
                                    n_ep=args.episodes, n_steps=args.steps)
        print(f"  {P:>9.1f} {r['servable_frac']*100:>10.1f} {r['qos_direct']*100:>9.1f} "
              f"{r['qos_irs']*100:>9.1f} {r['blocked_frac']*100:>9.1f} "
              f"{r['rtot_direct']:>10.2f} {r['rtot_irs']:>10.2f}")
    print("=" * 78)
    print("  Đọc: Servable% = trần QoS vật lý (ANY policy). Direct% cao = IRS chưa cần.")
    print("       Hạ P để Direct% TỤT (IRS trở nên cần) NHƯNG giữ Servable% >= ~95%.")


if __name__ == '__main__':
    main()
