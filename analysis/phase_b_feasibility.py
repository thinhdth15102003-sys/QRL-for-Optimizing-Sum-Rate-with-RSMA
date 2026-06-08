"""
phase_b_feasibility.py
----------------------
Phase B prep: probe environment feasibility at higher R_LoS_km ramps (0.25, 0.3,
0.35) BEFORE committing a training run. Reports Servable% / Direct-only QoS /
All-IRS QoS, and recommends a λ_D value per the [I-3] curriculum rule:

  Direct-only QoS↓  ⇒  λ_D↑  (penalty must tighten when easy baseline degrades)

This is a CPU-light probe — no actor, no training, just env transitions under
fixed reference policies. Run before launching a ramp-0.3 training to calibrate.

Usage:
  python analysis/phase_b_feasibility.py
  python analysis/phase_b_feasibility.py --ramps 0.25,0.3,0.35 --episodes 30 --steps 15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from params import make_config
from train import _env_feasibility_report


def _lambda_D_recommendation(rep, current_lambda_D: float, ramp_label: str) -> str:
    """Per [I-3]: if Direct-only QoS dropped substantially, λ_D should rise.
    Rough heuristic: tighten λ_D proportionally to QoS shortfall vs ramp-0.2 (where
    Direct-only ≈ 100%). Lower bound 1.5, upper bound 5.0 (avoid penalty-dom [I-2])."""
    qd = rep["qos_direct"]
    if qd >= 0.95:
        rec = current_lambda_D  # no change needed
        why = "Direct-only ≥ 95% → λ_D unchanged"
    elif qd >= 0.80:
        rec = min(2.5, current_lambda_D * 1.4)
        why = f"Direct-only {qd*100:.0f}% (mild drop) → λ_D ~×1.4 (cap 2.5)"
    elif qd >= 0.60:
        rec = min(4.0, current_lambda_D * 2.0)
        why = f"Direct-only {qd*100:.0f}% (significant drop) → λ_D ~×2 (cap 4.0)"
    else:
        rec = min(5.0, current_lambda_D * 3.0)
        why = f"Direct-only {qd*100:.0f}% (severe drop) → λ_D ~×3 (cap 5.0 to avoid [I-2])"
    return f"recommended λ_D ≈ {rec:.2f}  ({why})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ramps", default="0.25,0.3,0.35",
                    help="Comma-sep R_LoS_km values to probe (default: 0.25,0.3,0.35).")
    ap.add_argument("--episodes", type=int, default=30,
                    help="Episodes per ramp (default: 30 — twice the train.py default for stability).")
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ramps = [float(r.strip()) for r in args.ramps.split(",")]

    print("=" * 76)
    print("  PHASE B FEASIBILITY PROBE  (calibrate λ_D for higher R_LoS ramps)")
    print(f"  ramps={ramps}  episodes={args.episodes}  steps={args.steps}  seed={args.seed}")
    print("=" * 76)

    # Reference: current Case-2 params at R_LoS=0.2 use λ_D=1.5
    cfg0 = make_config()
    current_lambda_D = cfg0.lambda_D
    print(f"  reference: current cfg R_LoS={cfg0.R_LoS_km} km, λ_D={current_lambda_D}, "
          f"D_k={cfg0.D_k_bps_hz}")
    print()

    results = {}
    for R_LoS in ramps:
        cfg = make_config(R_LoS_km=R_LoS)
        rep = _env_feasibility_report(cfg, seed=args.seed,
                                      n_ep=args.episodes, n_steps=args.steps)
        results[R_LoS] = rep
        rec = _lambda_D_recommendation(rep, current_lambda_D, f"ramp_{R_LoS}")
        print(f"  → {rec}")
        print(f"  → blocking rate at this ramp: {rep['blocked_frac']*100:.1f}% of users")
        print()

    # Cross-ramp summary table
    print("=" * 76)
    print("  CROSS-RAMP SUMMARY")
    print("=" * 76)
    print(f"  R_LoS    Servable%   Direct-only%   All-IRS%    blocked%    R_tot(Dir)   R_tot(IRS)")
    print("  " + "-" * 74)
    for R_LoS, r in results.items():
        print(f"  {R_LoS:.2f}     "
              f"{r['servable_frac']*100:7.1f}    "
              f"{r['qos_direct']*100:8.1f}     "
              f"{r['qos_irs']*100:8.1f}    "
              f"{r['blocked_frac']*100:6.1f}    "
              f"{r['rtot_direct']:8.2f}     "
              f"{r['rtot_irs']:8.2f}")
    print()
    print("  ⭐ Picks the next ramp where Direct-only is NO LONGER trivially solving the problem")
    print("    (Direct-only QoS drops below ~95%) — that's where IRS+RSMA contribution becomes")
    print("    defensible. Per [I-3], also tighten λ_D so the policy can't take the easy way out.")


if __name__ == "__main__":
    main()
