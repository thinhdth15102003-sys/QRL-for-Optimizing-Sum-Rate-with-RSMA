"""
parse_lambda_trajectory.py
--------------------------
T1.1: Parse training_log.txt to extract λ trajectory + λgrad statistics.

Reads a results/result_N/training_log.txt and emits:
  (a) CSV: ep, lam_y_max, lam_z_max, grad_mag, grad_net, grad_r per diag block
  (b) Summary stats: range, mean, ρ_λ = ||λ_T − λ_0||_∞ / ||λ_0||_∞

Usage:
  python analysis/parse_lambda_trajectory.py --log results/result_11/training_log.txt
"""

import argparse
import os
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=None, help="CSV output path (default: <log dir>/lambda_trajectory.csv)")
    args = ap.parse_args()

    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()

    # Match each diag block — look for `diag[N]` then within ~10 lines find
    # `λ|max| y=X z=Y` and `λgrad: mag=A net=B r=C`.
    # Multi-line DOTALL with non-greedy.
    pat = re.compile(
        r"diag\[(\d+)\][^\n]*\n(?:[^\n]*\n){0,5}?[^\n]*λ\|max\| y=([\d.]+) z=([\d.]+)"
        r"[\s\S]{0,500}?λgrad: mag=([\d.eE+-]+) net=([\d.eE+-]+) r=([\d.]+)",
        re.MULTILINE,
    )
    rows = pat.findall(text)
    print(f"n_diag_blocks_matched = {len(rows)}")

    if not rows:
        print("ERR: no diag blocks parsed.")
        return

    eps = [int(r[0]) for r in rows]
    ymax = [float(r[1]) for r in rows]
    zmax = [float(r[2]) for r in rows]
    mags = [float(r[3]) for r in rows]
    nets = [float(r[4]) for r in rows]
    rs = [float(r[5]) for r in rows]

    # ρ_λ proxy from log (using |max| only, not full vector — true ρ needs ckpt diff).
    y0, z0 = ymax[0], zmax[0]
    yT, zT = ymax[-1], zmax[-1]
    rho_y = abs(yT - y0) / abs(y0) if abs(y0) > 0 else float("nan")
    rho_z = abs(zT - z0) / abs(z0) if abs(z0) > 0 else float("nan")
    rho_max = max(rho_y, rho_z)

    print(f"ep range:        {min(eps)} → {max(eps)}  (n={len(eps)})")
    print(f"λ|max| y:        {y0:.4f} (ep{eps[0]}) → {yT:.4f} (ep{eps[-1]})  | min/max over run: {min(ymax):.4f}/{max(ymax):.4f}")
    print(f"λ|max| z:        {z0:.4f} (ep{eps[0]}) → {zT:.4f} (ep{eps[-1]})  | min/max over run: {min(zmax):.4f}/{max(zmax):.4f}")
    print(f"ρ_λ (|max| only): y={rho_y:.4f}  z={rho_z:.4f}  max={rho_max:.4f} = {100*rho_max:.2f}%")
    print()
    print(f"λgrad mag:       min={min(mags):.3e}  mean={sum(mags)/len(mags):.3e}  max={max(mags):.3e}")
    print(f"λgrad net:       min={min(nets):.3e}  mean={sum(nets)/len(nets):.3e}  max={max(nets):.3e}")
    print(f"λgrad r=net/mag: min={min(rs):.4f}  mean={sum(rs)/len(rs):.4f}  max={max(rs):.4f}")

    out = args.out or os.path.join(os.path.dirname(args.log), "lambda_trajectory.csv")
    with open(out, "w", encoding="utf-8") as g:
        g.write("ep,lam_y_max,lam_z_max,grad_mag,grad_net,grad_r\n")
        for r in rows:
            g.write(",".join(r) + "\n")
    print(f"\nwrote {out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
