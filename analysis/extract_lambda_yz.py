"""
extract_lambda_yz.py  —  "pull data" step (run me to refresh the CSV)
--------------------------------------------------------------------
Pulls the VQC encoding-scale trajectory ‖λ_y‖_∞, ‖λ_z‖_∞ for the Case-1 curriculum
(R_LoS 0.2→0.5) from the training logs and writes a CSV. Then draw it yourself with
analysis/plot_lambda_yz.py (which only needs the CSV + matplotlib — no repo access).

Parses each run's diag blocks (`λ|max| y=.. z=..`, same regex as
parse_lambda_trajectory.py) and concatenates across the chain on a continuous
cumulative-episode axis — each segment truncated at the episode the NEXT ramp
resumed from, so the path is the true trajectory (no abandoned tail).

Pure stdlib (re, csv) — runs anywhere the logs are.

Usage:
  python analysis/extract_lambda_yz.py
  python analysis/extract_lambda_yz.py --out analysis/data/lambda_yz_case1.csv
"""

import argparse, os, re, csv

# (result dir, ramp R_LoS, max ep INCLUSIVE = ep next ramp resumed from; None = all)
CHAIN = [
    ("results/result_5",  0.2, 900),   # R_LoS 0.2 (fresh)
    ("results/result_8",  0.3, 300),   # R_LoS 0.3
    ("results/result_12", 0.4, 500),   # R_LoS 0.4
    ("results/result_18", 0.5, None),  # R_LoS 0.5 (final, lock ep_00400)
]

# diag block → ep, ‖λ_y‖_∞, ‖λ_z‖_∞   (the log's `λ|max| y=.. z=..`)
PAT = re.compile(
    r"diag\[(\d+)\][^\n]*\n(?:[^\n]*\n){0,5}?[^\n]*λ\|max\| y=([\d.]+) z=([\d.]+)",
    re.MULTILINE,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="analysis/data/lambda_yz_case1.csv")
    args = ap.parse_args()

    rows, offset = [], 0
    for rdir, ramp, max_ep in CHAIN:
        log = os.path.join(rdir, "training_log.txt")
        with open(log, encoding="utf-8", errors="ignore") as f:
            recs = PAT.findall(f.read())
        seg_last, kept = 0, 0
        for ep_s, ly, lz in recs:
            ep = int(ep_s)
            if max_ep is not None and ep > max_ep:
                continue
            rows.append([offset + ep, float(ly), float(lz), ramp, os.path.basename(rdir), ep])
            seg_last = max(seg_last, ep); kept += 1
        offset += seg_last
        print(f"{rdir:<22} ramp={ramp} matched={len(recs):3d} kept={kept:3d} (≤{max_ep}) last_ep={seg_last} cum_now={offset}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["cum_ep", "lam_y_inf", "lam_z_inf", "ramp", "src_result", "src_ep"])
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({len(rows)} rows, ep 0→{rows[-1][0] if rows else 0})")


if __name__ == "__main__":
    main()
