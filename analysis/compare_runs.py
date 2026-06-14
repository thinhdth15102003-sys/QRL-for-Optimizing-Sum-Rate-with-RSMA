"""
compare_runs.py  —  head-to-head comparison pulling DIRECTLY from saved result folders
--------------------------------------------------------------------------------------
Reads each run's results/.../training_log.txt + hyperparameters.json straight from disk
(NOT from docs — gold baselines are kept on disk and are the source of truth). Prints,
per run: config (K/M/N/P_S/spawn), feasibility probe (Servable/Direct/All-IRS), per-ckpt
SAMPLED QoS/reward/R_tot windows (±W ep), and peak-QoS. Use for fair lever comparisons.

Usage:
  python analysis/compare_runs.py "results/result_26" "results/Training-Case 2/result_11" \
         "results/result_27" --labels combo gold N48ctrl
"""
import sys, os, re, json, argparse, statistics as st

EP_PAT = re.compile(
    r"^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+).*?\s(\d\.\d{3})\s+([\d.]+)\s+"
    r"([\d.]+)/(\d+)\s+([\d.]+)/\d+\s+([\d.]+)/\d+")  # ep,rew,best,..,Rtot,R/u,QoS/K,irs,blk


def parse(dir_):
    hp = json.load(open(os.path.join(dir_, "hyperparameters.json"), encoding="utf-8"))
    s = hp.get("system", {})
    cfg = dict(K=s.get("K"), M=s.get("M"), N=s.get("N"), P_S=s.get("P_S_dBm"),
               R_LoS=s.get("R_LoS_km"), spawn=("balanced" if s.get("balanced_blocked_spawn")
               else "binomial"), cf=s.get("counterfactual_assign", False))
    log = os.path.join(dir_, "training_log.txt")
    probe, rows = {}, []
    with open(log, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if "Servable users" in ln: probe["serv"] = ln.split(":")[1].split("%")[0].strip()
            elif "Direct-only" in ln:  probe["dir"]  = ln.split(":")[1].split("%")[0].strip()
            elif "All-IRS" in ln:      probe["irs"]  = ln.split(":")[1].split("%")[0].strip()
            m = EP_PAT.match(ln)
            if m:
                ep, rew, rtot = int(m.group(1)), float(m.group(2)), float(m.group(4))
                qosK, K = float(m.group(6)), int(m.group(7))
                rows.append((ep, rew, rtot, 100.0 * qosK / K))
    return cfg, probe, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--win", type=int, default=25, help="±ep window for ckpt stats")
    ap.add_argument("--step", type=int, default=100)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(d) for d in args.dirs]

    runs = []
    for d, lab in zip(args.dirs, labels):
        cfg, probe, rows = parse(d)
        # robust peak = best ±win WINDOW mean (single-ep max is noise — e.g. r11 had a
        # lucky 73% ep while its stable level is ~60%). Slide window over all eps.
        eps = [r[0] for r in rows]
        best = None
        for c in range(eps[0], eps[-1] + 1, max(1, args.win)):
            w = [r for r in rows if c - args.win <= r[0] <= c + args.win]
            if len(w) >= 5:
                q = st.mean(r[3] for r in w)
                if best is None or q > best[3]:
                    best = (c, st.mean(r[1] for r in w), st.mean(r[2] for r in w), q)
        peak = best or (rows[-1][0], rows[-1][1], rows[-1][2], rows[-1][3])
        runs.append((lab, d, cfg, probe, rows, peak))
        print("=" * 78)
        print(f"  {lab}  ·  {d}")
        print(f"    K{cfg['K']} M{cfg['M']} N{cfg['N']} P_S{cfg['P_S']} R{cfg['R_LoS']} "
              f"spawn={cfg['spawn']} cf={cfg['cf']}")
        print(f"    feasibility: Servable {probe.get('serv','?')}% · Direct {probe.get('dir','?')}% "
              f"· All-IRS {probe.get('irs','?')}%   ← QoS ceiling/baseline")
        print(f"    ep {rows[0][0]}..{rows[-1][0]}  ·  PEAK QoS (±{args.win}ep window) "
              f"{peak[3]:.1f}% @ep{peak[0]} (R_tot {peak[2]:.2f}, rew {peak[1]:.0f})")
        print(f"    {'ckpt':>8} | {'QoS%':>6} | {'R_tot':>5} | {'rew':>7}")
        for ck in range(args.step, rows[-1][0] + 1, args.step):
            w = [r for r in rows if ck - args.win <= r[0] <= ck + args.win]
            if w:
                print(f"    ep_{ck:05d} | {st.mean(r[3] for r in w):5.1f}% | "
                      f"{st.mean(r[2] for r in w):5.2f} | {st.mean(r[1] for r in w):7.1f}")
    print("=" * 78)
    print("  PEAK-QoS (sampled) head-to-head:")
    for lab, d, cfg, probe, rows, peak in runs:
        print(f"    {lab:<10} {peak[3]:5.1f}% @ep{peak[0]:<5} (R_tot {peak[2]:.2f}) "
              f"| All-IRS ceiling {probe.get('irs','?')}% | N{cfg['N']} spawn={cfg['spawn']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
