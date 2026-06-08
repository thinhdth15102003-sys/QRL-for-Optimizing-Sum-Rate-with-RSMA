"""
pick_resume_ckpt.py
-------------------
Sweet-spot checkpoint picker for ramp transitions.

Problem (user 2026-06-08): khi resume sang ramp mới (R_LoS 0.2→0.3), chọn ckpt
nào? Quá NON (early) → policy chưa stable, transfer poor. Quá DETERMINISTIC (late)
→ ent_ph collapsed → không adapt được ramp mới. Hiện tại agent + critic
"unrecoverable" sau drift → khó pick.

Sweet-spot criteria (4-metric AND):
  1. reward μ (rolling-50) ≥ 80% of best_reward observed
  2. ent_ph in [40%, 60%] of M·N·ln(n_levels) max
     (not too high = idle [F]; not too low = collapsed)
  3. explVar > 0.5 stable (critic healthy)
  4. ≥ 20ep AFTER best_reward peak (stability margin, NOT lucky spike)

Outputs:
  - Sweet-spot ckpt path (max-ep candidate satisfying all criteria)
  - Diagnostic table cho all saved ckpts with per-criterion score
  - Verdict: PICK / NONE_FOUND / RUN_STILL_NON

Usage:
  python analysis/pick_resume_ckpt.py --run results/result_6
  python analysis/pick_resume_ckpt.py --run results/result_6 --reward-pct 0.7  # relax to 70%
  python analysis/pick_resume_ckpt.py --run "results/Training-Case 2/result_11" --target-ep 1000
"""

import sys, os, re, json, argparse
import numpy as np
from pathlib import Path


# ── log parsers ──────────────────────────────────────────────────────────────

RE_ROLLING50 = re.compile(r"rolling-50: reward μ=\s*([+-]?\d+\.\d+)\s+QoS μ=\s*(\d+)%\s+Rtot μ=([\d.]+)\s+\│\s+best=([+-]?\d+\.\d+)\s+\((\d+) ep ago\)")
RE_DIAG_EP = re.compile(r"^\s*┄\s*diag\[(\d+)\]")
RE_ENT = re.compile(r"ent q=([\d.]+) ph=([\d.]+) pw=([\d.]+) ck=([\d.]+)")
RE_CRITIC = re.compile(r"V̄=([+-]?\d+\.\d+)\s+σV=([\d.]+)\s+explVar=([+-]?[\d.]+)")
RE_STAR = re.compile(r"^\*\s+(\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)")
RE_BEHAV = re.compile(r"behaviour: grp dir=([\d.]+)")


def parse_log(log_path):
    """Parse training_log.txt → episodes dict keyed by diag ep."""
    if not log_path.exists():
        return {}, None
    lines = log_path.read_text(errors='ignore').split('\n')

    diags = {}  # ep → dict(ent_ph, explVar, V_bar, sigma_V, reward_mu, qos, rtot, eps_since_best)
    current_diag_ep = None
    best_reward = -float('inf')
    best_ep = None

    for ln in lines:
        # New diag block
        m = RE_DIAG_EP.match(ln)
        if m:
            current_diag_ep = int(m.group(1))
            diags.setdefault(current_diag_ep, {})
            continue

        # Critic V̄/σV/explVar inside current diag
        m = RE_CRITIC.search(ln)
        if m and current_diag_ep is not None:
            diags[current_diag_ep]['V_bar'] = float(m.group(1))
            diags[current_diag_ep]['sigma_V'] = float(m.group(2))
            diags[current_diag_ep]['explVar'] = float(m.group(3))
            continue

        # Entropy line
        m = RE_ENT.search(ln)
        if m and current_diag_ep is not None:
            diags[current_diag_ep]['ent_q'] = float(m.group(1))
            diags[current_diag_ep]['ent_ph'] = float(m.group(2))
            diags[current_diag_ep]['ent_pw'] = float(m.group(3))
            diags[current_diag_ep]['ent_ck'] = float(m.group(4))
            continue

        # Rolling-50
        m = RE_ROLLING50.search(ln)
        if m and current_diag_ep is not None:
            diags[current_diag_ep]['reward_mu'] = float(m.group(1))
            diags[current_diag_ep]['qos_pct'] = int(m.group(2))
            diags[current_diag_ep]['rtot_mu'] = float(m.group(3))
            diags[current_diag_ep]['best_so_far'] = float(m.group(4))
            diags[current_diag_ep]['eps_since_best'] = int(m.group(5))
            continue

        # Best-reward star line
        m = RE_STAR.match(ln)
        if m:
            r_val = float(m.group(2))
            if r_val > best_reward:
                best_reward = r_val
                best_ep = int(m.group(1))

    # Compute rolling-50 peak (for robust reward criterion)
    reward_mus = [d['reward_mu'] for d in diags.values() if 'reward_mu' in d]
    peak_rolling50 = max(reward_mus) if reward_mus else 0.0
    worst_rolling50 = min(reward_mus) if reward_mus else 0.0
    range_rolling50 = peak_rolling50 - worst_rolling50

    return diags, dict(best_reward=best_reward, best_ep=best_ep,
                       peak_rolling50=peak_rolling50, worst_rolling50=worst_rolling50,
                       range_rolling50=range_rolling50)


def get_ckpt_eps(ckpt_dir):
    """Return list of ckpt episode numbers from ep_NNNNN dir names."""
    if not ckpt_dir.exists():
        return []
    eps = []
    for d in ckpt_dir.iterdir():
        m = re.match(r"ep_(\d+)$", d.name)
        if m:
            eps.append(int(m.group(1)))
    return sorted(eps)


def nearest_diag(diags, target_ep, max_gap=80):
    """Find diag closest to target_ep (within max_gap)."""
    candidates = [(abs(ep - target_ep), ep) for ep in diags.keys() if abs(ep - target_ep) <= max_gap]
    if not candidates:
        return None
    return min(candidates)[1]


def get_case_dims(run_dir):
    """Read K, M, N from hyperparameters.json."""
    hp_path = run_dir / "hyperparameters.json"
    if not hp_path.exists():
        return None
    hp = json.loads(hp_path.read_text())
    sys_c = hp.get('system', {})
    return sys_c.get('K', 10), sys_c.get('M', 2), sys_c.get('N', 24)


# ── scoring ──────────────────────────────────────────────────────────────────

def score_ckpt(ep, diag_ep, diags, best_info, ent_ph_max, args):
    """Score a checkpoint against 4 sweet-spot criteria."""
    if diag_ep is None or diag_ep not in diags:
        return None
    d = diags[diag_ep]
    if 'reward_mu' not in d or 'explVar' not in d or 'ent_ph' not in d:
        return None

    best_ep = best_info['best_ep']

    # Criterion 1: reward μ within tolerance of PEAK ROLLING-50 (more robust than
    # comparing to single-ep "*" lucky spike which can be far above sustained reward).
    # Tolerance scales with reward-range across run → naturally handles negative scales.
    r_mu = d['reward_mu']
    peak_r_mu = best_info['peak_rolling50']      # best rolling-50 reward μ ever observed
    range_r = best_info['range_rolling50']       # peak - worst rolling-50
    tolerance = range_r * (1.0 - args.reward_pct)
    crit1 = r_mu >= (peak_r_mu - tolerance)
    best_r_local = peak_r_mu                     # display field

    # Criterion 2: ent_ph in [40%, 60%] of max
    ent_ph_pct = d['ent_ph'] / ent_ph_max
    crit2 = args.ent_low <= ent_ph_pct <= args.ent_high

    # Criterion 3: explVar > threshold stable
    explVar = d['explVar']
    crit3 = explVar > args.explvar_min

    # Criterion 4: stability margin (≥ stability_margin ep after peak)
    crit4 = (best_ep is None) or (ep >= best_ep + args.stability_margin)

    score = sum([crit1, crit2, crit3, crit4])

    return dict(
        ep=ep, diag_ep=diag_ep, reward_mu=r_mu, best_local=best_r_local,
        ent_ph=d['ent_ph'], ent_ph_pct=ent_ph_pct, explVar=explVar,
        eps_since_best=d.get('eps_since_best', None),
        crit1_reward=crit1, crit2_ent_ph=crit2, crit3_explVar=crit3, crit4_stability=crit4,
        score=score,
        qos_pct=d.get('qos_pct'), rtot_mu=d.get('rtot_mu'),
        V_bar=d.get('V_bar'),
    )


def emit_table(scores, ent_ph_max, args):
    """Pretty-print diagnostic table + verdict."""
    print()
    print("=" * 100)
    print(f"  SWEET-SPOT CKPT PICKER — criteria: reward ≥{int(args.reward_pct*100)}% best  AND  "
          f"ent_ph ∈ [{int(args.ent_low*100)},{int(args.ent_high*100)}]% max  AND  "
          f"explVar > {args.explvar_min}  AND  ≥{args.stability_margin}ep post-peak")
    print(f"  (ent_ph max = M·N·ln(4) = {ent_ph_max:.2f})")
    print("=" * 100)
    print(f"  {'ep':>6s}  {'reward μ':>9s}  {'best loc':>9s}  {'ent_ph':>7s}({'%max':>4s})  "
          f"{'explVar':>8s}  {'QoS':>4s}  {'R_tot':>6s}  {'V̄':>7s}  {'crit':>10s}  {'score':>5s}")
    print(f"  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*7}({'-'*4})  {'-'*8}  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*10}  {'-'*5}")
    for s in scores:
        crit_str = ("R" if s['crit1_reward'] else ".") + \
                   ("E" if s['crit2_ent_ph'] else ".") + \
                   ("V" if s['crit3_explVar'] else ".") + \
                   ("S" if s['crit4_stability'] else ".")
        v_bar = f"{s['V_bar']:+7.2f}" if s.get('V_bar') is not None else "   N/A"
        print(f"  {s['ep']:>6d}  {s['reward_mu']:>+9.2f}  {s['best_local']:>+9.2f}  "
              f"{s['ent_ph']:>7.2f}({s['ent_ph_pct']*100:>3.0f}%)  "
              f"{s['explVar']:>+8.3f}  {s.get('qos_pct','—'):>4}  {s.get('rtot_mu',0):>6.3f}  "
              f"{v_bar}  {crit_str:>10s}  {s['score']:>5d}/4")
    print("=" * 100)


def emit_verdict(scores, args):
    """Pick sweet-spot ckpt or report NONE_FOUND/RUN_STILL_NON."""
    perfect = [s for s in scores if s['score'] == 4]
    if perfect:
        # Latest perfect-score ckpt
        pick = max(perfect, key=lambda s: s['ep'])
        print(f"\n  ✅ PICK: ep_{pick['ep']:05d}  (4/4 criteria met, latest)")
        print(f"     reward μ={pick['reward_mu']:+.2f}  ent_ph={pick['ent_ph_pct']*100:.0f}% max  "
              f"explVar={pick['explVar']:+.3f}  R_tot={pick['rtot_mu']:.2f}")
        return pick

    # Relax to 3/4
    three = [s for s in scores if s['score'] == 3]
    if three:
        pick = max(three, key=lambda s: s['ep'])
        # Identify which crit missed
        missed = [name for name, k in
                  [("reward", "crit1_reward"), ("ent_ph", "crit2_ent_ph"),
                   ("explVar", "crit3_explVar"), ("stability", "crit4_stability")]
                  if not pick[k]]
        print(f"\n  🟡 BEST-AVAILABLE: ep_{pick['ep']:05d}  (3/4 criteria, missed: {missed[0]})")
        print(f"     reward μ={pick['reward_mu']:+.2f}  ent_ph={pick['ent_ph_pct']*100:.0f}% max  "
              f"explVar={pick['explVar']:+.3f}  R_tot={pick['rtot_mu']:.2f}")
        return pick

    # Worse
    two = [s for s in scores if s['score'] == 2]
    if two:
        print(f"\n  ⚠ NONE-FOUND: best score 2/4 (run may be too non, drifted, or critic dying)")
        print(f"     Consider continuing training OR relaxing criteria with --reward-pct 0.6")
        return None

    print(f"\n  ❌ RUN-STILL-NON: best score ≤ 1/4 (training likely premature)")
    print(f"     Wait until reward μ stabilizes near peak before picking resume ckpt.")
    return None


def main():
    ap = argparse.ArgumentParser(description="Auto-pick sweet-spot checkpoint for ramp transition.")
    ap.add_argument("--run", required=True, help="Path to results/result_N (or with subdir)")
    ap.add_argument("--reward-pct", type=float, default=0.8,
                    help="Reward floor as fraction of best (default 0.8 = 80%%)")
    ap.add_argument("--ent-low", type=float, default=0.40, help="ent_ph lower bound as %% of max")
    ap.add_argument("--ent-high", type=float, default=0.65, help="ent_ph upper bound as %% of max")
    ap.add_argument("--explvar-min", type=float, default=0.5, help="Min explVar threshold")
    ap.add_argument("--stability-margin", type=int, default=20,
                    help="Min ep AFTER best peak (avoid lucky spike)")
    ap.add_argument("--target-ep", type=int, default=None,
                    help="Optional: print recommendation focused near this ep")
    args = ap.parse_args()

    run_dir = Path(args.run).resolve()
    if not run_dir.is_dir():
        print(f"✗ Not a directory: {run_dir}")
        sys.exit(1)

    log_path = run_dir / "training_log.txt"
    ckpt_dir = run_dir / "checkpoints"

    case_dims = get_case_dims(run_dir)
    if case_dims is None:
        print(f"✗ Cannot read hyperparameters.json from {run_dir}")
        sys.exit(1)
    K, M, N = case_dims
    n_levels = 4  # 2-bit phase
    ent_ph_max = M * N * np.log(n_levels)

    print(f"\n  Run: {run_dir.name}  (K={K}, M={M}, N={N}, ent_ph_max={ent_ph_max:.2f})")

    diags, best_info = parse_log(log_path)
    if not diags:
        print(f"✗ No diag blocks parsed from {log_path}")
        sys.exit(1)
    if best_info['best_ep'] is None:
        print(f"✗ No best-reward marker found (no '*' lines)")
        sys.exit(1)

    ckpts = get_ckpt_eps(ckpt_dir)
    if not ckpts:
        print(f"✗ No ep_NNNNN checkpoints found in {ckpt_dir}")
        sys.exit(1)

    print(f"  Best so far: reward {best_info['best_reward']:+.2f} @ ep_{best_info['best_ep']:05d}")
    print(f"  Saved ckpts: {len(ckpts)} (ep_{ckpts[0]:05d} .. ep_{ckpts[-1]:05d})")

    # Score each ckpt
    scores = []
    for ep in ckpts:
        diag_ep = nearest_diag(diags, ep)
        s = score_ckpt(ep, diag_ep, diags, best_info, ent_ph_max, args)
        if s is not None:
            scores.append(s)

    if not scores:
        print(f"✗ Could not score any ckpts (diag/ckpt ep mismatch?)")
        sys.exit(1)

    emit_table(scores, ent_ph_max, args)
    pick = emit_verdict(scores, args)

    if pick:
        ckpt_path = ckpt_dir / f"ep_{pick['ep']:05d}"
        if ckpt_path.exists():
            print(f"\n  📂 Resume path: {ckpt_path}")
            print(f"     Use: --resume \"{ckpt_path}/agents\"")
        else:
            print(f"  ⚠ Ckpt dir {ckpt_path} not found on disk")


if __name__ == "__main__":
    main()
