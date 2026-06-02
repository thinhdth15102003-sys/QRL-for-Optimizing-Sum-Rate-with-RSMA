"""
sweep_oracle_critic.py
----------------------
Tier-0 COMPREHENSIVE sweep: train the oracle critic over a GRID of
architecture × grad-clip × target-normalisation on ONE fixed clean dataset.

Purpose
-------
Disambiguate the persistent [B-2b] unstable-critic on CLEAN data, free of the
policy-drift confound that plagues live training.  Answers, in one cheap CPU
run, which knob actually moves explVar toward the Tier-1 ceiling (~0.81 under
a trained policy):

  H1  grad_clip   : does clipping at norm-1 (live default) strangle the fit?
  H2  architecture: is [512,256,128,64] over-parameterised for a 44-dim input
                    (ReLU dead 46-63% observed live)?  Does [128,64] fit better?
  H3  target-norm : does predicting z-scored return (PopArt-style) fit better
                    than raw return (return scale drift observed live)?

Output
------
  results/oracle_sweep_K{K}M{M}_R{R}.txt   — ranked table + verdict
  results/oracle_sweep_K{K}M{M}_R{R}.csv   — machine-readable grid

CPU-only (random policy + numpy critic) → safe to run alongside GPU training.

Usage
-----
    python sweep_oracle_critic.py
    python sweep_oracle_critic.py --n_episodes 500 --epochs 200
    python sweep_oracle_critic.py --dataset results/oracle_dataset_K10M2_R0.20.npz
"""
from __future__ import annotations

# ── path bootstrap: make project root importable when run as script ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ─────────────────────────────────────────────────────────────────────────

import os
import csv
import json
import time
import argparse
import itertools
import numpy as np

import params as P
from params import make_config
from train_oracle_critic import collect_oracle_dataset, train_oracle


# ── Grid definition ─────────────────────────────────────────────────────────────
# Keep terse; expand only if a knob proves interesting.
GRID = {
    'arch': [
        [512, 256, 128, 64],   # current live arch
        [256, 128],
        [128, 64],
        [64, 64],              # Schulman-style tiny critic
    ],
    'grad_clip': [
        1.0,        # live default (binds hard)
        100.0,
        1000.0,     # ≈ live diagnostic value
        0.0,        # OFF (unclipped)
    ],
    'normalize_targets': [True, False],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--r_los', type=float, default=None)
    ap.add_argument('--n_episodes', type=int, default=400)
    ap.add_argument('--steps_per_ep', type=int, default=50)
    ap.add_argument('--reward_noise_avg', type=int, default=128)
    ap.add_argument('--gamma', type=float, default=getattr(P, 'gamma', 0.95))
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--seed', type=int, default=20260531)
    ap.add_argument('--dataset', type=str, default=None,
                    help="Reuse an existing dataset npz instead of collecting.")
    ap.add_argument('--out_dir', type=str, default='results')
    args = ap.parse_args()

    overrides = {} if args.r_los is None else {'R_LoS_km': args.r_los}
    cfg = make_config(**overrides)
    case_tag = f"K{cfg.K}M{cfg.M}_R{cfg.R_LoS_km:.2f}"
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Dataset (collect once, reuse across all grid cells) ──
    ds_path = (args.dataset
               or os.path.join(args.out_dir, f"oracle_dataset_{case_tag}.npz"))
    if args.dataset and os.path.isfile(args.dataset):
        print(f"Loading dataset {ds_path}")
        d = np.load(ds_path)
        dataset = {k: (d[k].item() if d[k].shape == () else d[k]) for k in d.files}
    else:
        print(f"Collecting clean dataset (reward_noise_avg={args.reward_noise_avg}) "
              f"→ {ds_path}")
        dataset = collect_oracle_dataset(
            cfg, n_episodes=args.n_episodes, steps_per_ep=args.steps_per_ep,
            reward_noise_avg=args.reward_noise_avg, seed=args.seed,
            gamma=args.gamma)
        np.savez(ds_path, **{k: np.asarray(v) for k, v in dataset.items()})
        print(f"  saved {len(dataset['states'])} samples")

    # ── Sweep ──
    combos = list(itertools.product(
        GRID['arch'], GRID['grad_clip'], GRID['normalize_targets']))
    print(f"\nSweeping {len(combos)} configs "
          f"({len(GRID['arch'])} arch × {len(GRID['grad_clip'])} clip × "
          f"{len(GRID['normalize_targets'])} norm)\n")

    rows = []
    t_start = time.time()
    for i, (arch, clip, norm) in enumerate(combos):
        t0 = time.time()
        res = train_oracle(dataset, hidden=arch, epochs=args.epochs,
                           lr=args.lr, batch_size=args.batch_size,
                           seed=args.seed, normalize_targets=norm,
                           grad_clip=clip)
        dt = time.time() - t0
        row = {
            'arch'      : 'x'.join(map(str, arch)),
            'grad_clip' : clip,
            'norm_tgt'  : norm,
            'best_EV'   : round(res['best_ev_raw'], 4),
            'final_EV'  : round(res['final_ev_raw'], 4),
            'sec'       : round(dt, 1),
        }
        rows.append(row)
        print(f"  [{i+1:2d}/{len(combos)}] arch={row['arch']:18s} "
              f"clip={clip:8.1f} norm={str(norm):5s} → "
              f"best_EV={row['best_EV']:+.3f} final={row['final_EV']:+.3f} "
              f"({dt:.0f}s)")

    rows.sort(key=lambda r: r['best_EV'], reverse=True)

    # ── CSV ──
    csv_path = os.path.join(args.out_dir, f"oracle_sweep_{case_tag}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── TXT report ──
    txt_path = os.path.join(args.out_dir, f"oracle_sweep_{case_tag}.txt")
    L = []
    L.append("=" * 78)
    L.append("  ORACLE CRITIC SWEEP  ·  Tier-0 comprehensive diagnostic")
    L.append("=" * 78)
    L.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  "
             f"({time.time()-t_start:.0f}s total)")
    L.append(f"  Case K={cfg.K} M={cfg.M} R_LoS_km={cfg.R_LoS_km}  "
             f"dataset N={len(dataset['states'])} "
             f"(reward_noise_avg={args.reward_noise_avg})")
    L.append(f"  Tier-1 ceiling (trained policy ref) ≈ 0.81  ·  "
             f"online (result_5/6) ≈ 0.40")
    L.append("")
    L.append("  RANKED BY best_EV (raw-scale validation explained-variance)")
    L.append("  " + "-" * 74)
    L.append(f"  {'arch':18s} {'clip':>9s} {'norm':>6s} {'best_EV':>9s} {'final_EV':>9s}")
    for r in rows:
        L.append(f"  {r['arch']:18s} {r['grad_clip']:9.1f} "
                 f"{str(r['norm_tgt']):>6s} {r['best_EV']:+9.3f} {r['final_EV']:+9.3f}")
    L.append("")
    # ── Marginal analysis: which knob matters most? ──
    L.append("  MARGINAL EFFECT (mean best_EV holding one knob) ")
    L.append("  " + "-" * 74)
    def _marg(key):
        vals = {}
        for r in rows:
            vals.setdefault(r[key], []).append(r['best_EV'])
        return {k: float(np.mean(v)) for k, v in vals.items()}
    for knob in ('arch', 'grad_clip', 'norm_tgt'):
        m = _marg(knob)
        spread = max(m.values()) - min(m.values())
        L.append(f"  {knob:10s} (spread {spread:+.3f}): " +
                 "  ".join(f"{k}={v:+.3f}" for k, v in
                          sorted(m.items(), key=lambda kv: -kv[1])))
    L.append("")
    L.append("  VERDICT GUIDE")
    L.append("  " + "-" * 74)
    L.append("  • Knob with LARGEST spread = the dominant lever → apply live.")
    L.append("  • If best_EV ≈ 0.81 ceiling → arch CAN fit; live instability is")
    L.append("    pure training-dynamics (clip / lr / policy-drift) → fixable.")
    L.append("  • If best_EV plateaus ≪ 0.81 even unclipped+normalized+small-arch")
    L.append("    → arch/feature bottleneck → redesign critic input or capacity.")
    L.append("  • grad_clip=1.0 row ≪ grad_clip=1000/off row → confirms clip was")
    L.append("    strangling the live critic (H1) → keep grad_clip_critic high.")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))

    print(f"\nReports → {txt_path}\n          {csv_path}")
    print("\nTOP 3:")
    for r in rows[:3]:
        print(f"  {r['arch']:18s} clip={r['grad_clip']:.0f} "
              f"norm={r['norm_tgt']} → best_EV={r['best_EV']:+.3f}")


if __name__ == '__main__':
    main()
