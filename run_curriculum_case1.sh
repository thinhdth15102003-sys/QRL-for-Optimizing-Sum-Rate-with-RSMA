#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Automated Case-1 (no-ae) curriculum:  R_LoS 0.3 → 0.4 → 0.45 → 0.5
#   • 1000 episodes per ramp
#   • after each ramp, --resume <prev run-dir> auto-picks the BEST sweet-spot ckpt
#     (train.py runs analysis/pick_resume_ckpt.py internally)
#   • final ramp 0.5: λ_D 1.5→2.0  +  spawn radii → 1.0 (unlock the full LoS disk)
#
# USER runs this (it launches train.py sequentially for many hours):
#   cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" && nohup bash run_curriculum_case1.sh > curriculum.log 2>&1 &
# Resume point = the converged ramp-0.2 run (edit PREV below if different).
# ──────────────────────────────────────────────────────────────────────────────
set -uo pipefail
PY="/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11"
PROJ="/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
cd "$PROJ" || { echo "FATAL: cannot cd $PROJ"; exit 1; }

PREV="results/result_67"     # ⭐ converged ramp-0.2 Case-1 no-ae run (resume its best ckpt)
EPISODES=1000
PWFAIR=0.5                    # Case-1 power-fairness α

run_ramp () {
  local rlos="$1"; local lam="$2"; shift 2; local extra=("$@")
  echo "════════════════════════════════════════════════════════════════════"
  echo "  RAMP R_LoS=$rlos  λ_D=$lam  resume=$PREV  extra=${extra[*]:-none}"
  echo "════════════════════════════════════════════════════════════════════"
  local logf="curriculum_ramp_${rlos}.log"
  "$PY" train.py --no-ae --resume "$PREV" \
        --R-LoS "$rlos" --lambda-D "$lam" --power-fairness "$PWFAIR" \
        --episodes "$EPISODES" "${extra[@]}" 2>&1 | tee "$logf"
  # parse THIS run's output dir (the final "Results → results/result_N" line)
  local out
  out=$(grep -oE "results/result_[0-9]+" "$logf" | tail -1)
  if [ -z "$out" ] || [ ! -d "$out/checkpoints" ]; then
    echo "FATAL: could not resolve output run-dir after ramp $rlos (got '$out')"; exit 1
  fi
  PREV="$out"
  echo "──── ramp $rlos COMPLETE → next resume = $PREV ────"
}

# ── intermediate ramps: λ_D=1.5 (⚠ [I-3] suggests gently raising λ 1.5→1.8→2.0 if
#    QoS drifts at 0.4/0.45 — edit per-ramp λ below if you see user-abandonment drift)
run_ramp 0.3  1.5
run_ramp 0.4  1.5
run_ramp 0.45 1.5

# ── ramp 0.5: bump λ→2.0 (result_45 precedent) — converge with current spawn first
run_ramp 0.5  2.0

# ── FINAL UNLOCK: spawn radii → 1.0 (IRS + free users fill the full LoS disk).
#    Done as a SEPARATE step (resume the 0.5-converged policy) so it adapts to the
#    harder spawn gradually instead of jumping ramp+λ+spawn all at once.
#    (To merge into the 0.5 ramp instead, move these two flags up to the run above.)
run_ramp 0.5  2.0  --irs-spawn-frac 1.0 --user-free-frac 1.0

echo "✅ CURRICULUM COMPLETE.  Final run-dir: $PREV"
