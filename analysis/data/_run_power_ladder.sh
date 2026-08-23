#!/bin/bash
# Private-power allocation across the transmit-power ladder, both cases.
#
# Checkpoints are the ones behind Table VI / Table IX, taken from
# TABLE-PROVENANCE.md. Every cell was verified to carry the shrunk sub-actor
# widths (power [64,32], phase [64,64]), so the sweep varies P_S and nothing
# else; the earlier ladder runs that mixed widths (r317, r336/r353, r329/r349)
# were superseded and are not used.
#
# At P_S = 50 the table cells are five-seed means but the shifted rungs are
# single-seed, so seed 0 is used at every rung to keep the sweep comparable.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
OUT=analysis/data/power_ladder.jsonl
LOG=analysis/data/power_ladder_run.log
: > "$OUT"; : > "$LOG"

run () {   # K M P_S  VQC_dir  DNN_dir  FLAT_dir
    echo "── K=$1 M=$2 P_S=$3" | tee -a "$LOG"
    $PY analysis/probe_power_concentration.py --K "$1" --M "$2" --P "$3" \
        --seeds 42 43 44 --steps 200 \
        --hier "VQC=$4" "DNN=$5" --flat "flat=$6" 2>>"$LOG" \
      | tee -a "$LOG" | grep '^JSON ' | sed 's/^JSON //' >> "$OUT"
}

R=results
# ---- Case 1 (K=5, M=1) ----
run 5 1 40 $R/result_325            $R/result_374/checkpoints/ep_12600 $R/result_398/checkpoints/ep_17700
run 5 1 50 $R/result_328/checkpoints/ep_02700 $R/result_234/checkpoints/ep_07400 $R/result_301/checkpoints/ep_06000
run 5 1 60 $R/result_384/checkpoints/ep_00600 $R/result_364/checkpoints/ep_13100 $R/result_397/checkpoints/ep_12300
run 5 1 70 $R/result_373/checkpoints/ep_04400 $R/result_375/checkpoints/ep_02000 $R/result_399/checkpoints/ep_20004

# ---- Case 2 (K=10, M=2) ----
run 10 2 40 $R/result_404/checkpoints/ep_02400 $R/result_386/checkpoints/ep_04200 $R/result_400/checkpoints/ep_40008
run 10 2 50 $R/result_278/checkpoints/ep_00100 $R/result_231/checkpoints/ep_08700 $R/result_376/checkpoints/ep_40008
run 10 2 60 $R/result_405/checkpoints/ep_01000 $R/result_387/checkpoints/ep_06000 $R/result_401/checkpoints/ep_11100
run 10 2 70 $R/result_406/checkpoints/ep_02300 $R/result_389/checkpoints/ep_09500 $R/result_402/checkpoints/ep_40008

echo "wrote $(wc -l < "$OUT") rung(s) to $OUT" | tee -a "$LOG"
