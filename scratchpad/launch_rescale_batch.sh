#!/bin/bash
# Rescale-fix validation batch (8 runs). Power head input now scale-free (h/mean|h|)
# so it can rank users; closed-form-ck isolates the power fix from the ck head.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
export QRL_GPU_MLP=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
BASE="--actor-mode classical --oracle-warmup --assign-warmup-episodes 16 --assign-warmup-epochs 30 --phase-warmup-episodes 16 --phase-warmup-epochs 1500 --assign-aux-weight 0.3 --phase-aux-weight 0.5 --closed-form-ck --power-priv-frac 0.60"
# config = "power_aux_weight power_fairness"
for cfg in "0.3 0.0" "0.3 0.1" "0.5 0.0" "0.5 0.1" "0.5 0.3" "0.7 0.1" "0.0 0.3" "0.0 0.1"; do
  set -- $cfg
  nohup $PY train.py $BASE --power-aux-weight "$1" --power-fairness "$2" --power-fairness-priv "$2" \
      > "analysis/data/rescale_paw${1}_a${2}.log" 2>&1 &
  sleep 25
done
wait
