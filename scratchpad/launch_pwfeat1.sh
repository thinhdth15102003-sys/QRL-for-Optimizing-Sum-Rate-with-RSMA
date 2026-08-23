#!/bin/bash
# Single scout run: DNN Case 2 with --power-scale-feats, seed 0.
#
# No loop. The looped version launched twice (two processes, one log file, the
# second truncating the first), so this one is deliberately a straight-line
# script with a guard: if a run with this flag is already alive it refuses to
# start a second one rather than silently duplicating.
set -e
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
LOG=analysis/data/dnn_c2_pwfeat_s0.log

if pgrep -f "train\.py.*--power-scale-feats" >/dev/null; then
  echo "REFUSING: a --power-scale-feats run is already alive"; exit 1
fi

cp params_master.py params.py
sed -i "s/^K        = [0-9]*/K        = 10/; s/^M        = [0-9]*/M        = 2/" params.py
sed -i "s/^n_hidden_phase = \[.*\]/n_hidden_phase = [64, 64]/" params.py
sed -i "s/^n_hidden_power = \[.*\]/n_hidden_power = [64, 32]/" params.py
sed -i "s/^n_hidden_ck    = \[.*\]/n_hidden_ck    = [64, 32]/" params.py

CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
nohup $PY -u train.py \
  --actor-mode classical --episodes 10000 --oracle-warmup \
  --assign-warmup-episodes 16 --assign-warmup-epochs 30 \
  --phase-warmup-episodes 16 --phase-warmup-epochs 1500 --phase-warmup-target 85 \
  --assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
  --power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80 \
  --power-scale-feats --seed 0 > "$LOG" 2>&1 &

echo "launched pid=$! -> $LOG"
sleep 50                       # let it import params.py before restoring
cp params_master.py params.py
echo "params.py restored; train procs=$(pgrep -fc 'train\.py')"
