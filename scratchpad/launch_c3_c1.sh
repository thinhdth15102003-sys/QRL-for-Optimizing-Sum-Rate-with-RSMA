#!/bin/bash
# Two runs:
#   A. Case 3 DNN, BASELINE recipe (no --power-scale-feats). Case 3 has never
#      been trained; tab:case3_main does not exist and the Case-3 column of
#      tab:param_budget is all em-dashes. Running it with an unvalidated flag
#      would confound "does Case 3 train at all" with "does the flag help", so
#      the baseline goes first. Smoke measured ~1.0 s/ep -> 20k in ~5.6 h.
#   B. Case 1 DNN WITH the flag, to pair against the existing r234/238/239/240.
#
# Guards are per-log rather than per-flag: the Case-2 scout (r270) is already
# running with --power-scale-feats, so a flag-wide guard would block B.
set -e
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL"
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

setp () {   # $1=K $2=M
  cp params_master.py params.py
  sed -i "s/^K        = [0-9]*/K        = $1/; s/^M        = [0-9]*/M        = $2/" params.py
  sed -i "s/^n_hidden_phase = \[.*\]/n_hidden_phase = [64, 64]/" params.py
  sed -i "s/^n_hidden_power = \[.*\]/n_hidden_power = [64, 32]/" params.py
  sed -i "s/^n_hidden_ck    = \[.*\]/n_hidden_ck    = [64, 32]/" params.py
}

WARM="--oracle-warmup --assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 --phase-warmup-target 85"
AUX="--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80"

# ── A. Case 3, baseline ──
L3=analysis/data/dnn_c3_base_s0.log
if [ -s "$L3" ] && pgrep -f "train\.py.*--episodes 20000" >/dev/null; then
  echo "SKIP A: already running"
else
  setp 15 3
  nohup $PY -u train.py --actor-mode classical --episodes 20000 $WARM $AUX \
      --seed 0 > "$L3" 2>&1 &
  echo "[$(date +%H:%M:%S)] A launched: Case 3 DNN baseline -> $L3"
  sleep 60
fi

# ── B. Case 1, with the new power features ──
L1=analysis/data/dnn_c1_pwfeat_s0.log
setp 5 1
nohup $PY -u train.py --actor-mode classical --episodes 20000 $WARM $AUX \
    --power-scale-feats --seed 0 > "$L1" 2>&1 &
echo "[$(date +%H:%M:%S)] B launched: Case 1 DNN +power-scale-feats -> $L1"
sleep 60

cp params_master.py params.py
echo "[$(date +%H:%M:%S)] params.py restored; train procs=$(pgrep -fc 'train\.py')"
