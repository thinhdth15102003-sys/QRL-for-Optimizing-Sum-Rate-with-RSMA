#!/bin/bash
# Refill the four slots freed by killing r256-r259.
#
#   1 x VQC Case 2, seed 0, AE decoder RESTORED to [128,64]   <- the test
#   3 x VQC Case 1, seeds 2/3/4                               <- fills the rest
#
# The Case-2 run uses seed 0 deliberately: r261 is also seed 0 and keeps running
# untouched as the control, so the pair differs in exactly one variable
# (n_hidden_ae) and the comparison needs no seed correction.
#
# params.py is GLOBAL and read at import time, hence cp+sed immediately before
# each launch and a sleep long enough for the process to finish importing before
# the next sed rewrites the file.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# $1=K $2=M $3=ae   ("shrink" = current recipe, "full" = restored decoder)
setp () {
  cp params_master.py params.py
  sed -i "s/^K        = [0-9]*/K        = $1/; s/^M        = [0-9]*/M        = $2/" params.py
  sed -i "s/^n_hidden_phase = \[.*\]/n_hidden_phase = [64, 64]/" params.py
  sed -i "s/^n_hidden_power = \[.*\]/n_hidden_power = [64, 32]/" params.py
  sed -i "s/^n_hidden_ck    = \[.*\]/n_hidden_ck    = [64, 32]/" params.py
  if [ "$3" = "shrink" ]; then
    sed -i "s/n_hidden_ae=\[128, 64\]/n_hidden_ae=[32, 16]/" params.py
  fi
  sed -i "s/n_hidden_ae=\[32\]/n_hidden_ae=[16]/" params.py
  sed -i "s/^n_qubits = [0-9]*/n_qubits = 8/" params.py
}

BASE="--actor-mode quantum --episodes 20000 --oracle-warmup \
--assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 --phase-warmup-target 85 \
--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80"

# ── the AE test: Case 2, big decoder ──
setp 10 2 full
grep -n "n_hidden_ae=" params.py | sed 's/^/  [C2-full] /'
nohup $PY -u train.py $BASE --seed 0 > analysis/data/vqc_c2_aefull_s0.log 2>&1 &
echo "[$(date +%H:%M:%S)] launched VQC C2 seed0 AE=[128,64]"
sleep 60

# ── fill remaining slots: Case 1 seeds 2,3,4 (recipe unchanged) ──
setp 5 1 shrink
for s in 2 3 4; do
  nohup $PY -u train.py $BASE --seed $s > "analysis/data/vqc_c1_s$s.log" 2>&1 &
  echo "[$(date +%H:%M:%S)] launched VQC C1 seed$s"
  sleep 45
done

cp params_master.py params.py
echo "[$(date +%H:%M:%S)] params.py restored; running=$(pgrep -fc 'train\.py')"
