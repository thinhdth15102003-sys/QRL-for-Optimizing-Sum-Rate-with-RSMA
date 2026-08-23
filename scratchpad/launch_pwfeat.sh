#!/bin/bash
# SCOUT: does giving the power head the absolute scale it currently cannot see
# close any of the 73% of the Case-2 gap that sits on the power axis?
#
# ONE seed: this is a scout, and the five Case-1 seeds plus the four Case-2
# ones all converged to within their own spread, so a single run is enough to
# tell a real effect from nothing. Seed 0 pairs with the control r231.
#
# DNN, not VQC — the question is about the power head, which is identical in both
# arms, and the DNN reaches the same operating point far faster. If it moves
# nothing here it will not move anything with a VQC in front of it.
#
# CONTROL already exists and needs no compute: r231/232/233/236, same recipe,
# same widths, same n_latent, WITHOUT --power-scale-feats. Compare at ep_10000.
# The only differing variable is the flag.
#
# n_qubits is left at the params_master default 12 so n_latent = 24 matches the
# control's power d_s of 2K+24 = 44; the DNN has no circuit, but n_latent still
# sets the rep the power head receives.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cp params_master.py params.py
sed -i "s/^K        = [0-9]*/K        = 10/; s/^M        = [0-9]*/M        = 2/" params.py
sed -i "s/^n_hidden_phase = \[.*\]/n_hidden_phase = [64, 64]/" params.py
sed -i "s/^n_hidden_power = \[.*\]/n_hidden_power = [64, 32]/" params.py
sed -i "s/^n_hidden_ck    = \[.*\]/n_hidden_ck    = [64, 32]/" params.py

BASE="--actor-mode classical --episodes 10000 --oracle-warmup \
--assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 --phase-warmup-target 85 \
--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80 \
--power-scale-feats"

for s in 0; do
  nohup $PY -u train.py $BASE --seed $s > "analysis/data/dnn_c2_pwfeat_s$s.log" 2>&1 &
  echo "[$(date +%H:%M:%S)] launched DNN C2 power-scale-feats seed $s"
  sleep 45
done

cp params_master.py params.py
echo "[$(date +%H:%M:%S)] params.py restored; running=$(pgrep -fc 'train\.py')"
