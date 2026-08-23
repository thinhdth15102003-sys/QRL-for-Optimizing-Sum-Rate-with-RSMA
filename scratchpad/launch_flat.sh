#!/bin/bash
# Flat-PPO baseline (Meng 2024 method) — 3 seeds per case.
#
# params.py is GLOBAL and read at import time, so K/M must be sed-ed in and the
# process given time to import before the next launch overwrites it. Same
# constraint the VQC scheduler works around; the sleep is what makes it safe.
#
# 3 seeds because a number that goes into a paper table needs training-seed
# variance, and infer-seed cannot substitute for it.
#
# Measured cost under 8-core contention: Case 1 ~1.25 s/ep (~7h/20k),
# Case 2 ~1.75 s/ep (~10h/20k).
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
export CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

setp () { sed -i "s/^K        = [0-9]*/K        = $1/; s/^M        = [0-9]*/M        = $2/" params.py; }

# Case 1 (K=5, M=1)
setp 5 1
for s in 0 1 2; do
  nohup $PY -u train_flat.py --episodes 20000 --seed $s \
      > "analysis/data/flat_c1_s$s.log" 2>&1 &
  echo "[$(date +%H:%M:%S)] launched flat_c1_s$s"
  sleep 30
done

# Case 2 (K=10, M=2)
setp 10 2
for s in 0 1 2; do
  nohup $PY -u train_flat.py --episodes 20000 --seed $s \
      > "analysis/data/flat_c2_s$s.log" 2>&1 &
  echo "[$(date +%H:%M:%S)] launched flat_c2_s$s"
  sleep 30
done

echo "[$(date +%H:%M:%S)] done; flat runs=$(pgrep -fc 'train_flat')"
echo "params.py left at K=10 M=2 — restore with: cp params_master.py params.py"
