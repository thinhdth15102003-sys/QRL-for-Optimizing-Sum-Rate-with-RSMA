#!/bin/bash
# T3.4 — probe_lambda_landscape across 5 result_11 ckpts for variance band
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
for ep in 00300 00600 01000 01400 01700; do
  echo "========== ep_${ep} =========="
  "$PY" analysis/probe_lambda_landscape.py \
    --ckpt "results/result_11/checkpoints/ep_${ep}" \
    --states 60 2>&1 | tee "results/result_11/lam_landscape_ep${ep}.txt" | tail -20
  echo
done
echo "[T3.4 DONE]"
