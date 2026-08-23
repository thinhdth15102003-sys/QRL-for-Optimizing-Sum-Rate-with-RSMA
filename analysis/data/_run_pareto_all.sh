#!/bin/bash
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
LOG=analysis/data/pareto_all_run.log
: > "$LOG"
echo "start $(date -Is)" >> "$LOG"
nice -n 5 $PY analysis/probe_pareto_exhaustive.py --K 5 --M 1 --shard 0/1 \
    > analysis/data/pareto_c1_rerun.log 2>&1
echo "C1 done $(date -Is)" >> "$LOG"
for i in 0 1 2 3 4 5 6 7; do
    nice -n 5 $PY analysis/probe_pareto_exhaustive.py --K 10 --M 2 --shard "$i/8" \
        > "analysis/data/pareto_c2_shard$i.log" 2>&1 &
done
wait
echo "C2 done $(date -Is)" >> "$LOG"
grep -h "states in" analysis/data/pareto_c1_rerun.log analysis/data/pareto_c2_shard*.log >> "$LOG"
