#!/bin/bash
# Wait for five of the eight training slots to free, then launch the classical
# policy-width sweep that fills the five empty rows of tab:abl_modelscale.
#
# Recipe is r231's (the (40) row already in that table) to the flag, differing
# only in --pol-hidden, so the new points are comparable to their own baseline.
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1

LOG=analysis/data/waiter_dnn_width.log
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
say() { echo "$(date -Is)  $*" >> "$LOG"; }

: > "$LOG"
say "waiter up (pid $$) — waiting for the train.py count to reach 3"
say "pattern is 'train\\.py': it matches -u runs and does NOT match train_flat.py"

while true; do
    n=$(pgrep -fc 'train\.py')
    [ "${n:-0}" -le 3 ] && break
    sleep 300
done
say "slots free — live train.py count = ${n:-0}"

# params.py carries the active case and has NO CLI flag for K/M, so a hand-edit
# between now and launch would silently run the wrong case. Check before
# committing five 15-hour runs.
KM=$($PY -c 'import params; print(params.K, params.M)' 2>/dev/null)
if [ "$KM" != "10 2" ]; then
    say "ABORT — params.py active case is ($KM), expected (10 2)"
    exit 1
fi
say "active case OK: K,M = $KM"

# Sub-actor widths also have no CLI flag; Tables VI/VII were trained on the
# shrunk set and a wide params.py would inflate every count by ~8x.
SUB=$($PY -c 'import params; print(params.n_hidden_phase, params.n_hidden_power, params.n_hidden_ck)' 2>/dev/null)
say "sub-actor widths: $SUB  (expect [64, 64] [64, 32] [64, 32])"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0

i=0
for W in "256 128" "128 128" "64 64" "64 32" "32 16"; do
    i=$((i + 1))
    $PY -u train.py --actor-mode classical --pol-hidden $W \
        --episodes 20000 --seed 0 \
        --oracle-warmup --assign-warmup-episodes 16 --assign-warmup-epochs 30 \
        --phase-warmup-episodes 16 --phase-warmup-epochs 1500 \
        --assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
        --power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80 \
        > "analysis/data/dnn_width_${i}.log" 2>&1 &
    say "launched  --pol-hidden $W   pid $!"
    # let each run claim its results/result_N before the next starts, or the
    # scheduler hands the same number to two processes
    sleep 25
done

say "all five launched; waiting for them to finish"
wait
say "all five finished"
for f in analysis/data/dnn_width_*.log; do
    d=$(grep -oE "results/result_[0-9]+" "$f" | head -1)
    say "  $f -> ${d:-?}"
done
