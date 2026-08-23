#!/bin/bash
# Launch the two remaining Case-1 readout ablation arms: nn-zz and full-zz.
# Both need IDENTICAL params.py (K=5, M=1, nq=8, n_hidden_ae=[16]); only --readout
# differs and that is a CLI flag. So: one lock, set once, fire both, restore once.
# Mirrors results/result_410 (C1 single-z) exactly apart from --readout.

set -u
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
META=analysis/data/launch_ro_c1_pair.meta
BAK=/tmp/params_before_ro_c1_pair.py

# Match the environment every other run in this comparison was trained under.
# GPU is ~12x SLOWER here (n_q=8 -> 256 amplitudes, all kernel-launch overhead),
# and unpinned BLAS threads oversubscribe the 8-core box AND change reduction
# order, which would confound an ablation whose premise is "only readout differs".
export CUDA_VISIBLE_DEVICES=""
export QRL_GPU_MLP=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

RECIPE="--actor-mode quantum --episodes 20000 --seed 0 \
--oracle-warmup \
--assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 \
--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80"

echo "=== launch ro_c1 pair $(date '+%F %T') ===" >> "$META"

exec 9>/tmp/params_launcher.lock
flock -w 60 9 || { echo "  !! lock timeout" >> "$META"; exit 1; }
echo "  lock acquired $(date '+%F %T')" >> "$META"

cp params.py "$BAK"
$PY - <<'EOF'
import io, re
p = "params.py"
s = io.open(p, encoding="utf-8", newline="").read()
subs = [
    (r"^K        = 10", "K        = 5 "),
    (r"^M        = 2",  "M        = 1"),
    (r"^n_qubits = 12", "n_qubits = 8 "),
    (r"^n_hidden_ae   = _hyp\['n_hidden_ae'\]", "n_hidden_ae   = [16]"),
]
for pat, rep in subs:
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, "FAILED to patch: " + pat
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("params.py patched -> K=5 M=1 nq=8 ae=[16]")
EOF
if [ $? -ne 0 ]; then
  echo "  !! params patch FAILED, restoring and aborting" >> "$META"
  cp "$BAK" params.py; exit 1
fi
grep -E "^(K|M|n_qubits|n_hidden_ae)\b" params.py | sed 's/^/  /' >> "$META"

for RO in nn-zz full-zz; do
  TAG=$(echo "$RO" | tr -d '-')
  LOG="analysis/data/ro_c1_${TAG}.log"
  setsid $PY -u train.py $RECIPE --readout "$RO" > "$LOG" 2>&1 9>&- &
  echo "  FIRED C1 $RO  pid=$!  log=$LOG" >> "$META"
  sleep 45          # let this process finish importing params.py before the next
done

sleep 75
cp "$BAK" params.py
echo "  params.py restored $(date '+%F %T')" >> "$META"
grep -E "^(K|M|n_qubits|n_hidden_ae)\b" params.py | sed 's/^/  /' >> "$META"

sleep 120
echo "--- after launch ---" >> "$META"
for d in $(ls -dt results/result_* 2>/dev/null | head -4); do
  nq=$(grep -m1 -oE "[0-9]+ qubits" "$d/training_log.txt" 2>/dev/null)
  ro=$(grep -m1 -oE "\-> [0-9]+-dim[^ ]*" "$d/training_log.txt" 2>/dev/null)
  kk=$(grep -m1 -oE "K=[0-9]+ users, M=[0-9]+" "$d/training_log.txt" 2>/dev/null)
  echo "  $d  $kk  $nq  $ro" >> "$META"
done
echo "  live: $(pgrep -fc 'train\.py')" >> "$META"
