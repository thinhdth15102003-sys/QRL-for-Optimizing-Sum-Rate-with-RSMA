#!/bin/bash
# Waiter: fire the Case-2 VQC depth sweep (L=1, L=2) as the two PPO-flat C2
# N-shift runs free their cores. Fills the empty depth block of tab:abl_modelscale.
#
# Matched to the five Case-2 main-table seeds (r278/r320/r291/r323/r324), all of
# which run n_q=8, L=3, n_hidden_ae=[32,16] and POWER-FAIRNESS f=0.80.
# NOTE f=0.80, not the 0.60 an older note claimed -- verified from the log header
# of all five runs before writing this.
#
# n_var_layers and n_qubits and n_hidden_ae have NO CLI flag, so params.py is
# patched under the launcher lock and restored after each fire. L=1 and L=2 need
# different params, hence the lock is re-acquired per fire.
#
# Do NOT self-setsid here: under `wsl -e bash script` that makes setsid the
# initial process, it forks and exits, WSL tears the session down and the orphan
# dies. This script is launched AS a setsid'd child of a short-lived parent.

set -u
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
META=analysis/data/waiter_vqc_c2_depth.meta

export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RECIPE="--actor-mode quantum --episodes 20000 --seed 0 \
--oracle-warmup \
--assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 \
--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80"

echo "=== waiter VQC C2 depth armed $(date '+%F %T') pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') ===" >> "$META"
echo "  triggers: pid 4977 (r424, flat C2 N=16) -> L=1 ; pid 5098 (r425, flat C2 N=32) -> L=2" >> "$META"

alive () {
  [ -d "/proc/$1" ] || return 1
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'train_flat\.py'
}

fire () {   # $1 = L
  local LL="$1" LOG="analysis/data/vqc_c2_L$1.log" BAK=/tmp/params_before_vqc_L$1.py
  exec 9>/tmp/params_launcher.lock
  flock -w 3600 9 || { echo "  !! L=$LL lock timeout, skipped" >> "$META"; return 1; }
  echo "  lock acquired for L=$LL $(date '+%F %T')" >> "$META"
  cp params.py "$BAK"
  $PY - "$LL" <<'EOF'
import io, re, sys
want = sys.argv[1]
p = "params.py"
s = io.open(p, encoding="utf-8", newline="").read()
subs = [(r"^K\s*=\s*\d+",            "K        = 10"),
        (r"^M\s*=\s*\d+",            "M        = 2"),
        (r"^N\s*=\s*\d+",            "N = 24"),
        (r"^n_qubits\s*=\s*\d+",     "n_qubits = 8 "),
        (r"^n_hidden_ae\s*=.*$",     "n_hidden_ae   = [32,16]"),
        (r"^n_var_layers\s*=.*$",    "n_var_layers  = " + want)]
for pat, rep in subs:
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, "FAILED to patch " + pat
io.open(p, "w", encoding="utf-8", newline="").write(s)
EOF
  if [ $? -ne 0 ]; then
    echo "  !! params patch FAILED for L=$LL, restoring" >> "$META"
    cp "$BAK" params.py; exec 9>&-; return 1
  fi
  grep -E "^(K|M|N|n_qubits|n_hidden_ae|n_var_layers)\b" params.py | sed "s/^/  [L=$LL] /" >> "$META"
  setsid $PY -u train.py $RECIPE > "$LOG" 2>&1 9>&- &
  echo "  FIRED VQC C2 L=$LL  pid=$!  log=$LOG" >> "$META"
  sleep 90
  cp "$BAK" params.py
  echo "  params.py restored $(date '+%F %T')" >> "$META"
  exec 9>&-
}

for spec in "4977:1" "5098:2"; do
  TPID="${spec%%:*}"; LL="${spec##*:}"
  echo "  waiting on pid $TPID for L=$LL ..." >> "$META"
  while alive "$TPID"; do sleep 120; done
  echo "  trigger pid $TPID exited $(date '+%F %T')" >> "$META"
  fire "$LL"
done

sleep 120
echo "--- verify (expect n_q=8, L as fired, core = 48+16L) ---" >> "$META"
for d in $(ls -dt results/result_* 2>/dev/null | head -4); do
  grep -m1 -oE "QC     :.*" "$d/training_log.txt" 2>/dev/null \
    | sed "s|^|  $d  |" >> "$META"
done
echo "  waiter done $(date '+%F %T')" >> "$META"
