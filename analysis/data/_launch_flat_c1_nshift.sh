#!/bin/bash
# PPO-flat Case-1 N-shift cells for Table IX: N=16 and N=32.
# PPO-flat's state width is 10N+38 at C1, so an N shift changes the input layer
# and the policy cannot be evaluated zero-shot -- each N needs its own training.
# Baseline to match: results/result_301 / 307 (K=5, M=1, N=24, P_S=50, 20k eps).
#
# Unlike the readout pair, these two runs need DIFFERENT params.py (N differs),
# so the lock is held across BOTH launches and params is re-patched in between.
# Each fire is followed by a sleep long enough for that process to finish its
# import before the next sed touches the file -- this is the r394 failure mode.

set -u
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
META=analysis/data/launch_flat_c1_nshift.meta
BAK=/tmp/params_before_flat_nshift.py

export CUDA_VISIBLE_DEVICES=""
export QRL_GPU_MLP=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# train_flat.py defaults already equal the Case-1 recipe:
#   --lr 3e-4  --target-kl 0.05  --hidden 512,256
#   --power-fairness 0.0  --power-priv-frac 0.8
RECIPE="--episodes 20000 --seed 0"

echo "=== launch flat C1 N-shift $(date '+%F %T') ===" >> "$META"

# Lock a LAUNCHER-only file, and close fd 9 in every child (9>&- below).
# A child that inherits the lock fd keeps the lock held for its whole training
# run, which silently blocks the next launcher forever -- observed 2026-08-20.
exec 9>/tmp/params_launcher.lock
flock -w 60 9 || { echo "  !! could not acquire launcher lock in 60s, aborting" >> "$META"; exit 1; }
echo "  lock acquired $(date '+%F %T')" >> "$META"
cp params.py "$BAK"

patch_params () {   # $1 = N
  $PY - "$1" <<'EOF'
import io, re, sys
want_N = sys.argv[1]
p = "params.py"
s = io.open(p, encoding="utf-8", newline="").read()
subs = [
    (r"^K        = 10", "K        = 5 "),
    (r"^M        = 2",  "M        = 1"),
    (r"^N = 24",        "N = " + want_N),
]
for pat, rep in subs:
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, "FAILED to patch: " + pat
io.open(p, "w", encoding="utf-8", newline="").write(s)
EOF
}

for NN in 16 32; do
  cp "$BAK" params.py            # always patch from the pristine copy
  if ! patch_params "$NN"; then
    echo "  !! params patch FAILED for N=$NN, restoring and aborting" >> "$META"
    cp "$BAK" params.py; exit 1
  fi
  grep -E "^(K|M|N) " params.py | sed "s/^/  [N=$NN] /" >> "$META"
  LOG="analysis/data/flat_c1_n${NN}.log"
  setsid $PY -u train_flat.py $RECIPE > "$LOG" 2>&1 9>&- &
  echo "  FIRED flat C1 N=$NN  pid=$!  log=$LOG" >> "$META"
  sleep 75                        # let this process import params.py before the next sed
done

cp "$BAK" params.py
echo "  params.py restored $(date '+%F %T')" >> "$META"
grep -E "^(K|M|N) " params.py | sed 's/^/  /' >> "$META"

sleep 90
echo "--- verify ---" >> "$META"
for NN in 16 32; do
  ds=$(( 10*NN + 38 ))
  echo "  N=$NN expects d_s=$ds" >> "$META"
done
echo "  live train_flat: $(pgrep -fc 'train_flat\.py')" >> "$META"
