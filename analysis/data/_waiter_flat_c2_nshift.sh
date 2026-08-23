#!/bin/bash
# Waiter: fire PPO-flat Case-2 at N=16 and N=32 as r422 / r423 (Case-1 N-shift)
# free their cores. Fills the remaining 6 cells of tab:abl_genshift.
#
# Baseline matched to the Case-2 flat family used in Table IX
# (r376/377/388/390/391): 40000 episodes, lr 1e-4. Everything else is
# train_flat.py default (lr-critic 3e-4, target-kl 0.05, hidden 512,256,
# power-fairness 0.0, power-priv-frac 0.8).
# Expected d_s = 40N + 75 -> 715 at N=16, 1355 at N=32 (baseline N=24 is 1035).

# NOTE: do NOT self-setsid with `exec setsid "$0"`. Under `wsl -e bash script`
# that makes setsid the initial process; it forks and exits immediately, WSL
# tears the session down and the orphan dies. This script is instead launched
# AS a setsid'd background child of a short-lived parent, the same shape that
# keeps the training processes alive.

set -u
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
META=analysis/data/waiter_flat_c2_nshift.meta

export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "=== waiter flat C2 N-shift armed $(date '+%F %T') pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') ===" >> "$META"
echo "  triggers: pid 3704 (r422, C1 N=16) -> C2 N=16 ; pid 3750 (r423, C1 N=32) -> C2 N=32" >> "$META"

alive () {   # $1 = pid; true only while it is still a train_flat.py process
  [ -d "/proc/$1" ] || return 1
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'train_flat\.py'
}

fire () {    # $1 = N
  local NN="$1" LOG="analysis/data/flat_c2_n$1.log" BAK=/tmp/params_before_flat_c2_n$1.py
  exec 9>/tmp/params_launcher.lock
  flock -w 3600 9 || { echo "  !! N=$NN lock timeout, skipped" >> "$META"; return 1; }
  echo "  lock acquired for N=$NN $(date '+%F %T')" >> "$META"
  cp params.py "$BAK"
  $PY - "$NN" <<'EOF'
import io, re, sys
want = sys.argv[1]
p = "params.py"
s = io.open(p, encoding="utf-8", newline="").read()
# idempotent: set the value whatever it currently is
for pat, rep in ((r"^K\s*=\s*\d+", "K        = 10"),
                 (r"^M\s*=\s*\d+", "M        = 2"),
                 (r"^N\s*=\s*\d+", "N = " + want)):
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, "FAILED to patch " + pat
io.open(p, "w", encoding="utf-8", newline="").write(s)
EOF
  if [ $? -ne 0 ]; then
    echo "  !! params patch FAILED for N=$NN, restoring" >> "$META"
    cp "$BAK" params.py; exec 9>&-; return 1
  fi
  grep -E "^(K|M|N) " params.py | sed "s/^/  [C2 N=$NN] /" >> "$META"
  setsid $PY -u train_flat.py --episodes 40000 --seed 0 --lr 1e-4 \
      > "$LOG" 2>&1 9>&- &
  echo "  FIRED flat C2 N=$NN  pid=$!  log=$LOG" >> "$META"
  sleep 75
  cp "$BAK" params.py
  echo "  params.py restored $(date '+%F %T')" >> "$META"
  exec 9>&-
}

for spec in "3704:16" "3750:32"; do
  TPID="${spec%%:*}"; NN="${spec##*:}"
  echo "  waiting on pid $TPID for C2 N=$NN ..." >> "$META"
  while alive "$TPID"; do sleep 120; done
  echo "  trigger pid $TPID exited $(date '+%F %T')" >> "$META"
  fire "$NN"
done

sleep 90
echo "--- verify ---" >> "$META"
$PY - <<'EOF' >> "$META"
import json, glob, os
for f in sorted(glob.glob("results/result_*/hyperparameters.json"))[-6:]:
    try: h = json.load(open(f))
    except: continue
    if h.get("actor", {}).get("actor_mode") != "flat": continue
    e = h["env"]; d = os.path.basename(os.path.dirname(f))
    exp = 40 * e["N"] + 75
    print("  %s K=%s M=%s N=%s d_s=%s expected=%s %s"
          % (d, e["K"], e["M"], e["N"], h["actor"]["d_s"], exp,
             "OK" if h["actor"]["d_s"] == exp else "MISMATCH"))
EOF
echo "  waiter done $(date '+%F %T')" >> "$META"
