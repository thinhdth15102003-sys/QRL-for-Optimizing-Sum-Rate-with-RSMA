#!/bin/bash
# Waiter: fire the entanglement ablation at the FULL budget when r418 frees a core.
#
# Why this and not L=4/L=6: the entanglement row of tab:abl_modelscale currently
# rests on r322, which ran only 2700 episodes against a reference chain selected
# at global 10300-19800, and lands 1.7 sigma low. That caveat is what stops the
# "quantum correlations are inert" contribution from being a measured result.
# The depth axis already has L=1,2,3 and is expected flat, so extra depth points
# do not move any claim.
#
# Config matched to the five Case-2 main-table seeds (r278/r320/r291/r323/r324):
# n_q=8, L=3, n_hidden_ae=[32,16], POWER-FAIRNESS f=0.80, 20000 episodes, seed 0.
# The ONLY difference is --no-entangle, which clears the CZ chain and the cross
# block bridges while leaving the parameter count and the R1 readout untouched
# (verified: probe_param_count gives core 96 / total 1,833 for both r322 and the
# reference r291).
#
# Do NOT self-setsid: under `wsl -e bash script` setsid becomes the initial
# process, forks, exits, and WSL tears the session down with the orphan.

set -u
cd "/mnt/c/Project/IRS-assisted RSMA Quantum-RL" || exit 1
PY=/home/thinhduong/miniconda3/envs/IRS_QRL/bin/python3.11
META=analysis/data/waiter_noentangle.meta

export CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RECIPE="--actor-mode quantum --episodes 20000 --seed 0 --no-entangle \
--oracle-warmup \
--assign-warmup-episodes 16 --assign-warmup-epochs 30 \
--phase-warmup-episodes 16 --phase-warmup-epochs 1500 \
--assign-aux-weight 0.3 --phase-aux-weight 0.5 --ck-aux-weight 0.3 \
--power-fairness 0.3 --power-fairness-priv 0.3 --power-priv-frac 0.80"

echo "=== waiter no-entangle armed $(date '+%F %T') pid=$$ sid=$(ps -o sid= -p $$ | tr -d ' ') ===" >> "$META"
echo "  trigger: pid 611 (r418, C1 single-z) -> VQC C2 no-entangle, 20000 ep" >> "$META"

alive () {
  [ -d "/proc/$1" ] || return 1
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | grep -q 'train\.py'
}

echo "  waiting on pid 611 ..." >> "$META"
while alive 611; do sleep 120; done
echo "  trigger pid 611 exited $(date '+%F %T')" >> "$META"

exec 9>/tmp/params_launcher.lock
flock -w 3600 9 || { echo "  !! lock timeout" >> "$META"; exit 1; }
echo "  lock acquired $(date '+%F %T')" >> "$META"
BAK=/tmp/params_before_noent.py
cp params.py "$BAK"
$PY - <<'EOF'
import io, re
p = "params.py"
s = io.open(p, encoding="utf-8", newline="").read()
for pat, rep in ((r"^K\s*=\s*\d+",         "K        = 10"),
                 (r"^M\s*=\s*\d+",         "M        = 2"),
                 (r"^N\s*=\s*\d+",         "N = 24"),
                 (r"^n_qubits\s*=\s*\d+",  "n_qubits = 8 "),
                 (r"^n_hidden_ae\s*=.*$",  "n_hidden_ae   = [32,16]"),
                 (r"^n_var_layers\s*=.*$", "n_var_layers  = 3")):
    s, n = re.subn(pat, rep, s, count=1, flags=re.M)
    assert n == 1, "FAILED to patch " + pat
io.open(p, "w", encoding="utf-8", newline="").write(s)
EOF
if [ $? -ne 0 ]; then
  echo "  !! params patch FAILED, restoring" >> "$META"; cp "$BAK" params.py; exit 1
fi
grep -E "^(K|M|N|n_qubits|n_hidden_ae|n_var_layers)\b" params.py | sed 's/^/  /' >> "$META"
setsid $PY -u train.py $RECIPE > analysis/data/vqc_c2_noent_full.log 2>&1 9>&- &
echo "  FIRED VQC C2 no-entangle  pid=$!" >> "$META"
sleep 90
cp "$BAK" params.py
echo "  params.py restored $(date '+%F %T')" >> "$META"
exec 9>&-

sleep 120
echo "--- verify (header MUST read NO-CZ(ablation), not +2xCZ-bridge) ---" >> "$META"
for d in $(ls -dt results/result_* 2>/dev/null | head -2); do
  grep -m1 -oE "QC     :.*" "$d/training_log.txt" 2>/dev/null | sed "s|^|  $d  |" >> "$META"
done
echo "  waiter done $(date '+%F %T')" >> "$META"
