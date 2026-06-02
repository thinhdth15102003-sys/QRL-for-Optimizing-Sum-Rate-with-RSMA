"""V1-V5 verification harness runner for the architecture-2 VQC rebuild
(Δ1 λ-unfreeze affine, Δ2 R1 readout, Δ4 SoftmaxPQC head, Δ5 bypass).

Runs each verify_*.py as a subprocess and reports PASS/FAIL. GATE: all must
pass before launching a fresh training run on architecture-2.

  V1/V2  verify_r1_readout.py      — R1 observable correctness + jacobian (FD vs param-shift/SPSA)
  V3/V4  verify_softmax_head.py    — head shapes + gradcheck (W_sm/b_sm/β + λ/affine through head)
  Δ1     verify_delta1_gradcheck.py — affine+LN+λ chain-rule gradcheck (float64)
  Δ1-b   verify_delta1_batch.py    — Δ1 batch path runs, finite grads, persists
  arch2  verify_arch2_batch.py     — full arch-2 batch path, PPO-consistency, save/load

Usage (WSL):  python analysis/verify/run_all.py
"""
import sys, os, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCRIPTS = [
    "verify_r1_readout.py",
    "verify_delta1_gradcheck.py",
    "verify_delta1_batch.py",
    "verify_softmax_head.py",
    "verify_arch2_batch.py",
]
results = []
for s in SCRIPTS:
    print(f"\n{'='*70}\nRUN {s}\n{'='*70}")
    r = subprocess.run([PY, os.path.join(HERE, s)], capture_output=True, text=True)
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-1500:])
    ok = (r.returncode == 0)
    results.append((s, ok))

print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
for s, ok in results:
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {s}")
all_ok = all(ok for _, ok in results)
print(f"\n{'✅ ALL VERIFICATIONS PASSED — safe to train architecture-2' if all_ok else '❌ SOME FAILED — DO NOT TRAIN'}")
sys.exit(0 if all_ok else 1)
