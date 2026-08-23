"""Assignment-actor parameter count, VQC vs DNN, read off the trained checkpoints.

Scope is the ASSIGNMENT ACTOR ONLY — the block that maps the state to the
per-user assignment logits. The phase / power / C_k sub-actors and the critic
are identical in both arms and are deliberately excluded; including them would
dilute the very quantity the comparison is about.

Counts come from the checkpoints actually used in the results tables, not from
params.py, so they reflect the shrunk configuration that was trained.

Reported per arm:
  core     the parameters of the assignment mechanism itself
           (VQC: variational angles + encoding scales; DNN: policy MLP)
  encoder  the state compressor feeding it
           (VQC: autoencoder; DNN: encoder MLP)
  total    core + encoder — the whole assignment actor, the honest head-to-head
"""
import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from RL import QuantumActor, ClassicalActor


def breakdown(run_dir):
    d = os.path.join(run_dir, 'agents')
    ac = json.load(open(os.path.join(d, 'actor_config.json')))
    mode = ac.get('mode', 'quantum')
    if mode == 'classical':
        a = ClassicalActor.from_dir(d, seed=0)
        enc = sum(v.size for k, v in a.params.items() if '_enc_' in k)
        core = sum(v.size for k, v in a.params.items() if '_pol_' in k)
        extra = a.num_params() - enc - core
        return dict(mode='DNN', core=core, enc=enc, other=extra,
                    total=a.num_params(), nq=0, train_only=0,
                    note=f"enc{list(a.ENC_HIDDEN)} pol{list(a.POL_HIDDEN)}")
    a = QuantumActor.from_dir(d, seed=0)
    # The AE DECODER never runs at inference — it appears only in the
    # reconstruction branch (quantum_actor._forward_ae / its batch twin), while
    # _dual_encode, the deployed path, touches only the encoder weights below.
    # Counting the decoder in a deployment-parameter claim would overstate the
    # VQC arm by more than 2k parameters.
    ENC = ('W_irs_1', 'b_irs_1', 'W_irs_2', 'b_irs_2',
           'W_u_1', 'b_u_1', 'W_u_2', 'b_u_2', 'W_proj')
    enc = sum(v.size for k, v in a._p_ae.items() if k in ENC)
    dec = sum(v.size for k, v in a._p_ae.items() if k not in ENC)
    quant = sum(v.size for v in a._p_qc.values())
    readout = sum(v.size for v in a._p_xi.values())
    return dict(mode='VQC', core=quant, enc=enc, other=readout,
                total=quant + enc + readout, train_only=dec,
                nq=getattr(a, 'N_QUBITS', 0),
                note=f"nq={getattr(a,'N_QUBITS',0)} readout={getattr(a,'READOUT_MODE','?')}"
                     f" (+{dec} train-only decoder)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='label=run_dir (dir containing agents/)')
    a = ap.parse_args()

    print("=" * 92)
    print("  ASSIGNMENT-ACTOR PARAMETER COUNT  (sub-actors + critic EXCLUDED)")
    print("=" * 92)
    print(f"  {'run':<20}{'arm':>6}{'core':>10}{'encoder':>10}{'other':>8}"
          f"{'TOTAL':>10}   detail")
    print("  " + "-" * 88)
    out = {}
    for spec in a.runs:
        lbl, rd = spec.split('=', 1)
        b = breakdown(rd)
        out[lbl] = b
        print(f"  {lbl:<20}{b['mode']:>6}{b['core']:>10,}{b['enc']:>10,}"
              f"{b['other']:>8,}{b['total']:>10,}   {b['note']}")
    print("=" * 92)
    # pair up VQC/DNN by the case suffix in the label
    cases = sorted({l.split('-')[-1] for l in out})
    for c in cases:
        v = next((b for l, b in out.items() if l.endswith(c) and b['mode'] == 'VQC'), None)
        d = next((b for l, b in out.items() if l.endswith(c) and b['mode'] == 'DNN'), None)
        if v and d:
            print(f"  {c}:  total DNN/VQC = {d['total']/v['total']:.1f}x"
                  f"   |  core-only = {d['core']/max(v['core'],1):.0f}x"
                  f"  (core-only compares unlike units — report TOTAL)")


if __name__ == '__main__':
    sys.exit(main())
