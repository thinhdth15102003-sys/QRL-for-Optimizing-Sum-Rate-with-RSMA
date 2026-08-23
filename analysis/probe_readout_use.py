"""
probe_readout_use.py
--------------------
Is the R1 readout actually being used, and which parts of it carry the signal?

VQC-only, and it is the analysis that pre-figures the entanglement ablation.
R1 emits four blocks:

    R1-a  <Z_u>        one-qubit, user qubits
    R1-d  <Z_r>        one-qubit, IRS qubits
    R1-b  <Z_u Z_r>    TWO-qubit, user x IRS
    R1-c  mean_u' <Z_u Z_u'>   TWO-qubit, user cluster

At Case 2 (nq=8, M=2) the two-qubit blocks are 18 of the 26 entries — over two
thirds. On a product state they collapse to products of R1-a/R1-d exactly
(measured residual 3.3e-07 against 0.351 with the CZ chain), so if the policy
does not depend on them, entanglement is buying nothing and the ablation will
come out flat.

Three measurements, none of which need training:

  VARIANCE   how much each block actually moves across states. A block pinned
             near a constant cannot carry information regardless of its width.
  SENSITIVITY how much the emitted assignment changes when a block is replaced by
             its per-state product surrogate <Z_i><Z_j>. This is exactly what the
             no-entangle circuit would deliver, so it estimates the ablation's
             effect WITHOUT running it.
  LAMBDA     the encoding scales. Frozen lambda is a known failure mode here
             ([E] in Common-Knowledge); a circuit whose input scales never moved
             is not really reading the state.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CSI.env import ISTNEnv
from infer import load_training_cfg, load_agents, _compute_irs_favored
import RL.quantum_circuit as qc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--states", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = load_training_cfg(a.run)
    env = ISTNEnv(cfg, seed=a.seed, n_steps_ep=a.states + 5)
    actor, phase_net, power_net, ck_net = load_agents(a.run, seed=a.seed)
    demand = np.full(cfg.K, cfg.D_k_bps_hz)
    nq, M = actor.N_QUBITS, cfg.M
    nu = nq - M
    if getattr(actor, "READOUT_MODE", "") != "r1":
        print("  run is not using the R1 readout; nothing to decompose")
        return

    # block layout, mirroring _obs_from_batch: [R1-a | R1-b | R1-c | R1-d]
    blocks = [("R1-a  <Z_u>   1-qubit", 0, nu),
              ("R1-b  <Z_uZ_r> 2-qubit", nu, nu + nu * M),
              ("R1-c  <Z_uZ_u'> 2-qubit", nu + nu * M, nu + nu * M + nu),
              ("R1-d  <Z_r>   1-qubit", nu + nu * M + nu, nu + nu * M + nu + M)]

    O, A, Z = [], [], []
    obs = env.reset(seed=a.seed)
    for _ in range(a.states):
        blocked = _compute_irs_favored(env)
        s_t = actor.extract_state(obs, demand, blocked)
        phi, _, info = actor.forward(s_t, greedy=True)
        o = info.get("o_hat")
        if o is None:
            print("  actor.forward did not return o_hat; cannot decompose")
            return
        O.append(np.asarray(o, dtype=float))
        A.append(np.asarray(phi, dtype=int))
        Z.append(np.asarray(info["z_t"], dtype=float))
        env.user_pos = env._walk_users(env.user_pos)
        env.channels = env.channel_model.update_user_channels(
            env.user_pos, env.irs_pos, env.channels)
        obs = env._get_obs()
    O, A = np.array(O), np.array(A)

    print("=" * 76)
    print(f"  READOUT USE · {a.run} · nq={nq} M={M} · {len(O)} states · dim={O.shape[1]}")
    print("=" * 76)
    print(f"  {'block':<26}{'width':>7}{'sd':>10}{'|mean|':>10}{'share of sd':>13}")
    tot = O.std(axis=0).sum()
    for lbl, lo, hi in blocks:
        b = O[:, lo:hi]
        print(f"  {lbl:<26}{hi-lo:>7}{b.std(axis=0).mean():>10.4f}"
              f"{np.abs(b.mean(axis=0)).mean():>10.4f}"
              f"{100*b.std(axis=0).sum()/tot:>12.1f}%")
    two_q = (nu * M + nu)
    print(f"\n  two-qubit blocks are {two_q}/{O.shape[1]} = "
          f"{100*two_q/O.shape[1]:.0f}% of the readout width")

    # SENSITIVITY: replace the two-qubit blocks by the product surrogate and see
    # how many assignments flip. This is what a product-state circuit would give.
    flips, changed = 0, 0
    for i in range(len(O)):
        o2 = O[i].copy()
        zu, zr = O[i][0:nu], O[i][blocks[3][1]:blocks[3][2]]
        k = nu
        for u in range(nu):
            for r in range(M):
                o2[k] = zu[u] * zr[r]
                k += 1
        for u in range(nu):
            others = [zu[u] * zu[up] for up in range(nu) if up != u]
            o2[k] = float(np.mean(others)) if others else 0.0
            k += 1
        logits, _ = actor._head_forward(Z[i], o2)
        a2 = logits.reshape(cfg.K, actor.n_choices).argmax(axis=1)
        d = int((a2 != A[i]).sum())
        flips += d
        changed += int(d > 0)
    print(f"\n  SENSITIVITY  replacing both 2-qubit blocks by <Z_i><Z_j>:")
    print(f"    per-user assignment changes : {100*flips/(len(O)*cfg.K):.1f}%")
    print(f"    states whose action changes : {100*changed/len(O):.1f}%")
    print("    (this is what --no-entangle would deliver; near 0 predicts a flat"
          " ablation)")

    ly, lz = np.asarray(actor.lam_y), np.asarray(actor.lam_z)
    print(f"\n  LAMBDA (encoding scales, init ~U(-0.5,0.5))")
    print(f"    |lam_y| max {np.abs(ly).max():.4f}  mean {np.abs(ly).mean():.4f}")
    print(f"    |lam_z| max {np.abs(lz).max():.4f}  mean {np.abs(lz).mean():.4f}")
    print("=" * 76)


if __name__ == "__main__":
    sys.exit(main())
