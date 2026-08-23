# VQC Case 2, three seeds — 2026-08-09

Not yet in the paper (user asked to hold). Recorded so the numbers are traceable.

## Chains and picks

Each seed is a three-part resume chain, so the checkpoints were scanned across all
nine directories (606 checkpoints, 100-episode grid) and stitched onto one global
episode axis before applying the rule. `analysis/pick_ckpt_chain.py` does the
stitching; scanning each directory separately would pick the best of each *segment*
instead of the best of the *seed*.

⚠ Segments ran past the point the next segment resumed FROM, and those extra episodes
are NOT part of the lineage — r285 logged to 1636 but the resume took `ep_01600`, r290
to 14279 but the resume took `ep_14200`, r261 to 10235 but `ep_10200`. The stitcher
caps each segment at its resume point; without that it can pick a checkpoint that was
discarded from the chain.

| seed | chain (dir : global range) | pick | global ep | val reward / QoS |
|---|---|---|---|---|
| 0 | r261 `0–10200` → r278 `10200–13100` → r289 `13100–20200` | **r278/ep_00100** | 10 300 | 349.19 / 96.3% |
| 1 | r285 `0–1600` → r290 `1600–15800` → r320 `15800–20000` | **r320/ep_04000** | 19 800 | 344.42 / 93.9% |
| 2 | r286 `0–1700` → r291 `1700–15800` → r321 `15800–20000` | **r291/ep_13700** | 15 400 | 338.16 / 94.5% |

⚠ The seed-0 pick moved after the scan finished: an interim run on a partially scanned
tree picked r289/ep_04100 (344.42). r278 is *mid*-chain (global 10200–13100), not an
early low-reward segment, and it held a better checkpoint. Never run the picker on an
incomplete scan.

## Results (600 states, test env seeds 42/43/44, `--skip-ao`)

Rule: argmax episode return s.t. QoS ≥ 90%, selected on validation seeds 7/21/99.
Same rule, protocol and states as the DNN/AO numbers, which were NOT re-measured.

| | Reward | R_tot | QoS | Shortfall |
|---|---|---|---|---|
| **VQC** (3 seeds) | **339.5 ± 11.5** | 1.748 ± 0.027 | **94.7 ± 1.9%** | 0.81% / 4.0e-4 |
| DNN (5 seeds) | 337.5 ± 2.3 | 1.747 ± 0.011 | 92.1 ± 2.0% | 1.08% / 5.4e-4 |
| AO | 330.1 | 1.651 | 100.0% | 0 |
| Greedy | 281.2 | 1.434 | 99.4% | 0.21% |

Per seed: s0 350.15 @ 96.8% · s1 341.20 @ 94.2% · s2 327.30 @ 93.0%.

## VQC vs DNN

| axis | Δ (VQC − DNN) | SE_diff | |
|---|---|---|---|
| Reward | +2.0 | 6.72 | 0.30σ — indistinguishable |
| R_tot | +0.001 | 0.014 | 0.07σ — indistinguishable |
| **QoS** | **+2.6 pp** | 1.41 | **1.84σ** |

The QoS gap is the largest method separation measured anywhere in this project. It is
still only 1.8σ on 3-vs-5 seeds, so it is a lean, not a proof.

## Where the QoS lean does and does not appear

| setting | who leads QoS |
|---|---|
| Case 1, base | DNN, +1.9 pp |
| Case 1, blockage stress (`d_block` 0.10) | **VQC, +5.6 pp** |
| Case 2, base | **VQC, +2.6 pp** |

So the paper's "the VQC leans toward QoS" is defensible with a qualifier — it holds at
the larger case and under environmental stress, not at the Case-1 nominal point. Three
independent measurements support that shape.

## Open

* Seeds 3 and 4 (r323 → , r324 → ) finish Tue 11/08; five seeds would firm up the
  QoS number, whose sd is currently 5× the DNN's (11.5 vs 2.3) partly because n=3.
* The QoS floor barely binds here — 191–199 of ~201 candidates per chain clear 90%,
  versus Case 1 where it bound hard enough to push two of five seeds back to the start
  of their chain. VQC Case 2 holds QoS throughout training; it does not shave.
* The 4 200 episodes that just completed helped seed 1 (pick at global 19 800) and did
  nothing for seed 2 (pick at 15 400, before that segment began).
