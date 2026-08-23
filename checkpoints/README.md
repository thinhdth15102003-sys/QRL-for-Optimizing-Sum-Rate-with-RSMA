# Reported checkpoints

The exact weights behind every learned number in the paper, one directory per
(case, method, training seed). Selected by the rule of Section V-A: highest
greedy episode return subject to a 90% QoS floor, searched on validation
environment seeds 7 / 21 / 99 and reported on test seeds 42 / 43 / 44.

These are **inference weights**. `critic_*.npz` is not included: `infer.py` never
loads it, and it accounted for roughly 70% of the payload. The full checkpoints,
including critics and every non-selected episode, stay in the untracked
`results/` tree.

Run and episode for each directory are also written to its `provenance.json`.
The authoritative record, including how each row was measured, is
`analysis/data/TABLE-PROVENANCE.md`.

## Contents

| case | method | seed | run | checkpoint | global ep |
|---|---|---|---|---|---|
| Case 1 | VQC | 0 | `r328` | `ep_02700` | 16300 |
| Case 1 | VQC | 1 | `r260` | `ep_09000` | 9000 |
| Case 1 | VQC | 2 | `r265` | `ep_04500` | 4500 |
| Case 1 | VQC | 3 | `r266` | `ep_04200` | 4200 |
| Case 1 | VQC | 4 | `r267` | `ep_06800` | 6800 |
| Case 1 | DNN | 0 | `r234` | `ep_07400` | -- |
| Case 1 | DNN | 1 | `r238` | `ep_12100` | -- |
| Case 1 | DNN | 2 | `r239` | `ep_19500` | -- |
| Case 1 | DNN | 3 | `r240` | `ep_09500` | -- |
| Case 1 | DNN | 4 | `r294` | `ep_17400` | -- |
| Case 1 | PPO-flat | 0 | `r301` | `ep_06000` | -- |
| Case 1 | PPO-flat | 1 | `r307` | `ep_06000` | -- |
| Case 1 | PPO-flat | 2 | `r308` | `ep_06000` | -- |
| Case 1 | PPO-flat | 3 | `r347` | `ep_03000` | -- |
| Case 1 | PPO-flat | 4 | `r356` | `ep_03000` | -- |
| Case 2 | DNN | 0 | `r231` | `ep_08700` | -- |
| Case 2 | DNN | 1 | `r232` | `ep_11900` | -- |
| Case 2 | DNN | 2 | `r233` | `ep_16200` | -- |
| Case 2 | DNN | 3 | `r236` | `ep_17800` | -- |
| Case 2 | DNN | 4 | `r309` | `ep_19800` | -- |
| Case 2 | VQC | 0 | `r278` | `ep_00100` | 10300 |
| Case 2 | VQC | 1 | `r320` | `ep_04000` | 19800 |
| Case 2 | VQC | 2 | `r291` | `ep_13700` | 15400 |
| Case 2 | VQC | 3 | `r323` | `ep_13200` | 16200 |
| Case 2 | VQC | 4 | `r324` | `ep_09800` | 12800 |

## Case 2 has no PPO-flat row

PPO-flat does not learn at (K, M) = (10, 2). Both attempts, `r304` at learning
rate 3e-4 and `r305` at 1e-4, settle at an objective near -2.2 with QoS between
34% and 41%, so there is no checkpoint worth archiving and no five-seed row to
report. That failure is itself a result and is discussed in Section V-D.

## Caveat on Case 2 DNN seed 4

`r309` was trained with the wider sub-actor configuration that later drifted out
of the pinned set. It is the run behind the reported Table VII row, so it is
archived as it was measured, but its sub-actor widths differ from the other four
seeds. See `analysis/data/TABLE-PROVENANCE.md`.

## Loading

Each directory has the same `agents/` layout the training code writes, so it can
be passed straight to `infer.py`. The configuration files travel with the
weights, so the architecture is recovered from the directory rather than from
`params.py`, which has since moved on.
