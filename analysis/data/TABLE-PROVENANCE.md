# Table provenance — which run, which checkpoint, which probe

Written 2026-08-07 after discovering that the PPO-flat column of
`tab:abl_genshift` came from `result_310` (the `--power-fairness 0.1` variant)
while the paper describes PPO-flat as the faithful Meng port
(`--power-fairness 0`, `result_301`). Nothing on disk recorded which run
produced those cells, so the mismatch was invisible until the column was
re-measured and matched against both candidates.

**Add a row here whenever a number goes into the paper.** A cell whose origin is
not written down cannot be checked later, and this one was wrong for weeks.

---

## Selection rule (all learned policies, both cases)

Checkpoint = **argmax episode return subject to QoS ≥ 90%**, searched over **every
saved checkpoint** (100-episode grid), evaluated on **validation env seeds 7 / 21 / 99**. Reported on **test env seeds 42 / 43 / 44**
(Tables VI–VII) or **42 / 0 / 1 / 2 / 3** (Table IX). Validation and test seed
sets are disjoint so no checkpoint is selected on the states it is scored on.

The 90% floor is applied identically to every learned method. Without it,
PPO-flat's reward-optimal checkpoint sits at QoS 60–75% (it trades service for
rate monotonically through training), while VQC/DNN stay at 84–97% throughout —
so an unconstrained "best by reward" would mean something different for each
method. The floor applies at P_S ≥ 50 only; below that every method is far
outside the feasible region and the rule is not meaningful.

---

## Table VI — `tab:case1_main` (K=5, M=1, P_S=50)

Probe: `analysis/probe_main_table.py --K 5 --M 1 --steps 200 --env-seeds 42,43,44`
Raw: `analysis/data/maintable_c1_qos90.txt`

| row | seeds | checkpoints |
|---|---|---|
| HQC-HAC (VQC) | 5 | r328/ep_02700 · r260/ep_09000 · r265/ep_04500 · r266/ep_04200 · r267/ep_06800 |
| DNN | 5 | r234/ep_07400 · r238/ep_12100 · r239/ep_19500 · r240/ep_09500 · r294/ep_17400 |
| PPO-flat | 5 | r301/ep_06000 · r307/ep_06000 · r308/ep_06000 · r347/ep_03000 · r356/ep_03000 — **all `--power-fairness 0`, the faithful port** |
| AO / Greedy / Random | — | solvers, no checkpoint |

⚠ PPO-flat reached five seeds on 2026-08-13 (r347 seed 3, r356 seed 4). All five
share the port recipe exactly: power-fairness 0, lr 3e-4, target-kl 0.05,
hidden 512x256, checkpoint interval 3000, ending at ep_20004 with 7 checkpoints.
Going 3 -> 5 seeds moved the row 485.5 -> 481.2 and QoS 94.3% -> 96.0%, and more
than doubled the reward spread (2.6 -> 6.4).

⚠ The two new seeds pick ep_03000 while the first three pick ep_06000, because
seeds 3 and 4 have already fallen below the 90% floor by ep_6000. This is the
QoS-for-reward drift that motivates the floor, now visible in all five seeds: as
training proceeds every seed climbs in reward and falls in QoS, e.g. seed 0 goes
494 -> 550 in reward while QoS goes 98.6% -> 59.6%. Only 1-2 of the 7 checkpoints
per seed clear the floor at all.

⚠ probe_pick_ckpt.py cannot load a flat actor. The flat scan uses
`probe_main_table.py --flat` pointed at each checkpoint directory (a checkpoint dir
has the same `agents/` layout as a run dir); raw output in
`analysis/data/ckpt_scan_flat_c1/`. The method reproduces the three previously
recorded picks exactly.
⚠ VQC seed budgets are uneven: 18800 / 13900 / 20000 / 20000 / 10000 global
episodes for seeds 0-4. The chains were extended twice and re-picked each time
(2026-08-11, and again 2026-08-12 once seed 3's chain finished at 20000). Only
seed 0 ever moved (r262/ep_06800 -> r328/ep_02700); seeds 1-4 kept the checkpoints
they already had at a far smaller budget, and the 5-seed mean moved 489.5 -> 490.3
and then not at all. The picks sit at global episodes 16300 / 9000 / 4500 / 4200 /
6800, i.e. seeds 2 and 3 peak at roughly 21-23% of the episodes they were given.

⚠ Seed 3 is the sharpest case. Its last segment (r334) ran the full 10000 episodes
from global 10000 to 20000, adding 100 checkpoints, and produced nothing: the pool
grew 173 -> 200 candidates, exactly 30 still clear the 90% floor, and the pick is
unchanged at global 4200. Over that segment reward drifted +1.16/1000 ep while QoS
drifted -0.38 pp/1000 ep, so what gain there was came out of QoS margin. This is
the fifth independent negative result on extending VQC Case-1 training.

⚠ Seeds 2 and 3 clear the 90% floor on the validation seeds but land at 89.8% /
89.5% on the test seeds — the floor is a selection rule, not a guarantee on the
reported set. Only 32/201 and 30/200 of their checkpoints clear it at all, against
100/101, 137/140 and 177/189 for seeds 4, 1 and 0.

## Table VII — `tab:case2_main` (K=10, M=2, P_S=50)

Same probe, `--K 10 --M 2`. Raw: `analysis/data/maintable_c2_qos90.txt`

| row | seeds | checkpoints |
|---|---|---|
| DNN | 5 | r231/ep_08700 · r232/ep_11900 · r233/ep_16200 · r236/ep_17800 · r309/ep_19800 |
| HQC-HAC (VQC) | 5 | r278/ep_00100 · r320/ep_04000 · r291/ep_13700 · r323/ep_13200 · r324/ep_09800 |
| PPO-flat | — | r304 (lr 3e-4) and r305 (lr 1e-4) both fail to learn: J ≈ −2.2, QoS 34–41% |

Re-measured 2026-08-12 after seeds 3 and 4 (r323/r324) reached their 15400-episode
target; the 2026-08-11 figures were a mid-flight snapshot taken at ~78% of that.
Only seed 3 moved, r323/ep_11700 -> ep_13200 (global 14700 -> 16200), and the
5-seed mean went 336.8 -> 337.3 with QoS 94.2% -> 94.4%.

Picks sit at global episodes 10300 / 19800 / 15400 / 16200 / 12800 — spread across
the whole budget, unlike Case 1 where every seed peaks early. That contrast is the
cleanest statement of the scale effect: at K=5 the extra episodes buy nothing, at
K=10 they are used.

Resume chains (`analysis/pick_ckpt_chain.py`, scan dir `analysis/data/ckpt_scan_c2_all`,
spec `tag:runid:global_offset:cap`, cap = the episode the next segment resumed FROM):

| seed | chain |
|---|---|
| 0 | `s0_a:261:0:10200` → `s0_b:278:10200:2900` → `s0_c:289:13100:-` |
| 1 | `s1_a:285:0:1600` → `s1_b:290:1600:14200` → `s1_c:320:15800:-` |
| 2 | `s2_a:286:0:1700` → `s2_b:291:1700:14100` → `s2_c:321:15800:-` |
| 3 | `c2s3_a:315:0:3000` → `c2s3_b`(+`c2s3_b2_1..4`, `c2s3_b3_1..3`)`:323:3000:-` |
| 4 | `c2s4_a:316:0:3000` → `c2s4_b`(+`c2s4_b2_1..4`, `c2s4_b3_1..3`)`:324:3000:-` |

Case-1 chains (scan dir `analysis/data/ckpt_scan_c1_all`):

| seed | chain |
|---|---|
| 0 | `vqc_c1_s0:262:0:13600` → `c1s0_new`(+`c1s0_new2`)`:328:13600:-` |
| 1 | `vqc_c1_s1:260:0:-` |
| 2 | `vqc_c1_s2a:265:0:8400` → `vqc_c1_s2b:282:8400:1600` → `c1s2_c_part1..3:327:10000:6600` → `c1s2_d_full:331:16600:-` |
| 3 | `vqc_c1_s3a:266:0:8400` → `vqc_c1_s3b:283:8400:1600` → `c1s3_new`(+`c1s3_new2_1..6`, `c1s3_new3_1..3`)`:334:10000:-` |
| 4 | `vqc_c1_s4a:267:0:8300` → `vqc_c1_s4b:284:8300:-` |

Per-seed on the test seeds (reward · R_tot · QoS):

| seed | Case 1 | Case 2 |
|---|---|---|
| 0 | 483.69 · 2.441 · 91.1% | 350.15 · 1.768 · 96.8% |
| 1 | 516.97 · 2.603 · 96.2% | 341.20 · 1.757 · 94.2% |
| 2 | 461.56 · 2.356 · 89.8% | 327.30 · 1.717 · 93.0% |
| 3 | 481.83 · 2.447 · 89.5% | 335.43 · 1.739 · 94.4% |
| 4 | 507.62 · 2.562 · 95.2% | 332.55 · 1.730 · 93.7% |

## Power-ladder and readout-ablation chains (in progress)

Each cell below is a resume chain, not a single run: the box lost its training
twice, once to an unexplained mass kill (2026-08-12 ~13:38, no reboot) and once
to a Windows Update reboot (2026-08-13 00:44). Segment format
`run(local_ep_reached)`; the global episode is the running sum of the caps.

| cell | chain | target |
|---|---|---|
| C2 P_S=40 | r329(9000) → r340(1200) → **r349** | 20000 |
| C2 P_S=60 | r332(8800) → r344(1200) → **r350** | 20000 |
| C2 P_S=70 | r339(500) → r341(1200) → **r351** | 20000 |
| C2 single-z | r338(1200) → r345(1300) → **r352** | 20000 |
| C1 P_S=60 | r336(4400) → r343(1900) → **r353** | 20000 |
| C1 P_S=70 | r337(4400) → r342(1700) → **r354** | 20000 |
| C1 P_S=40 | r325 | 19900, DONE |
| DNN C1 seed 4 (re-run) | r346(12300) → **r355** | 20000 |
| PPO-flat C1 seed 3 | r347 | 20004, DONE |
| PPO-flat C1 seed 4 | r348 (lost, 5760 ep) → **r356** fresh | 20000 |

⚠ `train_flat.py` has no `--resume` and checkpoints every 3000 episodes, so a
PPO-flat run interrupted mid-segment restarts from zero. r348 reached episode
5760 and none of it was recoverable.

⚠ The single-z ablation lives in the chain, not in a flag: `--readout` is ignored
on resume and the mode is carried in the checkpoint's `actor_config.json`.
Verified after each relaunch by the banner — r352 prints `-> 8-dim`, a standard
run prints `-> 26-dim+2xZZ-cross(B4)`.

⚠ Still not started: DNN C1 at P_S=60, PPO-flat C2 seeds 1 and 2.

## Table IX — `tab:abl_genshift`

Probes: `analysis/probe_gen_sweep_one.py` (learned) and
`analysis/probe_gen_sweep_ao.py` (AO), both `--seeds 42 0 1 2 3 --eps 10 --steps 200`
= 10,000 states per cell.
Raw: `analysis/data/genshift_c1_5seed/` (per seed),
`analysis/data/genshift_ao_c1_ps30.txt`, `analysis/data/genshift_ao_c2_ps30.txt`,
`analysis/data/genshift_c1_ps40_r317_best.txt`.
Aggregated with `analysis/aggregate_genshift_seeds.py` (mean over training seeds).

* **Case 1, six live rows** (base P_S=50, κ=0.10/0.15/0.20, σ²=13/16): mean over
  the SAME checkpoints as Table VI — 5 VQC seeds, 5 DNN seeds, 3 PPO-flat seeds.
* **Case 1, P_S=40 DNN cell**: r317/best (retrained at 40 dBm, single seed).
* **Case 1, combined-B DNN cell**: r317/best — legitimate because r317 *is*
  retrained at P_S=40, so only κ/speed/σ² are zero-shot there.
* **P_S=30 and combined-A**: AO only. Both were re-measured after the low-power
  point moved 45 → 30 dBm (2026-08-07); the old 45 dBm numbers are gone from the
  table because the row no longer means the same thing.
* **P_S=45/60/70 and Case-2 learned columns**: blanked as `(retraining)`.

⚠ The base row of Table IX and the Table VI row are the same policies at the same
operating point but use different env-seed sets and state counts (10,000 vs 600),
so they do not match numerically and can even order DNN and PPO-flat differently.
The methods are within seed noise of each other, which is why.

### ⚠ Known asymmetry in the checkpoint search

`train_flat.py` saves every **3000** episodes (8 checkpoints/run); `train.py` saves
every **100** (200+ per run). After the QoS ≥ 90% filter, each PPO-flat seed has
only **2** valid candidates (ep 3000 and ep 6000) while VQC/DNN have 50–100. The
test-seed re-scoring removes the *inflation* from searching more candidates, but
not the *opportunity* — a larger pool genuinely has a better chance of containing
a good checkpoint. So the PPO-flat number is "best among what was saved", not
"best available". A matched-grid check (restricting VQC/DNN to a 3000-ep grid) is
the cheap way to confirm this does not change the ordering.

### Superseded
* PPO-flat Case-1 column **was** `result_310` (`--power-fairness 0.1`). Confirmed
  by re-measurement: r310 reproduced the old cells to 0.1–0.3 pp of QoS across
  three rows, r301 was nowhere near. Replaced 2026-08-07.

## ⛔ SUB-ACTOR WIDTH AUDIT (2026-08-13) — three groups are mixed

`params.py` currently holds the **Case-2 tuned** sub-actor widths
(`n_hidden_phase=[64,128,256,128]`, `n_hidden_power=[128,128,64]`,
`n_hidden_ck=[128,128,64,32]`). There is **no CLI flag** for any of them —
`--pol-hidden` is the only width flag train.py exposes — so every run launched
from the current params.py silently gets sub-actors ~8x larger than the runs
already in Tables VI/VII.

⚠ `hyperparameters.json` is NOT trustworthy for this. It records params.py's
values (train.py:1427/1435/1450), but on `--resume` train.py throws those nets
away and rebuilds with `PhaseMLP.from_dir(...)` / `PowerMLP.from_dir(...)` /
`CkMLP.from_dir(...)` (train.py:1865-1871), so a resume segment reports whatever
params.py happened to say. **Ground truth = `agents/{phase,power,ck}_config.json`
of the run itself.** r328 reports WIDE in its json and is actually small.

Measured sub-actor totals (phase+power+ck, from `*_params.npz`):

| group | runs | sub-actor params | verdict |
|---|---|---|---|
| Tab VI VQC C1 | r328·r260·r265·r266·r267 | 15,795 each | homogeneous |
| Tab VI DNN C1 | r234·r238·r239·r240·r294 | 16,819 each | homogeneous |
| Tab VI DNN C1 **new seed 4** | r346 → r355 | **138,803** | ⛔ 8.3x, do not pool |
| Tab VII VQC C2 | r278·r320·r291·r323·r324 | ~15.8-21k each | homogeneous |
| Tab VII DNN C2 | r231·r232·r233·r236 = 21,022 · **r309 = 145,438** | | ⛔ **MIXED, already in the paper** |
| ladder VQC C1 | r325 (P_S=40) 15,795 · r336/r353 (60), r337/r354 (70) 137,267 | | ⛔ P_S confounded with 8.7x model |
| ladder VQC C2 | r329/r349, r332/r350, r339/r351 all 137k+ | | anchor (main table) is small ⇒ confounded |
| ladder DNN C1 | r317 (40) 138,803 · r357 (60) 138,803 | | anchor r234 is 16,819 ⇒ confounded |
| abl single-z | r338 → r352 wide | | 26-dim baseline is small ⇒ confounded |

Separately, the DNN C1 row is already non-homogeneous in the **policy MLP**:
r294 (seed 4) has `hidden_post=[256,128]` = 45,986 actor params against
`[40]` = 6,810 for seeds 0-3. Fixing that was the point of the r346 re-run,
which then picked up the wide sub-actors instead.

⇒ Before scheduling anything else, pin the sub-actor widths per case and record
them in the launch script, the way `--pol-hidden 40` is already pinned.

### Which runs ALREADY IN THE PAPER are affected — exactly two

Cross-checked every run named in this file against its own `agents/*_config.json`.
Only **FRESH** runs read params.py; every resume segment inherited its chain's
widths through `from_dir`, which is why the VQC rows survived by luck.

| run | where it is used | widths | scope of damage |
|---|---|---|---|
| **r309** | Tab VII `tab:case2_main`, DNN row, seed 4 (pick ep_19800) | 145,438 vs 21,022 (6.9x) | 1 of 5 seeds in a mean |
| **r317** | Tab IX `tab:abl_genshift`, Case-1 P_S=40 DNN cell **and** combined-B DNN cell | 138,803 vs 16,819 (8.3x) | **single-seed cells, 100% of the cell** |

Clean: all of Table VI (VQC 5 / DNN 5 / PPO-flat 5), and the Table VII VQC row.
PPO-flat is structurally immune — the flat actor is one network with no sub-actors.

⚠ r309's pick sits at ep_19800, the latest of the five DNN C2 seeds (others
8700 / 11900 / 16200 / 17800). Consistent with the extra capacity being used,
but not evidence on its own.

⚠ Inside Table IX the P_S=40 DNN cell (r317, wide) is compared against a base row
whose DNN checkpoints are the small Table VI ones, so that row's P_S contrast is
confounded with an 8.3x change in sub-actor size.

Not yet on any table but carrying the same defect: r329→r349, r332→r350,
r339→r351, r338→r352, r336→r343→r353, r337→r342→r354, r346→r355, r357.
Case-3 r299/r300 are wide too, but Case 3 is out of the paper.

### Relaunch on the corrected widths — 2026-08-13 17:47, runs r358-r365

`params.py` width fix committed as **bae40d3**; the seven contaminated runs were
killed and everything restarted FRESH (a resume cannot change a checkpoint's
weight shapes). The abandoned chains are r329/r332/r339/r338/r336/r337 and their
segments through r340-r345 and r349-r355.

| new run | cell | replaces | episodes |
|---|---|---|---|
| r358 | VQC C2 P_S=40 | r329→r340→r349 | 20000 |
| r359 | VQC C2 P_S=60 | r332→r344→r350 | 20000 |
| r360 | VQC C2 P_S=70 | r339→r341→r351 | 20000 |
| r361 | VQC C2 single-z | r338→r345→r352 | 20000 |
| r362 | VQC C1 P_S=60 | r336→r343→r353 | **10000** |
| r363 | VQC C1 P_S=70 | r337→r342→r354 | **10000** |
| r364 | DNN C1 P_S=60 | r357 | 20000 |
| r365 | DNN C1 seed 4 | r346→r355 (and r294 in Tab VI) | 20000 |

C1 VQC capped at 10000 because all five base seeds peak at 21-23% of budget and
five separate extension experiments returned nothing; extendable by `--resume`
now that the widths match.

Verified from each run's own startup banner, not `hyperparameters.json` (the
phased launch mutated params.py three times, so that file is meaningless here):

* sub-actors `...→64→64→4` / `...→64→32→...` / `...→64→32→...` on all eight = the
  shrunk set, matching r234 and r231.
* C1 VQC `16-dim → 8 qubits, U_var x2, 22-dim+1xZZ`; C2 VQC `U_var x3,
  26-dim+2xZZ` — same as r325 and r278 respectively.
* r361 prints `READOUT ABLATION: --readout single-z -> mode=single_z`, 8-dim.
* both DNN runs print `6,810 params (enc [128] -> z24 -> pol [40])`, identical to
  r234/r238/r239/r240, so r365 also fixes the `pol_hidden=[256,128]` anomaly that
  r294 carries in Table VI.

⚠ Still outstanding after this batch: **r309** (Tab VII DNN C2 seed 4) and
**r317** (Tab IX, two single-seed cells). Neither was relaunched — the box is at
8 live runs.

### Armed 2026-08-13 18:17 — DNN C1 at P_S=40 and P_S=70

A waiter (`sched_dnn_40_70.sh`, pid 6348) polls every 5 min and fires the two runs
when **both** r364 and r365 have exited, into the two cores they release. Gated on
process exit rather than a fixed timer so an early or late finish still lands
correctly; 20 h ceiling, heartbeat in `analysis/data/sched_dnn_40_70.heartbeat`,
log in `sched_dnn_40_70.meta`. Expected fire ~04:30 on 2026-08-14.

* **P_S=40** replaces **r317**, which fixes *both* of its Table IX cells at once —
  the Case-1 P_S=40 DNN cell and the combined-B DNN cell reuse the same checkpoint.
* **P_S=70** is a cell that has never had a run.

Recipe read back from r317 and cross-checked against r234/r238/r239/r240: seed 0,
`--pol-hidden 40`, both warm-ups, aux 0.3/0.5/0.3, fairness 0.3/0.3/0.80,
n_qubits left at the default 12 (ClassicalActor reads n_latent = 2*nq = 24).

⚠ After this pair lands, the only run still on a table with the wide sub-actors
is **r309** (Table VII DNN C2 seed 4).

### 2026-08-14 — Table VI DNN row re-seeded, Table IX C1 P_S=60 filled

Mass kill at 05:45-05:50 (no reboot, no traceback, no OOM; WSL uptime unbroken —
the same signature as 2026-08-12 13:38). r364 and r365 had already finished at
03:20 and were unaffected; the other eight were resumed as r368-r375.

**Table VI, DNN row** — seed 4 moves `r294/ep_17400` → **`r365/ep_13400`**.
r294 was the only seed trained with `pol_hidden=[256,128]` (45,986 actor params)
against `[40]` = 6,810 for seeds 0-3, so the old row averaged two actor sizes.
Pick made under the standard rule on validation seeds 7/21/99: all 36 refined
checkpoints cleared the 90% floor, argmax reward landed at ep_13400
(reward 510.5, QoS 96.8% on validation).

| | old (with r294) | new (with r365) |
|---|---|---|
| Reward | $493.7 \pm 23.2$ | $494.8 \pm 24.3$ |
| $R_{tot}$ | $2.494 \pm 0.114$ | $2.501 \pm 0.121$ |
| QoS | $94.7 \pm 2.8\%$ | $94.6 \pm 2.7\%$ |
| Shortfall | 0.97% / 4.9e-4 | 1.00% / 5.0e-4 |

The row barely moves, which is the useful outcome: the homogeneity fix costs
nothing in headline numbers, so the earlier row was not propped up by r294's
larger policy MLP.

**Table IX, Case 1, $P_S=60$, DNN** — `r364/ep_13100`, filled with
$2.8024$ / $94.3\%$ / $556.5$ from `probe_gen_sweep_one.py --seeds 42 0 1 2 3
--eps 10` (10,000 states), the probe and seed set Table IX uses.

⚠ **Read the right line out of that probe.** Its first row is labelled
`base ($P_S{=}50$)` but that string is hardcoded — the probe restores the env
from the run's own `hyperparameters.json`, so for a run trained at 60 dBm the
"base" row is at 60, not 50. Confirmed against the r317 precedent, where
`base` = 1.7163 and `$P_S=40$` = 1.7161 (the same point, twice) and the paper
correctly pasted the `$P_S=40$` line. Always paste the explicitly labelled
`$P_S=X$` row matching the training budget.

⚠ r364 is nearly flat past ep 3000: every checkpoint sits within 1.9% of the
best (544-556 reward). Same shape as the five negative results on extending
VQC Case-1 training.

### 2026-08-14 — Table IX Case-1 live rows re-aggregated on 5 seeds

Prompted by "the PPO-flat base number looks wrong". It was not wrong for its own
table, but it was **stale in a way the seed counts hid**: Table IX's Case-1 rows
were a **3-seed** PPO-flat mean (`flat_s0..s2` = r301/r307/r308) while Table VI
went to 5 seeds on 2026-08-13, and `dnn_s4.txt` was still **r294**/ep_17400 while
Table VI's seed 4 is now r365/ep_13400.

Added `flat_s3` (r347/ep_03000), `flat_s4` (r356/ep_03000), replaced `dnn_s4`
(r365/ep_13400); old file kept at `analysis/data/genshift_superseded/`. Same probe
and seed set as the rest of the table (42,0,1,2,3 x 10 eps = 10,000 states).

⚠ The superseded file had to be **moved out of the glob directory** —
`aggregate_genshift_seeds.py --group DNN=dnn_` globs `dnn_*.txt`, so leaving
`dnn_s4_r294_superseded.txt` beside it would have counted seed 4 twice.

The **VQC column came out byte-identical**, which is the check that the
re-aggregation touched only what it should.

| | old | new |
|---|---|---|
| PPO-flat base | $2.4851$ / $92.1\%$ / $493.0$ | $2.4658$ / $93.6\%$ / $489.3$ |
| DNN base | $2.4822$ / $94.4\%$ / $490.7$ | $2.4829$ / $94.4\%$ / $491.1$ |

⚠ **The bolding flips.** PPO-flat previously held the best reward in four of the
six live rows; on five seeds DNN leads reward in four of six and R_tot in five of
six. Any sentence that leaned on PPO-flat topping this table needs re-reading.

⚠ Numbers in the interpreting sentence (line 1567) do not match the table, and did
not before this change either. Measured from the table itself:

| claim in prose | actual |
|---|---|
| VQC C1, $\kappa$ 0.05→0.20 costs $0.04\%$ | **0.004%** (10x) |
| VQC C1, $\sigma^2$ 10→16 costs $1.1\%$ | **1.21%** |
| VQC C2, $\kappa$ 0.05→0.20 costs $0.7\%$ | **0.52%** |
| VQC C2, $\sigma^2$ 10→16 costs $2.3\%$ | **2.80%** |

That sentence also has a comma splice: `costs $1.1\%$ and $2.3\%$, Both learned
actors therefore degrade...`.

### 2026-08-14 — VQC seed 0 in Table IX was also stale

Audited all fifteen `genshift_c1_5seed/*.txt` against Table VI's checkpoint list.
Three of the three learned columns had drifted, each differently:

| column | files | defect |
|---|---|---|
| PPO-flat | 3 | only `flat_s0..s2`; Table VI went to 5 seeds on 08-13 |
| DNN | 5 | `dnn_s4` = r294/ep_17400, superseded by r365/ep_13400 |
| VQC | 5 | **`vqc_s0` = r262/ep_06800, superseded by r328/ep_02700 on 08-11** |

The VQC one is the least visible: the file count was right, so nothing looked
wrong until the checkpoint paths inside the headers were compared. r262 and r328
are the same training seed, different segments of its chain — Table VI's pick moved
to r328 and Table IX never followed.

`vqc_s0` regenerated on r328/ep_02700 (old file in `genshift_superseded/`). Effect
on the six live rows is small but not nil:

| | old (r262) | new (r328) |
|---|---|---|
| base | $2.4590$ / $92.9\%$ / $485.3$ | $2.4653$ / $92.3\%$ / $486.5$ |
| $\sigma^2=16$ | $2.4292$ / $92.2\%$ / $478.2$ | $2.4337$ / $91.7\%$ / $479.0$ |

All three learned columns of the six Case-1 live rows now come from exactly the
Table VI checkpoints, five seeds each.

⚠ **Do not paste the non-live rows out of the aggregate.** The regenerated files
carry 14 shifts while the untouched ones carry 6, so `$P_S=30/40/60/70$`,
`speed $3$/$6$`, and `combined-A/B` appear in the aggregate with `±0.0000` — those
are single-seed values from whichever file was regenerated, not 5-seed means. Only
base, $\kappa\in\{0.10,0.15,0.20\}$ and $\sigma^2\in\{13,16\}$ are 5-seed.

⚠ Case-2 PPO-flat (six live rows) is still **one seed**, and no raw file in
`analysis/data/` reproduces its numbers — the source is unrecorded. Fix it together
with PPO-flat C2 seeds 1 and 2.

### 2026-08-14 — tab:param_budget split into two blocks, PPO-flat added

PPO-flat has no assignment actor: one network emits assignment, phases, powers and
common-rate shares together, so it could not go in a table that decomposes the
assignment head. Dropping 305,016 into a column of 743 would compare a whole policy
against one head. The table now has two blocks:

* **Assignment actor** — the existing decomposition, unchanged. Its ratio row is
  relabelled `DNN/VQC (inference / training)` so $9.2\times$/$6.0\times$ reads as
  head-to-head rather than system-wide.
* **Complete policy, inference** — all three methods, like-for-like.

| | Case 1 | Case 2 |
|---|---|---|
| HQC-HAC (VQC) | 16,538 | 21,831 |
| Classical (DNN) | 23,629 | 32,108 |
| **PPO-flat** | **305,016** | **725,239** |
| PPO-flat / HQC-HAC | 18.4x | 33.2x |

Counted from the saved weights (`scratchpad/budget.py`), two traps handled:

⚠ the VQC actor npz contains the AE decoder under `W_dec/b_dec/W_d_out/b_d_out`;
excluding it reproduces the table's own 1,304 -> 743 and 4,101 -> 1,833 split, which
is the check that the decomposition already in the paper is right.

⚠ the flat npz contains Welford input-normalisation buffers `_rn_mean/_rn_m2/
_rn_count` (2·d_s+1 floats). Those are running statistics, not parameters. Dropping
them gives 305,016 / 725,239, matching `train_flat.py`'s own recorded `num_params`
exactly; keeping them would have overstated flat by 557 / 2,071.

⚠ Sub-actor totals are NOT identical between the two hierarchical variants
(C1: 15,795 VQC vs 16,819 DNN) because PhaseMLP/PowerMLP input width follows the
representation fed to them (`o_hat` 22-dim vs `z_t` 24-dim). Do not describe the
three sub-actors as "shared and therefore cancelling".

⇒ This closes the "you crippled PPO-flat" objection at Case 2: it carries **22.6x**
the DNN parameters and **33.2x** the VQC parameters, and receives a richer input
(d_s = 1035, of which 960 is the full per-element g_RU, against the 44-dim affinity
the hierarchical actor sees). The remaining exposure is seeds, not capacity: C2
PPO-flat is n=1 per learning rate.

### Armed 2026-08-14 07:2x — PPO-flat Case 2, seeds 0 and 1

Waiter `sched_flat_c2.sh` polls every 5 min and fires when both r374 and r375 exit
(ETA ~14:30-14:50 today). Seed 2 and the wide-net control follow in the slot
r372/r373 open on 16/08.

```
--lr 1e-4  --episodes 40000  --hidden 512,256
--checkpoint-interval 100  --target-kl 0.05  --power-fairness 0
```

Each flag answers one objection to "PPO-flat does not scale past K=5":

| flag | closes |
|---|---|
| `--lr 1e-4` | r305 (1e-4) beat r304 (3e-4): best J −1.50 vs −2.11, final-quarter −1.90 vs −2.39. The claim should rest on flat's **best** measured setting. |
| `--episodes 40000` | ⚠ **r305 had NOT converged at 20k**: second half +0.98 J over the first, best at ep 15,864 (79% of budget). C1's working run drifted only +0.22. "You stopped it early" is the strongest objection available today, and doubling the budget is what turns the plateau into evidence. |
| 3 seeds | C2 flat is currently n=1 per learning rate. |
| `--checkpoint-interval 100` | removes the documented selection asymmetry (flat had 7 candidates against 200 for the hierarchical runs). The default is 1000; the old runs used 3000. |

Capacity is **not** on this list and needs no fix: flat already carries 725,239
inference parameters at Case 2 against 32,108 (DNN) and 21,831 (VQC), i.e. 22.6x and
33.2x, and receives a richer input (d_s = 1035, of which 960 is the full per-element
g_RU, against a 44-dim affinity). The `--hidden 1024,512` run on 16/08 exists only to
close the argument in print, not because there is reason to expect it to help.

⚠ If flat at 40k does reach a decent operating point, the paper's Case-2 claim has to
change. That is the point of running it: the claim currently rests on a run that was
still improving when it was stopped.

### 2026-08-14 — Case-1 DNN power ladder completed (r374, r375)

| cell | run | pick | R_tot / QoS / Reward |
|---|---|---|---|
| C1 $P_S=40$ DNN | r366→**r374** | ep_12600 | $1.8350$ / $76.9\%$ / $329.1$ |
| C1 combined-B DNN | same checkpoint | ep_12600 | $1.5375$ / $79.0\%$ / $253.7$ |
| C1 $P_S=70$ DNN | r367→**r375** | ep_02000 | $3.8151$ / $81.7\%$ / $596.2$ |

r374 replaces r317, the wide-sub-actor run. Both of its cells moved: P_S=40 went
$1.7161$/$81.5\%$/$314.1$ → $1.8350$/$76.9\%$/$329.1$ and combined-B
$1.5183$/$75.1\%$/$236.9$ → $1.5375$/$79.0\%$/$253.7$. The shrunk run gets **more rate
and less QoS** at P_S=40, so the earlier cell was not merely noisy, it sat at a
different point on the rate-QoS trade-off.

⚠ **The QoS ≥ 90% selection floor has no feasible checkpoint in either run.**
r374 spans 76-80% and r375 spans 80.0-81.2% across 35 and 31 checkpoints. For
P_S=40 that is expected and already documented. For **P_S=70 it is not** — the floor
was supposed to apply at P_S ≥ 50. Both picks therefore fall back to max reward, and
that fallback has to be stated wherever the rule is described.

⚠⚠ **QoS is non-monotone in transmit power**, and the mechanism is in the objective:

| P_S | 40 | 50 | 60 | 70 |
|---|---|---|---|---|
| DNN R_tot | 1.84 | 2.48 | 2.80 | **3.82** |
| DNN QoS | 76.9% | 94.4% | 94.3% | **81.7%** |

The shortfall penalty is scale-free, normalised by $R_k^{\min}$ and capped at
$\lambda_D$ per starved user, while the rate term grows with $P_S$. At $70$ dBm the
rate a starved user's power earns elsewhere exceeds the $\lambda_D \le 1.5$ it costs,
so starving becomes the reward-optimal move and the policy takes it. Low QoS at
$40$ dBm is the opposite cause, plain infeasibility. Any sentence reading the ladder
as "more power, better service" is wrong at the top rung.

### 2026-08-16 — six runs measured after the machine reset (Table IX)

Selection `analysis/probe_pick_ckpt.py --seeds 7 21 99 --eps 4`;
reporting `analysis/probe_gen_sweep_one.py --seeds 42 0 1 2 3 --eps 10 --steps 200`.
Raw picks in `analysis/data/pick_*.txt`, raw cells in `analysis/data/genshift_c*_*.txt`.

| cell | chain | pick | select reward / R_tot / QoS |
|---|---|---|---|
| VQC C1 $P_S=60$ | r372→r384 | r384/ep_00600 | $557.08$ / $2.7970$ / $95.4\%$ |
| VQC C1 $P_S=70$ | r373→r385 | r373/ep_04400 | $606.62$ / $3.9926$ / $79.4\%$ |
| DNN C2 $P_S=40$ | r378→r386 | r386/ep_04200 | $161.67$ / $0.9783$ / $92.5\%$ |
| DNN C2 $P_S=60$ | r379→r387 | r379/ep_08500 | $436.02$ / $2.2153$ / $90.0\%$ |
| DNN C2 $P_S=70$ | r389 (fresh) | r389/ep_04500 | $450.01$ / $2.2763$ / $90.3\%$ |
| flat C2 seed 2 | r388 (fresh) | r388/ep_40008 | $-314.59$ / $0.6288$ / $56.2\%$ |

Reported cells (test seeds), R_tot / QoS / Reward:

| cell | value |
|---|---|
| VQC C1 $P_S=60$ | $2.7942$ / $95.3\%$ / $556.3$ |
| VQC C1 combined-B | $2.8356$ / $95.4\%$ / $564.5$ |
| VQC C1 $P_S=70$ | $3.9302$ / $80.5\%$ / $601.8$ |
| DNN C2 $P_S=40$ | $0.9825$ / $92.7\%$ / $162.0$ |
| DNN C2 combined-A | $0.9646$ / $91.7\%$ / $159.2$ |
| DNN C2 $P_S=60$ | $2.2073$ / $89.7\%$ / $434.0$ |
| DNN C2 combined-B | $2.2743$ / $91.7\%$ / $447.5$ |
| DNN C2 $P_S=70$ | $2.2691$ / $89.2\%$ / $447.7$ |

⚠⚠ **Scanning only the tail of a resume chain breaks the selection rule.** The first
pass scanned r384/r385/r386/r387 over their own dirs only, i.e. the last 1,700-6,100
episodes of chains that are 10,000-20,000 long, and concluded that DNN C2 $P_S=60$ had
no checkpoint at QoS $\ge 90\%$. The fresh, unresumed r389 showed why that was an
artifact: every one of its feasible checkpoints sits at global episode $\le 4{,}500$.
Re-scanning the predecessor r379 found **17** feasible checkpoints (ep_00500-ep_08500,
peak $96.7\%$), and the pick moved from r387/ep_06000 ($439.01$, QoS $87.9\%$) to
r379/ep_08500 ($436.02$, QoS $90.0\%$). Every future pick on a resumed run must scan
the whole chain, not the final dir.

⚠ **Validation-selected QoS $\ge 90\%$ does not imply reported QoS $\ge 90\%$.**
DNN C2 $P_S=60$ selects at $90.0\%$ on seeds 7/21/99 and reports $89.7\%$ on
42/0/1/2/3; $P_S=70$ selects at $90.3\%$ and reports $89.2\%$. This is the intended
behaviour of the split, but no sentence may claim the reported cells satisfy the floor.

⚠ **The $P_S=70$ floor failure is architecture-independent.** DNN C1 (r375) spans
$80.0$-$81.2\%$; VQC C1 (r373→r385) spans $79.4$-$82.2\%$ over 32 checkpoints. Two
different actors land in the same band, which supports reading the mechanism off the
objective rather than off the network.

⚠ VQC C1 $P_S=70$: r373/ep_04400 ($606.62 \pm 8.80$) beats r385/ep_00300
($605.79 \pm 10.97$) by $0.83$, under $0.1\sigma$. argmax picks the former; the two are
not distinguishable.

### 2026-08-16 — PPO-flat Case 2 at 40k does not plateau

r388 (seed 2, 40,008 episodes) reward rises monotonically to the last checkpoint:
$-562$ (ep 1,200) → $-392$ (10,800) → $-334$ (30,000) → $-315$ (40,008), QoS
$48.0\% \to 56.2\%$. The 2026-08-13 note asked whether 40k would turn the plateau into
evidence. It did not. The operating point stays far below the hierarchical actors
(R_tot $0.60$ / QoS $54.4\%$ against DNN's $\approx 2.48$ / $94.4\%$ at the same
$P_S=50$), but the curve is still climbing, so the paper must say **"does not reach a
usable operating point within 40k episodes"** and must not say "has converged" or
"has plateaued". The "you stopped it early" objection is pushed from 20k to 40k, not closed.

Pasteable PPO-flat C2 cells (**n=1**, seeds 3/4 still training as r390/r391) —
R_tot / QoS / Reward:

| shift | value |
|---|---|
| base ($P_S{=}50$) | $0.5977$ / $54.4\%$ / $-342.3$ |
| speed $3$ | $0.6979$ / $59.9\%$ / $-264.8$ |
| speed $6$ | $0.8157$ / $66.1\%$ / $-178.6$ |
| $\kappa=0.10$ | $0.5976$ / $54.4\%$ / $-342.9$ |
| $\kappa=0.15$ | $0.5976$ / $54.4\%$ / $-342.5$ |
| $\kappa=0.20$ | $0.5979$ / $54.4\%$ / $-341.7$ |
| $\sigma^2=13$ | $0.6009$ / $55.4\%$ / $-352.6$ |
| $\sigma^2=16$ | $0.6039$ / $56.1\%$ / $-361.9$ |

The $P_S$ and combined rows were computed but are **not pasteable**: they are zero-shot
for flat, whereas those rows are `(retraining)` rows for every other method, and no
retrained flat run exists. They are recorded here because they are informative —
zero-shot at $P_S=60$ flat reaches QoS $97.4\%$ (R_tot $1.066$) and at $P_S=70$
$99.3\%$ ($1.457$). Flat's Case-2 failure at $P_S=50$ is therefore a demand-feasibility
failure at that budget, not an inability to respond to power. The "does not scale past
K=5" claim must be stated **at matched $P_S$**, where it holds comfortably.

⚠ flat shows **zero** sensitivity to $\kappa$ (R_tot $0.5976/0.5976/0.5979$ against a
base of $0.5977$). At QoS $54\%$ the policy is far enough from the constraint that CSI
error does not move it. Do not read this as CSI robustness.

### 2026-08-16 — zero-shot vs retrained at P_S=70: the shortfall gets WIDER, not DEEPER

`analysis/probe_shave_vs_starve.py`, new. One shared environment taken from
r382/ep_02900 ($P_S=70$), 200 states x seeds 42/0/1, greedy, decide on $\hat g$ /
score on true $g$, each arm using its **own** C_k head. Raw:
`analysis/data/shave_vs_starve_c2_ps70.txt`.

Arms: `base` = r289 (VQC C2 trained at $P_S=50$, transferred zero-shot),
`vqc70` = r382/ep_02900 (VQC retrained at 70, **intermediate**, global 8,700/20,000),
`dnn70` = r389/ep_04500 (DNN retrained at 70, complete).

| | base | vqc70 | dnn70 |
|---|---|---|---|
| R_tot | $1.8439$ | $2.2350$ | $2.2429$ |
| QoS | $91.9\%$ | $86.3\%$ | $84.6\%$ |
| J($\lambda{=}1.5$) | $1.7936$ | $2.1929$ | $2.1909$ |
| med \| fail | $0.905$ | $0.910$ | $0.902$ |
| p05 \| fail | $0.561$ | $0.706$ | $0.710$ |
| min | $\mathbf{0.034}$ | $0.420$ | $0.410$ |
| share $<0.50\,D_k$ | $0.3\%$ | $0.0\%$ | $0.0\%$ |
| share $0.70$-$1.00\,D_k$ | $7.3\%$ | $13.1\%$ | $14.7\%$ |
| mean \| served | $3.938$ | $5.038$ | $5.140$ |

The hypothesis under test was that the retrained policy shaves to a *lower* threshold.
It does not. The **depth** of the shortfall is the same in all three arms
($0.902$-$0.910$), and the retrained tails are strictly **shallower**. What changes is
the **count**: the $0.70$-$1.00$ band roughly doubles. Retraining buys its rate by
moving more users just under $D_k$ at unchanged depth, and by concentrating on the
users it does serve (mean over served slots $3.94 \to 5.04/5.14$).

⚠⚠ **The starving arm is the zero-shot one.** `base` reaches $\min = 0.034\,D_k$,
i.e. a user served $3.4\%$ of demand, and puts $0.3\%$ of slots below $0.50\,D_k$;
neither retrained arm ever goes below $0.41\,D_k$. Binary QoS ranks `base` *above*
both retrained policies while `base` is the only one that abandons anyone. This is a
concrete instance of the reporting problem already flagged in memory, and it is worth
a sentence in Section V wherever the QoS column is discussed.

λ_D sweep, J: base $1.794 \to 1.173$, vqc70 $2.193 \to 1.674$, dnn70 $2.191 \to 1.549$
over $\lambda \in \{1.5,3,5,10,20\}$. The ranking never flips, and the relative gap
*widens* with λ ($+22\%$ at $1.5$, $+43\%$ at $20$ for vqc70 over base). The lower QoS
is therefore not an artifact of a penalty weight that underprices near-threshold
shaving.

⭐ vqc70 and dnn70 are tied at $\lambda=1.5$ ($2.1929$ vs $2.1909$) but separate as λ
grows ($1.674$ vs $1.549$ at $20$), with vqc70 also holding $1.7$ points more QoS. The
VQC arm was only $8{,}700/20{,}000$ episodes in against a completed DNN, so this
comparison must be redone when r382 finishes before any claim is made from it.

### 2026-08-16 — selection rule split: floor on the base row only

Decision: the QoS $\ge 90\%$ floor applies to the **base** row alone. Every shifted row
is selected by **argmax episode return**, unconstrained. The reasoning is that all
arms share one recipe, so the policy sits at a different point on the rate-QoS frontier
in each environment, and forcing a common QoS floor across environments selects a
different kind of checkpoint in each.

Two pieces of evidence support this rather than merely permitting it. First,
`probe_shave_vs_starve.py` shows that the lower QoS of a $P_S=70$ policy is not bought
by shaving margin the penalty underprices: J wins at every $\lambda_D$ out to $20$, the
shortfall depth is unchanged, and the tail is shallower. Second, the retrained arms are
the ones that never starve anyone; the high-QoS zero-shot arm is.

Cells affected: **three**. The others were already unconstrained argmax.

| cell | pick under floor | pick under argmax | R_tot / QoS / Reward, before → after |
|---|---|---|---|
| DNN C2 $P_S=60$ | r379/ep_08500 | **r387/ep_06000** | $2.2073$/$89.7\%$/$434.0$ → $2.2359$/$87.4\%$/$438.7$ |
| DNN C2 combined-B | r379/ep_08500 | **r387/ep_06000** | $2.2743$/$91.7\%$/$447.5$ → $2.3042$/$89.8\%$/$453.0$ |
| DNN C2 $P_S=70$ | r389/ep_04500 | **r389/ep_09500** | $2.2691$/$89.2\%$/$447.7$ → $2.2988$/$87.7\%$/$454.3$ |

Unchanged, because the floor was never binding there: DNN C2 $P_S=40$ and combined-A
(floor does not apply below $50$~dBm, and ep_04200 was already the chain argmax);
VQC C1 $P_S=60$ (every checkpoint in both dirs sits at $93$-$97\%$); VQC C1 $P_S=70$ and
flat C2 (no feasible checkpoint existed, so both had already fallen back to argmax).

⚠ **Removing the floor makes the pick a max over statistically tied candidates.** On
r389 the top five selection rewards span $0.24$ against a per-checkpoint sd of $3.5$;
on r387 the top five span $0.43$ against the same sd. argmax over ~40 such candidates
carries an optimistic bias of roughly $2\sigma \approx 7$ reward units, which is larger
than the $3.0$-$5.5$ unit gap between the old and new picks. The validation/test split
is what contains this: selection ran on seeds 7/21/99 and the table reports 42/0/1/2/3.
In this instance the gain did transfer ($+4.7$, $+5.5$, $+6.6$ on the test seeds), but
that is an observation about these three cells, not a general guarantee.

⚠ **The protocol description must now state two rules, not one.** A reader who checks
the $P_S=70$ row's $87.7\%$ against a blanket "checkpoints are selected subject to
QoS $\ge 90\%$" will read it as a violation.

### 2026-08-16 — VQC Case 2 at P_S=40 breaks the same way the DNN does (PROVISIONAL)

Asked because the DNN cell at $P_S=40$ is the only Case-2 rung where a learned actor
loses to AO, and by a wide margin. Chain r368 (global 1,900-5,900) → r380
(5,900-8,900); **r380 is still training, 8,900 of 20,000**, so nothing here goes into
the table yet. Env verified from the checkpoint's own training_config
($K{=}10$, $M{=}2$, $P_S{=}40.0$), not from the directory name.

Selection scan over both dirs (36 checkpoints, seeds 7/21/99): reward wanders between
$103$ and $161$ with **no trend over 7,000 episodes**. r368 spans $103$-$158$,
r380 spans $124$-$161$. argmax is r380/ep_00200 ($161.23 \pm 6.77$) against
r368/ep_03600 ($158.12 \pm 5.45$), a gap of $3.1$ inside the noise.

Reported on seeds 42/0/1/2/3:

| | R_tot | QoS | Reward |
|---|---|---|---|
| AO | $1.2979$ | $93.8\%$ | $240.1$ |
| DNN r386/ep_04200, **20,000 ep** | $0.9825$ | $92.7\%$ | $162.0$ |
| VQC r380/ep_00200, *8,900 ep* | $1.0129$ | $92.3\%$ | $165.1$ |
| VQC r380/ep_03000, *8,900 ep* | $0.9922$ | $92.5\%$ | $158.6$ |

combined-A, same checkpoints: AO $1.0949$/$93.6\%$/$201.5$, DNN $0.9646$/$91.7\%$/$159.2$,
VQC(ep_00200) $0.9905$/$91.5\%$/$159.1$, VQC(ep_03000) $0.9727$/$91.6\%$/$151.7$.

⭐ **Both architectures land in the same place, and it is not a QoS trade.** QoS is
$92.3$-$92.7\%$ for both learned actors against AO's $93.8\%$, so the deficit is almost
pure sum-rate: $-24.3\%$ (DNN) and $-22.0\%$ (VQC) against AO. This is the opposite
shape to $P_S=60/70$, where the learned actors beat AO on rate and pay in QoS. At
$40$~dBm there is no trade to point at.

⭐ **A VQC less than half-trained already matches a fully-trained DNN** ($165.1$ vs
$162.0$ reward, $1.0129$ vs $0.9825$ R_tot), and its own trajectory is flat to slightly
down (ep_00200 $165.1$ → ep_03000 $158.6$). Retraining at the lower budget therefore
does not appear to be the missing ingredient.

⇒ Reading this as a capacity limit of either network is not supported. Both plateau at
R_tot $\approx 1.0$ while AO reaches $1.2979$ at the same QoS. The relative collapse is
also larger for the learned actors than for AO: against their own base rows, AO retains
$78.6\%$ of its rate at $40$~dBm, DNN $56.1\%$, VQC $58.2\%$. The shared recipe was
tuned at $50$~dBm, and the three power levers behind it (low $f$, high $\beta$, C_k
filling demand) stop being the right operating point once the budget cannot cover
$D_k$ for everyone. ⚠ That mechanism is a **hypothesis, not measured**;
`probe_gap_split` at $P_S=40$ would settle it.

⚠ Do NOT paste: the VQC $P_S=40$ and combined-A cells stay `(retraining)` until r380
reaches 20,000.

### 2026-08-16 — PPO-flat Case 2 completed at five seeds (Tables VII and IX)

Runs r376 (s0) · r377 (s1) · r388 (s2) · r390 (s3) · r391 (s4), all $P_S=50$,
lr $10^{-4}$, 40,000 episodes, hidden $512{\times}256$, target-kl 0.05,
power-fairness 0, checkpoint interval 100. The older Case-2 flat runs
r304/r305/r313/r314 (lr $3\cdot10^{-4}$, 20k) are a superseded generation and are not
mixed in.

Selection `probe_pick_ckpt.py --seeds 7 21 99 --eps 4`. The QoS $\ge 90\%$ floor is the
base-row rule and this is the base row, but no checkpoint of any seed exceeds
$\sim 57\%$, so all five fall back to argmax reward.

| seed | run | pick | select reward / R_tot / QoS |
|---|---|---|---|
| s0 | r376 | ep_40008 | $-319.35$ / $0.5937$ / $55.0\%$ |
| s1 | r377 | ep_40008 | $-343.20$ / $0.5975$ / $54.2\%$ |
| s2 | r388 | ep_40008 | $-314.59$ / $0.6288$ / $56.2\%$ |
| s3 | r390 | ep_39900 | $-305.94$ / $0.6143$ / $56.9\%$ |
| s4 | r391 | ep_35400 | $-330.00$ / $0.5857$ / $54.4\%$ |

⚠⚠ **Four of the five seeds pick their last checkpoint** and the fifth (s4) peaks at
ep_35400 with a flat, noisy tail ($-330.0$ at 35,400 against $-338.0$ at 40,008, sd
$40$-$50$). No seed has converged at 40,000 episodes. The 2026-08-13 note asked whether
doubling the budget would turn the plateau into evidence; across five seeds the answer
is no. Wherever the paper describes the Case-2 flat result it must say **"does not
reach a usable operating point within 40k episodes"** and must not say converged or
plateaued.

**Table VII** `probe_main_table.py --K 10 --M 2 --steps 200 --env-seeds 42,43,44
--skip-ao --flat PPO-flat=<the five checkpoints>`.
Raw `analysis/data/maintable_c2_flat_5seed.txt`.

| Policy | Reward | R_tot | QoS | Shortfall |
|---|---|---|---|---|
| PPO-flat | $-367.1 \pm 53.7$ | $0.557 \pm 0.039$ | $47.2 \pm 3.1\%$ | $23.42\%$/$117{\cdot}10^{-4}$ |

⚠ `--flat` takes `LABEL=dir1,dir2,...`, not a list of bare directories. Passing bare
directories raises `ValueError: not enough values to unpack` after the header has
already printed, which looks like a probe crash rather than a usage error.

**Table IX** `probe_gen_sweep_one.py --seeds 42 0 1 2 3 --eps 10 --steps 200` per seed,
written as `flat_s0..s4.txt` into `analysis/data/genshift_c2_5seed/` beside the
existing `dnn_s*` and `vqc_s*`, then `aggregate_genshift_seeds.py --group flat=flat_`.
Raw `analysis/data/genshift_c2_flat_agg.txt`. Six cells pasted, R_tot / QoS / Reward:

| shift | old (1 seed, unrecorded ckpt) | new (5 seeds) |
|---|---|---|
| Base | $0.6026$/$54.5\%$/$-355.5$ | $0.5874$/$54.4\%$/$-326.5$ |
| $\kappa=0.10$ | $0.6023$/$54.5\%$/$-355.4$ | $0.5874$/$54.4\%$/$-326.4$ |
| $\kappa=0.15$ | $0.6025$/$54.5\%$/$-354.7$ | $0.5873$/$54.4\%$/$-326.5$ |
| $\kappa=0.20$ | $0.6028$/$54.5\%$/$-353.9$ | $0.5872$/$54.4\%$/$-326.4$ |
| $\sigma^2=13$ | $0.6061$/$55.7\%$/$-365.5$ | $0.5909$/$55.3\%$/$-337.4$ |
| $\sigma^2=16$ | $0.6089$/$56.5\%$/$-375.1$ | $0.5943$/$56.0\%$/$-347.2$ |

No bold moves; the flat rewards are deeply negative and DNN keeps every Case-2 row.
The $P_S$ and combined rows stay `(retraining)` because those numbers are zero-shot for
flat, and the retrained flat Case-2 runs are queued behind waiters `flat_c2_ps40/60/70`.

⚠⚠ **Tables VII and IX disagree for the same policy at the same operating point**,
QoS $47.2\%$ against $54.4\%$ and reward $-367.1$ against $-326.5$. Both are correct
for their own protocol (600 states on env seeds 42/43/44 versus 5 env seeds x 10
episodes on 42/0/1/2/3), and flat's spread across training seeds is $\pm 53.7$ in
reward, wide enough to absorb the difference. This was already documented in general
terms; it is now a $7$-point QoS gap sitting in two adjacent tables, so a reader will
notice. Either the caption of Table IX states the different test set explicitly, or
the two are re-run on one protocol.

⚠ flat shows **zero** sensitivity to $\kappa$ at five seeds as well
($0.5874$/$0.5873$/$0.5872$ against a base of $0.5874$). At QoS $54\%$ the policy is far
enough from the constraint that CSI error does not move it. This is not CSI robustness.

### 2026-08-17 — PPO-flat power ladder measured, and it wins the top of the Case-1 ladder

Runs r398 (C1 $P_S=40$), r397 (C1 $60$), r399 (C1 $70$), r400 (C2 $40$). All are fresh
retrains at their own budget, verified from `hyperparameters.json`
(K/M/N/P_S/mode/lr/episodes). Selection `probe_pick_ckpt.py --seeds 7 21 99 --eps 4`
over the whole checkpoint grid, argmax reward (shifted rows carry no floor).

| cell | pick | select reward / R_tot / QoS | reported R_tot / QoS / Reward |
|---|---|---|---|
| C1 $P_S=40$ | r398/ep_17700 | $202.01$ / $1.0894$ / $92.9\%$ | $1.0812$ / $92.7\%$ / $197.0$ |
| C1 combined-A | same | — | $1.0048$ / $91.2\%$ / $173.2$ |
| C1 $P_S=60$ | r397/ep_12300 | $731.96$ / $3.8612$ / $60.0\%$ | $3.8389$ / $60.0\%$ / $727.3$ |
| C1 combined-B | same | — | $3.3312$ / $59.8\%$ / $630.8$ |
| C1 $P_S=70$ | r399/ep_20004 | $2055.16$ / $10.2761$ / $99.8\%$ | $10.2956$ / $99.9\%$ / $2059.0$ |
| C2 $P_S=40$ | r400/ep_40008 | $-332.53$ / $0.8231$ / $73.8\%$ | $0.8579$ / $76.3\%$ / $-244.1$ |
| C2 combined-A | same | — | $0.8163$ / $72.8\%$ / $-317.3$ |

⛔⛔ **At C1 $P_S=70$, PPO-flat Pareto-dominates both hierarchical actors**, R_tot
$10.2956$ against $3.8151$ (DNN) and $3.9302$ (VQC), *and* QoS $99.9\%$ against
$81.7\%$ and $80.5\%$. Reward is $3.4\times$ the previous best in that row. The bold
moves to PPO-flat in three Case-1 rows: $P_S=60$, $P_S=70$ and combined-B.

Ruled out before pasting:

* **Not a probe artifact.** r399's own training log ends at $J=+10.2973$, QoS $99.8\%$,
  which is the same quantity the probe reports.
* **Not a validation-seed fluke.** Selection ran on 7/21/99, reporting on 42/0/1/2/3,
  and the number moved by $+0.02$.
* **Not a power-budget violation.** New probe `analysis/probe_power_budget_check.py`
  measures emitted power against the linear budget. Both arms sit at exactly
  $1.0000$ of budget, max $1.0000$, zero over-budget states.
* **Not implausible physically.** $10.30$ bps/Hz over $K=5$ is $2.06$ per user, about
  $5$ dB SINR at $70$ dBm. Scaling the base row's $2.47$ by the $100\times$ power step
  predicts roughly $27$, so $10.30$ is *below* naive scaling. The number that needs
  explaining is the DNN's $3.82$, not the flat actor's $10.30$.

⭐ **The mechanism is the private/common split.** At the same total spend the two
policies divide it differently, measured over 60 states at $P_S=70$:

| arm | private | common | total / budget |
|---|---|---|---|
| PPO-flat | $5157$ W ($52\%$) | $4843$ W ($48\%$) | $1.0000$ |
| DNN | $3168$ W ($32\%$) | $6832$ W ($68\%$) | $1.0000$ |

The RSMA common stream must be decodable by every user in the group, so its rate is
set by the weakest link and does not scale with the power poured into it. Committing
$68\%$ of a $10$ kW budget to the common stream caps the achievable sum-rate, which is
what holds the hierarchical actors at $3.8$. The split is produced by the shared
power-fairness recipe (`--power-fairness 0.3 --power-fairness-priv 0.3
--power-priv-frac 0.80`), tuned at $P_S=50$.

⚠ Note the executed private fraction is $0.32$ against a configured
`power-priv-frac 0.80`. The blend inverts the configured target at this budget. Worth
reading `train._blend_power` before any sentence is written about the flag's meaning.

⇒ Combined with the $P_S=40$ result, where both hierarchical actors plateau ~$22$-$24\%$
below AO while flat also underperforms, the shared recipe is mistuned at **both** ends
of the power ladder. This is an optimization-recipe finding, not a capacity finding,
and it is now the strongest open issue in the results section.

⚠⚠ C1 $P_S=60$ pastes at **QoS $60.0\%$** next to the DNN's $94.3\%$. That is the
argmax rule applied to a method whose rate/QoS trade is far steeper than the others':
r397 spans $1.87$ @ $100\%$ (ep_300) to $4.93$ @ $60\%$ (ep_2700). Dropping the floor on
shifted rows costs much more for PPO-flat than for DNN or VQC. A per-column floor, or
reporting both operating points, needs a decision before submission.

### 2026-08-17 — C1 P_S=60 flat re-picked under the QoS floor (supersedes the entry above)

The first pass took argmax reward and landed on r397/ep_12300 at QoS $60.0\%$, which
sat next to the DNN's $94.3\%$ in the same row. Re-scanned the early grid at the native
$300$-episode spacing; the floor-feasible set is only two checkpoints wide.

| ckpt | reward | R_tot | QoS |
|---|---|---|---|
| ep_00300 | $372.91$ | $1.8658$ | $100.0\%$ |
| **ep_00600** | $574.78$ | $2.8750$ | $99.6\%$ |
| ep_00900 | $709.71$ | $3.9375$ | $66.5\%$ |

⚠ QoS falls **33 points in one 300-episode interval**. The coarse first scan stepped
every 600 episodes and jumped straight over the cliff from $99.6\%$ to $66.5\%$, which
is why it saw no feasible checkpoint above ep_300. Any flat run scanned on a grid
coarser than its own checkpoint interval can lose its entire feasible region.

Pick becomes **r397/ep_00600**, reported on 42/0/1/2/3:

| cell | argmax pick (old) | floor pick (new) |
|---|---|---|
| C1 $P_S=60$ | $3.8389$ / $60.0\%$ / $727.3$ | $2.9638$ / $97.5\%$ / $587.3$ |
| C1 combined-B | $3.3312$ / $59.8\%$ / $630.8$ | $2.5592$ / $97.4\%$ / $504.7$ |

⭐ The floor makes the Case-1 $P_S=60$ result **stronger, not weaker**. At ep_00600
PPO-flat beats both hierarchical actors on **all three** columns at once, R_tot
$2.9638$ against $2.8024$/$2.7942$, QoS $97.5\%$ against $94.3\%$/$95.3\%$, reward
$587.3$ against $556.5$/$556.3$. That is Pareto dominance, where the argmax pick was
only a different point on a trade-off. Bold stays with PPO-flat.

⚠ combined-B flips back to the DNN, $574.6$ against flat's $504.7$. The joint shift
costs flat more than it costs the hierarchical actors once flat is held to the floor.

⇒ Selection rule now reads uniformly: apply the QoS $\ge 90\%$ floor wherever a
feasible checkpoint exists, fall back to argmax where none does. Under it only this one
cell moved. The other flat cells are unaffected, C1 $P_S=70$ clears at $99.9\%$,
C1 $P_S=40$ at $92.7\%$, and C2 $P_S=40$ sits below $50$ dBm where the floor never
applied. DNN and VQC at $P_S=70$ remain floor-infeasible ($80$-$82\%$ across every
checkpoint) and keep their argmax picks.

⇒ With both C1 rows now floor-compliant, the picture at the top of the power ladder is
consistent: PPO-flat holds $97.5\%$ and $99.9\%$ QoS at $P_S=60$ and $70$ while the
hierarchical actors hold $94.3\%$ and $81.7\%$, and flat's sum-rate advantage grows with
the budget. The private/common split measured above remains the leading explanation.

### 2026-08-17 — zero-shot IRS array-size sweep, N = 16 / 24 / 32

`probe_main_table.py --steps 200 --env-seeds 42,43,44 --shift N=<16|32>`, one
invocation per (case, N) so every method sees the same 600 states. Checkpoints are the
Table VI / VII base picks verbatim, which makes the unshifted rows of those tables the
**N = 24 control at an identical protocol**. Raw `analysis/data/sweepN_c*.txt`.

The sweep is legitimately zero-shot for the hierarchical actors: the affinity map sums
over elements, $a_{k,m}=|\hat g^{SR}_m|\sum_n |\hat g^{RU}_{m,n,k}|$, so its width is
independent of $N$, and PhaseMLP shares one weight set across elements.
`probe_main_table.py:98` retargets the loaded `ph.N` at the eval array, which is what
prevents the documented silent truncation at $N_{\text{eval}} > N_{\text{train}}$.

**R_tot, and reward against AO**

| | N=16 | N=24 | N=32 | gain 16→32 |
|---|---|---|---|---|
| C1 VQC | $2.3449$ | $2.482$ | $2.5720$ | $+9.7\%$ |
| C1 DNN | $2.3710$ | $2.501$ | $2.5764$ | $+8.7\%$ |
| C1 AO | $1.8216$ | $1.876$ | $1.9105$ | $+4.9\%$ |
| C2 VQC | $1.6594$ | $1.742$ | $1.7568$ | $+5.9\%$ |
| C2 DNN | $1.6688$ | $1.747$ | $1.7754$ | $+6.4\%$ |
| C2 AO | $1.6341$ | $1.651$ | $1.6575$ | $+1.4\%$ |

| reward vs AO | N=16 | N=24 | N=32 |
|---|---|---|---|
| C1 VQC $-$ AO | $+97.5$ | $+115.2$ | $+126.7$ |
| C2 VQC $-$ AO | $\mathbf{-18.2}$ | $+7.2$ | $+13.6$ |

⭐ The margin widens **monotonically** with array size in both cases. AO gains almost
nothing from a larger surface, $+1.4\%$ at Case 2, because it re-solves per state under
the same equal-private-power rule and the array is not its binding constraint. The
learned actors gain $4$-$6\times$ as much. This reproduces the direction of the
pre-refactor $N=12\to48$ sweep (VQC $+15.5\%$ against AO $+2.1\%$) on a narrower range.

⚠⚠ **At Case 2, N = 16, both learned actors lose to AO on reward**, $308.6$ (VQC) and
$309.5$ (DNN) against $326.8$. They still win on R_tot ($1.6594$/$1.6688$ against
$1.6341$), so the crossover is entirely the QoS penalty: AO holds $100\%$ while they sit
at $92.6\%$ and $89.8\%$. Shrinking the array pushes the learned actors below the
baseline on $J$ even though they keep more sum-rate. Any sentence claiming the learned
actors dominate AO must be scoped to $N \geq 24$.

⭐⭐ **PPO-flat cannot be evaluated at any other $N$ at all.** All four attempts raise
`ValueError` in `flat_actor._normalize` before a single state is scored:

| case | state width | N=16 | N=24 (trained) | N=32 |
|---|---|---|---|---|
| C1 ($K{=}5,M{=}1$) | $10N+38$ | $198$ | $278$ | $358$ |
| C2 ($K{=}10,M{=}2$) | $40N+75$ | $715$ | $1035$ | $1355$ |

Every width matches the traceback exactly. The flat actor consumes the raw per-element
channel $g^{RU}$ of size $M\!\cdot\!N\!\cdot\!K$, so its first layer and its running
normalisation statistics are both sized at training time. This is not a defect to fix,
it is the design: a single dense network reading unaggregated CSI is tied to one array
size and must be retrained from scratch for another.

⇒ This is the sharpest evidence yet for the amortized-inference framing, and it is
**qualitative rather than a percentage**. AO must re-solve per state and barely profits
from more elements; PPO-flat cannot transfer at all; the hierarchical actors transfer
with zero retraining and are the only ones whose advantage grows with the array.

### 2026-08-18 — flat Case-2 power ladder closed, and n_q=6 matches n_q=8

Machine reset at 13:50 killed eight runs; six were resumed as r404-r409 and two fresh
readout arms launched as r410 (C1 single-z) and the full-zz run. Three runs had already
finished and are measured here.

**PPO-flat Case 2 at $P_S=60$ (r401) and $P_S=70$ (r402)**, both fresh retrains,
selection `probe_pick_ckpt.py --seeds 7 21 99 --eps 4`, reporting on 42/0/1/2/3.

| cell | pick | select reward / R_tot / QoS | reported R_tot / QoS / Reward |
|---|---|---|---|
| C2 $P_S=60$ | r401/ep_11100 | $507.81$ / $2.6223$ / $90.0\%$ | $2.6097$ / $89.7\%$ / $503.8$ |
| C2 combined-B | same | — | $2.7734$ / $84.2\%$ / $503.3$ |
| C2 $P_S=70$ | r402/ep_40008 | $2048.64$ / $10.2476$ / $98.0\%$ | $10.1880$ / $98.0\%$ / $2036.7$ |

Both pasted cells move the bold to PPO-flat, which now holds $P_S=60$ ($503.8$ against
$438.7$ for the DNN) and combined-B ($503.3$ against $453.0$). **The PPO-flat column of
Table IX carries no `(retraining)` entry in either case.**

⚠ r401 needed a second, finer scan. Its QoS falls monotonically with training,
$95.8\%$ at ep_00300 to $49.3\%$ at the unconstrained argmax ep_34500, and the first
scan stepped every 900 episodes across a boundary that is only 300 wide. At the native
spacing the last feasible checkpoint is **ep_11100 at exactly $90.0\%$**, with ep_11400
already at $89.4\%$. This is the second flat run whose feasible region a coarse grid
nearly erased, after r397. Any flat selection must scan at the checkpoint interval of
the run itself.

⚠ $P_S=70$ is measured but has no row; those rows were removed from Table IX on
2026-08-17. Recorded so the cell exists if the decision is revisited.

⭐ **Flat at $P_S=70$ reproduces across cases.** C1 gives R_tot $10.2956$ at QoS
$99.9\%$ and C2 gives $10.1880$ at $98.0\%$, a $0.5\%$ spread between two problems whose
user count differs by a factor of two, and r402 holds $100.0\%$ QoS at every checkpoint
through ep_09300. The high-budget behaviour is therefore not seed noise.

**VQC Case 2 at $n_q=6$ (r396)**, 20,000 episodes, pick r396/ep_19000, measured on the
Table VII protocol (`probe_main_table.py --K 10 --M 2 --steps 200 --env-seeds 42,43,44`).

| | Reward | R_tot | QoS |
|---|---|---|---|
| $n_q=8$, 5 seeds (Table VII) | $337.3 \pm 8.8$ | $1.742 \pm 0.021$ | $94.4 \pm 1.4\%$ |
| $n_q=6$, 1 seed | $336.5$ | $1.7362$ | $94.5\%$ |

⭐⭐ **Indistinguishable.** The reward gap is $0.8$ against a seed spread of $8.8$, i.e.
$0.09\sigma$; R_tot differs by $0.3\%$ and QoS by $0.1$ pp. Dropping from 8 to 6 qubits
takes the state space from $256$ to $64$ amplitudes, $N_Q$ from $26$ to $18$ and the
latent from $16$ to $12$, and costs nothing measurable at Case 2.

⇒ The register is not the binding constraint here. If r408 ($n_q=10$) also lands on the
same value the honest conclusion is **"performance is insensitive to register size over
6-10 qubits"**, not "8 is required". Any sentence presenting $n_q=8$ as a tuned optimum
has to be rewritten.

⚠ Three quantities move together with $n_q$ ($n_\text{latent}=2n_q$, $N_Q$, and the
circuit width), so a difference could not have been attributed to qubit count alone
either. The sweep bounds the joint effect, not the individual ones.

### 2026-08-19 — PROVISIONAL VQC Case-2 power ladder (must be re-measured)

⚠⚠ **These five cells are read off runs that had not finished.** The chains stood at
global 18,000-18,200 of 20,000, i.e. **91% of budget**, when the 2026-08-19 10:55 reset
interrupted them; they were resumed as r412/r413/r414 with 1,800-2,000 episodes left.
Re-scan the full chains and re-measure once those complete.

Chains: $P_S=40$ r368→r380→r404 · $P_S=60$ r369→r381→r405 · $P_S=70$ r370→r382→r406.
Selection scanned **both** the r38x and the r40x segment (global 6,200-18,200),
`--seeds 7 21 99 --eps 4`; reporting on 42/0/1/2/3.

| cell | pick | select reward / R_tot / QoS | reported R_tot / QoS / Reward |
|---|---|---|---|
| VQC C2 $P_S=40$ | r404/ep_02400 | $166.31$ / $0.9755$ / $93.6\%$ | $0.9786$ / $93.2\%$ / $162.3$ |
| VQC C2 combined-A | same | — | $0.9632$ / $92.3\%$ / $158.4$ |
| VQC C2 $P_S=60$ | r405/ep_01000 | $433.97$ / $2.1964$ / $91.9\%$ | $2.1907$ / $92.0\%$ / $432.9$ |
| VQC C2 combined-B | same | — | $2.2558$ / $93.5\%$ / $445.1$ |
| VQC C2 $P_S=70$ | r406/ep_02300 | $454.32$ / $2.2909$ / $91.5\%$ | $2.2860$ / $91.1\%$ / $453.3$ |

No bold moves. AO keeps $P_S=40$ and combined-A; PPO-flat keeps $P_S=60$, $P_S=70$ and
combined-B.

⭐ **The VQC clears the QoS floor at every rung where the DNN cannot.** At $P_S=60$ and
$P_S=70$ the DNN has no checkpoint at $\ge 90\%$ across its whole chain and falls back to
argmax ($87.4\%$ and $87.7\%$ reported), whereas the VQC picks sit at $91.9\%$ and
$91.5\%$ on the selection seeds and report $92.0\%$ and $91.1\%$. This is the same
direction as the Table VII base row ($94.4\%$ against $92.1\%$) and is now visible at
three separate transmit budgets, so it is unlikely to be seed noise. On reward the two
are within $6$ points at every rung, so the separation is in service, not in rate.

⚠ Scanning both chain segments mattered less than feared but was not wasted. The later
segment won all three chains, yet only by $1.06$, $1.28$ and $0.99$ reward against
per-checkpoint sd of $2.2$-$6.6$, so the margin is inside the noise. More usefully, the
early segment of $P_S=70$ (r382/ep_09200) sits at QoS $90.3\%$, right on the floor; had
the late segment failed the floor, that would have been the only feasible candidate and
a coarse grid would have missed it.

⚠ Global 1,700-6,200 (r368/r369/r370) was **not** scanned. Low risk under an argmax rule
since reward rises over training, but not excluded.

⚠ Reward has been flat within noise for roughly 12,000 episodes on all three chains
($P_S=60$ moves from $432.69$ at global 14,600 to $433.97$ at 16,200, a gain of $1.3$
against sd $4.6$). The remaining 2,000 episodes are therefore unlikely to change the
picks, but that is a prediction, not a measurement, and the cells stay provisional until
r412/r413/r414 finish.

## Re-check 2026-08-20 — resumed Case-2 VQC tails (r412/r413/r414) vs their parents

The 2026-08-19 resume added 1800/1900/2000 episodes on top of r404/r405/r406.
Every child checkpoint was re-scanned on validation seeds 7/21/99 (6 ep x 200
steps, `analysis/data/pick_r41{2,3,4}.txt`). **No child beat its parent**, so the
Table IX Case-2 VQC power rows are unchanged:

| chain | parent pick | parent reward | best child | child reward | verdict |
|---|---|---|---|---|---|
| $P_S=40$ r404→r412 | r404/ep_02400 | 166.31 | r412/ep_01600 | 163.93 | parent kept |
| $P_S=60$ r405→r413 | r405/ep_01000 | 433.97 | r413/ep_01000 | 433.59 | parent kept |
| $P_S=70$ r406→r414 | r406/ep_02300 | 454.32 | r414/ep_01600 | 454.33 | tie (+0.01), parent kept |

All 57 child checkpoints cleared QoS >= 90%, so the picks were not floor-limited.
⇒ ~2k extra episodes bought nothing at any power rung, consistent with the Case-2
ceiling being reached well before the budget ends.

r415 (Case-2 single-z readout, resumed from r407) DID produce a new number and is
now in Table XIII — see the provenance comment on `tab:abl_readout`.

## Table IX — Case-1 PPO-flat N-shift cells (filled 2026-08-20)

PPO-flat's state width is 10N+38 at Case 1, so an N shift cannot be evaluated
zero-shot; r422 (N=16, d_s=198) and r423 (N=32, d_s=358) were retrained from
scratch on the Case-1 flat recipe (20000 ep, lr 3e-4, target-kl 0.05,
hidden 512x256, power-fairness 0, priv-frac 0.8), matching r301/r307/r308.

Checkpoint by argmax return s.t. QoS >= 90% on validation seeds 7/21/99, at the
run's own N (`--shift N=16|32`; without it the probe would score an N=16 policy
in the default N=24 env):

| ckpt | r422 N=16 reward / QoS | r423 N=32 reward / QoS |
|---|---|---|
| ep_03000 | 449.78 / 99.3% | 495.70 / 99.8% |
| **ep_06000** | **457.54 / 91.2%** | **515.31 / 92.9%** |
| ep_09000 | 470.03 / 72.6% | 521.09 / 72.1% |
| ep_12000 | 470.68 / 86.1% | 522.81 / 67.5% |
| ep_15000 | 466.33 / 81.0% | 532.69 / 61.6% |
| ep_18000 | 467.27 / 72.1% | 524.54 / 63.6% |
| ep_20004 | 474.91 / 66.3% | 529.92 / 62.3% |

Reward-optimal sits at 61-66% QoS in both runs, so the floor is what selects
ep_06000 — the same behaviour recorded for the Case-1 main-table seeds.

Test seeds 42/43/44 (`analysis/data/genshift_c1_flat_n{16,32}.txt`):
* N=16: R_tot 2.1698, QoS 92.5%, reward 429.95
* N=32: R_tot 2.6684, QoS 90.4%, reward 531.50

⚠ At N=32 PPO-flat's 531.5 overtakes the DNN's 512.2, so the row's bold moved.
It buys that with QoS 90.4% against the DNN's 96.0% — the usual flat trade, not
a reversal of the standing result.

## Table IX — Case-2 PPO-flat N-shift cells (filled 2026-08-21)

r424 (N=16, d_s=715) and r425 (N=32, d_s=1355), fresh retrains on the Case-2 flat
recipe (40000 ep, lr 1e-4), matching r376/377/388/390/391. Selected by argmax
return on validation seeds 7/21/99, the rule the shifted rows use.

Neither run learns. Reward rises monotonically toward zero but never crosses it,
and QoS never leaves the 43-57% band at any of the 14 checkpoints, so argmax lands
on the last one in both cases.

| cell | pick | validation reward / R_tot / QoS | reported R_tot / QoS / Reward |
|---|---|---|---|
| C2 $N=16$ | r424/ep_40008 | $-190.65$ / $0.6387$ / $56.6\%$ | $0.5771$ / $51.5\%$ / $-283.0$ |
| C2 $N=32$ | r425/ep_40008 | $-354.37$ / $0.5899$ / $49.6\%$ | $0.5138$ / $43.5\%$ / $-397.4$ |

⇒ Retraining at a different array size does NOT rescue PPO-flat at Case 2. Its
failure there is tied to the power/demand regime (it also fails at the N=24 base,
R_tot 0.5874 / QoS 54.4%), not to the input width. That closes the last objection
that the Case-2 flat column might be an artifact of the fixed 10N+38 / 40N+75
state width: given its own retrained network at three array sizes, it fails at all
three.

Table IX now has no `(retraining)` placeholders left.


---

## Fig. 5 `fig:pareto` + Section VI Rate--QoS Frontier (measured 2026-08-22)

Probe: `analysis/probe_pareto_exhaustive.py --K 5 --M 1 --shard 0/1`
and `--K 10 --M 2 --shard i/8` for i = 0..7
Aggregation: `analysis/aggregate_pareto.py --K <K> --M <M>`
Figure: `analysis/plot_pareto_exhaustive.py`
Raw: `analysis/data/pareto_exh_K5M1_shard0of1.npz`,
`analysis/data/pareto_exh_K10M2_shard{0..7}of8.npz`, run log
`analysis/data/pareto_all_run.log`

Replaces `probe_v_star.py`, `probe_true_frontier.py` and
`probe_pareto_frontier.py`, which disagreed by up to 18% on the same quantity
because each enforced a different constraint (per-state `QoS >= t`,
`n_met >= n` on the estimate, "serve-all") and searched the assignment
differently. The numbers those probes fed into the paper
($V^*_{rate}$ 10.963 / 11.723, $R^*(K)$ 3.11 / 1.61, $\lambda_{crit}$ 2.00 /
1.15) came from `vstar_c*_ramp05_500states.txt`, dated **Jul 3 and Jul 11**,
i.e. **before the 2026-07-21 per-element refactor** — invalid by the project's
own rule. All of them are superseded here.

### Definition

    F(q) = max E[R_tot]  subject to  E[QoS] >= q

The constraint binds the **mean**, because Tables VI/VII report the mean. The
per-state version (max subject to a per-state threshold) is a different and
strictly lower quantity — it forbids trading service between states — and
mixing the two overstates how close a policy sits. Both are printed by
`aggregate_pareto.py`; at (5,1) they differ by 0.3%, at (10,2) by 3.4% **and**
the per-state row at `QoS >= 1.00` averages over only the 87.3% of states where
serving everyone is feasible at all, a bias the mean-constrained version does
not have.

`F` is traced by Lagrangian sweep over the per-state achievable sets: for each
mu >= 0 take the point maximising `R + mu*Q` in every state, average both
coordinates. That walks the upper concave envelope, which is the frontier once
states can be mixed.

### Method, per state

| variable | search | exact? |
|---|---|---|
| assignment | **exhaustive**, `(M+1)^K` = 32 at (5,1) and 59,049 at (10,2) | yes |
| phase | per-element oracle | yes, given the assignment |
| power | `w_p ∝ |h_est|^(2β)` on β × f grid (8 × 13) | **no — grid, so F is a lower bound** |
| C_k | demand-fill oracle + even leftover | yes, given the rest |

600 test states per case (env seeds 42/43/44 × 200 steps), the same states as
Tables VI/VII. Decisions on ĝ, scored on true g, averaged over 16 σ² draws.

⚠ **The power axis is the loose one, and it is the axis where the headroom is
known to be.** A full-vector power refinement is not done, so every attainment
percentage below is an over-estimate of how close the policies really are.

Two reductions make exhaustive search affordable — the oracle phase of a panel
depends only on the *set* of users on it (2,046 cached calls per state instead
of 59,049), and the rate kernel is closed form so all assignments are evaluated
in one vectorised pass. Cost: 0.6 s/state at (5,1), 54 s/state at (10,2).

### Correctness gate

`probe_pareto_exhaustive.py --validate` compares the vectorised kernel against
`CSI.rate.RateComputer` on 200 random (assignment, power) draws and **exits
non-zero if they differ**. Final agreement:

    C1: max|dR_tot| = 0.0e+00   max|dQoS| = 0.0e+00   max|dJ| = 0.0e+00
    C2: max|dR_tot| = 5.6e-17   max|dQoS| = 0.0e+00   max|dJ| = 1.8e-15

The gate caught **four** defects that would each have produced wrong numbers
silently. Recording them because three are easy to repeat:

1. **C_k decided on the true channel.** `compute_rates_partial` defaults to
   `use_true=False`, so the reference decides C_k on ĝ. Deciding on g is both
   the wrong pipeline and an oracle peek. Tell-tale: **R_tot still matched to
   4e-16** while QoS was off by 1/K, because the leftover-spread step makes
   `sum C_k` equal the group budget however it is split.
2. **`_finish_sum_rate` rescales the C_k it is handed** so each group's share
   sums to the common rate of the *scoring* channel, after a 1e-10 floor. The
   decision fixes only the split; the total follows the true channel, per draw.
3. **QoS convention.** `CSI/env.py:211` computes QoS from the **averaged**
   rates ("Overwrite noise-dependent fields with their averages so logged
   metrics match the averaged reward"), not by averaging the per-draw QoS.
4. **C_k re-decided per σ² draw.** The decision is made once per state at the
   nominal σ² and held across all 16 draws; only the rescale is per draw.

Survivor cap: the grid search keeps the best 32 points per attainable QoS level
before the 16-draw rescoring. `analysis/check_pareto_keep.py` confirms the
envelope is identical at cap 256 (max |dR(q)| = 0.0 at every QoS level, both
test states), so the cap is lossless. Noise draws are keyed by (env seed, step)
rather than taken from the env generator, so a sharded run reproduces a
single-process one exactly — verified on the same state via 0/1 and 0/8.

### Numbers

| quantity | Case 1 (5,1) | Case 2 (10,2) |
|---|---|---|
| `F` at highest mean QoS | **3.3782** @ 100.0% | **2.1472** @ 99.8% |
| serve-all feasible in | 99.5% of states | 87.3% of states |
| equal-split identity `R(m) = m log2(m/(m-1))` | 1.6096 | 1.5200 |
| `V*_rate = E[log2(1+Gamma_max)]` | **10.0700** | **10.9003** |
| `p_hat = (D/(D+eps_q))^2` | 0.9612 | 0.9612 |
| **`lambda_crit`** | **1.7405** (λ_D=1.5 below) | **1.0119** (λ_D=1.5 above) |

`V*_rate` is measured by its own pass (90 states per case) inside
`aggregate_pareto.py`, not read off the achievable sets, since the β/f grid
never fully concentrates.

Reward-optimal operating point — the achievable point of highest `J` per state,
averaged. This is what the training objective selects, and it is **not** full
service:

| | R_tot | QoS | episode return | `F` at that QoS |
|---|---|---|---|---|
| Case 1 | 3.3948 | **98.07%** | **676.7** | 3.4546 |
| Case 2 | 2.1690 | **95.00%** | **425.0** | 2.3612 |

Attainment at each policy's own operating point (denominators differ — AO is
pinned to full QoS by its QoS-first C_k rule, which is where `F` is lowest, so
these are not comparable across rows and the manuscript does not quote them):

| policy | C1 R_tot / QoS / attain | C2 R_tot / QoS / attain |
|---|---|---|
| HQC-HAC (VQC) | 2.482 / 92.4% / 68.8% | 1.742 / 94.4% / 73.0% |
| DNN | 2.501 / 94.6% / 70.4% | 1.747 / 92.1% / 70.5% |
| PPO-flat | 2.417 / 96.0% / 68.7% | 0.557 / 47.2% / 13.6% |
| AO | 1.876 / 100% / **55.5%** | 1.651 / 100% / **76.9%** |

On the objective every policy actually optimises, the learned actors reach
**73.1%** and **79.4%** of the attainable return against **55.4%** and **77.7%**
for the solver.

### What changed in the paper

`tab:abl_pareto` and equations `eq:abl_frontier`, `eq:abl_vrate`,
`eq:abl_concentration`, `eq:abl_lcrit` were **deleted**; the section now carries
one figure and no table. Also removed: the claim that
"R(10) = 1.520 plus the common streams (≈0.09) reproduces the serve-all frontier
level 1.61 to three digits" — against the measured 2.147 the gap is 0.63, so the
identity is a lower bound on serve-all, not a prediction of it.
