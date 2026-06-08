---
name: irs-rsma-log-reading
description: >-
  Structured PROTOCOL for reading & assessing a training run's log in this repo
  (results/result_N/training_log.txt). Use this skill WHENEVER the user wants to
  inspect or judge the health of a training run — e.g. "check result_N", "how's
  training going", "read/assess the training log", "is QoS improving", "is the run
  healthy / stuck / collapsing", "should I keep training or stop", "is it stable
  enough to advance the ramp", or any request to summarize what a run's log shows.
  It produces a consistent assessment covering: (1) the R_tot & QoS trend (improving
  or not), (2) overall agent status with a LONG-TERM view when the R_LoS curriculum
  isn't finished, plus what to change and how, and (3) whether any sub-actor/head is
  struggling or showing signs of it. For the meaning of each signal and the fixes,
  it leans on docs/Common-Knowledge.txt; for per-case hyperparameter context, on the
  matching docs/Training-Case-<N>.txt. When it finds a problem, hand off to the
  irs-rsma-training-playbook skill for the symptom→fix detail.
---

# IRS-RSMA Training-Log Reading Protocol

A repeatable procedure to read `results/result_N/training_log.txt` and produce a
consistent health assessment. Always follow Steps 0→4 in order, then emit the report
template. For signal meanings / fixes see `docs/Common-Knowledge.txt`; for per-case hyper
context see `docs/Training-Case-<N>.txt` (match by K,M in params.py / hyperparameters.json).

## ★ PRIORITY ORDER (judge & fix in THIS order — Common-Knowledge Part 5 #8)
**CRITIC → PhaseMLP (learning to optimize IRS) → NO-IDLE heads (esp. post-VQC:
phase/power/ck) → QoS → SUM-RATE.**
This is the order to ASSESS a run and to decide what to fix. A lower-priority metric
being bad is NOT actionable until every higher-priority subsystem is healthy:
- A bad QoS while the **critic is unstable** (explVar wobbling / <0.5) is EXPECTED — do
  not tune QoS levers (λ_D, readout) yet; fix the critic first. QoS ⊥ critic: QoS stuck
  at one level ACROSS runs regardless of explVar ⇒ bottleneck is downstream (PhaseMLP/
  heads), not the critic — but you still cannot MEASURE a downstream fix cleanly until
  the critic is stable.
- QoS at ramp 0.2 is **not forced by tuning** — it is traded off via `λ_D` at each LATER
  ramp (per feasibility probe). Don't over-optimize QoS at 0.2; secure critic + PhaseMLP.
- Sum-rate is the LAST polish — only meaningful once everything above is healthy.
So in the report: lead the ACTION with the highest-priority unhealthy subsystem.

## Log format reminder
Per-episode rows: `Ep Reward BestReward L_pg L_ae L_phs L_pw L_ck TD ClpQ ClpP ClpW ClpC R_tot R/user QoS/K IRS/K Blk/K s/ep`.
Every PPO update prints a `diag[...]` block (grad norms ‖∇‖, KL, ent, V̄/σV/explVar,
λ|max|, R̄−qp), an `ent/pg` line, a `rolling-50` line, and a `behaviour` line.

--------------------------------------------------------------------------------
## STEP 0 — Context (what am I even looking at?)
--------------------------------------------------------------------------------
Establish before judging anything:
- Run config: `grep -E '"K"|"M"|"R_LoS_km"|"lambda_D"|"gae_lambda"|"n_episodes"' results/result_N/hyperparameters.json`
  → which **Case** (K,M), which **R_LoS ramp**, λ_D, gae, episode target.
- Resume source: `grep "RESUMED actors from" .../training_log.txt` (fresh vs resumed).
- Feasibility probe (ceiling & baselines): `grep -E "Servable users|All-IRS|Direct-only" .../training_log.txt`.
- Progress & liveness: `wc -l` the log, last `checkpoints/` dir, and is it still running?
  (`ps -eo pid,stat,cmd | grep train`; compare log mtime to now — if frozen + process
  alive at high CPU → STOP, this is a GPU hang `[H]`, see playbook.)
- **Is the curriculum finished?** Note current R_LoS vs the 0.2→0.5 target, and episodes
  done vs target. This decides whether to judge by long-term trend (mid-ramp) or by
  final stable performance (last ramp, near done).
- **★ READ THE RUN-SUMMARY TABLE** at the top of `docs/Training-Case-<N>.txt`
  ("BẢNG TÓM TẮT RUN") + the "LEVER ĐÃ LOẠI" notes. This is the memory of every lever
  ALREADY TRIED, its result, and the limitation it did NOT fix. You MUST consult it
  before proposing any hyper change (see the DON'T-REPEAT guard in Step 4).

--------------------------------------------------------------------------------
## STEP 1 — R_tot & QoS trend  (improving or not?)
--------------------------------------------------------------------------------
Extract the trajectory:
`grep -nE "rolling-(50|[1-4][0-9]): reward" .../training_log.txt | awk 'NR%5==1'`
(also read the last few raw `diag` blocks for the current state.)

### ★★ EARLY-FLAT PATTERN — compare against baseline result_11 BEFORE calling failure
⚠ DO NOT diagnose "policy fail" from a flat early trajectory alone. The known-good
baseline result_11 (D_k=0.10, R_LoS=0.2, Case 2 main reference) was ALSO flat for
~400 episodes before climbing:
  ep~50    QoS 52%  R_tot 1.37  reward -286
  ep~250   QoS 52%  R_tot 1.40  reward -243   ← STILL FLAT after 200 ep
  ep~350   QoS 53%  R_tot 1.43  reward -215   ← still flat
  ep~500+  QoS 53→60% climbing begins
  ep~1200  QoS 60%  R_tot 1.75  PEAK
That's ~400 episodes of "looking failed" before genuine learning kicked in.

Protocol when assessing a run at < ep600 with flat QoS:
  1. Compare the run's (QoS, R_tot, reward μ) trajectory shape to result_11's same-ep
     range. If the SHAPES match (both flat, both slowly drifting), the run is in the
     normal early-learning phase — NOT failed.
  2. Only call "policy fail" if EITHER:
     • QoS / R_tot / reward is STRICTLY WORSE than result_11 at the same ep, AND the
       gap is widening (not closing); OR
     • the run has run ≥ 600 ep and is still flat (result_11 had started climbing
       by then).
  3. Different regimes (D_k, R_LoS) shift the absolute floor — what matters is the
     SHAPE OF THE CURVE relative to baseline, not the absolute value at any one ep.
  4. Cross-check with critic health (Step 2): if explVar trending up and ‖∇‖V
     decreasing, the policy is still learning even if QoS is flat — wait it out.

This guard prevents premature SIGINT + wasted re-runs (real cost). ALWAYS reference
result_11's early-flat phase before declaring early-stop on a < ep600 run.

### ★ EARLY-RAMP IRS CHECK (always run on R_LoS=0.2/0.3 — critical for whole curriculum)
The agent MUST learn to optimize IRS at the easy ramps; if it can't here (where QoS is
slack), it never will at hard ramps. So on early ramps, the key question is NOT "is QoS
high" but **"is PhaseMLP learning to optimize IRS, or idle?"** Trading a few early
episodes of lower QoS to explore/learn IRS phases is WORTH it — but only if that
exploration is PRODUCTIVE (phase converging), not random thrashing.
Check two things:
- **PhaseMLP idle?** Track phase entropy over the run (`ent ph` in diag, max = M·N·ln4):
  `grep -oE "ph=[0-9.]+ pw=" log | grep -oE "ph=[0-9.]+"`.
  DECREASING over the run → learning to pick phases (GOOD, optimizing IRS).
  FLAT & HIGH (~>60% of max) while `KL ph`/`‖∇ph‖` nonzero → **IDLE / thrashing**
  (gets gradient but never converges) → NOT optimizing IRS `[F]/[J]`.
- **IRS being used?** `behaviour` line `irs=[..]` and `Ck tot` over time: held ~mid
  (still exploring) vs collapsing toward 0 (direct-attractor `[J]`).
Root causes if phase idle: blind critic `[B]` (no directional signal) and/or
entropy-domination `[C]` (β keeps phase random). BOTH must clear before phase can learn
— so phase won't converge until critic works (explVar>0.5) AND β anneals to floor.
Verdict: state explicitly whether the agent is LEARNING TO OPTIMIZE IRS or just
using-IRS-with-random-phases (idle). This gates curriculum readiness more than QoS.

Read the three together — they tell different things:
- **QoS μ** = the PRIMARY objective metric. This is what "improving" means. Compare
  early vs mid vs late: rising / flat / declining / oscillating.
- **R_tot μ** (sum-rate) = capacity used. It TRADES OFF with QoS (serving weak users
  costs rate). Interpret jointly:
    QoS↑ & R_tot↓  → redistributing toward QoS (usually GOOD).
    QoS↑ & R_tot↑  → genuinely improving (great).
    QoS↓ & R_tot↓  → COLLAPSE `[A]`.
    both flat       → plateau / converged `[D]`.
- **Reward μ** = λ_D-scale-dependent → do NOT compare across runs with different λ_D;
  use it only within-run for direction. With low D_k, reward ≈ tracks QoS via penalty.
- Anchor to the probe: how far is QoS from the **servable ceiling** and above the
  **All-IRS / Direct** baselines? (Beating All-IRS = the agent is adding value.)
- Note **best QoS achieved and at which episode/checkpoint** (best model is by QoS from
  `checkpoints/`, NOT the reward-based `best/` dir).

Verdict for Step 1: one of {improving, holding/plateaued, collapsing, oscillating},
with the QoS numbers and the R_tot interpretation.

--------------------------------------------------------------------------------
## STEP 2 — Agent status & what to change  (LONG-TERM view if ramp unfinished)
--------------------------------------------------------------------------------
Time horizon matters:
- **Ramp NOT finished** (more R_LoS steps coming, or episodes remaining): judge the
  LONG-TERM trajectory and stability, not the instantaneous value. A mid-ramp dip is
  fine if the trend is healthy and critic is sound. Don't over-react to one bad window.
- **Last ramp / near done**: judge final stable performance and pick the best checkpoint.

Check stability/health signals (meanings in Common-Knowledge Part 3/4). Run BOTH
health checklists — critic and agent are SEPARATE subsystems (critic = V(s), independent
of the actor/readout):

★ CRITIC HEALTH — read explVar + V̄ + σV + ‖∇‖V TREND together (over several diags, not
  one). The critic gates everything: a sick critic → noisy advantages → agent can't learn,
  no matter how good the actor. Three "diseases" (fixes in playbook):
  - explVar ≈0 + V̄ wrong scale (≈±1 when returns are ±tens), esp. right after resume →
    BLIND / cold-start `[B-1]` → critic warm-up / load saved critic / tau↑.
  - explVar ≈0 PERSISTING after entropy-domination cleared (cờ ⚠ off) over hundreds of ep →
    CAN'T-FIT `[B-2]` → reward_noise_avg↑ + critic warm-up (incl. fresh).
  - explVar WOBBLING (e.g. 0.1↔0.6, not holding >0.5) AND `‖∇‖V` PLATEAU LARGE (e.g.
    700-1000, clipped hard) NOT decreasing → UNSTABLE `[B-2b]` → GAE-return critic target
    (regress t['ret'] not TD(0)), + lr_critic↓ if still wobbling.
  - HEALTHY: explVar >0.5 stable · ‖∇‖V decreasing/small · V̄ ≈ return scale · σV has spread.

★ AGENT/POLICY HEALTH — map to the symptom codes:
  - Collapse `[A]`: QoS peaks then falls + `behaviour` wc→~24% / Ck→~0.46 (common-stream sink).
  - Entropy-dom `[C]`: `ent/pg` phase > ~0.3 / harness "⚠ ENTROPY-DOMINATED".
  - Plateau `[D]`: `L_pg` ≈0 + reward/QoS flat + best unbeaten.
  - Direct-attractor / IRS idle `[J]`: dir→K, irs→~0, phase entropy stuck high (early-ramp check).
  - Frozen-λ `[E]`: `λ|max|` stuck at init. PhaseMLP weak `[F]`: ent phase stuck high + KL/grad large.

Then decide ONE action and justify it:
- **Keep training** (trend healthy, ramp unfinished) — say what to watch & for how long.
- **Adjust a hyper** — name the lever via the fix code: λ_D `[D]/[I]`, β `[C]`,
  gae `[A]`, spsa/lr_qc `[E]`, critic warm-up/tau `[B]`. Give the concrete change.
- **Stop & take checkpoint** (plateaued/converged or collapsing) — name the best QoS
  checkpoint.
- **Advance the ramp** — only if STABLE at current R_LoS (QoS holding/plateaued cleanly,
  no entropy-domination). Else stabilize first. (See playbook curriculum protocol.)

--------------------------------------------------------------------------------
## STEP 3 — Per-head health  (is any sub-actor đuối / showing signs?)
--------------------------------------------------------------------------------
From the latest `diag` block read `ent q/ph/pw/ck`, `KL`, `‖∇‖` per head, `λ|max|`, and
the `behaviour` line. Judge each head two ways: (a) entropy as % of its MAX (random
baseline), and (b) does its behaviour show STRUCTURE.

Max-entropy baselines (general — recompute per Case from K,M):
- IRS-select (q): K·ln(M+1).      [Case1: 5·ln2≈3.47 · Case2: 10·ln3≈10.99]
- PhaseMLP (ph):  M·N·ln(levels). [Case1: 24·ln4≈33.3 · Case2: 48·ln4≈66.5]
- PowerMLP (pw):  ln(#outputs).   [≈ln(G+1+K)]
- CkMLP (ck):     ln(group size).

Signs a head is **đuối / not learning**:
- Entropy stuck HIGH (>~60% of max) AND not decreasing over the run, while its `KL` and
  `‖∇‖` stay large → it's "vùng vẫy" (getting strong gradient but can't converge).
  Most often PhaseMLP `[F]`. Disambiguate: flat landscape (OK, leave) vs under-converged
  (sharpen) → confirm with greedy eval (greedy QoS ≈ sampled → leave; ≫ → sharpen).
- A head's `behaviour` stays UNstructured (e.g. power spread ~uniform: wp-top2≈2/K;
  Ck flat) → not learning to prioritize.
- Quantum scales **frozen**: `λ|max| y/z` stuck at init (~0.50/0.45) across the run →
  QC not learning `[E]` (worse at larger nq).
- Healthy head: entropy LOW %-of-max (converged) OR high but with clearly structured
  behaviour + small KL.

Report which heads are healthy, which are borderline, which are đuối, with the evidence.

--------------------------------------------------------------------------------
## STEP 3.5 — CONFIRM ROOT-CAUSE WITH A TARGETED PROBE  (before proposing a fix)
--------------------------------------------------------------------------------
The training_log alone shows SYMPTOMS; the standalone probes in `analysis/` are cheap
COUNTERFACTUALS that isolate the ROOT (hold the policy, swap one head / measure the
physical layer, recompute the metric — no ablation training needed). When a symptom
below appears and you're about to recommend a fix, RUN the matching probe first and
report its verdict — it has repeatedly overturned a wrong diagnosis (e.g. power-qos found
the QoS bottleneck was PowerMLP, not phase/critic). Run on the best-QoS checkpoint.

  Symptom (from Steps 1-3)                              | Probe (analysis/)              | Confirms
  -----------------------------------------------------+--------------------------------+-----------------------------
  QoS stuck at one level ACROSS runs regardless of     | probe_power_qos.py             | is it PowerMLP power-
  critic health (QoS ⊥ critic); users miss D_k         |  + probe_qos_by_assignment.py  | concentration? which link?
  -----------------------------------------------------+--------------------------------+-----------------------------
  "Is low IRS-usage optimal / should agent use IRS     | probe_irs_vs_direct.py         | physical IRS-opt vs direct
  more?"  (IRS/K ≈ block-count, phase idle)            |  (optimal-phase, physical)     | per user (block/non-block)
  -----------------------------------------------------+--------------------------------+-----------------------------
  Critic loạn dai dẳng — env-bound (aleatoric) vs      | probe_critic_ceiling.py        | explVar ceiling under
  trainable (training-dynamics)?                       |  (Tier-1 ceiling)              | random vs trained policy
  -----------------------------------------------------+--------------------------------+-----------------------------
  λ frozen [E] (λ|max| đứng init, λgrad r low) —        | probe_lam_interference.py      | per-sample ||g_b|| + mean
  INTERFERENCE (random dirs cancel) vs VANISHING       |  (mean pairwise cos)           | pairwise cos: cos≈0+mag OK
  (barren-plateau)? confirm BEFORE any λ lever          |                                | =interference; small mag=vanish
  -----------------------------------------------------+--------------------------------+-----------------------------
  PhaseMLP: entropy dropping but IS the phase actually  | probe_phase_quality.py         | |Σφ|live/N coherence +
  ALIGNING the channel? (priority-#2 — necessary check  |  (greedy live PhaseMLP)        | alignment% (0 rand→1 opt);
  that arch-2 unblocked phase, not just lower entropy)  |                                | live-IRS beats direct?

  PRINCIPLE: phase entropy dropping is NECESSARY but NOT SUFFICIENT for "PhaseMLP learning
  to optimize IRS" — entropy can converge to a BAD phase. ALWAYS confirm phase QUALITY
  (channel alignment a = (|Σφ|/N − 1/√N)/(1 − 1/√N)), not just entropy, before declaring
  priority-#2 solved. BASELINE (result_8, old arch frozen-λ): alignment ≈ 25% (partial, far
  from optimal) — arch-2 must raise this. a≥60% = optimizing; a≈0% = idle [F] despite entropy.

--------------------------------------------------------------------------------
## ★ DON'T-REPEAT GUARD  (run BEFORE writing any "→ ACTION: adjust hyper" line)
--------------------------------------------------------------------------------
Cross-check every proposed hyper change against the run-summary table + "LEVER ĐÃ LOẠI"
in `docs/Training-Case-<N>.txt` (read in Step 0). For each lever you're about to recommend:
- **Already tried & FAILED to fix this symptom?** → DO NOT recommend it again. Either pick
  a different lever, or explicitly state what is DIFFERENT now that makes it worth a retry
  (e.g. a precondition that wasn't met before). Name the prior run(s) in your reasoning.
- **In LEVER ĐÃ LOẠI (ruled out with evidence)?** → off the table; cite the evidence.
- **The symptom's true root is a DIFFERENT priority** (Part 5 #8 order: critic→PhaseMLP→
  no-idle→QoS→sum-rate)? → fix the root first; don't tune a downstream knob on a broken base.
Examples already in the Case-2 table (do NOT re-propose without a new reason):
  λ_D↑ (penalty-dom [I-2], r1) · grad_clip as the critic lever (r7, symptom-only) ·
  smaller critic arch (oracle) · spsa↑ alone to unfreeze λ (r11, STRUCTURAL not noise) ·
  PowerMLP init-bias too strong (r9, locks wc → starves phase).
State in the ACTION line that you checked the table (e.g. "not previously tried" or
"differs from rN because …"). This keeps the knowledge base from looping.

--------------------------------------------------------------------------------
## STEP 4 — Emit the assessment report  (ALWAYS use this template)
--------------------------------------------------------------------------------
```
RUN: result_N | Case <K,M> | R_LoS <x> | λ_D <..> | gae <..> | ep <done>/<target> | <running?/done>
RAMP: <finished? / step k of curriculum>

1. R_tot & QoS TREND: <improving | holding | collapsing | oscillating>
   QoS <early>→<late>% (best <..>% @ep<..>) · R_tot <..>→<..> · vs ceiling <..>%/All-IRS <..>%
   <one-line joint interpretation>

2. CRITIC HEALTH: <healthy | blind [B-1] | can't-fit [B-2] | unstable [B-2b]>
   explVar <..>(stable/wobble) · V̄ <..> · σV <..> · ‖∇‖V <..>(decreasing/plateau-large)
3. AGENT STATUS: <healthy | plateaued [D] | collapsing [A] | entropy-dom [C] | direct-attractor [J]>
   entropy-dom? <ent/pg ph ..> · common-stream wc<..>%/Ck<..> · L_pg <..>
   LONG-TERM (ramp unfinished?): <judgement>
   → ACTION: <keep / adjust <hyper> to <value> [code] / stop+checkpoint ep_<..> / advance ramp>

4. PER-HEAD: q=<ok/..> phase=<..> power=<..> ck=<..> · λ-quantum=<moving/frozen>
   đuối/borderline: <which + evidence, or "none">

   IRS-OPT (early ramps): <LEARNING TO OPTIMIZE IRS | IDLE: using-IRS-with-random-phases>
   evidence: phase entropy <early>→<late> (max <..>, <decreasing/flat>) · irs=[..] trend · Ck

BOTTOM LINE: <1-2 sentences: what's happening + the single most important next step>
```

--------------------------------------------------------------------------------
## STEP 5 — Persist the findings  (report → Training-Case; new issue → Common-Knowledge)
--------------------------------------------------------------------------------
After emitting the report, RECORD it so the knowledge base compounds. Do this every
time unless the user only asked for a quick glance. Use Edit/Write to actually update
the files — don't just say you will.

A) **Write the run outcome into `docs/Training-Case-<N>.txt`** (match N by K,M in params/
   hyperparameters.json):
   - **★ Add/refresh the run's row in the "BẢNG TÓM TẮT RUN" quick-reference table** (top of
     file): the lever tried (what's NEW vs prior run, use "=" for carried-over hypers) +
     RESULT + LIMITATION columns. This is the table Step 0 / the DON'T-REPEAT guard reads —
     keeping it current is what prevents future hyper-loops. If a new core limitation emerged
     or one was fixed, update the "4 LIMITATION CỐT LÕI" list under the table too.
   - Append a one-line entry to its "NHẬT KÝ RUN" table in the SAME terse format as the
     existing rows, e.g.:
     `result_N  R_LoS<x> λ_D<..> gae<..> β<..> ←<resume>  → QoS<peak>%@ep<..> [sự cố/mã]`
   - If this run is a curriculum-ramp milestone, also fill the matching per-ramp section
     (resume source, λ_D used, Feasibility Servable%/All-IRS%, QoS peak @ep, best
     checkpoint by QoS, lesson / hyper changed).
   - **Idempotent**: if an entry for this result_N already exists, UPDATE it — don't add
     a duplicate.

B) **Update `docs/Common-Knowledge.txt` ONLY if something NEW & GENERAL emerged** (true for any
   case, not a per-case number):
   - A failure mode not covered by codes `[A]`–`[I]` → add the next code `[J]`/`[K]`… to
     Part 3 (symptom → cause → fix), and if it has a tell-tale signal, a row to Part 4.
   - A new general signal/threshold or invariant → add to Part 4 / Part 5.
   - Do NOT put case/ramp-specific hyper values here (those belong in Training-Case-<N>),
     and do NOT duplicate an existing code — extend the existing one instead.
   - Keep the terse, case-agnostic style; bump the "Cập nhật lần cuối" date.

C) If a known problem `[A]`–`[I]` was implicated, hand off to the
   **irs-rsma-training-playbook** skill for the fix detail, and record the chosen fix +
   its result back in the Training-Case run log (so the next reader sees what worked).
