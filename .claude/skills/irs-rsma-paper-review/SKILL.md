---
name: irs-rsma-paper-review
description: >-
  Section-by-section REVIEW & LOCK protocol for the HQC-HAC research paper in this repo
  (Research Paper/HQC-HAC...IRS-Assisted.tex). Use this skill WHENEVER the user is reviewing,
  tightening, proof-reading, or "locking" any part of the paper — e.g. "review this section",
  "rà/siết phần X", "check the abstract/intro/system model/method/results", "lock section",
  "check citations", "does this claim need a cite", "is this overclaiming", "fix the writing
  style", "is the paper consistent with the code". It runs five checks on the passage:
  (1) CITATION integrity — every \cite exists in reference.bib and actually supports the claim,
  no fake/mismatched cites; (2) MISSING-CITATION flag — mark assertions/opinions that should be
  backed by a reference and suggest (the user finds the actual cite); (3) WRITING STYLE — academic,
  clear, concise, no informal or verbose wording; (4) CLAIM STRENGTH — no overclaim, keep to soft
  claims (observed ≥0.5 → soften; empirical → "Observation" not "Proposition"); (5) PAPER↔CODE
  consistency — the passage must match reality (the code in params.py / RL/ / CSI/ / istn/), cross-
  referenced against the known-error list in Research Paper/PAPER-SYNC-DIFFS.md. Emits a per-section
  punch-list the user can act on. Reality (code) is ground truth; the paper is fixed to match it.
---

# HQC-HAC Paper Review & Lock Protocol

A repeatable, section-scoped review of `Research Paper/HQC-HAC for Optimizing Sum-Rate with IRS-Assisted.tex`.
Run the five checks below on the passage the user names, then emit the punch-list template. Do NOT
rewrite silently — surface findings so the user tightens and locks each part deliberately.

## ⛔ GROUND RULE 1 — reality (code) is the source of truth
The **code is the ground truth**; the paper is corrected to match it (paper follows code, code follows
reality). For every factual/architectural claim, verify against the actual implementation, not against
what the paper "should" say:
- Hyperparameters → `params.py`
- Actor / VQC / AE / critic / SPSA → `RL/quantum_actor.py`, `RL/quantum_circuit.py`, `RL/critic.py`
- Channel / SINR / rate / reward / CSI split → `CSI/rate.py`, `CSI/env.py`, `istn/channel.py`
- Archived run configs / numbers → `results/result_N/agents/actor_config.json`, `hyperparameters.json`
- **Known open mismatches → `Research Paper/PAPER-SYNC-DIFFS.md`** (read FIRST; if the passage touches
  an item there, confirm the user has applied that diff before re-flagging — don't duplicate).

## ⛔ GROUND RULE 2 — numbers are DEFERRED; never block on quantitative data
The user fills and verifies ALL quantitative data (result numbers, table-cell values, reported
percentages/ratios/gains, `---` placeholders) in a SEPARATE later pass. During this review:
- Do **NOT** flag a missing / placeholder / unfilled / unverified numeric value as an issue, and do
  **NOT** try to re-derive or cross-check the numbers themselves against logs.
- Check 5 still verifies **structural** claims — architecture, formulas, symbols, which
  channel/readout/head/estimator, and parameter-count **formulas** (not the arithmetic result) —
  against code. Only the numeric VALUES are out of scope here.
- Checks 1–4 (citations, missing-cite, style, claim-strength) apply to the prose as normal, regardless
  of numbers.
- If a number is load-bearing for a claim's meaning, note it once as `deferred-number` (non-blocking)
  so the user remembers to verify it later — never as a FIX-THEN-LOCK blocker.

## Contribution framing (memory — every claim must map to one, and must NOT exceed it)
- **C1** sum-rate max, multi-IRS grouped RSMA/ISTN, 4-var sequential chain under QoS/discrete-phase/blockage/imperfect-CSI.
- **C2** HQC-HAC: VQC actor for assignment `(M+1)^K` + 3 classical sub-actors + entropy-anneal.
- **C3** AE-VQC: AE compresses state→latent→VQC (NISQ register).
- **Framing = FEASIBILITY + PARAM-EFFICIENCY + AMORTIZED-INFERENCE** (vs Tan 2026 solve-per-state).
  **NOT** a claim of quantum task-advantage. **Honest prior: PQC ≈ DNN** in task performance (Jerbi 2021).
  Any sentence implying quantum superiority in achievable objective is an OVERCLAIM — flag it.

--------------------------------------------------------------------------------
## STEP 0 — Scope & context
--------------------------------------------------------------------------------
- Confirm which section/paragraph/equation/table the user wants reviewed. Review THAT scope only
  (the user works section-by-section and locks incrementally).
- Read `Research Paper/PAPER-SYNC-DIFFS.md` and note any listed item that falls in this scope, so
  checks ①–⑤ don't re-report an already-known/queued fix (say "already in sync-list #X" instead).
- Pull the passage from the `.tex`. Identify: the claims made, the \cite keys used, the equations/
  numbers stated, and which code file is the ground truth for each factual assertion.

--------------------------------------------------------------------------------
## CHECK 1 — Citation integrity (no fake / no mismatched cite)
--------------------------------------------------------------------------------
For every `\cite{key}` in the passage:
- **Exists?** `key` must be a real entry in `Research Paper/reference.bib`. List any key with no bib
  entry (a dangling/fake cite). Quick audit of the whole file when useful:
  `grep -oE "\\cite\{[^}]+\}" <tex> | grep -oE "\{[^}]+\}" | tr -d '{}' | tr ',' '\n' | sort -u`
  vs `grep -oE "^@[a-z]+\{[^,]+" reference.bib` — every used key must appear as a bib key.
- **Appropriate?** Read the bib entry (title/venue/year/authors) and judge whether it plausibly
  supports the sentence's claim. Flag mismatches: wrong topic, wrong result attributed, a survey cited
  for a specific numeric claim, a method cited for a claim it doesn't make, year/venue that looks off.
- **Verifiable?** You cannot read the cited PDF — so for any cite whose SUPPORT you cannot confirm from
  the bib metadata + the paper's own framing, mark it `⚠ VERIFY` for the user to check against the
  source, rather than asserting it is fine. Never invent or "fix" a cite key.
- **Placement.** The cite should sit on the specific clause it backs, not floated at paragraph end when
  several distinct claims each need their own source.

Output: per-cite → {OK | ⚠ VERIFY (why) | ✗ DANGLING (no bib) | ✗ MISMATCH (claim vs source)}.

--------------------------------------------------------------------------------
## CHECK 2 — Missing-citation flag (suggest; user finds the cite)
--------------------------------------------------------------------------------
Mark sentences that assert something needing external support but currently have NO cite. The user
will locate the actual reference — your job is only to FLAG and SUGGEST what kind of source is needed.
Needs a citation:
- A factual claim about the world / prior art / a standard (channel model, ITU-R, RSMA/SIC, IRS N²
  scaling, NISQ limits, a "well-known" result).
- A method/technique borrowed from literature (PPO, GAE, PopArt, SPSA, soft actor-critic, autoencoder,
  data re-uploading, SoftmaxPQC) — first mention should cite its origin.
- A comparative/positioning claim ("unlike prior work", "no existing study", "state of the art").
- A property asserted as generally true rather than shown in THIS paper's experiments.
Does NOT need a citation (do not over-flag):
- The paper's own definitions, derivations, its own experimental results/numbers, its own design
  choices, and statements clearly scoped to "in our system / here".

Output: per-flagged sentence → the claim + "suggest cite: <kind of source>" (e.g. "original PPO paper",
"an ITU-R propagation reference", "a barren-plateau reference"). Do not fabricate a specific citation.

--------------------------------------------------------------------------------
## CHECK 3 — Writing style (academic, clear, concise, no informal)
--------------------------------------------------------------------------------
Flag and propose a tightened rewrite for:
- **Informal / colloquial** wording, contractions, hype adjectives ("hugely", "amazing", "a lot of",
  "nowadays", "cutting-edge"), vague intensifiers.
- **Verbosity / redundancy**: padding phrases ("it is important to note that", "in order to", "due to
  the fact that"), repeated ideas, sentences that restate the previous one. Prefer the shortest precise
  form. (Memory PROOF-DISCIPLINE: result → formula, few words, clean-result + link; no long forward
  derivations.)
- **Ambiguity / imprecision**: undefined symbols/acronyms at first use, dangling "this/it", inconsistent
  notation (a symbol used two ways), tense drift.
- **Structure**: run-on sentences; a paragraph with no topic sentence; a list that should be prose or
  vice-versa. Match the terse, Jerbi-2021-style used elsewhere (manual Lemma + 1-knob ablation +
  variance bands + bulleted observations).
Propose the concrete edit (OLD → NEW), keep it minimal, preserve meaning.

--------------------------------------------------------------------------------
## CHECK 4 — Claim strength (no overclaim; soft claim)
--------------------------------------------------------------------------------
Enforce the NO-OVERCLAIM rule (memory `feedback_paper_overclaim`):
- Claim only what the data/experiments in THIS paper support. A single-seed / single-config result must
  not be phrased as a general law.
- **Soften when the observed effect is ≥ 0.5** (or otherwise not negligible): replace "≪", "negligible",
  "eliminates", "guarantees", "proves", "always/never" with hedged forms ("small", "reduces", "tends to",
  "in our experiments", "suggests", "is consistent with").
- **Empirical finding → "Observation"**, NOT "Proposition/Theorem" (reserve Proposition/Lemma for
  actually-proved statements; the paper's Lemma 1/2 + Prop 1 are the only proved ones — check any new
  Proposition is genuinely a proof, else demote to Observation).
- **No quantum task-advantage claim.** The value is param-efficiency + amortized inference, and PQC≈DNN
  in task performance. Flag any sentence that reads as "quantum does better on the objective".
- Check superlatives ("first", "best", "optimal", "significantly") are literally justified; "optimal"
  should mean provably optimal, not "good".
Output: per-overclaim → the phrase + the softened version + the reason.

--------------------------------------------------------------------------------
## CHECK 5 — Paper ↔ code consistency (match reality)
--------------------------------------------------------------------------------
For every factual/architectural/numeric assertion in the passage, verify against the ground-truth code
(see GROUND RULE). Typical things to re-derive from source, not trust:
- Hyperparameter values / table entries → `params.py` (and archived `hyperparameters.json` for the run
  that produced a reported number).
- Architecture details: state/affinity content, AE branch dims, VQC gates/entangling/observables
  (`readout_mode` in `actor_config.json` = R1, not generic), head type (SoftmaxPQC), parameter counts,
  SPSA-of-Jacobian vs SPSA-of-loss, GAE/PopArt (no target net), CSI split (decide on ĝ, reward on true g).
- Formulas: SINR (`CSI/rate.py::_sinr_all`), reward (`CSI/env.py::step`), channel/path-loss/noise
  (`istn/channel.py`), CSI error model.
- Reported results numbers → the training log / infer output they came from; check the checkpoint,
  seed, eval-env (`⚙ eval env` line — see infer-env-fix), and that inference used the current pipeline.
Cross-check against `PAPER-SYNC-DIFFS.md`: if the discrepancy is already listed, cite the item number
instead of re-deriving; if it is NEW, add it to the punch-list AND propose appending it to the sync file.
Output: per-assertion → {matches code | ✗ mismatch (paper says X, code does Y) | already sync-list #N}.

--------------------------------------------------------------------------------
## STEP 6 — Emit the review punch-list (ALWAYS use this template)
--------------------------------------------------------------------------------
```
SECTION: <name / label / line range>   |   sync-list items in scope: <#… / none>

1. CITATIONS
   <key> : <OK | ⚠ VERIFY: why | ✗ DANGLING | ✗ MISMATCH: claim vs source>
   ...
2. MISSING CITES (suggest — user finds)
   "<sentence>" → suggest: <kind of source>
   ...  (or "none")
3. STYLE
   OLD: "<...>"  →  NEW: "<tightened>"   (<informal|verbose|ambiguous|structure>)
   ...  (or "clean")
4. CLAIM STRENGTH
   "<phrase>" → soften to "<...>"   (reason)   |   OVERCLAIM: <quantum-advantage / general-law / …>
   ...  (or "appropriately hedged")
5. PAPER↔CODE
   "<assertion>" : <matches | ✗ paper says X / code does Y (file:line) | sync-list #N>
   ...

LOCK VERDICT: <READY TO LOCK  |  FIX-THEN-LOCK: the N blocking items above>
```
- If nothing is wrong in a check, say so explicitly (don't pad).
- End with the ONE thing that most blocks locking this section, if any.

--------------------------------------------------------------------------------
## STEP 7 — Persist newly-found mismatches
--------------------------------------------------------------------------------
If Check 5 (or 1) surfaces a NEW paper↔code discrepancy not already in `PAPER-SYNC-DIFFS.md`, append it
there in the same OLD→NEW format (location, reality, diff) so the user's single edit pass stays complete.
Keep the file ordered by `.tex` line. Do not edit the `.tex` itself unless the user asks — this skill
produces the review; the user tightens and locks.
