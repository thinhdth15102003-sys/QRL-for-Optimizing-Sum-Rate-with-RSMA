---
name: irs-rsma-training-playbook
description: >-
  Troubleshooting + hyperparameter reference for training the IRS-assisted RSMA
  Quantum-RL agent in THIS repo (train.py / params.py / RL/). Use this skill
  WHENEVER the user reports or asks about RL training trouble here, even if they
  don't name it precisely: QoS/reward collapse after an early peak, a training
  plateau (reward flat / best unbeaten / L_pg≈0), the "ENTROPY-DOMINATED" log flag,
  a blind or mis-scaled critic (explVar≈0, V̄ wrong scale), frozen quantum encoding
  scales (λ|max| stuck near init 0.50/0.45), a random or struggling PhaseMLP,
  GPU/CLI hangs (training_log frozen but process alive at high CPU), suspiciously
  low-reward episodes, or "is this physical-ceiling or agent variance?". Also
  trigger for: "how do I stabilize training", "which hyperparameters should I
  change", "is it stable enough to raise R_LoS / advance the curriculum", checking
  a result_N run's health, scaling to a new Case (K=10/M=2 etc.), or setting up a
  new training run. Knowledge is split across docs/Common-Knowledge.txt (case-agnostic)
  and docs/Training-Case-<N>.txt (per-(K,M)-case hyperparameters) at the project root —
  read both, AND keep them up to date after resolving issues.
---

# IRS-RSMA Training Playbook

A two-tier knowledge base for diagnosing and fixing training issues, plus the proven
hyperparameter settings per case. Distilled from Case 1 (K=5,M=1, runs result_1→11).

## Knowledge layout (project root)

- **docs/Common-Knowledge.txt** — case-AGNOSTIC: model mechanisms, the symptom→cause→fix
  table `[A]`–`[I]`, the diagnostic-signal table, and core invariants. True for any
  (K,M) / R_LoS.
- **docs/Training-Case-1.txt** — K=5, M=1: hyperparameters per R_LoS ramp (0.2→0.5) + the
  actual result_1→11 history and per-ramp lessons.
- **docs/Training-Case-2.txt** — K=10, M=2: setup + watch-outs (per-ramp table to fill in).
- **params.py** defines Case 1/2/3 = (K=5,M=1)/(K=10,M=2)/(K=20,M=4).

## How to use this skill (diagnose → fix → record)

1. **Identify the current case** from `params.py` ACTIVE CASE (K, M) → pick the
   matching `docs/Training-Case-<N>.txt`.
2. **Read both layers**: `docs/Common-Knowledge.txt` for the mechanism + general fix, and
   the matching `docs/Training-Case-<N>.txt` for case/ramp-specific hyperparameter context
   (what was set, what worked, what to watch).
3. **Diagnose from the log**: read the `diag[...]` blocks, the `rolling-50` line, and
   the `behaviour` line in `results/result_N/training_log.txt`. Map symptoms via the
   cheat-sheet below to a `[X]` code, then apply the fix.
4. **RECORD what you learn** (this is essential — the playbook must compound):
   - New insight that holds for ANY case → update **docs/Common-Knowledge.txt** (Part 3/4/5).
   - Hyperparameter/ramp lesson specific to a case → update that **docs/Training-Case-<N>.txt**
     (the per-ramp section + the run log at the bottom).
   - When a new run finishes, append it to the case's run-log table.

## Symptom → fix index (full detail in docs/Common-Knowledge.txt Part 3)

- `[A]` **Collapse** (QoS peaks then falls; wc→~24%, Ck→~0.46) → `gae_lambda=0.9`.
- `[B]` **Blind critic** (explVar≈0 / V̄ wrong scale after resume) → critic warm-up + load saved critic + `critic_target_tau=0.02`.
- `[C]` **Entropy-dominated** (⚠ flag, ent/pg phase>1) → lower `beta_entropy`(→0.001) + faster anneal.
- `[D]` **Plateau** (reward flat, L_pg≈0) → raise `lambda_D` + bigger rollout (+ gae>0).
- `[E]` **Frozen quantum λ** (λ|max| stuck at init) → raise `spsa_n_reps`(→8), maybe `lr_actor_qc`.
- `[F]` **PhaseMLP random/weak** → confirm via greedy eval; if greedy≫sampled, sharpen, else flat landscape (leave it).
- `[G]` **Low-reward episodes** (physical vs agent-variance) → check Blk/K; report greedy eval not rollout.
- `[H]` **GPU/CLI hang** (log frozen, process alive) → deadlock; `kill -9` both, `nvidia-smi`, resume. One GPU job; stop with Ctrl+C.
- `[I]` **λ_D wrong at new R_LoS** → check feasibility "Servable %"; if <~95%, lower `lambda_D`.

## Diagnostic cheat-sheet (healthy → alarm → code)

- `explVar` >0.5 · ≈0 → `[B]`  |  `V̄` ≈return scale · ≈±1 when returns large → `[B]`
- `ent/pg phase` <~0.3 · >1 → `[C]`  |  `L_pg` ≈0 + flat reward → `[D]`
- `λ|max|` moves off 0.50/0.45 · stuck → `[E]`  |  `wc`/`Ck tot` → ~24%/~0.46 → `[A]`
- `QoS/K` rising to ceiling · falling → `[A]`/`[C]`

## Core invariants (see docs/Common-Knowledge.txt Part 5)

gae=0.9 + critic warm-up + low β/fast-anneal + tau=0.02 from ep 1 · R_LoS curriculum
0.2→0.5 (stabilize before advancing) · no cross-Case weight transfer · don't deepen
MLPs broadly · re-tune λ_D per ramp via feasibility probe · one GPU job, stop with
Ctrl+C · pick best model by QoS from `checkpoints/` (not `best/`).
