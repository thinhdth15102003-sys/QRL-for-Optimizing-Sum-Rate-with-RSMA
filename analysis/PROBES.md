# Probe registry

Index of `analysis/`, so a probe that already exists is not written a second
time. **Check here before creating any new probe.**

Each entry: what question it answers, and — where one exists — the result it last
measured, so a number can be looked up instead of re-run.

Runs are referenced as `rNNN` = `results/result_NNN`.

---

## Conventions every probe here must follow

These are not style rules; each one was learned by getting a wrong answer first.

1. **Compare on `J`, never on `R_tot` alone.** `J = ΣR − λ_D·Σ(shortfall/(D_k+ε))²`.
   Sum-rate at unequal QoS is meaningless.
2. **Decide on ĝ, score on `g`.** `RateComputer` defaults to `use_true=False`
   (the *estimated* channel). Searching and scoring on the same view lets a
   solution exploit estimation error it will not have at run time — that is how
   the phase and power teachers came to prefer targets that lose. Note that
   `compute_rates_partial` has no `use_true`: C_k is a design-time choice from ĝ
   in both views, and only the achieved sum rate is paid on `g`.
3. **Scouts are one seed, controls are four.** Report `z` against the control
   *band*, and read the **trend across windows** — one `z` is not a verdict
   (r270 read −1.64 at ep 4400 and −0.20 at convergence).
4. **End a run analysis with the gap split**: `phase N% · power N% · routing N%`
   plus the distance to AO. `probe_gap_split.py` produces it.
5. **Median + IQR for timing**, never mean: one scheduler preemption ruins a mean.

---

## Written 2026-08-01 (this cleanup)

| probe | question | last measured |
|---|---|---|
| `probe_gap_split.py` | Where is a run losing, and how far from AO? Swaps each of assignment / phase / power to its oracle, one at a time, on paired states. | r275, 3 env seeds: **AO Δ +0.033 ± 0.024 J** (parity, slight lean). Gap split **assign 0% · phase 100% · power 0%**. ⚠ oracle-assign (−0.203) and oracle-power (−0.121) are *worse than the policy* — ĝ-selected oracles lose on `g`. |
| `probe_phase_anatomy.py` | Is the phase gap a bad TARGET or bad TRACKING? Scores head / teacher-target / J-oracle and per-element agreement. | Phase gap 0.097 J = **0.036 tracking + 0.061 target**. Head agrees with the teacher at **chance** (25.2% vs 25.0% for 4 levels) after warm-up reached 65.7%. |
| `probe_power_headroom.py` | Is there room left on the power axis? Holds assignment+phase at the policy's choice and varies only power. | ⚠ Numbers from this probe were **inflated** by an uncapped β grid; `probe_gap_split.py` supersedes it. Keep for the AO-scheme comparison (AO allocates private power *equally* — it optimises only the split). |
| `probe_dirichlet_commit.py` | Does the head *intend* a concentrated allocation, or only fail to execute one? Splits the raw net output from the fairness blend. | Raw intent **1.37×** (r232) / **1.99×** (r275) dynamic range. The fairness blend flattens by only 33–38%, so `--power-fairness-priv 0` is not a lever. |
| `probe_ao_saturate.py` | Where does AO stop improving, and what does that setting cost? Sweeps `rounds × phase_sweeps × n_split`. | **`rounds=1, phase_sweeps=1, n_split=11`** is the plateau, J 1.6427 at 76.7 ms. `n_split` is inert (11=21=41). `phase_sweeps` is the only knob that matters. **Publish AO here** — deeper settings buy no J and inflate our latency advantage. |
| `probe_ab_band.py` | Is a single-seed scout outside the control seeds' own band? Reports z + the trend across windows. | Replaces `scratchpad/pwfeat_ab.py` and `scratchpad/final_ab.py`. |
| `probe_latent_rank.py` | Does the 2·nq latent starve the assignment head? PCA rank of the input vs task accuracy. | Input effective rank **7** (PCA-8 = 99.97% var); latent rank 10; readout rank 23/26. ⚠ **Its routing target is degenerate** (69.8% majority, changes in 20/499 steps) so the accuracy columns say nothing — only the PCA spectrum is usable. Needs a harder target before reuse. |
| `probe_decision_latency.py` | Wall-clock and rate-kernel evaluations for one complete four-head action. | Idle machine, 1 thread, AO at its plateau: **AO 26.9 / 56.6 / 107.4 ms**, DNN 0.656 / 0.902 / 1.036, VQC 2.238 / 2.877 / — (C1/C2/C3). Kernel evals **482 / 896 / 1496 for AO vs 3 for the policy, constant in K**. |
| `probe_pick_ckpt.py` | Which checkpoint of ONE run is best, by the ENV's own reward (greedy, the policy's **own** C_k, env noise draws) — not by the cross-method J. | ⛔ **Use this, not `probe_vs_ao`, to choose a checkpoint.** `probe_vs_ao` substitutes the oracle C_k for every arm (correct for comparing *methods*, since AO is locked to that rule) and that **erases the C_k head's own improvement**, which made r261 look plateaued from ep 6000. Measured on its own C_k, r261 rises monotonically to ep 10200 (reward 289.7 → 329.5, R_tot 1.501 → 1.724, QoS 97.8% → 93.2%). λ-crossover vs the ep-4000 model is at **λ≈10**, not λ≈3 as the oracle-C_k reading suggested. |
| `probe_split_frac.py` | What private/common split `f` does AO pick, what does the policy execute, and what would the policy gain by moving? Adds `--oracle-phase` to test the same sweep under a J-optimal phase. | C3 r299: AO picks **f\*=0.05** (IQR [0.05,0.05]), policy executes **0.73**. But on the policy's OWN routing/phase the best f is **0.50** worth only **+0.018 J**, and f=0.05 *collapses* it to 0.779. ⇒ **the split is NOT a C3 lever**; f's optimum is set by the phase/grouping above it (same f=0.05 gives 1.617 under AO's config, 0.779 under the policy's). ⚠ Also shows the axes interact strongly, so one-at-a-time gap attribution understates the joint effect. |
| `probe_flat_vs_hier.py` | Is the flat-PPO baseline really better, or just at a different point on the rate/QoS frontier? Rate+QoS side by side, the `R_k/D_k` distribution, a λ_D sweep, and the equal-QoS subset. | C1, r301 vs r238: flat **R_tot 2.9126 @ QoS 77.2%** vs hier **2.5578 @ 99.3%**. **NOT Pareto** — flat serves more users in **0.0%** of states, fewer in 96.7%. Ranking flips at **λ_D ≈ 9.6** (flat wins at 1.5/3/5, hier wins at 10/20). flat leaves **18.6% of users at 0.5–0.9·D_k** ⇒ real under-service, not the near-threshold shaving seen in our own runs. |
| `probe_shave_vs_starve.py` | Two policies, **one** environment (`--env-run`), the policy's **own** C_k. Built for zero-shot-vs-retrained comparisons that `probe_flat_vs_hier.py` cannot do, because that probe takes each arm's own cfg and substitutes the oracle C_k. Reports the `R_k/D_k` band histogram, `med\|fail` / `p05\|fail` / `min`, `mean\|serv`, and a λ_D sweep. | C2 at $P_S=70$, 200 states × 3 seeds: retraining raises R_tot $1.84\to2.24$ and drops QoS $91.9\to86.3\%$, but shortfall **depth is unchanged** (`med\|fail` $0.905$ vs $0.910$) and the retrained tail is **shallower** (`p05\|fail` $0.561\to0.706$, `min` $0.034\to0.420$). The zero-shot arm is the only one that starves anyone. Ranking never flips out to λ=20. |
| `probe_link_budget.py` | Where does the sum-rate physically come from, and how big is the gap to Tan *in dB* rather than in %? Per-branch SNR + the SINR implied by a given `R_tot`. | Case 1: direct median **−5.67 dB** (capacity 0.346 bps/Hz ≈ AO's per-user 0.367); IRS opt-phase median +25.4 dB but only **60% of users** have IRS > direct. Our AO 1.8366 vs Tan 1.52 = **+20.8% rate but only +0.92 dB SINR**. Used to correct `docs/Dk-Noise-Calibration.md`: Tan's missing `g_RU` path loss is **median −17.4 dB**, so dropping it makes *their* branch stronger — that bullet had the sign backwards. |

---

## Pre-existing, by topic

**Assignment / routing** — `probe_assignment_oracle.py` (⭐ the coordinate-ascent
oracle; VQC's gap at ep 10000 was **+0.274** vs DNN's +0.435/+0.509),
`probe_assignment_decomp.py`, `probe_assignment_distill.py`,
`probe_assignment_enumerate.py`, `probe_assignment_quality.py`,
`probe_assignment_scatter.py`, `probe_assignment_sensitivity.py`,
`probe_qos_by_assignment.py`, `probe_overassign_cf.py`.

**Phase** — `phase_oracle.py` (the oracle itself; `oracle_phase_idx(n_sweeps=)`
and `oracle_phase_search`), `probe_phase_quality.py`,
`probe_phase_supervised_ceiling.py`, `phase_b_feasibility.py`.

**Power / C_k** — `oracle_alloc.py` (`oracle_ck_met`, `phi_from_idx`,
`_est_channels`), `probe_power_qos.py`, `probe_wc_optimality.py`,
`probe_oracle_alloc.py`, `probe_feasibility_at_P.py`.

**Baselines** — `probe_ao_rate.py` (⭐ the AO policy class), `probe_ao_baseline.py`,
`probe_tan_sa.py`, `probe_main_table.py`, `probe_gen_sweep.py`,
`probe_gen_sweep_ao.py`.

**Pareto / ceilings** — `probe_v_star.py`, `probe_true_frontier.py`,
`probe_pareto_frontier.py`, `probe_pareto_figure.py`, `probe_pareto_plot.py`,
`probe_pareto_agent.py`, `probe_realistic_ceiling.py`, `verify_vstar_formulas.py`.

**IRS capacity / geometry** — `probe_irs_group_capacity.py`, `probe_bstar_hyper.py`,
`probe_bstar_confirm.py`, `probe_group_mix_threshold.py`,
`probe_variable_b_seats.py`, `probe_irs_vs_direct.py`, `probe_irs_overload.py`,
`probe_irs_overassign.py`, `probe_irs_attribution.py`,
`probe_blocked_direct.py`, `probe_blocked_direct_quality.py`,
`probe_blocked_spawn_balance.py`, `probe_crossover_blocked.py`,
`probe_orderstat_marginal.py`.

**Critic** — `probe_critic_ceiling.py`, `probe_critic_repr_ablation.py`,
`probe_critic_temp_sweep.py`, `train_oracle_critic.py`, `sweep_oracle_critic.py`,
`analyze_critic_run.py`.

**Representation / architecture** — `probe_representation_stability.py`,
`probe_uenc_ablation.py`, `probe_param_count.py`, `probe_inference_latency.py`
(assignment head only — `probe_decision_latency.py` covers the full action).

**λ / D_k** — `probe_lambda_landscape.py`, `probe_lam_interference.py`,
`d_k_hypothesis_probe.py`, `extract_lambda_yz.py`,
`parse_lambda_trajectory.py`, `plot_lambda_yz.py`.

**Housekeeping** — `compare_runs.py`, `auto_analyze.py`, `pick_resume_ckpt.py`,
`probe_sample_eff.py`, `plot_case1_curriculum.py`.

---

## Findings kept, code not promoted

These answered their question once; the answer is recorded here and the script
stays in `scratchpad/`.

- `pwfeat_why.py` — is `--power-scale-feats` hurting via tanh saturation or
  LayerNorm dilution? **Both refuted**: saturation 69.3% vs 67.9% (C1 vs C2,
  essentially equal), ranking retention unchanged (91.2→91.2, 95.2→95.0).
- `power_representable.py` — can the head express the teacher target? Fitting
  the target's log-shares: current input **R² 0.245**, `+|h|²` **0.601**,
  `+log|h|` 0.559. ⚠ Its MLP column diverged (negative R²) — only the linear
  column is usable.
- `ln_dilution.py`, `ln_check.py` — the mandatory LayerNorm check before adding
  anything to the power state. Raw `|h|²` cut the h-block std 19%, raw `log|h|²`
  49%; standardised per state both land at +1.5% and ranking retention *rises*
  (95.7→96.7%).
- `opcount.py` — superseded by `probe_decision_latency.py`, which counts kernel
  evaluations as part of its normal output.
- `gap_decomp2.py` — the older gap tool. **Superseded** by `probe_gap_split.py`,
  which fixes the ĝ/g split; treat its numbers as pre-correction.

---

## Launchers

`scratchpad/launch_*.sh`, one per experiment, each carrying the measurement that
motivated it. All follow the same shape: `cp params_master.py params.py` + `sed`
(params.py is global and read at import), a guard that refuses to start a
duplicate, and `cp params_master.py params.py` again to restore.

⚠ Run them with PowerShell `Start-Process -WindowStyle Hidden wsl.exe -ArgumentList
"-e","bash","<script>"` — `nohup` inside `wsl -e bash -c` is killed when the tool
call returns.

⚠ `pgrep -f "python3.11 train.py"` **misses** runs started with `-u`. Use
`train\.py`.

`launch_vqc_nq.sh <nq>` is parameterised and self-verifying: it aborts if
`params.py` did not come out as intended rather than training the wrong
architecture for hours.
