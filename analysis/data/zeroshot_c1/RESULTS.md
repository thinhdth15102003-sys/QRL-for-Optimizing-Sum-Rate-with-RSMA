# Zero-shot environment sweep, Case 1 — 2026-08-08

Not yet in the paper. Recorded so the numbers are traceable if they go in.

## What makes these zero-shot

Every axis below leaves all four networks' parameter shapes untouched, so the
trained policies run as-is:

| head | input dim | depends on N? |
|---|---|---|
| QuantumActor | `K(M+2)+2M` = 17 | no (affinities sum over N) |
| PhaseMLP | `2K+n_latent+K` = 41 **per element**, `d_out = n_levels` | no — shared per-element weights |
| PowerMLP | `K` = 5 | no |
| CkMLP | `3K` = 15 | no |
| Critic | 17 | no |

⚠ **`PhaseMLP.N` must be retargeted at eval time.** The weights are N-agnostic but the
loaded instance keeps its training N and `forward` loops `range(self.N)`. Without the
retarget, `N_eval > N_train` **silently leaves the extra elements at their previous
phase** — plausible numbers for a policy that only steered part of the surface.
Patched in `analysis/probe_main_table.py` (`ph.N = cfg.N`) and `infer.py`
(`run_hqchac_episode`, covers `probe_gen_sweep_one`). Any earlier N-sweep result
predating 2026-08-08 is suspect.

Also patched: `probe_main_table --shift` parsed every value as float, so `N=12`
became `12.0` and numpy rejected it as an array size. Integer literals now stay int.

## Axes screened and dropped

`beta_IRS`, `path_loss_exp` — collinear with `N` (all three scale the IRS branch only);
sweeping them adds no independent information. `beta_blocking` — inert (VQC−AO moved
+0.601 → +0.489 across a 10× change). `user_speed` — excluded by the user, no Doppler
in the model.

## Protocol

`probe_main_table.py --K 5 --M 1 --steps 200 --env-seeds 42,43,44` (600 states/cell),
5 training seeds per learned method, checkpoints = the QoS≥90% fine-grid picks that
Tables VI/IX report:
- VQC: r262/ep_06800 · r260/ep_09000 · r265/ep_04500 · r266/ep_04200 · r267/ep_06800
- DNN: r234/ep_07400 · r238/ep_12100 · r239/ep_19500 · r240/ep_09500 · r294/ep_17400

## Results (R_tot / QoS; AO R_tot only)

| shift | VQC | DNN | AO | VQC−AO |
|---|---|---|---|---|
| **base** (N=24, rain=4, d_block=0.05) | 2.4766 / 92.8% | 2.4941 / 94.7% | 1.8757 | +0.601 |
| N=12 | 2.2590 / 93.9% | 2.3088 / 94.3% | 1.8119 | +0.447 |
| N=36 | 2.5455 / 93.0% | 2.5490 / 95.1% | 1.8486 | +0.697 |
| N=48 | **2.6094** / 91.9% | 2.5855 / 94.0% | 1.8506 | **+0.759** |
| rain=0 dB | 2.7461 / 93.3% | 2.7565 / 95.2% | 1.9485 | +0.798 |
| rain=8 dB | 1.9149 / 89.1% | 1.9494 / 90.6% | 1.6637 | +0.251 |
| rain=12 dB | 1.1202 / 76.8% | 1.1584 / 76.8% | **1.2642** | **−0.144** |
| d_block=0.025 | 2.4108 / 93.2% | 2.4315 / 94.5% | 1.8828 | +0.528 |
| d_block=0.075 | 1.5822 / **89.9%** | 1.6501 / 86.3% | 1.6359 | −0.054 |
| d_block=0.10 | 1.3252 / **83.8%** | 1.3920 / 78.2% | 1.4843 | −0.159 |

## Three findings

1. **Array size: the learned policies scale, AO does not.** N=12→48 gives VQC +15.5%,
   DNN +12.0%, AO +2.1%; the margin over AO widens monotonically +0.447 → +0.759.
   Mechanism: the per-element rule applies at O(N) whatever N is, while AO at its
   published `phase_sweeps=1` covers a shrinking fraction of a growing search space.
   N=48 is also the ONLY point in the sweep where VQC beats DNN on R_tot (+0.9%,
   probably inside seed noise).

2. **Rain and blockage density both cross over.** AO overtakes at roughly rain ≈ 10 dB
   and d_block ≈ 0.073. Same mechanism for both: AO re-solves per state and adapts to
   the shifted channel; fixed weights do not. The crossovers sit at large perturbations
   (3× the nominal rain, 2× the nominal building width), so the honest reading is
   "robust over roughly a 2× perturbation, degrades past it".
   ⚠ A 1-seed / 60-step screen had put the rain gap at −0.539; the 5-seed number is
   **−0.144**. Use the 5-seed number.

3. **The VQC's QoS lean is conditional, not general.** At base, DNN leads QoS
   (94.7% vs 92.8%). Under blockage stress the order reverses: VQC +3.6 pp at
   d_block 0.075 and **+5.6 pp** at 0.10. The paper's current claim that the VQC leans
   toward QoS holds under stress but not at the nominal operating point — it needs the
   qualifier.

## Why rain is legitimately zero-shot

The paper justifies retraining for `P_S ≠ 50` with "transmit power is a deployment
parameter rather than a runtime environmental variable". Weather is not deployed, so
rain is a genuine runtime shift by that same criterion and the degradation there cannot
be waved off as "needs retraining". `N`, by contrast, IS a deployment parameter — the
defensible claim for the N axis is architectural transferability of the per-element
phase rule across array sizes, not runtime robustness.
