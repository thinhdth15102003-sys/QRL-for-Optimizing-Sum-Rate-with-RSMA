# HQC-HAC

A hybrid quantum–classical hierarchical actor–critic that maximises downlink sum rate in an
IRS-assisted grouped-RSMA integrated satellite–terrestrial network, under per-user QoS
demands, quantised IRS phases, building blockage and imperfect CSI.

Four decisions are coupled at every scheduling interval: which reflecting surface serves
which user, how each surface is configured, how the transmit budget is split, and how each
group's common rate is shared. Together they form a mixed-integer non-convex problem with no
tractable closed form. The usual remedy is to decompose it and alternate between the
subproblems, which is priced **per channel realisation**: the solver keeps nothing between
invocations, so a link whose geometry follows user mobility pays the iterative cost again
every time the geometry moves.

This repository takes the opposite trade. A policy is trained once offline and thereafter
answers each realisation with a single forward pass. The claim being tested is **decision
cost** — amortised inference and parameter efficiency — not a quantum advantage on the
objective itself. On the task, the variational actor and the classical actor it replaces
land within a quarter of the spread across training seeds, which is what the prevailing
prior for parametrised quantum policies anticipates.

## Structure of the policy

The four variables form a strict sequential dependency. The assignment fixes the grouping
that governs the phase design; the phase design shapes the effective channels that enter the
power allocation; the resulting budget delimits the feasible common-rate split. That ordering
is what the hierarchy mirrors.

| stage | module | resolves |
|---|---|---|
| high level | variational quantum circuit | IRS–user assignment, a space of size $(M{+}1)^K$ |
| low level | `PhaseMLP` | per-element discrete phase shifts |
| low level | `PowerMLP` | private/common budget split across users |
| low level | `CkMLP` | common-rate shares within each group |

The discrete complexity concentrates in the assignment, which is where the circuit sits. Its
input would otherwise scale with both the user population and the panel count — a direct
angle encoding of the affinity state needs 17 qubits at $(K,M)=(5,1)$, 44 at $(10,2)$ and 81
at $K{=}15$ — so an autoencoder learns the latent embedding the assignment actually depends
on and the register stays at eight throughout.

The circuit is a **hand-written state-vector simulator in numpy**. There is no PennyLane,
Qiskit or JAX dependency. On an 8-qubit register the whole state is 256 amplitudes, small
enough that the CPU path outruns the GPU one by roughly an order of magnitude.

## Install

```bash
conda env create -f environment.yml
```

Then `conda activate IRS_QRL`. Dependencies are numpy, scipy and matplotlib. `cupy` is
optional and only reached when `QRL_GPU_MLP=1`.

## Selecting a network scale

There is **no CLI flag for $K$, $M$, the qubit count or the sub-actor widths**. They are
edited in `params.py`, at the block marked `ACTIVE CASE`, where only `K` and `M` are set by
hand and the per-case hyperparameters derive from them.

```python
K = 10     # Case 1: 5 | Case 2: 10 | Case 3: 15
M = 2      # Case 1: 1 | Case 2: 2  | Case 3: 3
```

`--resume` detects $K$ and $M$ from the checkpoint, so a resumed run does not need the file
edited to match.

> `params.py` currently defaults to `n_qubits = 12`, whereas every reported run used 8. The
> published checkpoints carry their own `actor_config.json`, so they load at the width they
> were trained at regardless of what `params.py` says. A **fresh** run reproducing the paper
> needs `n_qubits = 8` set in `params.py` first. There is no environment override; the
> `QRL_NQ` mentioned in the comments there was never implemented.

## Training

```bash
python train.py --episodes 20000 --seed 0
```

Output goes to `results/result_<N>/`, allocated at start-up, holding `training_log.txt`,
`hyperparameters.json`, periodic `checkpoints/ep_*/agents/` and a final `agents/`. A run that
has not finished has no top-level `agents/`; use `best/agents` or a numbered checkpoint.

The CPU path is the fast one:

```bash
CUDA_VISIBLE_DEVICES="" QRL_GPU_MLP=0 python train.py --episodes 20000 --seed 0
```

The flat baseline is a separate entry point:

```bash
python train_flat.py --episodes 20000 --seed 0 --lr 3e-4 --target-kl 0.05
```

### `train.py` flags

**Run control**

| flag | type | default | meaning |
|---|---|---|---|
| `--episodes` | int | 15000 | training episodes |
| `--steps` | int | 200 | environment steps per episode |
| `--gamma` | float | 0.95 | TD discount |
| `--seed` | int | 0 | global random seed |
| `--resume` | path | – | an `agents/` directory to continue from, or a run directory, in which case the best checkpoint is picked automatically |
| `--no-plots` | flag | off | skip matplotlib output |

**Environment overrides** — each overrides the `params.py` value for this run and is recorded in `hyperparameters.json`.

| flag | type | default | meaning |
|---|---|---|---|
| `--P-S` | float | – | satellite transmit power, dBm |
| `--D-k` | float | – | per-user QoS demand, bps/Hz |
| `--R-LoS` | float | – | satellite line-of-sight coverage radius, km |
| `--lambda-D` | float | – | fixed QoS penalty weight $\lambda_D$ |
| `--irs-spawn-frac` | float | – | IRS spawn region as a fraction of $R_{\mathrm{LoS}}$ |
| `--user-free-frac` | float | – | unblocked-user spawn region as a fraction of $R_{\mathrm{LoS}}$ |

**Architecture and ablation** — fresh runs only; on `--resume` the saved actor config wins.

| flag | type | default | meaning |
|---|---|---|---|
| `--actor-mode` | `quantum`/`classical` | `quantum` | which high-level actor to build. `classical` is the DNN arm of the comparison, identical in role, input and output |
| `--pol-hidden` | int… | – | policy-MLP widths for the classical arm, for the width sweep |
| `--readout` | `r1`/`single-z`/`nn-zz`/`full-zz` | `r1` | observable set. `r1` is the structured per-action readout; `single-z` drops every two-qubit term; the others are generic patterns |
| `--no-entangle` | flag | off | build the circuit with no two-qubit gates. Parameter count is unchanged, so this isolates whether the quantum correlations matter |
| `--no-ae` | flag | off | drop the latent bypass, so heads read the quantum readout alone |
| `--no-ae-keep-pretrain` | flag | off | with `--no-ae`, keep the encoder pre-training |
| `--no-pretrain` | flag | off | skip autoencoder hot-start |

**Auxiliary teachers** — each adds $w\cdot(-\log \pi(\text{oracle}\mid s))$ to one head's loss at every update, alongside PPO. Weight 0 disables.

| flag | type | default | meaning |
|---|---|---|---|
| `--assign-aux-weight` | float | 0.0 | assignment teacher. This is the one that prevents routing-identity collapse, and it deliberately has no anneal flag |
| `--phase-aux-weight` | float | 0.0 | phase teacher |
| `--power-aux-weight` | float | 0.0 | power teacher |
| `--ck-aux-weight` | float | 0.0 | common-rate teacher |
| `--phase-aux-anneal-end` | float | – | decay the phase teacher to 0 by this fraction of training |
| `--power-aux-anneal-end` | float | – | the same for the power teacher |
| `--phase-aux-hspread` | float | 0.0 | shift the phase teacher toward high channel-spread states without changing its total strength |
| `--phase-aux-sweeps` | int | 0 | coordinate-ascent sweeps used to build the phase target |
| `--power-aux-beta-cap` | flag | off | restrict the power teacher grid to $\beta \le 1$ |

**Supervised warm-up**

| flag | type | default | meaning |
|---|---|---|---|
| `--oracle-warmup` | flag | off | supervised warm-up of both the assignment head and the phase head before PPO |
| `--phase-warmup` | flag | off | phase head only |
| `--assign-warmup-episodes` | int | – | rollout episodes for the assignment warm-up |
| `--assign-warmup-epochs` | int | – | cross-entropy epochs for it |
| `--phase-warmup-episodes` | int | 8 | rollout episodes for the phase warm-up |
| `--phase-warmup-epochs` | int | 30 | cross-entropy epochs for it |
| `--phase-warmup-target` | float | – | train the phase warm-up until per-IRS match reaches this percentage |
| `--full-phase-warmup` | flag | off | disable early stopping and run the full epoch budget |
| `--freeze-phase` | flag | off | hold the phase head fixed so the assignment trains against a stationary phase |

**Power and common-rate shaping**

| flag | type | default | meaning |
|---|---|---|---|
| `--power-fairness` | float | 0.0 | blend the applied split toward equal, $\alpha\cdot\text{uniform} + (1-\alpha)\cdot\text{learned}$ |
| `--power-fairness-priv` | float | – | the same $\alpha$ for the private distribution only |
| `--power-priv-frac` | float | 0.8 | target private fraction used by that blend |
| `--power-scale-feats` | flag | off | append $2K{+}1$ absolute-scale features to the power head input |
| `--power-feat-scale` | float | 1.0 | divisor inside the feasibility-margin tanh of the above |
| `--power-mag-feats` | `sq`/`log` | – | append the per-user magnitude ordering |
| `--power-clean-input` | flag | off | feed the power head only the effective channels |
| `--closed-form-ck` | flag | off | replace the learned common-rate head with the exact demand-fill allocator |
| `--ck-logit-spread` | float | 8.0 | clamp on the within-group softmax logit spread, which stops it collapsing to one-hot |

**Credit assignment**

| flag | type | default | meaning |
|---|---|---|---|
| `--counterfactual-assign` | flag | off | COMA-style per-user counterfactual baseline for the assignment head |
| `--cf-coef` | float | 0.3 | its strength |
| `--cf-anneal-end` | float | 0.0 | decay it to 0 by this fraction of training |
| `--routing-shape` | flag | off | directed shaping toward the IRS-favoured link |
| `--routing-shape-coef` | float | 0.1 | its strength |
| `--phase-counterfactual` | flag | off | weight each phase gradient by expected occupancy, so phase credit tracks the routing distribution rather than the sampled activation |
| `--irs-bonus` | float | 0.0 | reward shaping bonus per IRS-routed user |

**Entropy**

| flag | type | default | meaning |
|---|---|---|---|
| `--beta-entropy` | float | – | initial entropy coefficient |
| `--beta-entropy-min` | float | – | floor after annealing |
| `--beta-entropy-anneal-end` | float | – | fraction of training over which it anneals |
| `--beta-entropy-pwr-private` | float | – | extra bonus on the private power axis alone |

**Constrained variant** — off by default; the reported runs use a fixed $\lambda_D$.

| flag | type | default | meaning |
|---|---|---|---|
| `--lagrangian` | flag | off | treat $\lambda_D$ as a multiplier and adapt it by dual ascent on a QoS target |
| `--qos-target` | float | 0.8 | that target |
| `--lambda-lr` | float | 0.05 | dual-ascent step per rollout |
| `--lambda-min` / `--lambda-max` | float | 1.0 / 3.0 | clipping range |

### `train_flat.py` flags

The flat baseline ports the single-network method of Meng et al. onto this environment. It
shares the rest of the training stack — GAE, entropy bonus, PopArt, Dirichlet heads, the
richer state — so what is being compared is the architecture rather than the surrounding
machinery.

| flag | type | default | meaning |
|---|---|---|---|
| `--episodes`, `--steps`, `--gamma`, `--seed` | | as `train.py` | |
| `--lr` | float | 3e-4 | actor learning rate. **Tuned**, not the source paper's 1e-4; use `--lr 1e-4` for their setting |
| `--lr-critic` | float | from `params.py` | critic learning rate |
| `--hidden` | str | `512,256` | trunk widths, comma-separated |
| `--ppo-epsilon` | float | Meng default | PPO clip |
| `--ppo-epochs` | int | from `params.py` | epochs per update |
| `--rollout-episodes` | int | from `params.py` | episodes per rollout |
| `--target-kl` | float | 0.05 | KL early stop. **Leave it on.** A joint density over roughly 250 action dimensions makes the ratio compound multiplicatively; without the guard about 98% of samples land clipped and have their gradient zeroed. 0 disables |
| `--power-fairness` | float | 0.0 | off by default, which is faithful to the source, where QoS is shaped purely through the reward |
| `--power-priv-frac` | float | 0.8 | as `train.py` |
| `--checkpoint-interval` | int | 1000 | episodes between checkpoints |
| `--D-k`, `--R-LoS-km` | float | – | environment overrides |

## Evaluation

`infer.py` takes a run directory or any checkpoint directory and evaluates the policy against
AO, greedy and random baselines.

```bash
python infer.py checkpoints/case2_K10_M2/hqc-hac-vqc/seed0 --episodes 3 --steps 200 --seed 42
```

The environment is read from the checkpoint rather than from `params.py`, so evaluation
matches training without editing anything. Check the `⚙ eval env:` line it prints.

| flag | type | default | meaning |
|---|---|---|---|
| `run_dir` | path | required | run or checkpoint directory |
| `--episodes` | int | 50 | evaluation episodes |
| `--steps` | int | 200 | steps per episode |
| `--seed` | int | 42 | evaluation seed |
| `--stochastic` | flag | off | sample instead of taking the greedy action |
| `--shots` | int | trained value | VQC measurement shots at deployment |

**Robustness overrides.** These perturb the environment after training and need no retraining.

| flag | type | meaning |
|---|---|---|
| `--kappa` | float | CSI error coefficient, training value 0.05 |
| `--noise-var` | float | receiver noise variance in dBW, training value 10.0 |
| `--R-LoS` | float | line-of-sight coverage radius |
| `--irs-spawn-frac` | float | IRS spawn region |
| `--user-free-frac` | float | unblocked-user spawn region |
| `--n-elements` | int | IRS elements per panel |
| `--p-s-dbm` | float | satellite transmit power |
| `--user-speed` | float | user mobility, training value 1.5 m/s |
| `--natural-spawn` | flag | disable balanced blocked spawning |

Every head is independent of the element count $N$, so sweeping `--n-elements` is a genuine
zero-shot test. Sweeping $K$, $M$ or the phase quantisation is not, and needs retraining.

## Published checkpoints

`checkpoints/` holds the exact weights behind every learned number in the paper, one
directory per case, method and training seed, with a `provenance.json` naming the run and
episode each came from. `checkpoints/README.md` has the full table and two caveats worth
reading. Critic weights are omitted because `infer.py` never loads them.

## Repository layout

| path | contents |
|---|---|
| `train.py` | hierarchical actor–critic training loop |
| `train_flat.py` | flat-PPO baseline |
| `infer.py` | evaluation against AO, greedy and random |
| `params.py` | active case and all derived hyperparameters |
| `CSI/` | environment, rate and SINR kernel, model-based baselines |
| `istn/` | channel model, path loss, blockage, CSI error |
| `RL/` | quantum actor, circuit, classical sub-actors, Dirichlet heads, flat actor |
| `analysis/` | roughly 50 probes indexed by `analysis/PROBES.md`, plus the figure scripts |
| `analysis/data/` | probe output and the launch provenance logs |
| `docs/` | per-case training logs and the symptom-to-fix notes |
| `Research Paper/` | manuscript source, figures, and `build_manuscript.py` |

Two things are deliberately absent. `results/` is the full run tree, roughly 20 MB tracked
and far more on disk, of which the reported slice is published under `checkpoints/`. The
Telegram training monitor is operational tooling with no bearing on the method.

## Reproducing a reported number

Read `analysis/data/TABLE-PROVENANCE.md` first. It names, for every table and figure in the
paper, the probe that produced it, the checkpoints it ran on, the environment and validation
seeds, and the caveats that apply. Checkpoints are selected by the rule in Section V-A:
highest greedy episode return subject to a 90% QoS floor, searched on validation environment
seeds and reported on a disjoint test set.

Two conventions matter when comparing against a baseline. Decisions are taken on the
estimated channel $\hat g$ and scored on the true $g$, and policies are compared on the
penalised objective $J = \sum_k R_k - \lambda_D \sum_k (\text{shortfall}_k / D_k)^2$ rather
than on sum rate alone, since a rate obtained by underserving users is not comparable with
one that meets the demand.

## Building the paper

```bash
python "Research Paper/build_manuscript.py"
```

One source carries both a reading version and the submission version behind an `\iffull`
switch. The script copies the source, flips the switch and the document class, compiles, and
reports pages, errors, undefined references and overfull boxes. `--full` builds the reading
version instead. It keeps the `.aux`, which is the only reliable record of what each version
actually typeset.
