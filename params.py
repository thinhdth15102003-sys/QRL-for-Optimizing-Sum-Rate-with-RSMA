"""
params.py
---------
Central tuning file — edit here to adjust the simulation without touching model code.

Usage
-----
    from params import make_config
    cfg = make_config()                  # all defaults
    cfg = make_config(K=15, kappa=0.0)   # override specific values

Training cases
--------------
    Change the four lines under "ACTIVE CASE" to switch scenarios.
    All dependent params (extra_cz/zz_pairs, d_aff, hidden dims) auto-update.

    Case 1 | K=5,  M=1, P=50 dBm, nq=8,  n_latent=16  (small  — fast debug)
    Case 2 | K=10, M=2, P=70 dBm, nq=12, n_latent=24  (medium — paper baseline)
    Case 3 | K=20, M=4, P=100dBm, nq=16, n_latent=32  (large  — full capacity)
"""

from istn.config import SystemConfig

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVE CASE  ←  change these four lines to switch training scenario
# ══════════════════════════════════════════════════════════════════════════════
K        = 10    # Case 1: 5  | Case 2: 10  | Case 3: 20
M        = 2     # Case 1: 1  | Case 2: 2   | Case 3: 4
P_S_dBm  = 70.0  # Case 1: 50 | Case 2: 70  | Case 3: 100
n_qubits = 12    # Case 1: 8  | Case 2: 12  | Case 3: 16
n_latent = 24    # Case 1: 16 | Case 2: 24  | Case 3: 32  (= 2 × n_qubits)
# ══════════════════════════════════════════════════════════════════════════════

# ── Network topology ────────────────────────────────────────────────────────────
N = 24    # reflection elements per IRS (unchanged across cases)

# ── Satellite RF  (ITU-R model) ─────────────────────────────────────────────────
# P_S_dBm defined above (case-dependent)
f_GHz         = 1.58   # carrier frequency (GHz)
h_SR_km       = 800.0  # satellite altitude (km)
G_S_dBi       = 60.0   # satellite TX antenna gain (dBi)
G_U_dBi       = 60.0   # user RX antenna gain (dBi)
noise_mean_dBW = -30.0  # 0 dBm = -30 dBW = 10^-3 W  (paper: n_0 ~ N(0,10) dBm)
noise_var_dBW  = 10.0   # noise variance in dB^2; sampled per step: n_0 ~ N(-30,10) dBW

# ── Rain attenuation  (lognormal, ITU-R) ────────────────────────────────────────
rain_mean   = 4.0   # μ (dB)
rain_var_SR = 0.1   # σ²  satellite → IRS link
rain_var_SU = 0.1   # σ²  satellite → user link
rain_var_RU = 0.1   # σ²  IRS → user link

# ── IRS-to-user propagation  (urban Rayleigh + path loss) ──────────────────────
g_sf_dB       = 10.0   # small-scale fading gain (dB)
path_loss_exp = 3.5    # urban path-loss exponent
d0_ref_km     = 0.01   # reference distance (10 m) for path-loss model

# ── Imperfect CSI (multiplicative error: Δg = κ|g|ε, ε~CN(0,1)) ────────────────
kappa = 0.05   # multiplicative CSI error coefficient κ; applies to g_SR, g_RU, g_SU

# ── Discrete phase-shift ────────────────────────────────────────────────────────
quantization_bits = 2   # 2-bit → 4 levels: {0, π/2, π, 3π/2}

# ── QoS constraint & demand-aware reward (quadratic penalty) ────────────────────
# QoS: R_private[k] + C_k[k] >= D_k for every user k
D_k_bps_hz = 0.10   # per-user QoS demand D_k (bps/Hz); penalised if R_tot < D_k
lambda_D   = 1.5    # penalty weight λ_D (QoS vs sum-rate knob). ↑ → agent keeps weak
                  # [Case 2 R_LoS=0.2: 3.0→1.5. result_1 stuck at QoS54% with λ_D=3.0
                  #  because Direct-only ALREADY = 100% QoS → λ_D=3.0 made qp(4.2)≫
                  #  sum-rate(1.5) → penalty-dominated noisy reward → critic blind
                  #  (explVar~0.1) → no learning. See Common-Knowledge [I-2]. Calibrate
                  #  λ_D vs the BEST trivial baseline (incl. Direct-only), not just All-IRS.]
                  # users instead of trading their QoS for sum-rate. Per fully-failed
                  # user the penalty ≈ λ_D, so λ_D must exceed the marginal sum-rate
                  # gained by abandoning that user (~1-3 bps/Hz here). Ramp gradually.
epsilon_qp = 0.001  # positive error tolerance ε for QoS denominator guard

# ── Ground geometry & satellite LoS zone ────────────────────────────────────────
R_LoS_km = 0.2     # satellite LoS coverage radius on ground (km)
                  # users and IRS are both placed within this circle
                  # curriculum: 0.2 → 0.3 → 0.4 → 0.45 → 0.5 (resume each step)
                  # [Case 2: START fresh at 0.2 (easiest), then ramp & resume each step.
                  #  Read feasibility probe at each R_LoS; set λ_D per docs/Training-Case-2.txt.]
h_IRS_km = 0.02   # IRS/building rooftop height (km) = 20 m

# ── Spawn-region bounds (FRACTION of R_LoS, 0.0→centre … 1.0→full LoS disk) ──────
# Keeps IRS central (urban core) and free users inner-half so nothing spawns at
# the LoS edge with no coverage — critical as R_LoS grows toward 0.5.
irs_spawn_radius_frac = 0.667      # IRS spawn within this·R_LoS, min-separated
user_free_radius_frac = 0.4        # free (non-confined) users within this·R_LoS

# ── User mobility  (random-walk pedestrian model) ───────────────────────────────
user_speed_mps = 1.5   # walking speed (m/s)
dt_s           = 1.0   # time step duration (s)
                       # → step size = user_speed_mps × dt_s = 1.5 m/step

# ── Building blocking & IRS reflection efficiency  (two distinct quantities) ─────
beta_blocking = 0.1   # power attenuation of sat→user direct link when blocked (0–1)
beta_IRS      = 1.0    # IRS element reflection efficiency; ideal passive IRS = 1.0
d_block_km    = 0.02   # building footprint half-width (km) — each building is a
                       # 3-D box (2·d_block × 2·d_block × h_IRS_km).  Shadow on
                       # the ground is computed by 3-D ray–box intersection with
                       # the satellite at (0,0,h_SR_km).

# ── Reinforcement learning — episode counts (also used by test suite) ────────────
n_episodes           = 6000   # training episodes
n_steps_per_ep       = 200    # max environment steps per episode
n_rollout_episodes   = 12     # collect this many episodes before each PPO update
                              # [result_8: 8→12 — near the optimum the advantage
                              #  signal is small; more episodes/update lowers its
                              #  variance → more reliable climb direction.]
reward_noise_avg     = 16     # average reward over this many per-step noise draws
                              # (cuts reward variance ~1/√R; preserves E[reward]; 1=off)
                              # [Case 2 result_2: 8→16 — critic MÙ dai dẳng (explVar~0.08)
                              #  dù đã gỡ entropy-domination (H1 bác). Giả thuyết H2: return
                              #  quá nhiễu để critic fit → tăng averaging giảm variance ~1/√2.
                              #  Kèm bật critic warm-up cho fresh (dưới). WATCH explVar>0.5.]
checkpoint_interval  = 100    # save agent snapshot every N episodes (0 = end-only)
                              # [Case2 result_7: 200→100 — checkpoint dày để probe critic đa giai đoạn
                              #  (diagnostic run ~800ep → 8 checkpoints qua warmup/β-anneal/settle).]
n_eval_episodes   = 50     # episodes used for policy comparison (test 10)
n_irs_pos_samples = 20     # episodes to sample for IRS position diversity check
n_walk_steps      = 50     # steps to verify users stay within LoS after walking
n_signal_samples  = 5_000  # Monte Carlo samples for signal power verification

# ── Cross-block entanglement topology (B3 / B4) ──────────────────────────────────────
# Option-A layout: q0..q(nq-M-1) = user block  |  q(nq-M)..q(nq-1) = IRS block
# B3: M evenly-spaced CZ bridges applied after the NN chain per variational layer
# B4: same M pairs as extra ⟨Z_i Z_j⟩ observables → N_QUANTUM = (2·nq−1) + M

def _cross_block_pairs(nq: int, m: int) -> tuple:
    """Option-A: user q0..q(nq-m-1), IRS q(nq-m)..q(nq-1). Returns m evenly-spaced bridges."""
    if m == 0:
        return ()
    n_user  = nq - m
    spacing = max(n_user // m, 1)
    return tuple((i * spacing, nq - m + i) for i in range(m))

def _b1_zz_pairs(nq: int, m: int) -> tuple:
    """B1 full observables: (nq-m-1) user-block NN ZZ + (nq-m)*m all cross-block ZZ pairs."""
    n_user  = nq - m
    user_nn = tuple((i, i + 1) for i in range(n_user - 1))
    cross   = tuple((u, nq - m + j) for u in range(n_user) for j in range(m))
    return user_nn + cross

#   Case 1 (nq=8,  M=1): ((0, 7),)                            — 1 bridge
#   Case 2 (nq=12, M=2): ((0, 10), (5, 11))                   — 2 bridges
#   Case 3 (nq=16, M=4): ((0, 12), (3, 13), (6, 14), (9, 15)) — 4 bridges
extra_cz_pairs = _cross_block_pairs(n_qubits, M)   # B3 CZ bridges  (auto by case)
extra_zz_pairs = _cross_block_pairs(n_qubits, M)   # B4 Tier-1 ZZ  (auto by case)
# full_zz_pairs = _b1_zz_pairs(n_qubits, M)        # B1 full ZZ     (uncomment when B1 active)
n_hidden_ae   = [128, 64]                  # Case 2: d_aff=44 (was 17) → cần rộng hơn [32]; decoder 24→64→128→44
                                           #   Case 1 (d_aff=12):  [32]         → decoder 16→32→12
                                           #   Case 2 (d_aff=34):  [128, 64]    → decoder 24→64→128→34
                                           #   Case 3 (d_aff=108): [256,128,64] → decoder 32→64→128→256→108
n_hidden_post = [256, 128]                 # hidden-layer sizes for Post-NN after QC (list)
                                           #   (unused when vqc_softmax_head=True)
n_var_layers  = 3     # L: variational circuit depth — U_var = ∏_{ℓ=1}^{L} [U_ent · ∏_i Rz Ry]

# ── VQC readout + head (Δ2/Δ4 architecture-2, 2026-06-01) ───────────────────────
# [G3 core run] Architecture (2) AE-VQC-SoftmaxPQC (per docs/Ablation-Plan PHẦN A):
#  vqc_readout_mode='r1'   : physics-structured observables (per-user Z, per-IRS
#       Z·Z_IRS, cluster mean-ZZ, Z_IRS) instead of generic Z+NN-ZZ. N_QUANTUM auto
#       = (nq-M)(M+2)+M (Case2 42). Jacobian observable-agnostic (auto-adapts).
#  vqc_softmax_head=True   : replace classical post-NN MLP (~50K params) with a
#       single linear map + trainable inverse-temperature β (Jerbi et al. SOFTMAX-PQC).
#       logits = β·(W·[o_hat‖z_t] + b). ~2K params, clean quantum-vs-classical
#       attribution. z_t kept in head input = classical bypass [B] (Δ5).
# For A1 ablation, set readout_mode='generic' + softmax_head=False → recovers baseline.
vqc_readout_mode = 'r1'      # 'r1' | 'generic'
vqc_softmax_head = True      # True = SoftmaxPQC linear+β head; False = classical MLP post-NN
vqc_softmax_beta_init = 1.0  # initial inverse-temperature β (trainable). softmax(β·logits).

# ── SPSA gradient estimator  ─────────────────────────────────────────────────────────
# spsa_n_reps = 0  : exact parameter-shift rule (16,384 circuits/update at nq=16)
# spsa_n_reps > 0  : SPSA estimate with n_reps repetitions  (32× speedup at n_reps=4)
spsa_n_reps   = 16    # Case 2: 8→16 (G2 attempt 1, post-result_10 2026-06-01).
                      # λgrad r=0.04-0.09 ổn định ~0.06 ở result_10 → ZERO-MEAN NOISE pattern.
                      # Theo Training-Case-2 Ý TƯỞNG (B): r<0.5 do NOISE → tăng SPSA reps để
                      # gradient sạch hơn (giảm variance ~1/√n_reps). Nếu r jumps >0.2 ở result_11
                      # → NOISE confirmed → có thể tăng nữa. Nếu r stays ~0.06 → STRUCTURAL
                      # (z_enc layer-norm zero-mean self-cancels) → cần sửa normalization.
                      # Cost: ~2× VQC compute per step (~60s/ep vs 30s previously).
spsa_epsilon  = 0.1   # SPSA finite-difference step ε  (standard value; range 0.05–0.2)

# ── AE Hot-start pre-training ────────────────────────────────────────────────────
ae_pretrain_epochs = 5000    # steps of AE-only gradient descent before joint training
ae_pretrain_lr     = 1e-3   # Adam lr for hot-start (higher than joint lr_actor_ae)

# ── Generalized Advantage Estimation (GAE) ───────────────────────────────────────
gae_lambda = 0.9  # λ in Â_t = Σ (γλ)^l δ_{t+l}; 0→TD(0), →1 Monte-Carlo (low bias)
                  # [result_9: 0.0→0.9 — multi-step credit assignment (effective horizon
                  #  1/(1−γλ)≈7 steps). result_8 peaked at QoS~90%@ep1000 then collapsed via
                  #  the RSMA common-stream eroding (wc 70%→24%, Ck 1.7→0.46) — a long-horizon
                  #  credit failure that TD(0) can't fix. λ=0.9 credits the common stream's
                  #  delayed QoS benefit. Advantages are normalised per-rollout so the scale
                  #  change is absorbed; adv_clip=5 still guards tails.]
adv_clip   = 5.0  # clip normalised advantage to ±this (0/None = off). Bounds grad
                  # from extreme-advantage outliers (high spawn-variance at large
                  # R_LoS); ±5 only touches ~5σ tails → no effect on stable runs.

# ── Data re-uploading (paper Eq. 7): re-apply U_E(z) before each U_var layer ────
data_reuploading = True   # True: circuit = H→U_E→∏_ℓ(U_L(θ_ℓ)·U_E(z))

# ── Phase-shift MLP (IRS element phase policy) ──────────────────────────────────
n_hidden_phase = [64, 128, 256, 128]          # hidden units  31→64→128→256→96
lr_phase       = 1e-4  # Adam lr

# ── Power allocation MLP  ([w_c, w_p] summing to P_S) ───────────────────────────
n_hidden_power = [128, 128, 64]  # Case 2: nới lớp đầu (input 2K+nlat=44)
lr_power       = 1e-4  # Adam lrLem
# [Case2 result_8 2026-06-01] factored output (3-way: split + common + private),
# fix coupling-induced wp-concentration. Probe (probe_power_qos): equal-split private
# → QoS 58%→83% (+25pts) at same assignment/phase. Extra entropy bonus on π_private
# is the LEVER that targets wp-spread INDEPENDENTLY of common-vs-private split.
beta_entropy_pwr_private = 0.003 # added on top of global β_entropy for π_private only.
                                  # 0.0 = neutral (all 3 axes get β only).
                                  # ↑ = stronger spread pressure on private power.
                                  # Tune by watching wp-top2 (target < ~35% at K=10).
                                  # [Case2 smoke result_10 2026-06-01: 0.02 TRIGGERED
                                  #  ⚠ENTROPY-DOMINATED on pw (ent/pg=2.43, β_p was 21×
                                  #  global β → entropy term 8× PG). 0.02→0.003 → β_p~4×
                                  #  global, ent/pg expected <1.0. Probe target wp-top2<35%.]

# ── Common-rate split MLP  (C_k fractions, normalised within groups) ────────────
n_hidden_ck    = [128, 128, 64, 32]     # Case 2: nới lớp đầu (input 5K=50)
lr_ck          = 1e-4  # Adam lr

# ── Critic architecture ───────────────────────────────────────────────────────────
critic_hidden = [512, 256, 128, 64]   # hidden layer sizes (arbitrary depth)

# ── Agent training hyperparameters ───────────────────────────────────────────────
gamma         = 0.95    # TD discount factor γ
ae_weight     = 0.5     # autoencoder reconstruction weight in total actor loss
lr_actor_ae   = 1e-4    # Adam lr — AE weights ω
lr_actor_qc   = 1e-4    # Adam lr — quantum parameters λ, θ  [halved from 3e-4: SPSA gradients are noisy]
lr_actor_xi   = 3e-4    # Adam lr — post-NN weights ξ
lr_critic     = 3e-4    # Adam lr — critic ψ  (↑ from 3e-4: at R_LoS=0.4/λ_D=4 the
                        # return variance is higher → critic lagged (explVar ~0.2,
                        # TD ~0.3) → noisy advantages → reward wavering. Faster
                        # critic helps it track the noisier value target.)
n_shots_train = 1500   # measurement shots during training forward pass
n_shots_eval  = 3000   # measurement shots during evaluation / inference
eval_interval = 20     # run greedy evaluation every N training episodes

# ── PPO-clip ─────────────────────────────────────────────────────────────────────
ppo_epsilon = 0.2   # clip ratio ε: ratio clipped to [1−ε, 1+ε]
ppo_epochs  = 6     # K: passes over the episode buffer per episode

# ── Entropy regularization & mini-batch ──────────────────────────────────────────
beta_entropy            = 0.001  # entropy bonus β₀ at start (early exploration).
                                 # [Case 2 result_1 (λ_D=1.5): 0.002 GÂY entropy-domination DÙ
                                 #  fresh — phase L_pg quá nhỏ (critic mù + phase ít ảnh hưởng
                                 #  reward) → ⚠ENTROPY-DOMINATED cả run → phase IDLE (entropy
                                 #  phẳng ~62% max, không học optimize IRS) → 0.002→0.001 để gỡ.
                                 #  Giả thuyết: bớt random → return ổn định hơn → critic fit
                                 #  được (explVar leo) → phase có tín hiệu học. WATCH ent/pg ph<0.3.]
                                 # MUST ANNEAL toward ~0: a FIXED β over-regularises once
                                 # the policy converges (L_pg shrinks → β·H dominates →
                                 # policy drifts back to random → QoS un-learns). result_3
                                 # drifted with β=0.002 fixed; β=0.01 collapsed fast.
                                 # Rule: β·H must stay ≪ L_pg, incl. AT convergence.
beta_entropy_min        = 0.0003 # β floor (≈0): once converged, entropy must not lead
                                 # [result_8: 0.0002→0.0003, still negligible.]
beta_entropy_anneal_end = 0.1    # reach floor at 10% of episodes (@4000 ep → ~ep 400).
                                 # [Case 2 result_1: 0.15→0.1 — về sàn nhanh hơn để cờ
                                 #  entropy-domination tắt sớm, cho phase cơ hội hội tụ.]
                                 # NOTE: it's a FRACTION of --episodes.
batch_size              = 64     # mini-batch size within each PPO epoch pass

# ── Learning rate schedule (linear decay after warm-up) ──────────────────────────
lr_decay_start = 0.2   # LR stays at 100 % for this fraction of episodes, then decays
                       # [result_8: 0.15→0.2 — keep full LR a bit longer so the resumed
                       #  policy has room to move off the plateau before decaying.]
lr_min_frac    = 0.25  # LRs decay to at most lr_min_frac × initial_lr
                       # [result_8: 0.1→0.25 — don't let LR die too low; keep climb power.]

# ── Critic warm-up on --resume (calibrate V(s) before PPO) ────────────────────────
# A resumed actor is near-optimal, but a fresh critic starts blind (explVar≈0);
# its noisy 1-step advantages erode the good actor over the first few hundred
# episodes. Before PPO, roll the FROZEN resumed policy for N episodes and fit
# V(s) to Monte-Carlo returns (unbiased value scale, matched to the current
# env/reward params). Only runs with --resume; 0 = off. Especially important
# when the resume dir has no saved critic (e.g. result_6 → cold-start otherwise).
critic_warmup_episodes = 24   # Case 2: tăng từ 16 (state lớn hơn, variance cao hơn). 0 = off
critic_warmup_epochs   = 50   # supervised regression passes over the warm-up buffer

# ── PopArt: value-target normalisation (van Hasselt 2016) ──────────────────────────
# [Case2 result_8 2026-05-31] critic [B-2b] wobble KHÔNG do grad_clip (result_7 bác H1)
# NI do RETURN SCALE-DRIFT (live bias=μ(V−ret)=−0.79 trôi). Oracle: norm_tgt=True > False
# (+0.042). PopArt = chuẩn hóa target return (running μ,σ) + bảo toàn output khi stats đổi
# (rescale lớp cuối) → grad O(1) ổn định + theo kịp drift. PASS: explVar GIỮ >0.5 + bias→0.
popart_enabled     = True
popart_beta        = 0.1   # EMA rate của (μ,ν) mỗi rollout-update. Nhỏ=chậm/mượt, lớn=nhanh/nhiễu.
popart_sigma_floor = 1e-2  # sàn σ (tránh chia 0 lúc đầu / return gần hằng).

# ── Gradient norm clipping ────────────────────────────────────────────────────────
grad_clip_ae     = 0.1    # max L2 norm for AE encoder only (tight: prevents RL gradient spikes)
grad_clip_actor  = 0.5    # max L2 norm for QC/xi/phase/power/ck grad-dicts  [tightened: SPSA noise]
grad_clip_critic = 1000.0 # max L2 norm for critic grad-dict
                          # [Case2 result_7 DIAGNOSTIC 2026-05-31: 1.0→1000. Tier-2 đo preclip
                          #  ‖∇‖V=400-1300 ở result_6 → clip=1.0 (và cả 5.0) cắt ~99-100% MỌI
                          #  gradient → lr_critic vô nghĩa (grad đã rescale về clip-norm) → critic
                          #  [B-2b] không bao giờ settle. 1000 ≈2× median preclip: clipfrac 100%→~15%,
                          #  chỉ chặn spike >1000. Test dứt khoát H1 (clip có phải nút thắt). Xem
                          #  Common-Knowledge [B-2b] quy-tắc-clipfrac-trước-lr.]

# ── Random seeds ─────────────────────────────────────────────────────────────────
seed_default = 0    # general-purpose seed (multi-episode, training)
seed_eval    = 42   # seed used for policy evaluation resets
seed_config  = 42   # seed used in test_config user-position snapshot
seed_irs     = 1    # seed used in test_irs IRS-position snapshot
seed_env     = 7    # seed for test_env_core
seed_partial = 99   # seed for test_partial_action
seed_power   = 3    # seed for test_power_constraint
seed_csi     = 5    # seed for test_imperfect_csi


# ── Factory function ────────────────────────────────────────────────────────────

def make_config(**overrides) -> SystemConfig:
    """
    Build a SystemConfig from the values above with optional keyword overrides.

    Example
    -------
    >>> cfg = make_config(K=15, kappa=0.0)
    """
    base = dict(
        K=K, M=M, N=N,
        P_S_dBm=P_S_dBm, f_GHz=f_GHz, h_SR_km=h_SR_km,
        G_S_dBi=G_S_dBi, G_U_dBi=G_U_dBi,
        noise_mean_dBW=noise_mean_dBW, noise_var_dBW=noise_var_dBW,
        rain_mean=rain_mean, rain_var_SR=rain_var_SR,
        rain_var_SU=rain_var_SU, rain_var_RU=rain_var_RU,
        g_sf_dB=g_sf_dB, path_loss_exp=path_loss_exp, d0_ref_km=d0_ref_km,
        kappa=kappa,
        quantization_bits=quantization_bits,
        D_k_bps_hz=D_k_bps_hz, lambda_D=lambda_D, epsilon_qp=epsilon_qp,
        R_LoS_km=R_LoS_km, h_IRS_km=h_IRS_km,
        irs_spawn_radius_frac=irs_spawn_radius_frac,
        user_free_radius_frac=user_free_radius_frac,
        user_speed_mps=user_speed_mps, dt_s=dt_s,
        beta_blocking=beta_blocking, beta_IRS=beta_IRS,
        d_block_km=d_block_km,
    )
    base.update(overrides)
    return SystemConfig(**base)
