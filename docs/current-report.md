# CURRENT REPORT — DNN vs VQC, Case-1 & Case-2

> Cập nhật: 2026-06-29. Số liệu lấy từ `infer.py` (held-out eval, greedy, 50 ep, seed 42).
> Mục tiêu: bảng so sánh DNN (classical actor) vs VQC cho A1 (quantum-vs-classical) + bảng params no-ae VQC & DNN.
> ⚠ Baseline (Greedy/AllIRS/...) KHÁC nhau giữa Case-1 và Case-2 (env khác) → chỉ so trong cùng case.
> ⭐⭐ 2026-06-29: **A1 Case-2 UNIFIED-METHOD** (cả VQC & DNN: α0.9 full-③ + ck-fix + oracle+phase-warmup→95, ramp0.2, λ1.5, seed42).
>   r90 VQC-AE 96.8% ≈ r95 DNN 96.5% = **TASK-PARITY** (honest-prior). AE khử "extraction-cost gap −4.7pp" cũ (= no-ae r66 artifact).

---

## 1. KẾT QUẢ (infer held-out, 50 ep, seed 42)

| Case | Actor | Run | QoS | ΣRate (bps/Hz) | Feasibility | Reward | Greedy (QoS / ΣRate) |
|------|-------|-----|-----|----------------|-------------|--------|----------------------|
| ⭐ **Case-1** (K5M1) | **DNN** (α0.5 fair) | **r102** ep2000 | **97.4%** | 1.797 | 92.2% | 346.8 | 98.3% / 1.505 |
| ~~Case-1~~ (OLD, α=0 unfair) | DNN | r28/r29 | 76.0% | 1.883 | 20.5% | 282.1 | — |
| **Case-1** (K5M1) | **VQC** (AE) | r39 | **98.4%** | 1.713 | 95.0% | 334.1 | 98.3% / 1.505 |
| **Case-1** (K5M1) | **VQC** (no-AE) | r74 | **99.1%** | 1.665 | 96.5% | 328.6 | 98.3% / 1.505 |
| ⭐ **Case-2** (K10M2) | **VQC** (AE, unified) | **r90** ep2000 | **96.8%** | 1.512 | 75.9% | 280.1 | 91.8% / 1.374 |
| ⭐ **Case-2** (K10M2) | **DNN** (unified) | **r95** ep2000 | **96.5%** | 1.509 | 73.6% | 276.5 | 91.8% / 1.374 |
| ~~Case-2~~ (OLD, pre-unified) | DNN | r53 | 96.8% | 1.506 | 76.4% | 280.0 | — |
| ~~Case-2~~ (OLD, no-ae) | VQC no-ae | r66 | 92.1% | 1.420 | 50.1% | 211.2 | — |

**Trần servable (probe):** Case-1 ≈ 98.2% · Case-2 (oracle@P50) ≈ 99.8% (in-log "59.4%" là under-count default-phase).

### Ghi chú quan trọng
- ⚠ **Case-1 DNN (r28/r29) = pre-recipe, α=0 (concentrated power)** → QoS chỉ 76%, KHÔNG fair với VQC (95-98%). Đây KHÔNG phải giới hạn năng lực DNN mà là artifact α=0 (giống Case-2 α-too-low). **Cần train mới DNN Case-1 với `--actor-mode classical --power-fairness 0.5`** để có baseline fair (kỳ vọng ~95%, ngang VQC). Tạm dùng 76% + đánh dấu rõ.
- **Case-1 VQC** hiện là bản **AE** (r39, theo yêu cầu "bỏ qua việc Case-1 VQC đang là AE"). Bản no-ae Case-1 (r67/r69) đạt 98% nếu muốn thay sau.
- **Case-2 DNN (r53) = 96.8% > Greedy 91.8%** ✓ (xác nhận doc 97%). **Case-2 VQC no-ae (r66)** ≈ 90% (training-eval) — chờ infer chốt.
- ⭐⭐ **[06-29 UNIFIED] Case-2 VQC-AE (r90) 96.8% = DNN (r95) 96.5% = TASK-PARITY.** Gap "−4.7pp extraction-cost" (DNN r53 vs no-ae r66) là **artifact của no-ae**, KHÔNG phải bản chất VQC — AE-representation khử gap. r90 vs r95 unified-method (cùng α0.9 full-③ ck-fix oracle+phase-warmup→95) chỉ khác ACTOR → so A1 sạch. ⟹ honest-prior PQC≈DNN + param-eff (decision-core 300×).

### Bảng infer chi tiết (verbatim)

**Case-1 DNN — r28/r29** (giống hệt nhau, có thể là duplicate seed/config):
```
  Policy              Reward   ΣRate(bps/Hz)   Feasibility    QoS frac
  HQC-HAC            282.104          1.8832         20.5%       76.0%
  Greedy             295.085          1.5052         92.2%       98.3%
  AllIRS             -96.172          0.8786         14.9%       68.1%
  DirectOnly        -136.865          0.8229          5.9%       50.5%
  Random            -246.474          0.6830          9.0%       40.0%
```

**Case-2 DNN — r53:**
```
  Policy              Reward   ΣRate(bps/Hz)   Feasibility    QoS frac
  HQC-HAC            279.981          1.5059         76.4%       96.8%
  Greedy             220.309          1.3736         56.7%       91.8%
  AllIRS           -1218.715          0.5480          0.0%       31.1%
  DirectOnly       -1021.977          0.6689          0.0%       37.9%
  Random           -1624.847          0.3894          0.0%        9.8%
```

**Case-1 VQC (AE) — r39:**
```
  Policy              Reward   ΣRate(bps/Hz)   Feasibility    QoS frac
  HQC-HAC            334.100          1.7128         95.0%       98.4%
  Greedy             295.085          1.5052         92.2%       98.3%
  AllIRS             -96.172          0.8786         14.9%       68.1%
  DirectOnly        -136.865          0.8229          5.9%       50.5%
  Random            -246.474          0.6830          9.0%       40.0%
```
⟹ Case-1 VQC (AE) **98.4%** ≈ Greedy 98.3% nhưng **ΣRate 1.71 > 1.51** (VQC route IRS để tăng sum-rate, vẫn giữ QoS trần).

**Case-2 VQC (no-ae) — r66:**
```
  Policy              Reward   ΣRate(bps/Hz)   Feasibility    QoS frac
  HQC-HAC            211.218          1.4203         50.1%       92.1%
  Greedy             220.309          1.3736         56.7%       91.8%
  AllIRS           -1218.715          0.5480          0.0%       31.1%
  DirectOnly       -1021.977          0.6689          0.0%       37.9%
  Random           -1624.847          0.3894          0.0%        9.8%
```
⟹ Case-2 VQC no-ae **92.1%** > Greedy 91.8% ✓; **DNN 96.8% > VQC 92.1% (~4.7pp gap = VQC extraction-cost**, đúng honest-prior: parity biểu diễn + chi phí learnability). Cả hai vượt Greedy.

### A1 — CURRICULUM (correct-env, greedy seed42) [2026-06-29, sau infer.py env-fix]
⚠ **INFER.PY BUG-FIX 06-29**: `load_training_cfg` cũ CHỈ đọc K/M/N/P_S/D_k từ training_config.json → R_LoS+radius rơi về **params.py (ramp0.2)** → mọi infer curriculum (ramp≠0.2 / radius-unlock) **eval trên SAI env** (tell-tale: Greedy/AllIRS baseline giống hệt mọi ramp). ĐÃ FIX: auto-đọc R_LoS_km+2 radius_frac từ hyperparameters.json (walk-up) + CLI `--R-LoS/--irs-spawn-frac/--user-free-frac`. **Số curriculum dưới đây là POST-FIX (đúng env).** Bug đã PHÓNG ĐẠI curriculum-cost.

| Setting | VQC-AE QoS | DNN QoS | VQC ΣRate | DNN ΣRate | Greedy |
|---------|-----------:|--------:|----------:|----------:|-------:|
| **Case-1** ramp0.3 (r41 ep1000 / r104 ep2000) | 96.4% | 95.9% | 1.780 | 1.802 | 98.3% (C1-env) |
| ramp0.2 (ep2000) | 96.8% | 96.5% | 1.512 | 1.509 | 91.8% |
| ramp0.3 (ep1200) | 96.5% | 95.1% | 1.526 | 1.509 | 92.0% |
| ramp0.3 (ep2000, final) | **96.8%** | **95.8%** | 1.530 | 1.521 | 92.0% |
| ramp0.5+full-unlock (ep2000) | *(VQC pending)* | **95.3%** | — | 1.518 | 92.0% |

- ⭐ **DNN robust cả curriculum: 96.5→95.1→95.3%** (giữ ~95% tới full-unlock; bug cũ tưởng tụt 93.9%).
- ⭐ **Task-parity giữ**: VQC≈DNN @ramp0.2, VQC +1.4pp @ramp0.3. Cả hai Pareto-dominate Greedy mọi setting.
- ⚠ TODO: VQC chain mới tới ramp0.3 (r99); cần ramp0.4→0.5+unlock để đủ cặp A1 ở setting cuối.

### A1 — EVAL-SEED VARIANCE (C2 ramp0.2 ep2000, 50ep greedy × seeds 42-45) [2026-07-03]
- **VQC r90: QoS 96.5 ± 0.24%** (96.8/96.7/96.4/96.2) · ΣRate 1.509 ± 0.002
- **DNN r95: QoS 96.2 ± 0.23%** (96.5/96.2/95.9/96.0) · ΣRate 1.504 ± 0.004
- ⟹ error bars CHỒNG NHAU → **parity claim solid với eval-seed bars**. Paired-by-seed: VQC > DNN cả 4/4 seed (+0.2→+0.5pp) = edge nhỏ consistent (để observation, KHÔNG claim advantage — cần train-seed mới nói về method). Greedy dao động theo seed (91.6/91.5/90.5) → không trộn seed giữa bảng.
- Train multi-seed (2 seed × fresh ramp0.2 pair) vẫn nên có cho headline; eval bars đã đủ mức "mean±std over 4 eval seeds".

### A1 — Quantum vs Classical (đọc nhanh)
- ⭐⭐ **Case-1 [06-30 FAIR]**: DNN r102 (α0.5, mirror-r39 vanilla, 2000ep) **97.4%/1.797** ≈ VQC-AE r39 **98.4%/1.713** = **TASK-PARITY** (VQC +1.0pp QoS, DNN +4.9% rate — 2 Pareto points sát nhau; cả 2 ≈ Greedy-QoS 98.3% nhưng +14-19% rate). Số 76% (r28/r29 α=0) OBSOLETE.
- ~~Case-1 cũ: VQC(AE) 98.4% ≫ DNN 76.0%~~ — SUPERSEDED (α=0 artifact, r102 khử).
- ⭐⭐ **Case-2 [06-29 UNIFIED]**: VQC-AE (r90) **96.8%** ≈ DNN (r95) **96.5%** = **TASK-PARITY** (chênh +0.3pp QoS / +0.9% rate = trong noise). Cả hai Pareto-dominate Greedy (+5pp QoS & +10% rate). Gap "−4.7pp" cũ = no-ae artifact (r66), AE khử. ⟹ honest-prior PQC≈DNN XÁC NHẬN với cùng method; câu chuyện = **param-eff** (decision-core circuit 144 vs DNN policy-MLP 43K = 300×), KHÔNG task-advantage.
- ⚠ ~~Case-2 cũ: DNN 96.8% > VQC(no-ae) 92.1%~~ — SUPERSEDED (no-ae confound; dùng AE-unified r90/r95).

---

## 2. BẢNG PARAMS — no-ae VQC

| Param | Case-1 (r67/r69) | Case-2 (r66) |
|-------|------------------|--------------|
| K / M / N | 5 / 1 / 24 | 10 / 2 / 24 |
| Actor mode | quantum, **no-ae** | quantum, **no-ae** |
| Qubits n_q / latent 2n_q | 12 / 24 | 12 / 24 |
| Variational layers L | 2 | 3 |
| n_choices (M+1) | 2 | 3 |
| Data re-uploading | enabled | enabled |
| Shots (train / eval) | 1500 / 3000 | 1500 / 3000 |
| AE loss weight α_AE | **0** (no-ae) | **0** (no-ae) |
| **Power-fairness α (③)** | **0.5** | **0.95** |
| QoS penalty λ_D | 1.5 | 1.5 |
| Discount γ / GAE λ | 0.95 / **0.9** | 0.95 / **0.9** |
| Entropy β₀ → β_min | 1e-3 → 3e-4 | 1e-3 → 3e-4 |
| Reward-noise averaging | 16 | 16 |
| PPO clip / epochs / batch | 0.2 / 6 / 64 | 0.2 / 6 / 64 |
| Rollout episodes (per update) | 8 | 8 |
| LR (AE / VQC / head) | 1e-4 / 1e-4 / 3e-4 | 1e-4 / 1e-4 / 3e-4 |
| LR critic | 3e-4 | 3e-4 |
| LR (phase / power / ck) | 1e-4 / 1e-4 / 5e-5 | 1e-4 / 1e-4 / 5e-5 |
| Critic arch | [512,256,128,64], PopArt(β0.1) | [512,256,128,64], PopArt(β0.1) |
| SPSA step / reps | 0.1 / 16 | 0.1 / 16 |
| Episodes | 1500 | 4000 |
| Warm-up | assign-warmup (oracle routing), **KHÔNG phase-warmup** | oracle-warmup + ck-fix |

---

## 3. BẢNG PARAMS — DNN (classical actor)

| Param | Case-1 (r28/r29) | Case-2 (r53) |
|-------|------------------|--------------|
| K / M / N | 5 / 1 / 24 | 10 / 2 / 24 |
| Actor mode | classical (AE-enc + MLP) | classical (AE-enc + MLP) |
| Encoder hidden | [128] | [128] |
| Policy hidden | [256, 128] | [256, 128] |
| Qubits | 0 (no circuit) | 0 (no circuit) |
| Latent dim | 24 | 24 |
| AE loss weight α_AE | (default) | 0.5 (AE on) |
| **Power-fairness α (③)** | **0 (pre-recipe ⚠)** | **0.95** |
| QoS penalty λ_D | 1.5 | 1.5 |
| Discount γ / GAE λ | 0.95 / 0.9 | 0.95 / 0.9 |
| Reward-noise averaging | 16 | 16 |
| PPO clip / epochs / batch | 0.2 / 6 / 64 | 0.2 / 6 / 64 |
| LR (enc / head / critic) | 1e-4 / 3e-4 / 3e-4 | 1e-4 / 3e-4 / 3e-4 |
| Episodes | 1500 | 2000 |
| Note | ⚠ α=0 → QoS 76%, KHÔNG fair; cần rerun α0.5 | resume r52/ep1600, α0.6→0.95 |

---

## 4. PAPER `tab:hyper` — CÁC GIÁ TRỊ STALE CẦN SỬA (chưa đụng paper, chờ duyệt)

Bảng `tab:hyper` (label `tab:hyper`, ~dòng 1065 trong .tex) đang mô tả config CŨ (nq8 era). Để khớp config hiện tại (theo TODO "tab:hyper khớp Case2 K10/M2/λ1.5 + GAE"):

| Dòng | Hiện tại (paper) | Nên sửa thành |
|------|------------------|---------------|
| Users K / IRS M / N | 5 / 1 / 24 | **10 / 2 / 24** (nếu feature Case-2) |
| GAE λ | **0** | **0.9** |
| QoS penalty λ_D | **3.5** | **1.5** |
| Qubits n_q / latent | **8 / 16** | **12 / 24** |
| Reward-noise averaging | **8** | **16** |
| Entropy β₀→β_min | 2e-3 → 2e-4 | **1e-3 → 3e-4** |
| LR critic | 5e-4 | **3e-4** |
| SPSA reps | 4 | **16** |
| AE loss weight α_AE | 0.5 | giữ 0.5 cho DNN/AE-VQC; **0 cho no-ae VQC** |
| (THÊM) Power-fairness α | — | **0.95 (Case-2) / 0.5 (Case-1)** |
| Episodes | 6000 | per-run (Case-2 4000 / Case-1 1500) — interlock với prose dòng 1098 |
| Critic Polyak τ 5e-3 | — | hiện dùng **PopArt** (β0.1), không Polyak → cần sửa cả prose |

> ⚠ Cần bạn chốt: (a) bảng feature **case nào** (K5M1 hay K10M2), (b) **single-config hay 2 cột VQC|DNN**, (c) có đổi prose (episodes, LoS radius, Polyak→PopArt) không. Mình sẽ sửa paper sau khi bạn chốt.

---

## 5. PARAM-EFFICIENCY — no-ae VQC vs DNN (đếm trực tiếp từ `actor_params.npz`)

⚠ **DNN KHÔNG dùng AE** — encoder của DNN là MLP thường (44→128→24), không decoder, không L_ae. Chỉ VQC có AE (encoder+decoder). Vì vậy **decoder là VQC-only** ⇒ so full-trainable-có-decoder là KHÔNG apples-to-apples (phạt VQC vì thứ DNN không có).

| Thành phần | C1 VQC (r67) | C1 DNN (r29) | C2 VQC (r66) | C2 DNN (r53) |
|---|---|---|---|---|
| Encoder (VQC: AE dual-branch · DNN: plain MLP) | 352 | 5,400 | 606 | 8,856 |
| AE decoder † (VQC-only) | 1,361 | — | 15,596 | — |
| **Circuit-core** (θ+λ+enc-norm) | **120** | — | **144** | — |
| Post-readout head | 591 | — | 2,011 | — |
| Policy MLP | — | 40,586 | — | 43,166 |
| **TOTAL (trainable)** | **2,424** | **45,986** | **18,357** | **52,022** |
| **Inference (−decoder †)** | **1,063** | 45,986 | **2,761** | 52,022 |

† AE decoder: VQC-only; discarded tại inference; ở no-ae còn KHÔNG nhận gradient (ae_weight=0) ⇒ vestigial.
‡ Lúc TRAIN, AE của VQC (enc+dec=16.2K @C2) thực ra > encoder DNN (8.9K); VQC chỉ nhỏ hơn ở MỌI thành phần khi infer (decoder bỏ). ⇒ claim "param-eff tại inference / decision-core", KHÔNG phải "full-pipeline lúc train".
- Circuit-core breakdown: θ (variational) C1 48 / C2 72 · λ (encoding scales) 24 / 24 · enc LayerNorm γ,β 48 / 48.

**Tỷ lệ VQC-nhỏ-hơn-DNN:**
| Mức | C1 | C2 | Robust? |
|---|---|---|---|
| Decision-core (circuit vs policy-MLP) | **338×** | **300×** | ✅ robust, scale-invariant (θ ⟂ K) |
| Inference total (−decoder) | **43×** | **19×** | ✅ |
| Full trainable (gồm decoder) | 19× | **2.8×** ⚠ | ⚠ AE-decoder-bound ở C2 |

**Claim recommendation (NO-OVERCLAIM):**
1. Headline robust: *quantum decision core ~300× nhỏ hơn classical policy-MLP, scale-invariant theo K*.
2. Deployment: *inference actor nhỏ hơn 19–43×* (decoder bỏ).
3. ⚠ Caveat: C2 full-trainable chỉ 2.8× vì decoder (15.6K) — decoder ở no-ae KHÔNG train ⇒ structural-no-AE (deferred) sẽ đưa full-trainable = inference = 19×. KHÔNG claim "VQC param-efficient" trần trụi bằng full-pipeline.
> Số này supersede ước lượng thô handoff cũ; đếm trực tiếp từ `actor_params.npz`.
