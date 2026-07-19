# DNN vs VQC — PHÂN TÍCH SÂU (theory + tốc độ) — 2026-07-18 (·07-19 5-axis fill)

> Doc nội bộ, nguồn cho Discussion/rebuttal sau. KHÔNG phải claim mới cho paper — mọi câu định đưa
> vào paper phải qua no-overclaim filter. Phân biệt rõ ✅ EVIDENCE-BACKED vs 🔮 HYPOTHESIS.

## F. 5-AXIS BENCHMARK — TRẠNG THÁI + SỐ ĐO (07-19, đã điền vào paper sec:abl_qvc)

Cặp chuẩn: VQC r110/ep_01600 · DNN-40K r101/ep_01900 · DNN-2.2K r137/ep_02000 (width-run (40,)).

| Trục | Trạng thái | Số đo chính |
|---|---|---|
| ① Perf & Budget | ✅ | within-3%; train VQC 23-37 vs DNN 0.7-1.1 s/ep (~30×, statevector-sim); infer 0.17 vs 0.008 s/step |
| ② Generalization | ✅ ĐỦ | tab:abl_genshift 11 shift; VQC edge +1.9→+4.0pp @P40; natural-spawn free; N không zero-shot (PhaseMLP dim) |
| ③ Model Scalability | ✅ recount npz | decision-core 144 vs 43,166 = **300×** · infer-total 2,761 vs 52,022 = **19×** (C2; recount khớp §5) |
| ③b iso-perf point | ✅ MỚI | **DNN-2.2K r137 = 1.516/97.4%** ≈ DNN-40K 1.509/96.5 ≈ VQC 1.512/96.8 → 40K OVER-PARAM; matched-ratio ~**15×** (2,230 vs 144), 4 config còn lại chờ user |
| ④ Sample Eff | ✅ | ep-to-90%final ~25 cả 2 (ngang); **transient dip NÔNG hơn ở VQC**: min QoS 87/91% (transitions r0.4/r0.5) vs DNN 83/84% |
| ⑤ Representation | 🔶 nửa | VQC z_t recover **51%** oracle-gap (raw 27%, live 0%); DNN z_t counterpart = PENDING (probe cần patch load ClassicalActor) |

**Error-bars (per-episode std, n=10, base):** VQC 0.0014-0.0027 · DNN 0.0027-0.0055 across shifts →
VQC-DNN gap (0.011-0.029) > fluctuation MỌI shift. Đã ghi footnote tab:abl_genshift.

⭐ **Insight ③b (quan trọng nhất 07-19):** width-run cho thấy DNN 40K là over-parameterized —
DNN 2.2K vẫn match. Nên headline HONEST đổi từ "300× at chosen width" thành "matched-performance
ratio ~15× (+ structural O(n) vs O(d_h·n))". Paper A1 Model-Scalability đã viết đúng tinh thần này.
4 width còn lại ((64,64)/(64,32)/(32,16)/(16,8)) sẽ xác định điểm GÃY chính xác — chờ user launch.

## A. VÌ SAO PERFORMANCE BẰNG NHAU (within 3%) — và vì sao VQC NHỈNH ở Case 2

### A1. ✅ Honest prior (external evidence)
Jerbi 2021 (cite trong paper): parametrized quantum policies **match, not exceed** classical policies
trên natural control tasks. Kết quả within-3% của mình ĐÚNG prior — không cần cơ chế đặc biệt để
giải thích parity; cần cơ chế để giải thích nếu nó KHÁC prior (nó không khác).

### A2. ✅ Task-parity là STRUCTURAL (evidence mạnh nhất, của riêng repo này)
Bài toán assignment có **nghiệm closed-form biết trước** (assignment law: route đúng blocked set,
docs/IRS-Group-Capacity-Ablation.md §1-2 + fig:assignment_law trong paper):
- probe_assignment_quality (§2(6) worksheet): VQC r103 vs DNN r100 — Σirs 7.00/6.96 ≈ Blk 6.77,
  blocked→đúng-IRS **99.6%/99.0%**, UNDER 0.0/0.1%.
- Hai kiến trúc KHÁC NHAU hội tụ về CÙNG nghiệm đóng → performance ceiling là của BÀI TOÁN,
  không phải của actor. Trần chung ⟹ parity structural, không phải trùng hợp.
- Hệ quả viết paper: parity không phải "VQC yếu" cũng không phải "DNN yếu" — cả hai chạm trần
  task-optimal ở trục assignment; khác biệt còn lại nằm ở trục power/Ck (frontier §6.7: gap C1
  ở power-concentration, gap C2 ở Ck demand-fill).

### A3. ✅ Capacity KHÔNG phải constraint — gap là optimization-side
Distill probe (probe_assignment_distill, 06-05): MLP đọc z_t recover **51%** oracle-gap (raw 27%),
live VQC 0% trong training thường → representation ĐỦ thông tin; phần gap còn lại =
credit-assignment/non-stationarity của chain 4 actor — architecture-agnostic. Vì bottleneck không
nằm trong actor, hoán đổi VQC↔DNN không dịch chuyển bottleneck ⟹ thêm một lý do parity.

### A4. Vì sao VQC NHỈNH ở Case 2 (QoS +1.9pp; assignment-law deviation 0.10 vs 0.35;
### transient drop nhỏ hơn ở ramp transitions — Fig reward_log_case2)
- 🔮 **H-a Inductive bias khớp cấu trúc**: M cross-block CZ bridges nối user-register↔IRS-register +
  R1 readout action-aligned (⟨Z_u⟩, ⟨Z_uZ_r⟩, ⟨Z_r⟩) — đúng dạng coupling user–IRS mà assignment
  cần. Bias đúng → sample-efficiency cao hơn trong curriculum non-stationary → track assignment law
  chặt hơn (0.10 vs 0.35 = evidence CONSISTENT với H-a, chưa phải proof).
- 🔮 **H-b Implicit regularization từ param-count nhỏ**: 120-144 params (vs 40k) = hypothesis class
  hẹp → khó overfit vào stage hiện tại của curriculum → transient drop nhỏ khi ramp đổi
  (observed: "visibly smaller transient drops", Fig case-2). Cùng chiều với việc DNN C1 kẹt
  geometry-asymmetry trước đây (memory 07-12).
- 🔮 **H-c SoftmaxPQC trainable β**: inverse-temperature học được = "trainable greediness"
  (Jerbi) — điều tiết exploration mượt hơn head logits thô.
- ⚠ **H-d Single-seed**: mọi khác biệt ≤ vài pp nằm trong single-seed variability — paper ĐÃ hedge
  đúng ("within single-seed variability, not a systematic advantage"). GIỮ hedge này; H-a/b/c là
  cơ chế ứng viên để THẢO LUẬN, không phải để claim.

## B. TỐC ĐỘ TRAINING (wall-clock THẬT, median s/ep trên 100 eps cuối, đo 07-18)

| Actor | Case 1 | Case 2 |
|---|---|---|
| **VQC** | r45 23.3 · r111 23.8 · r117 30.6 | r90 29.8 · r99 28.5 · r110 37.1 · r131 29.6 · r132 27.3 · r133 27.0 · r134 30.6 |
| **DNN** | r102 0.8 · r106 0.9 · r107 0.8 · r126 0.7 | r98 1.0 · r100 1.1 |
| **Ratio** | ≈ **29×** | ≈ **27-35×** |

(Số dòng-cuối log KHÔNG dùng được — dính eval/diag overhead: r110 322.7s, r117 273.3s là artifact.)

### Cost model (giải thích con số ~30×)
- VQC mỗi episode: ~**38k statevector evaluations** (memory #12) = behavior sampling S_tr=1500-shot
  mỗi step × 200 steps + SPSA 2×16 evals/update × minibatch passes; mỗi eval = mô phỏng 2^{n_q}=4096
  amplitudes × L layers. CPU-bound; GPU (CuPy) chỉ constant-factor, KHÔNG phá được 2^{n_q}
  (trần thực dụng ~28-29 qubits). SPSA đã là lựa chọn rẻ: ~6× ít evals hơn param-shift @dim(η)=72-144.
- DNN mỗi episode: forward+backward MLP ~40k params × 200 steps — trivial (~0.7-1.1 s/ep, phần lớn
  là env/rate computation chứ không phải network).
- ⟹ **~30× là chi phí MÔ PHỎNG lượng tử, không phải chi phí thuật toán**: trên hardware thật,
  chi phí VQC = shot-cost (1500-3000 shots/decision), không phải 2^{n_q}. Câu paper Discussion
  "inference cost is measured in simulation, not wall-clock on quantum hardware" ĐÃ cover đúng —
  có thể mở rộng thêm cho training cost nếu reviewer hỏi.
- Ghi chú phụ: readout giàu hơn không làm chậm đáng kể (r133 single-z 27.0 ≈ r132 full-zz 27.3 ≈
  r131 nn-zz 29.6 — trong noise); AE (12q) là thứ giữ cho 30× khả thi — raw C2/C3 = 22/41 qubits
  → ×2^10/2^29 statevector = bất khả thi (đã ghi A2/Discussion).

## C. TỐC ĐỘ INFERENCE

- **Cả hai actor đều amortized**: 1 forward pass / channel state (policy đã train offline), đối lập
  Tan 2026 solve-per-state (SDR-POA + RMCG + AO + SA lặp đến hội tụ MỖI realization). Đây là claim
  chính của paper (amortized-inference) — KHÔNG đổi giữa VQC/DNN.
- Trong simulator: VQC inference = statevector hoặc S_ev=3000-shot estimate → chậm hơn MLP forward
  ~1-2 order (cùng tỷ lệ statevector-cost ở trên nhưng không có SPSA/backprop). Cả hai vẫn ≪ chi phí
  1 lần solve convex-pipeline per state.
- Trên hardware: VQC = 3000 shots × circuit depth ~O(L·n_q) gates — shot-count là đơn vị chi phí
  đúng; so sánh wall-clock hardware-vs-GPU nằm NGOÀI scope paper (đã ghi limitation).
- ⬜ Nếu cần SỐ inference wall-clock cho rebuttal: time infer.py (50 eps × 200 steps) cho 1 run VQC
  vs 1 run DNN cùng case — chưa đo hôm nay (không có trong logs sẵn).

## E. ZERO-SHOT GENERALIZATION SWEEP — CASE 2 (đo 2026-07-18 tối, benchmark trục ②)

Ckpt chuẩn: VQC r110/ep_01600 · DNN r101/ep_01900 (cặp A1 closing). Greedy inference 50 ep × 200
step seed 42; base env R_LoS 0.5 · 1/1 · α0.9. Zero-shot = KHÔNG retrain, override lúc eval
(infer.py flags mới: --p-s-dbm/--user-speed/--natural-spawn + --kappa/--noise-var sẵn có).
Logs: analysis/data/gensweep_c2/*.txt. VQC noise-rows (κ/nv) lấy từ Current-Result NOISE-SWEEP
(cùng ckpt; base 1.5300 cũ vs 1.5307 re-run hôm nay = trong noise ✓).
⚠ Trục N KHÔNG zero-shot được: PhaseMLP output dim phụ thuộc N (trained N=24) — index-error khi
N≠24; robustness theo N phải retrain-per-N (đúng cách tab:nsweep C1 đã làm).

| Shift | VQC ΣRate / QoS | DNN ΣRate / QoS | Δ QoS (VQC−DNN) |
|---|---|---|---|
| **base** | **1.5307 / 97.3%** | **1.5182 / 95.4%** | +1.9pp |
| P_S 40 dBm | 1.2947 / 74.5% | 1.2654 / 70.5% | **+4.0pp** |
| P_S 45 dBm | 1.4462 / 87.1% | 1.4268 / 83.4% | **+3.7pp** |
| P_S 60 dBm | 1.5818 / 99.6% | 1.5777 / 99.2% | +0.4pp |
| speed 3 m/s (2×) | 1.5244 / 97.1% | 1.5120 / 95.0% | +2.1pp |
| speed 6 m/s (4×) | 1.4928 / 95.3% | 1.4840 / 93.2% | +2.1pp |
| natural spawn | 1.5312 / 97.2% | 1.5122 / 95.3% | +1.9pp |
| κ = 0.10 | 1.5291 / 97.2% | 1.5170 / 95.4% | +1.8pp |
| κ = 0.20 | 1.5263 / 97.0% | 1.5150 / 95.3% | +1.7pp |
| nv = 13 | 1.5270 / 96.8% | 1.5150 / 94.8% | +2.0pp |
| nv = 16 | 1.5241 / 96.3% | 1.5119 / 94.3% | +2.0pp |

**Đọc kết quả (evidence-backed):**
1. **Cả hai actor generalize tốt** mọi trục trừ P_S: mobility 4× mất ≤2.2pp, κ×4 mất ≤0.3pp,
   natural-spawn ≈ FREE (−0.1pp cả 2 phía — policy không overfit balanced-spawn).
2. **P_S là trục stress duy nhất** (đổi regime interference→power-limited, đúng theory §6 worksheet):
   P40 mất ~23-25pp QoS; Greedy sập còn 42% → cả 2 actor giữ lead ~+28-33pp.
3. ⭐ **VQC edge NỞ RỘNG dưới distribution shift**: +1.9pp @base → +3.7/+4.0pp @P45/P40; đồng nhất
   +1.7…+2.1pp trên mọi trục còn lại — VQC không bao giờ thua DNN ở bất kỳ shift nào (11/11).
   Consistent với H-a (inductive bias) §A4 nhưng vẫn single-seed — hedge như cũ khi đưa vào paper.
4. Suy giảm tương đối VQC luôn ≤ DNN (vd sp6: −2.0 vs −2.2pp; nv16: −1.0 vs −1.1pp) — "degrades
   gracefully" giữ nguyên và có cặp so sánh.

## D. CÂU CHỮ SẴN DÙNG CHO DISCUSSION (đã no-overclaim, user duyệt trước khi chèn)
1. "The parity is structural rather than incidental: on the assignment axis both actors converge to
   the closed-form composition rule of Appendix A, so the achievable ceiling is set by the task."
2. "The VQC's Case-2 edge (tighter tracking of the composition rule, smaller transient drops at
   curriculum transitions) is consistent with the action-aligned inductive bias of the cross-block
   bridges and structured readout, though within single-seed variability."
3. "The ~30× wall-clock training overhead of the quantum actor is a cost of classical statevector
   simulation (≈38k evaluations of a 2^{12}-amplitude state per episode), not of the algorithm;
   on quantum hardware the corresponding unit is measurement shots."
