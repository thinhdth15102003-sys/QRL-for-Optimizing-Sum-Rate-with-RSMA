# SECTION-REVIEW PUNCH-LIST — full paper (2026-07-17, autonomous 5h-timer run)

Skill `irs-rsma-paper-review`, đủ 5 checks/section. **REVIEW-ONLY — chưa sửa .tex** (trừ 2 fix frozen-λ
đã làm TRƯỚC review theo yêu cầu riêng). User confirm từng mục rồi mới sửa.
Quy ước: 🔴 phải sửa · 🟡 nên sửa · 🟢 tuỳ chọn · `sync#X` = đã nằm trong docs/PAPER-SYNC-DIFFS.md
(KHÔNG re-flag, chỉ điểm danh) · `deferred-number` = số user tự verify sau (GR2, non-blocking).

**Cite-audit toàn cục: SẠCH** — 32 keys dùng = 32 keys bib, 0 dangling, 0 unused, 0 mismatch rõ.

---

## 1. ABSTRACT (dòng 35-49)   |   sync trong scope: none

1. CITATIONS — không có cite trong abstract (chuẩn IEEE, OK).
2. MISSING CITES — none.
3. STYLE
   - 🟡 "offering wide-area coverage and **seamless** connectivity" → bỏ "seamless" (hype; đã flag 07-14).
   - 🟡 "compresses the high-dimensional **channel state**" → "**system state**" (khớp ① affinity: state
     gồm channel + demand + IRS stats, không chỉ channel).
4. CLAIM STRENGTH
   - 🔴 **[Abstract #4 — OPEN từ 07-14]** "achieves **significant sum-rate gains over baseline methods**"
     — pairing "quantum"+"gains" thiếu value-prop → đọc như quantum-advantage. Sửa 2 phần:
     (a) "baseline methods" → "**heuristic baselines**";
     (b) THÊM 1 câu honest value-prop, gợi ý: *"and matches a classical deep-network actor within a few
     percent while using two orders of magnitude fewer assignment-actor parameters, amortizing the
     per-state optimization cost of model-based methods into a single offline training run."*
5. PAPER↔CODE — "under imperfect CSI and dynamic user mobility" ✓ đúng post-CSI-split (κ có teeth).

LOCK VERDICT: **FIX-THEN-LOCK — mục 4 (#4 framing) là blocker duy nhất.**

---

## 2. INTRODUCTION §I.A Background + Contributions + Organization (56-296)   |   sync: none

1. CITATIONS — tất cả OK (wang2019/zhu2021/kodheli2020/3gpp38811/liu2011/mao2018/cover1972/
   tan2025flexible/pan2022/sutton2018/levy2019/gong2022+2023/haarnoja2018/jerbi2021/hinton2006/nagy2025
   đều đúng chỗ, đúng claim).
   - ⚠ VERIFY: bảng tab:comparison_existing_works — **hàng "2026 \cite{tan2025flexible}"**: label năm
     2026 nhưng key đặt tên 2025 → user check field `year` trong bib entry (và các hàng 2026 khác:
     he2026weighted, zhang2026satellite) khớp label cột Ref.
2. MISSING CITES — none đáng kể (đoạn nào assert đều đã có cite).
3. STYLE
   - 🟢 dòng 79 "offer a **fundamentally different computational pathway**" — hơi hype; đề xuất
     "offer an alternative computational route" (giữ cũng chấp nhận được vì có cite jerbi ngay đó).
4. CLAIM STRENGTH
   - 🟡 dòng 270: "**no existing study** jointly addresses..." → "**none of the surveyed works**
     jointly addresses..." (universal→scoped vào bảng; an toàn reviewer).
5. PAPER↔CODE — contributions map đúng C1/C2/C3 ✓; (M+1)^K ✓; AE→latent→VQC ✓.

LOCK VERDICT: READY TO LOCK sau 2 mục nhỏ (270 + verify year-labels).

---

## 3. RELATED WORKS §II (297-331)   |   sync: none

1. CITATIONS — wu2019intelligent ✓ (đúng paper Wu&Zhang, N² scaling claim khớp nguồn gốc);
   tan2025flexible ✓; levy2019/bacon2017/gong2022/haarnoja2018/jerbi2021/nagy2025 ✓.
   - ⚠ VERIFY dòng 314: "**approximately 29% sum-rate gain** over the IRS-assisted RSMA baseline" —
     số attributed cho Tan; user check đúng con số trong paper gốc.
2. MISSING CITES — none.
3. STYLE (3 lỗi grammar thật — 🔴 cả 3)
   - 🔴 dòng 315: "susceptible to estimation errors arising from **estimation errors arising from**
     channel aging..." — LẶP CỤM. Bỏ 1 lần.
   - 🔴 dòng 317: "incurs computational complexity that scales prohibitively **with the numbers of
     users, IRS elements, and control variables grow**" → "...prohibitively **as the numbers of users,
     IRS elements, and control variables grow**" (hoặc bỏ "grow").
   - 🔴 dòng 318: "replacing model-based sub-problem decomposition with a hierarchical reinforcement
     learning approach **in which leverages variational quantum policy representations manages** the
     combinatorial complexity" → gãy ngữ pháp. Đề xuất: "...approach **whose variational quantum policy
     representation manages** the combinatorial complexity of multi-IRS user assignment."
   - 🟢 dòng 322: "~\cite{levy2019learning}, ~\cite{bacon2017option}" → gộp "\cite{levy2019learning,
     bacon2017option}" (spacing chuẩn).
4. CLAIM STRENGTH — appropriately hedged.
5. PAPER↔CODE — mô tả Tan (2-group, K1 IRS-only/K2 direct, SDR-POA/RMCG/AO/SA) khớp Tan-diff notes ✓.

LOCK VERDICT: FIX-THEN-LOCK — 3 lỗi grammar 🔴.

---

## 4. SYSTEM MODEL §III (332-773)   |   sync trong scope: **⑦ (dòng 603)**

1. CITATIONS — none trong section (formulas của mình).
2. MISSING CITES
   - 🟡 dòng 562: "large-scale path loss and rain attenuation **following the ITU-R model**" — chưa cite
     chỗ này. Suggest: ITU-R P.618 (rain) hoặc re-cite \cite{3gpp38811} tại câu.
3. STYLE
   - 🔴 dòng 556: "We **defined** set of scheduled user messages after **spliting** as:" →
     "We **define the** set of scheduled user messages after **splitting** as".
   - 🔴 dòng 705: subsection title "**Optimizating** Problem" → "**Optimization** Problem".
   - 🟡 symbol overload: bảng notation khai $g$ = "True channel coefficient" NHƯNG $g$ đồng thời là
     group index ($g \in \mathcal{G}$) khắp paper. Đề xuất: đổi channel thành $g^{X}_{\cdot}$ giữ
     nguyên (đã có superscript) nhưng trong bảng notation ghi rõ "channel coefficient (with link
     superscript)" để phân biệt với group index; hoặc đổi group index thành $\ell$. Tối thiểu: sửa mô tả
     bảng notation.
   - 🟡 $s$ overload: dòng 500 $s = \{s_1..s_K\}$ (set) rồi dòng 558 $s = [s_{c,1}..s_{p,K}]$ (vector
     sau split) — cùng ký hiệu 2 nghĩa; đặt tên khác cho vector sau split (vd $\tilde{s}$).
   - 🟢 dòng 591: "with a random propagation phase ... **which is tracked** via Doppler compensation"
     → ", tracked via Doppler compensation at the satellite".
4. CLAIM STRENGTH — OK (model assumptions khai rõ).
5. PAPER↔CODE
   - `sync#⑦` dòng 603: OLD text "The RL agent, however, observes the true channel magnitudes |g|..."
     — CONFIRMED chưa apply. (Không re-flag.)
   - 🔴 **NEW — eq common-rate (676-680) SAI NOTATION NỘI BỘ**: LHS viết $R_{k,c,g} = \min_{k\in
     \mathcal{K}_g}\{R_{1,c,g},...,R_{K_g,c,g}\}$ — (a) per-user rate $R_{k,c,g} = \log_2(1+
     \gamma_{k,c,g})$ **chưa từng được định nghĩa** trước khi lấy min; (b) LHS phải là **$R_{c,g}$**
     (group rate) chứ không phải $R_{k,c,g}$; (c) min-subscript k đè lên enumerate. Đề xuất:
     ```latex
     R_{k,c,g} = \log_2(1+\gamma_{k,c,g}), \qquad
     R_{c,g} = \min_{k \in \mathcal{K}_g} R_{k,c,g}.
     ```
     (khớp bảng notation: $R_{k,c,g}$ = per-user common-decoding rate, $R_{c,g}$ = group common rate;
     khớp code CSI/rate.py::_sinr_all → R_c^g = log2(1+min SINR).)
   - 🔴 **NEW — notation private stream (553)**: "$s_{p,k}$" ≠ "$s_{k,p,g}$" dùng mọi nơi khác
     (608/617/622-627/notation table). Thống nhất $s_{k,p,g}$.
   - Channel/CSI/SINR/power formulas còn lại: ✓ matches code (sync-file mục "Không đổi" đã verify).

LOCK VERDICT: FIX-THEN-LOCK — eq common-rate 🔴 + 2 typo 🔴 (+ sync ⑦ khi apply-pass).

---

## 5. FRAMEWORK §IV.A Overall (775-818)   |   sync trong scope: **① (795-814, 816)**

1. CITATIONS — vanhasselt2016popart ✓ đúng chỗ PopArt.
2. MISSING CITES
   - 🟡 dòng 818/969/1019: **PPO chưa cite nguồn gốc** ở first-mention → suggest: Schulman et al. 2017
     (PPO paper).
   - 🟡 dòng 780: "(P0) is **NP-hard**" — asserted không cite/proof → hoặc cite nguồn NP-hardness cho
     lớp bài toán (mixed-integer nonconvex QCQP), hoặc soften "is a mixed-integer non-convex program
     for which no polynomial-time exact method is known".
3. STYLE
   - 🟡 sau khi apply ①: đoạn 786 ("A direct quantum encoding...") và đoạn 823 (sec:ae mở đầu) sẽ
     TRÙNG rationale (d_s grows with N, impractical register) — gộp/rút một trong hai.
4. CLAIM STRENGTH
   - 🟢 dòng 890 (VQC intro): "using **far fewer** trainable parameters than a classical network **of
     comparable expressivity**" — "comparable expressivity" chưa được thiết lập; đề xuất "with a
     compact trainable-parameter budget (Section~\ref{sec:res_qvsc})".
5. PAPER↔CODE
   - `sync#①` eq:state (795-814) + critic s_t (816): CONFIRMED chưa apply.
   - 🔴 **NEW — XUNG ĐỘT KÝ HIỆU $\mathbf{a}_t$ (nổ ra khi apply ①)**: dòng 973 đặt **action**
     $\mathbf{a}_t = (\varphi,\Phi,\mathbf{w},\mathbf{C})$, trong khi eq:affinity (823) đặt **affinity**
     $\mathbf{a}_t$. Sau ① (state ≡ affinity a_t) thì state và action TRÙNG ký hiệu. Đề xuất: đổi action
     thành $\mathbf{u}_t$ (control) toàn cục — 973, 1055, Lemma 1 (1682: $\pi(\mathbf{a}_t|\mathbf{s}_t)$).
     **→ đã append vào PAPER-SYNC-DIFFS.md như mục ⑫** để user sửa cùng pass ①.

LOCK VERDICT: FIX-THEN-LOCK — chờ ① + ⑫ cùng lúc; PPO cite 🟡.

---

## 6. AE §IV.B (820-886)   |   sync: none

1-4. CITATIONS/MISSING/STYLE/CLAIM — sạch; prose end-to-end vs frozen-AE viết tốt, có lý do rõ.
5. PAPER↔CODE — $d_{\mathrm{aff}} = K(M{+}2)+2M$ ✓ = code extract_state (C1: 17 ✓ = actor d_s);
   dual-branch dims (16-hidden IRS branch, 8-hidden micro-encoder, P projection) — khớp actor_config
   hidden_ae; group-norm + demand un-normalized ✓ khớp _group_norm code (demand giữ raw — verified
   session 07-14 fix note). LN affine γ,β thuộc VQC encoding (sync ④), không phải AE — OK ở đây.

LOCK VERDICT: **READY TO LOCK** (sau khi ① apply thì rà lại câu mở đầu trùng rationale — xem §5.3).

---

## 7. VQC §IV.C (887-964)   |   sync trong scope: **④ (895-899) · ② (944-951) · ⑨ (953) · ③ (955-963)**

1. CITATIONS — OK.
2. MISSING CITES
   - 🟡 dòng 911/936: "**data re-uploading**" first mention chưa cite gốc → suggest: Pérez-Salinas
     et al. 2020 (data re-uploading paper). (Lemma 2 cite schuld2021effect cho Fourier-spectrum là
     ĐÚNG nguồn khác — vẫn nên thêm nguồn re-uploading.)
   - 🟢 dòng 894: LayerNorm chưa cite (Ba et al. 2016) — optional.
3. STYLE — clean (mô tả circuit rõ, H/Ry/Rz/CZ hardware-efficient ✓).
4. CLAIM STRENGTH — "without loss" (911: "$2n_q$ latent components are loaded **without loss**") —
   🟢 đề xuất "one-to-one" thay "without loss" (tanh squash + LN có mất thông tin biên độ).
5. PAPER↔CODE
   - `sync#④②⑨③`: eq:ln_z thiếu γ,β · eq:observables = generic 2n_q−1 (readout thật = R1 34/42) ·
     analytic-during-training câu sai (train dùng 1500-shot sampling, chỉ grad/ratio analytic) ·
     head = "MLP with ReLU" (thật = SoftmaxPQC linear+β). TẤT CẢ CONFIRMED chưa apply.
   - "M cross-block CZ bridges" (911) ✓ **matches code** `params._cross_block_pairs` (m evenly-spaced).
   - Fig. quantikz caption "⟨Z_i⟩ và ⟨Z_iZ_j⟩" — sau ② vẫn đúng loosely (R1 cũng là Z-basis) ✓.

LOCK VERDICT: FIX-THEN-LOCK — 4 mục sync là blockers chính của section này.

---

## 8. TRAINING §IV.D (966-1056)   |   sync trong scope: **① (973) · ⑩ (sau 987) · ⑧ (1042-1051)**

1. CITATIONS — vanhasselt2016popart ✓; jerbi (không trong scope này) OK.
2. MISSING CITES
   - 🟡 PPO (969, 1019) → Schulman 2017; **GAE** (1003, eq:gae) → Schulman et al. 2016 (GAE paper);
     **SPSA** (1042) → Spall 1992. Ba method mượn đều chưa có nguồn gốc.
3. STYLE — clean; eq:reward/objective/gae/critic_loss trình bày tốt.
4. CLAIM STRENGTH
   - 🔴 dòng 1051: "making each update **orders of magnitude cheaper** than the parameter-shift rule"
     — với dim(η)=72–144 và SPSA n_reps=16: 2·16=32 evals vs 2·dim=144–288 → **~4.5–9× = MỘT order,
     không phải "orders"** (log banner tự ghi "6x vs param-shift"). Sửa: "several-fold cheaper (about
     an order of magnitude at our circuit sizes)" hoặc "6× fewer circuit evaluations per update".
5. PAPER↔CODE
   - `sync#①` (973 state def) + `sync#⑩` (reward trên g thật — thêm sau 987) + `sync#⑧` (eq:spsa
     finite-diff-of-J → SPSA-of-Jacobian + chain rule): CONFIRMED chưa apply.
   - eq:gae/eq:critic_loss + PopArt no-target-net ✓ ĐÚNG code (fix 07-13 đã nằm).
   - eq:reward ✓ = CSI/env.py penalty (quadratic normalized shortfall, λ_D, ε_q).
   - Action notation $\mathbf{a}_t$ (973): xem §5.5 — sync ⑫.

LOCK VERDICT: FIX-THEN-LOCK — "orders of magnitude" 🔴 + 3 sync + cites 🟡.

---

## 9. SIM SETUP §V.A (1057-1100)   |   sync trong scope: **⑥ (1062, 1078) · ⑤ (1084)**

1. CITATIONS — none needed thêm.
2. MISSING CITES — none.
3. STYLE — 🟢 "\section{**Experiment**}" (1057) → "Experiments" / "Simulation Results and Analysis"
   (Organization hứa "presents simulation results").
4. CLAIM STRENGTH — OK.
5. PAPER↔CODE
   - `sync#⑥` P_S Case-3 60 dBm (1062 prose + 1078 tab:hyper) · `sync#⑤` lr_ck chưa tách 5e-5 (1084):
     CONFIRMED chưa apply.
   - Còn lại tab:hyper ✓ khớp params.py từng dòng (γ0.95, GAE0.9, clip 0.2/6, rollout 12/64,
     β 1e-3→3e-4 @10%, λ_D1.5, ε_q1e-3, avg16, LR 1e-4/1e-4/3e-4, 2000/200, clip 0.1/0.5, nq12/24,
     L 2/3/5, reup, SPSA 0.1/16, shots 1500/3000, α_AE 0.5, noise (−30,10), rain (4,0.1), κ0.05, B2,
     R_LoS 0.2-0.5, IRS 20m, speed 1.5, f/h 1.58/800, G 60) — values deferred-number nhưng cấu trúc ✓.
   - Eval protocol (greedy, 3000 shots, 50 ep, fixed seed) ✓ khớp infer.py.

LOCK VERDICT: FIX-THEN-LOCK — chỉ 2 sync items; còn lại sạch.

---

## 10. RESULTS §V.B (1102-1248)   |   sync: none

1. CITATIONS — jerbi2021 (1174) ✓ đúng honest-prior chỗ cần.
2. MISSING CITES — none (own results).
3. STYLE — B.1/B.2/B.3 gọn, đúng giọng. 🟢 B.1 "$7.6$–$59.0\%$" — range check với bảng (Case-1
   All-IRS 57.6 max? 59.0 lấy từ đâu) → deferred-number, user rà khi verify số.
4. CLAIM STRENGTH — chuẩn: "within 3%", "two distinct operating points", "neither dominates",
   caption ghi rõ sweet-spot ckpt từng actor (transparent) ✓. "visibly smaller transient drops" —
   acceptable (qualitative, có Fig).
5. PAPER↔CODE — tab:param_eff 120/144 + 338×/300× `sync#④-keep` ✓ đúng giữ; số bảng = deferred.
   tab:abl_readout C2 3/4 filled [07-17 session này] ✓ nhất quán 3 docs.

LOCK VERDICT: READY TO LOCK (numbers deferred riêng).

---

## 11. DISCUSSION + CONCLUSION (1251-1265)   |   sync: none

- ✅ Mâu thuẫn "essentially flat" đã FIX (07-17, session này) — giờ khớp appendix co-adapted optimum.
- "two orders of magnitude fewer parameters" ✓ đúng (338×/300×).
- Limitations paragraph ✓ honest (single-seed, C3 structural-only, sim-only inference cost).
- Conclusion ✓ khớp Discussion, không claim vượt.

LOCK VERDICT: **READY TO LOCK**.

---

## 12. APPENDIX A — Group-Composition Capacity (1274-1474)   |   sync: none

1-2. CITATIONS/MISSING — none needed (own math; validated bằng probe của mình).
3. STYLE — viết tốt, đúng .md-style đã chốt.
4. CLAIM STRENGTH — chuẩn ("exact by construction", "conservative special case", "in this geometry",
   reference-specific đã nêu trong bảng caption) ✓.
5. PAPER↔CODE
   - 🔴 **NEW — CLASH KÝ HIỆU $s_k$** (1401): shortfall $s_k = (R_k^{\min} - R_p(\Gamma_k))^+$ đè lên
     $s_k$ = user message (toàn paper + notation table). Đề xuất đổi shortfall → $\varsigma_k$ (hoặc
     $u_k^{\mathrm{sf}}$). **→ appended PAPER-SYNC-DIFFS.md mục ⑬.**
   - Nội dung khớp worksheet §1-§2 (order-stat corrected: E[Δ] increasing → corner survives ✓;
     two-point + E_1 closed forms ✓; general ≥ floor ✓; tab:abl_bstar per-draw 100% ✓).
   - deferred-number: 0.141-vs-0.137, bảng B* values.

LOCK VERDICT: FIX-THEN-LOCK — 1 mục (ς_k rename).

---

## 13. APPENDIX B — Pareto Frontier (1476-1569)   |   sync: none

1-4. Sạch. λ_crit notation nhất quán nội bộ; "achievable lower bounds" honest ✓; "even λ_D→∞ attains
   only ~92%" + waterfill parking đúng worksheet §6.5 ✓; opposite-sides-of-λ1.5 đúng §6.4 ✓.
5. PAPER↔CODE — SINR general-power form (1486-1488) ✓ = worksheet §6.1; deferred-number: V* 10.963/
   11.723 (worksheet bản cũ ghi 10.934/11.712 — có thể do env-set khác; USER VERIFY khi rà số),
   frontier 3.11/1.61, attainment 70/95/94.

LOCK VERDICT: READY TO LOCK (numbers deferred).

---

## 14. APPENDIX C — Frozen-λ (1571-1676)   |   sync: none

- ✅ 2 fix session này: (a) \cite{mcclean2018barren} đã thêm cạnh barren-plateau scale; (b) Discussion
  đã hết mâu thuẫn. Narrative co-adapted-optimum + sign-flip finding + KL-share declining — khớp
  probe refresh 07-12 (anchor r90→r110) ✓.
- 🟡 Số trong section = OLD-code anchor (r110 chain) — mechanism durable, nhưng khi có anchor new-code
  final → RE-RUN 4 probe (parse_lambda_trajectory / lam_interference / lambda_landscape ×5 /
  uenc_ablation) và refresh ρ_λ/cos/KL-share + tab:abl_frozen. ⚠ Probe scripts default --ckpt còn trỏ
  result_8/result_11 (stale) — repoint khi chạy.
- 🟢 Scope-caveat "rescaling direction only" GIỮ NGUYÊN khi user chỉnh — đừng generalize.

LOCK VERDICT: READY TO LOCK (số refresh sau retrain là việc riêng).

---

## 15. APPENDIX D — Architectural & Training Ablations (1678-1864)   |   sync trong scope: **④ (1702-1708)**

1. CITATIONS — jerbi2021 ✓ ×2 (honest-prior + trainable-greediness); schuld2021effect ✓ (Lemma 2).
2. MISSING CITES — none thêm.
3. STYLE — Jerbi-style Hypothesis/Setup/Observations nhất quán ✓.
4. CLAIM STRENGTH — A5 entropy "qualitatively... left to future work" ✓; A6/A7 pending ghi honest ✓;
   readout Observation (07-17) đã tự hedge "within single-seed variability" ✓.
5. PAPER↔CODE
   - `sync#④` eq:vqc_params (1702-1708) = 2n_q(L+1) cũ. ⚠ LƯU Ý THÊM: prose ngay dưới (1716) đã dùng
     số "120→144" = theo công thức MỚI (L+3) → **nội bộ đang mâu thuẫn cho tới khi ④ apply** (formula
     cho 72/96 nhưng prose nói 120/144). Apply ④ là hết.
   - 🔴 **NEW — DISTILLATION DANGLING**: 2 chỗ viện dẫn "distillation probe/result" (1757 →
     "Section~\ref{sec:abl_ae}" và 1774 "consistent with the distillation result") nhưng **paper KHÔNG
     trình bày thí nghiệm distillation ở đâu cả** (abl_ae không có). Chọn 1: (a) thêm 2-3 câu
     distillation vào abl_ae hoặc abl_hier (probe có thật: linear/MLP head đọc z_t recover phần lớn
     oracle-routing — finding #3), hoặc (b) bỏ cả 2 mệnh đề viện dẫn. Đề xuất (a) — nó đỡ cho cả
     câu "readout not the bottleneck".
   - 🔴 **NEW — A6 α-MISMATCH (1811, 1815, 1832)**: text ghi Case-2 main operating point =
     **α_pf 0.95** và sweep {0.5, 0.7, **0.95**}. Run main thật (r90→r110 chain = tab:case2_main) =
     **α_pf 0.90** (hyperparameters.json + infer banner "α=0.90 restored"; 0.95 là recipe DNN r53-era).
     Sửa: 0.95 → 0.9 ở 3 chỗ (prose ×2 + hàng bảng C2). **→ appended PAPER-SYNC-DIFFS.md mục ⑪.**
   - 🟡 **A2 hứa "(iii) the theoretical training cost" (1721) nhưng không deliver** — không có đoạn
     nào về cost. Chọn: (a) thêm 2 câu qubit-cost (raw C2/C3 = 22/41 qubits → 2^{n_q} statevector ×
     ~38k SPSA-evals/episode = intractable; AE giữ 12q => tractable — số liệu memory #12), hoặc
     (b) bỏ "(iii)" khỏi câu liệt kê. Đề xuất (a) — 1 câu là đủ và đây là support mạnh cho C3.
   - Lemma 1 ✓ (đúng factorization, proof sketch hợp lệ); Lemma 2 ✓ (schuld cite);
     Hypothesis 4b "M bridges" ✓ = code `_cross_block_pairs` (m bridges evenly-spaced).
   - A7 note "(ii) currently has no dedicated flag" ✓ ĐÚNG code (train.py chưa có assignment-only).
   - tab:abl_ae_params 17/44/81 → 9/22/41 qubits ✓ = d_aff formula; deferred-number L_AE floors.

LOCK VERDICT: FIX-THEN-LOCK — distillation 🔴 + α-mismatch 🔴 (+ sync ④).

---

## 16. GLOBAL

- 🟡 **Spelling EN-US/EN-GB trộn**: "quantisation"(×4)/"quantised"/"standardised"/"characterise"/
  "Generalise"/"factorises" vs đa số -ize ("summarizes", "regularization", "normalization"...).
  IEEE chấp nhận cả hai nhưng phải NHẤT QUÁN → chọn -ize (đa số hiện tại), sweep một lần.
- ✅ Cite integrity toàn cục: 32/32, 0 dangling, 0 unused.
- ✅ Labels Organization (sec:related/model/framework/results/discussion/conclusion) đều tồn tại.
- Numbers: toàn bộ giá trị bảng/percent = **deferred** (GR2) — user verify pass riêng; các chỗ đã
  điểm danh deferred-number ở trên chỉ để nhớ, KHÔNG blocker.

---

# TỔNG HỢP

| Section | 🔴 | 🟡 | 🟢 | Verdict |
|---|---|---|---|---|
| 1 Abstract | 1 | 2 | 0 | FIX-THEN-LOCK |
| 2 Intro | 0 | 1(+verify) | 1 | ~READY |
| 3 Related | 3 | 0(+verify) | 1 | FIX-THEN-LOCK |
| 4 System Model | 3 | 3 | 1 | FIX-THEN-LOCK |
| 5 Framework | 1(⑫) | 3 | 1 | FIX (với ①) |
| 6 AE | 0 | 0 | 0 | **READY** |
| 7 VQC | 0(4 sync) | 1 | 2 | sync-pass |
| 8 Training | 1 | 2(3 sync) | 0 | FIX-THEN-LOCK |
| 9 Sim Setup | 0(2 sync) | 0 | 1 | sync-pass |
| 10 Results | 0 | 0 | 1 | **READY** |
| 11 Disc+Concl | 0 | 0 | 0 | **READY** ✅ |
| 12 App A | 1(⑬) | 0 | 0 | FIX-THEN-LOCK |
| 13 App B | 0 | 0 | 0 | **READY** |
| 14 App C | 0 | 1(re-probe) | 1 | **READY** |
| 15 App D | 2(+④) | 1 | 0 | FIX-THEN-LOCK |
| 16 Global | 0 | 1 | 0 | — |

## TOP-10 ƯU TIÊN (thứ tự đề xuất)
1. **Apply PAPER-SYNC-DIFFS.md** (14 mục cũ + 3 mục mới ⑪⑫⑬ vừa append) — mở khoá §5/7/8/9/15.
2. **⑫ đổi action $\mathbf{a}_t \to \mathbf{u}_t$** — PHẢI làm CÙNG lúc với ① (nếu không state=action trùng ký hiệu).
3. **⑪ A6 α 0.95→0.9** (3 chỗ) — sai với run thật của tab:case2_main.
4. **Distillation dangling** (1757+1774) — thêm đoạn ngắn hoặc bỏ viện dẫn.
5. **Eq common-rate** (676-680) — định nghĩa $R_{k,c,g}$ rồi $R_{c,g}=\min$.
6. **⑬ shortfall $s_k \to \varsigma_k$** (App A) — hết đụng message $s_k$.
7. **Related Works 3 lỗi grammar** (315/317/318) + "Optimizating" (705) + "defined/spliting" (556).
8. **"orders of magnitude cheaper" → ~6×/one order** (1051) — khớp log.
9. **Missing cites**: PPO (Schulman'17), GAE (Schulman'16), SPSA (Spall'92), data re-uploading
   (Pérez-Salinas'20), ITU-R (P.618/3gpp) — user tìm bib entries.
10. **Abstract #4** (heuristic baselines + value-prop sentence) + "seamless" + spelling -ize sweep.

*Số liệu bảng/percent: verify pass riêng của user (GR2) — không nằm trong 10 mục trên.*
