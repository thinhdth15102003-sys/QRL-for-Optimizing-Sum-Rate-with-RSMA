# MATH-HYPO DIRECTIONS — hướng xây mathematical hypothesis THIÊN COMMS cho paper (2026-07-18)

Yêu cầu user: cần hypothesis toán về COMMS (không phải AI/ML) để tăng chất comms-theory.
Mỗi hướng: (i) phát biểu candidate · (ii) nền tảng sẵn có · (iii) khả thi chứng minh · (iv) probe validate.
Tiêu chí xếp hạng: chứng minh sạch + validate bằng probe ngắn + reviewer-value.
Style rule khi đưa vào paper: empirical → "Observation", chứng minh được → "Lemma/Proposition" (no-overclaim).

---

## ⭐ TOP-3 ĐỀ XUẤT

### H1. Discrete-phase quantization loss bound (sinc² law) — MỚI HOÀN TOÀN, comms-classic
- **(i) Candidate Lemma.** Với B-bit uniform phase quantization, phase error mỗi element
  δ_n ~ U[−π/2^B, π/2^B] độc lập. Kỳ vọng coherent-combining power của N elements:
  E|Σ_n e^{jδ_n}|² = N + N(N−1)·sinc²(π/2^B), với sinc(x)=sin(x)/x.
  ⟹ array gain giữ được **tỷ lệ sinc²(π/2^B) của gain N² liên tục**; B=2: sinc²(π/4) = (sin(π/4)/(π/4))² ≈ **0.811**.
  Corollary: rate loss ≤ log2(1/0.811) ≈ 0.30 bps/Hz trong high-SNR (thực tế nhỏ hơn nhiều vì interference-limited).
- **(ii) Nền tảng.** Paper đã có: N²-scaling (cite wu2019intelligent §II), 𝒬_θ B=2 trong model, phase oracle closed-form trong codebase.
- **(iii) Khả thi.** Chứng minh 5 dòng (independence + E[cos δ] = sinc). EXACT, không cần bound.
- **(iv) Probe.** ~phút: istn/channel.py — so Γ_cascade với phase oracle continuous vs quantized-2bit trên cùng draws; kiểm ratio ≈ 0.811·(1 + 1/((N−1)sinc²)) theo N=20-36 (khớp luôn tab:nsweep "no monotone trend": aperture không binding).
- **Reviewer-value:** giải thích ĐỊNH LƯỢNG vì sao 2-bit đủ (design choice đang thiếu justification trong paper) + nối kết literature discrete-phase IRS.

### H2. Misassignment robustness under CSI error (κ-bound) — giải thích 2 bảng kết quả sẵn có
- **(i) Candidate Proposition.** Assignment quyết định trên ĝ, chấm trên g. Với error model
  ĝ=g+κ|g|ε: xác suất flip lựa chọn giữa 2 route có margin SNR tương đối m_rel là
  P(flip) = Q(m_rel/(√2·κ)) + o(κ) — decisions chỉ đổi khi hai route gần hòa; rate gap kỳ vọng
  E[R(ĝ-opt) − R(g-opt)] = O(κ²) (flip chỉ xảy ra ở near-tie → mất mát bậc 2).
- **(ii) Nền tảng.** Design-versus-reality gap ĐÃ trong paper (sync ⑦/⑩); evidence: tab:noisesweep
  (κ 0.05→0.20 mất 0.3pp QoS, 0.2% rate) + g-check ≈null; verify session 07-14 đo gap κ0.3 ~2.8%, κ0.05 ~0.5%.
- **(iii) Khả thi.** 2-route case: closed form qua Gaussian tail (ε complex normal → |ĝ| Rician, xấp xỉ
  Gaussian cho κ nhỏ). General K,M: union bound. Mức "Proposition + sketch" đạt; exact = khó, KHÔNG cần.
- **(iv) Probe.** ~phút: CSI/env.py — sample margins giữa best và second-best route trên 500 states,
  đếm flip-rate theo κ ∈ {0.05..0.3}, khớp Q(·) curve. Data verify 07-14 tái dùng được.
- **Reviewer-value:** biến 2 bảng empirical robustness thành hệ quả CÓ CƠ CHẾ — đúng chỗ reviewer hay hỏi "why so robust?".

### H3. Order-statistics capacity band — FORMALIZE cái ĐÃ CÓ 80% (ít công nhất)
- **(i) Candidate Propositions** (từ App A + worksheet, nâng từ prose thành statement đánh số):
  P1: m_B ~ Exp(Λ), Λ=Σ1/Γ̄ (exact, heterogeneous) → E[R_c(m_B)] closed form qua E_1. [đã có eq]
  P2: E[Δ_{B→B+1}] tăng theo B (convexity) ⟹ optimum biên B=0 hoặc B_max dưới iid fading;
      interior optimum ⟺ heterogeneous candidates. [đã có prose + 27/27 probe]
  P3: Serve-all floor R(m)=m·log2(m/(m−1)) ↓ 1/ln2 (concentration identity) + drop-budget ceiling
      R*(d) với corner rule ⟹ capacity band [R*(0), R*(d)]; d_min = max(0,(K−K*)/K) dự đoán
      C3 (K15) không serve-all được (~13%). [worksheet §6; user đang tự làm — hướng này NỐI THẲNG]
- **(iii) Khả thi.** P1 exact ✓ (đã in paper). P2: chứng minh qua E[R_c(m_{B+1})]−E[R_c(m_B)] → 0 đơn điệu
  (log-concavity của Exp min); mức Lemma được. P3: two-point exact + general bound.
- **(iv) Probe.** verify_vstar_formulas.py + probe_orderstat_marginal.py (27/27) + probe feasibility K15 (~phút).
- **Reviewer-value:** nhóm App A từ "worksheet-style" thành 2-3 Proposition đánh số — chuẩn hoá cái sẵn có, rủi ro thấp nhất.

---

## CÁC HƯỚNG KHÁ (làm sau nếu còn thời gian)

### H4. Interference-limited invariance (P_S→∞ limit)
- **(i)** Lemma: lim_{P_S→∞} SINR_{p,k} = x_k/(1−x_k−y_g) — CHỈ phụ thuộc power fractions, |h|² cancel
  ⟹ high-power frontier R*(d) là hàm ĐẾM thuần (số served + splits), không phụ thuộc geometry.
- **(ii)** Đã có mầm trong eq SINR general-power App B + worksheet "@P50 interference-limited"; giải thích frontier C2 phẳng.
- **(iii)** Chứng minh 3 dòng (limit). **(iv)** tab:pssweep + frontier logs sẵn.
- Lưu ý: dạng limit-statement, reviewer có thể hỏi finite-P_S correction — thêm O(1/P_S) term nếu cần.

### H5. λ_crit phase transition theo K (extreme-value)
- **(i)** Corollary của eq:abl_lcrit: V*_rate = E[log2(1+Γ_max)] tăng ~log2 ln(KM) (Gumbel/extreme value
  cho Exp tails) trong khi R*(K) → 1/ln2 + common ⟹ λ_crit(K) có xu hướng GIẢM khi K tăng ở geometry cố định
  → giải thích C1 (λ_crit 2.00) vs C2 (1.15) nằm 2 phía λ_D=1.5 như hệ quả cấu trúc, không phải tình cờ.
- **(iii)** Asymptotic (không exact); mức "Observation + asymptotic argument". **(iv)** probe_v_star.py sweep K.
- ⚠ Rủi ro overclaim: Γ̄ heterogeneous theo geometry → chỉ claim xu hướng, kèm caveat.

### H6. Assignment law (blocked→its-IRS) như Proposition điều kiện
- **(i)** Proposition: nếu Γ_a^dir < Γ_th(A,B) thì re-assignment sang IRS là lợi đơn điệu (term (I)
  dominant + (II)≥0 khi a là worst); blocked ⟹ Γ^dir≈0 thỏa mọi Γ_th ⟹ optimal set ⊇ blocked set.
- **(ii)(iv)** App A đã có mọi mảnh + fig:assignment_law (99.4-100% trained-policy match). **(iii)** dễ, điều kiện hoá cẩn thận phần (III).
- Value vừa: chủ yếu "đóng dấu Proposition" cho cái đã thuyết phục.

### H7. Rain-attenuation outage (ITU-R lognormal)
- **(i)** Outage probability closed-form per link (lognormal CDF) → QoS floor không phụ thuộc policy.
- **(iii)** Dễ nhưng **giá trị thấp** — không chạm contribution chính; chỉ làm nếu reviewer đòi channel-level analysis.

---

## KHUYẾN NGHỊ TRIỂN KHAI
1. **H3 trước** (0.5 ngày): rename các kết quả App A thành Proposition 2/3 + câu dẫn — gần như zero rủi ro, nối với capacity-band user đang làm.
2. **H1 tiếp** (1 buổi: lemma 5 dòng + probe ~30 phút): lấp lỗ hổng justification 2-bit — độc lập mọi phần khác.
3. **H2 sau** (1 ngày: proposition + probe flip-rate): nâng imperfect-CSI từ "setting" thành "analyzed setting".
4. H4 gộp được vào phần Pareto expand (timer 15:27 hôm nay); H5/H6 để pass sau; H7 bỏ qua trừ khi bị đòi.
