# CITATION-NEED — các cite cần tìm & chèn (2026-07-18)

User tìm bib entry 1 lượt khi đi qua các phần; Claude chèn `\cite` + entry vào `reference.bib` khi có.
Vị trí ghi theo **search-string** (số dòng sẽ trôi). ⚠ KHÔNG sửa tex trước khi có entry.

> **STATUS 07-18 tối: ✅ XONG 7/7.** b (PPO ×2) · c (GAE ×2) · d (SPSA ×2) · e (re-uploading) ·
> g (LayerNorm) · **a (NP-hard → luo2008dynamic, câu rút gọn "generally NP-hard"; bonus SDR →
> luo2010semidefinite)** · **f (ITU-R → itu618 @techreport P.618-12)**. wu2019intelligent user cấp
> nhưng ĐÃ CÓ SẴN trong bib — bỏ qua, không thêm trùng. bibtex+2×pdflatex PASS, 0 undefined.

| # | Ưu tiên | Vị trí (search string trong tex) | Claim cần support | Ref gợi ý |
|---|---|---|---|---|
| a | 🔴 | `Consequently, (P0) is NP-hard` (§IV.A Overall Framework) | NP-hardness của lớp mixed-integer non-convex QCQP / joint IRS-assignment+beamforming. Nếu không tìm được cite khớp → fallback đã duyệt: soften thành "a mixed-integer non-convex program for which no polynomial-time exact method is known" | Cite NP-hardness cho sum-rate maximization / user-association (vd. Luo & Zhang 2008 "Dynamic spectrum management: Complexity and duality", IEEE JSTSP; hoặc paper chứng minh NP-hard cho IRS discrete phase) |
| b | 🔴 | `proximal policy optimization (PPO)` — first mention §IV.A (đoạn "The framework is trained as an actor--critic system") + §IV.D | Nguồn gốc PPO | Schulman et al. 2017, "Proximal Policy Optimization Algorithms", arXiv:1707.06347 |
| c | 🔴 | `generalized advantage estimation` quanh `eq:gae` (§IV.D Learning Objective and Critic) | Nguồn gốc GAE | Schulman et al. 2016, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016 (arXiv:1506.02438) |
| d | 🔴 | `simultaneous perturbation stochastic approximation (SPSA)` — first mention §IV.A + eq:spsa §IV.D | Nguồn gốc SPSA | Spall 1992, "Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation", IEEE TAC 37(3) |
| e | 🔴 | `data re-uploading` — first mention §IV.C Circuit Architecture (câu "Finally, $U_E$ is re-applied at the start of every layer (\emph{data re-uploading})") | Nguồn gốc kỹ thuật data re-uploading (Lemma 2 đã cite schuld2021effect cho Fourier spectrum — vẫn cần nguồn re-uploading gốc) | Pérez-Salinas et al. 2020, "Data re-uploading for a universal quantum classifier", Quantum 4, 226 |
| f | 🟡 | `large-scale path loss and rain attenuation following the ITU-R model` (System Model, trước eq channel) | Model rain attenuation lognormal ITU-R | ITU-R P.618 (propagation prediction Earth-space); hoặc re-cite `\cite{3gpp38811}` tại câu nếu không muốn thêm entry |
| g | 🟢 | `layer-normalized to stabilize the input range` quanh `eq:ln_z` (§IV.C Data Encoding) | LayerNorm | Ba, Kiros & Hinton 2016, "Layer Normalization", arXiv:1607.06450 |

## Mục liên quan đã xử lý khác (không cần cite)
- "orders of magnitude cheaper" SPSA → ĐÃ sửa thành "~6× fewer circuit evaluations" (07-18, khớp log banner).
- Bảng so sánh Intro: user tự verify field `year` các key tan2025flexible / he2026weighted / zhang2026satellite khớp label cột Ref (hàng 2026).
- Related Works: user tự verify số "approximately 29% sum-rate gain" đúng với paper Tan gốc.
