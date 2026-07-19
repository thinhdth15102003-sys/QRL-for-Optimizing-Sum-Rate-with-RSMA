# NOTATION GLOSSARY — toàn bộ ký hiệu trong paper (kể cả appendix/ablation) — 2026-07-18

Nguồn: đọc full tex 07-18 (sau khi apply 17 sync-diffs + rename affinity a_t→s_t + ς_k).
Mục đích: (1) tra cứu nhanh, (2) lộ các **overload** để user quyết có rename không, (3) danh sách
ký hiệu **chưa có trong bảng notation** (tab:variables) của paper.

## 1. Sets & indices
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| 𝒦, K | tập / số ground users | System Model |
| ℳ, M | tập / số IRS | System Model |
| 𝒩, N | tập / số reflecting elements mỗi IRS | System Model |
| 𝒢, G | tập / số active groups (G ≤ M+1) | System Model |
| 𝒦_g, K_g | tập users trong group g / size | System Model |
| k, m, g | index user / IRS / group | — (⚠ g overload, xem §9) |
| 𝒰, ℛ | tập index user-qubits {0..n_q−M−1} / IRS-qubits {n_q−M..n_q−1} | §IV.C eq:observables (R1) |

## 2. Channel & CSI
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| g^{SU}_k, g^{SR}_m, g^{RU}_{m,k} | true channel 3 link | eq channel (System Model) |
| ĝ (mũ) | estimated channel | eq CSI-error |
| Δg, κ, ε∼𝒞𝒩(0,1) | CSI error model ĝ=g+κ|g|ε | eq CSI-error |
| c | tốc độ ánh sáng | eq channel (⚠ c overload §9) |
| G_S, G_U | antenna gain satellite / user | eq channel |
| f_SU, f_SR | carrier frequency | eq channel |
| d^{SU}_k, d^{SR}_m, d^{RU}_{m,k}, d_0 | khoảng cách link / reference distance | eq channel |
| ε^{SU}_k, ε^{SR}_m, ε^{RU}_{m,k} | rain-attenuation coefficient (lognormal ITU-R) | eq channel (⚠ ε overload §9) |
| Ω_sf, g̃^{sf}_{m,k} | shadow-fading power gain / Rayleigh CN(0,1) | eq channel |
| α | path-loss exponent (RU link) | eq channel (⚠ α overload §9) |
| h | effective channel coefficient (sau IRS cascade) | notation table; SINR eqs |
| h_SR | altitude vệ tinh 800 km | tab:hyper (⚠ đè chữ h) |
| n_0, σ² | AWGN / noise variance | System Model |

## 3. RSMA signal
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| s_k | message user k | System Model |
| s = {s_1..s_K} | TẬP messages (set) | eq đầu System Model |
| s̃ (tilde, bold) | VECTOR messages sau split [s_{c,1..G}, {s_{k,p,g}}] | System Model (renamed 07-18) |
| s_{c,g} | common stream group g | System Model |
| s_{k,p,g} | private stream user k group g | System Model (thống nhất 07-18) |
| s_{k,g}, s_{k,c,g} | biến thể trong Fig RSMA_Encoder + caption (per-user message / per-user common sub-message) | fig:rsma_encoder ⚠ CHƯA định nghĩa formal trong text — nếu muốn sạch: định nghĩa s_k ≡ s_{k,g} khi k∈𝒦_g, và s_{k,c,g} là common sub-message trước joint encoding |
| w_{c,g}, w_{k,p,g}, 𝐰 | scalar precoding coefficients / tập | System Model |
| x, x_g | transmitted signal tổng / group | System Model |
| y_k | received signal user k | System Model |
| φ_k, 𝛗 | IRS-selection variable ∈{0..M} / vector | System Model |
| Φ_m, 𝚽 | phase-shift matrix IRS m / tập | System Model |
| θ_{m,n}, θ^{(q)}, 𝒬_θ, B | phase shift / mức lượng tử q / tập 2^B mức / số bit | System Model (⚠ B, θ overload §9) |

## 4. Rates & QoS
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| γ_{k,c,g}, γ_{k,p,g} | SINR decode common / private | eq SINR |
| R_{k,c,g} | per-user common-stream decoding rate = log2(1+γ_{k,c,g}) | eq common-rate (fixed 07-18) |
| R_{c,g} | group common rate = min_k R_{k,c,g} | eq common-rate |
| R_{k,p,g} | private rate = log2(1+γ_{k,p,g}) | System Model |
| C_{k,g}, 𝐂 | phần common rate cấp cho user k / tập | System Model |
| R_k, R_tot | tổng rate user k = R_{k,p,g}+C_{k,g} / sum-rate | System Model |
| R_k^min | QoS demand | System Model; = D_k trong affinity |

## 5. RL / MDP / Training
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| s_t (bold) | STATE = affinity feature vector (eq:affinity) | §IV.A + §IV.D MDP (renamed 07-18) |
| a_t (bold) | ACTION = (φ,Φ,w,C) | §IV.D MDP |
| r_t | reward = R_tot − λ_D Σ(shortfall/…)² | eq:reward |
| γ_d | discount factor | eq:reward para |
| λ_D, ε_q | QoS-penalty weight / guard | eq:reward |
| J(Θ), Θ | objective / toàn bộ tham số 4 actors | eq:objective |
| β, β_0, β_min | entropy coefficient + schedule | eq:objective (⚠ β overload §9) |
| H(π_j) | entropy actor j | eq:objective |
| Â_t, δ_t, λ_GAE | GAE advantage / TD error / GAE param | eq:gae |
| V(s_t;ψ), ψ | critic / tham số critic | §IV.D |
| G_t | λ-return regression target | eq:critic_loss (⚠ đè G số group!) |
| ρ_t, ε_c | PPO importance ratio / clip range | eq:ppo |
| η = {λ_y, λ_z, θ} | tham số quantum (SPSA) | §IV.D |
| c_s | SPSA perturbation step | eq:spsa (App C dùng ε_SPSA cho CÙNG đại lượng — ⚠ nên thống nhất) |
| Δ, Δ_i | Rademacher direction ±1 | eq:spsa |
| Ĵ_{o,i} | SPSA Jacobian estimate ∂ô_o/∂η_i | eq:spsa (renamed 07-18, hết đè ĝ) |
| S_tr=1500, S_ev=3000 | measurement shots train / eval | §IV.C (⑨) |
| α_AE | AE loss weight | eq AE-loss para |

## 6. AE & VQC
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| a_{k,m} | affinity entry = |ĝ^SR_m|·|ĝ^RU_{m,k}| (SCALAR, khác action bold a_t) | eq:affinity |
| D_k | per-user demand trong state | eq:affinity |
| q_m, p_m | mean affinity IRS m / satellite-IRS power |ĝ^SR_m|² | eq:affinity |
| d_aff = K(M+2)+2M | dim affinity | §IV.B |
| d_s = M(3+2K+2N)+3K | dim raw observation (baseline, không còn symbol riêng) | §IV.B |
| z_t, z̃_t | latent 2n_q / bản layer-normalized | §IV.B, eq:ln_z |
| ŝ_t, s_norm,t | reconstruction / group-normalized input của AE loss | eq:ae_loss (renamed 07-18) |
| L_AE | reconstruction loss | eq:ae_loss (⚠ đè L layers) |
| μ_z, σ_z, ϵ | LN statistics + guard | eq:ln_z |
| 𝛄, 𝛃 (bold) | LN affine học được ∈ ℝ^{2n_q} | eq:ln_z (④) (⚠ γ, β overload §9) |
| λ_y, λ_z | encoding scales ∈ ℝ^{n_q} | eq:encoding_angles |
| α_i, δ_i | rotation angles Ry/Rz = π·tanh(λ·z̃) | eq:encoding_angles (⚠ δ đè TD-error δ_t) |
| θ^y_i, θ^z_i | variational angles | §IV.C |
| L | số layers VQC (2/3/5) | §IV.C |
| n_q | số qubits (12) | §IV.B |
| U_E, U_L, U_ent | encoding / variational / entangling block | §IV.C |
| ô_t, N_Q | measurement vector / số observables =(n_q−M)(M+2)+M | eq:observables (②) |
| h_t = [z_t ∥ ô_t] | head input (skip connection) | §IV.C (⚠ đè h channel) |
| ξ, W, b, β | SoftmaxPQC head / linear map / inverse-temperature | §IV.C (③) |
| π_k | per-user assignment distribution ∈ Δ^M | eq:pi_k |

## 7. Appendix — capacity / Pareto / frozen-λ
| Symbol | Nghĩa | Định nghĩa ở |
|---|---|---|
| A, B | số users group direct / group IRS | App abl_capacity (⚠ B đè bits) |
| Γ = |h|²P_S/σ², Γ̄ | effective SNR / mean (geometry) | App abl_capacity |
| Γ_a^{IRS}, Γ_a^{dir}, Γ_wk, Γ_min, Γ_max | SNR theo route / weakest / min / max | App A+B |
| X_j ∼ Exp(1) | fading draw đơn vị | App abl_capacity |
| m, m_B | group-minimum SNR (⚠ đè index IRS m!) | App abl_capacity |
| Λ | rate của exponential min = Σ 1/Γ̄ | App abl_capacity |
| E_1 | exponential integral | App abl_capacity |
| ς_k (varsigma) | QoS shortfall (R_k^min − R_p(Γ_k))⁺ | App abl_capacity (⑬ renamed 07-18) |
| Δ R_tot(a), Δ_{0→1}, Δ_{B→B+1} | re-assignment marginals | eq:abl_dRtot (⚠ Δ đè Rademacher) |
| B*_rate(A), B*_QoS(A), B*(A) | composition thresholds | App abl_capacity, eq:abl_bstar |
| n*_mix | equal-share capacity R_c/(R^min − R_p) | App abl_capacity |
| x_k, y_g | power fractions w/P_S | App abl_pareto (⚠ x đè transmit signal, y đè received!) |
| V*_rate | rate-max endpoint E[log2(1+Γ_max)] | eq:abl_vrate |
| R(m), R*(m) | equal-split m-user sum-rate / best sum-rate serving m | eq:abl_concentration (⚠ m = số user served, đè index IRS) |
| λ_crit, p̂ | critical penalty weight / per-user penalty ≈0.98 | eq:abl_lcrit |
| ρ_λ | relative drift λ (0.14) | eq:abl_rho (⚠ đè ρ_t PPO ratio) |
| ε_SPSA | =c_s=0.1 (App C gọi tên khác) | eq:abl_rho ⚠ |
| ĝ_λ, ĝ_b | λ-gradient batch/per-sample (App C) | ⚠ đè ĝ estimated channel — cân nhắc đổi ĝ→ 𝐮 hoặc ghi rõ context |
| c (rescale), c (attenuation) | λ-rescale factor / encoder attenuation ∈{1..0} | App abl_frozen ⚠ |
| c_ω, ω, Ω | Fourier coefficient / frequency / spectrum | Lemma 2 |
| d_h | hidden width classical block | Prop 1 |
| n_q^{raw} = ⌈d_aff/2⌉ | qubits nếu không AE (9/22/41) | tab:abl_ae_params |
| α_pf | power-fairness blend | App abl_power |
| p, p*, p^eff, (p_s,p_c,p_p) | power probs / target / executed / split-common-private | App abl_power (⚠ p đè p_m) |

## 8. Ký hiệu CHƯA có trong bảng notation của paper (tab:variables)
s_t (state/affinity) · a_t (action) · z_t · a_{k,m} · D_k · q_m · p_m · d_aff · d_s · λ_y/λ_z ·
α_i/δ_i · θ^y/θ^z · L · n_q · ô_t · N_Q · 𝒰/ℛ · h_t · ξ/W/b/β(head) · 𝛄/𝛃 (LN affine) ·
S_tr/S_ev · α_pf · toàn bộ ký hiệu §7 (appendix — chấp nhận được vì appendix tự khai).
→ Đề xuất: thêm ~10 ký hiệu RL/VQC hay dùng nhất vào tab:variables (s_t, a_t, z_t, d_aff, n_q, L, ô_t, N_Q, η, β).

## 9. ⚠ BẢNG OVERLOAD (mức độ + đề xuất)
| Chữ | Các nghĩa đang dùng | Mức | Đề xuất |
|---|---|---|---|
| **g** | group index g∈𝒢 · channel g^{X} (có superscript) | TRUNG | Giữ (superscript phân biệt); ghi chú trong tab:variables "channel coefficient (with link superscript)" |
| **s** | s_k message · s set · s̃ split vector · s_{c,g}/s_{k,p,g} streams · **s_t state** · ŝ_t recon | TRUNG-CAO | Đã tách bằng subscript/decoration; chấp nhận được vì s_t luôn bold+t. KHÔNG dùng thêm s mới |
| **β** | entropy coef β · **LN affine 𝛃** · **head inverse-temperature β** · (β_0/β_min) | **CAO** — 3 nghĩa trong CÙNG khối VQC/training | Cân nhắc: head β → β_T (temperature) hoặc τ⁻¹; LN affine 𝛃 → 𝐛_LN. Tối thiểu: chú thích khi xuất hiện |
| **γ** | SINR γ_{k,·,g} · discount γ_d · **LN affine 𝛄** | TRUNG | γ_d có subscript; LN 𝛄 bold — chấp nhận, chú thích |
| **λ** | encoding scales λ_y/λ_z · λ_D · λ_GAE · Exp rate λ (E_1 formula) · λ_crit | TRUNG | Subscript đủ phân biệt |
| **α** | path-loss α · encoding angle α_i · α_AE · α_pf | TRUNG | Subscript đủ |
| **ε** | rain ε^{X} · CSI ε∼CN · LN guard ϵ · PPO clip ε_c · QoS guard ε_q · ε_SPSA | TRUNG-CAO | **ε_SPSA ≡ c_s: THỐNG NHẤT về c_s** (1 edit App C). Rain ε^{X} có superscript |
| **c** | speed of light · c_s SPSA · rescale factor c · attenuation c · Fourier c_ω | TRUNG | Speed of light chỉ xuất hiện 1 eq; appendix c local-scope |
| **B** | phase bits B · group count B, B* | TRUNG | Khác section; tab:variables ghi rõ |
| **Δ** | CSI Δg · Rademacher Δ · marginals Δ_{B→B+1} · Δ(c) | THẤP | Context đủ |
| **m** | IRS index m · **group-min SNR m/m_B** · **số users served R(m)** | TRUNG-CAO | Appendix-local; nếu muốn sạch: group-min → μ_B, served-count → n (R(n)) |
| **G** | số groups G · gains G_S/G_U · G_t λ-return | TRUNG | Subscript đủ |
| **h** | effective channel h · h_t head input · h_SR altitude | TRUNG | h_t bold+t; altitude chỉ trong bảng |
| **ĝ** | estimated channel · **ĝ_λ/ĝ_b gradient App C** | TRUNG | App C local; cân nhắc đổi gradient → 𝐮_b |
| **L** | VQC layers L · L_AE | THẤP | Subscript đủ |
| **x, y** | signal x/x_g · received y_k · **power fractions x_k, y_g App B** | TRUNG | App-local; nếu muốn sạch: fractions → ϑ_k, ϱ_g |
| **p** | p_m IRS power stat · power probs (p_s,p_c,p_p)/p^eff · p̂ penalty | TRUNG | Context đủ |
| **δ** | TD error δ_t · encoding angle δ_i | THẤP | Subscript đủ |
| **θ** | phase θ_{m,n} · variational θ^y/θ^z | THẤP | Superscript đủ |

**3 việc đáng làm nhất nếu muốn siết notation** (chờ user duyệt, chưa sửa):
1. ε_SPSA → c_s (App C, 1 chỗ) — thống nhất với §IV.D.
2. Head inverse-temperature β → β_T (2-3 chỗ §IV.C + eq:pi_k) — gỡ overload nặng nhất.
3. Thêm ~10 ký hiệu RL/VQC vào tab:variables.
