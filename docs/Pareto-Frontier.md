# PARETO FRONTIER — V*(λ) CLOSED-FORM WORKSHEET

> TÁCH từ docs/IRS-Group-Capacity-Ablation.md §6 ngày 2026-07-19 (yêu cầu user). Nội dung giữ
> NGUYÊN VĂN (kể cả ghi chú "GIỮ SỐ CŨ cho refs"); ký hiệu shortfall đã đổi ς theo paper.
> Paper counterpart: Appendix B (Rate--QoS Pareto Frontier), đã expand 07-18.

## §6. V*(λ) + PARETO FRONTIER CLOSED-FORM [2026-07-03 — chủ đề riêng, GIỮ SỐ CŨ cho refs]

> ⭐ **REFRESH 07-11 (paper-anchor regimes)** — probe re-run tại đúng env bảng chính:
> `vstar_c1_ramp03_500states.txt` (r111/ep1000) + `vstar_c2_ramp05_500states.txt` (r110/ep1600).
> KEY: ## ⭐⭐⭐ TRUE (QoS, R_tot) FRONTIER — MEASURED 2026-07-22, post-refactor

Tool: `analysis/probe_true_frontier.py --K .. --M .. --P 50 --R-LoS 0.5`
Method: routing **coordinate ascent** (started from the AO solution, soft-constrained)
+ per-element oracle phase + **(β, f) power sweep** + demand-fill C_k,
maximising R_tot **subject to QoS ≥ t**. Scored like `env.step`
(decide on ĝ, score on true g, 16 σ² draws).

⚠ Why this and not the AO probe: `probe_ao_rate` is **structurally limited** — it only
sweeps the split `f` and always uses EQUAL private power (β=0), and its `oracle_ck_met`
is hard-coded QoS-first. That is why sweeping λ_D from 0 → 8 gives AO the *same* point
every time: it cannot enter the rate-heavy region at all. It is a strong baseline, not a
ceiling.

| QoS ≥ | **Case 1** (K5/M1) max R_tot | vs AO 1.847 | **Case 2** (K10/M2) max R_tot | vs AO 1.651 |
|---|---|---|---|---|
| 100% | **2.714** | **+0.867 (+47%)** | **1.662** | **+0.011 (+0.7%)** |
| 99%  | 2.714 | +0.867 | 1.662 | +0.011 |
| 95%  | 2.714 | +0.867 | 1.662 | +0.011 |
| 90%  | 2.714 | +0.867 | 1.688 | +0.037 |
| 80%  | 2.956 | +1.110 | 1.773 | +0.122 |
| 50%  | 3.362 | +1.515 | 2.094 | +0.443 |
| 0%   | 5.271 | +3.425 | 5.089 | +3.438 |

### The two cases are qualitatively different — this drives paper strategy
- **Case 1: AO leaves 47% on the table.** Huge headroom at full QoS. Matches the
  exhaustive check (all 2^5=32 assignments): AO's greedy routing ascent found the true
  optimum in **0% of states**. This is where "the learned policy beats the per-state
  solver" can be claimed strongly.
- **Case 2: AO is within 0.7% of the ceiling.** Almost nothing to win on rate at high QoS.
  Realistic claim here is *parity with AO at ~1000× faster inference*, not "beats AO".

### Shape: hyperbolic, but FLAT in the high-QoS region
Both cases: R_tot climbs steeply only once QoS is allowed to fall below ~80%
(C2: 1.662 → 5.089 as QoS 100% → 0%). Between QoS 95% and 100% the frontier is
essentially flat. Physical reason: the link is **interference-limited** — concentrating
power barely raises SINR (interference rises with it) while starving weak users collapses
QoS immediately.

⇒ **A target of "AO + 0.2 R_tot at comparable QoS" is infeasible for Case 2**
(ceiling is AO + 0.011) but is **easily inside the frontier for Case 1** (AO + 0.867).

V*_rate 10.963/11.723 (formula ≡ oracle, verify_vstar_formulas UPDATED) · R*(K) 3.11/1.61 ·
> **λ_crit 2.00/1.15 → "2 phía của λ=1.5" GIỮ NGUYÊN ở regime mới** · identity C2: 1.520+0.090=1.610
> (khớp 3 chữ số) · attainment matched-QoS ĐỔI: C1 ≈70%/70% (frontier ramp0.3 CAO hơn: 2.546@99.7 vs
> agent 1.79 — gap vẫn trục power-concentration) · C2 95%/94% (row 1.610@99.3). Paper tab:abl_pareto
> đã dùng số MỚI; các mục 6.x dưới đây giữ số regime cũ (07-03) làm lịch sử.

Probe: `analysis/probe_v_star.py` (per-state 4-var oracle — assignment exhaustive
2^5 @C1 / hill-climb 1-flip @C2 + phase exact per-IRS + power grid + WTA endpoint
+ Ck closed-form; select nominal σ², re-score 16 draws). 500 states/case, logs
`analysis/data/vstar_c{1,2}_ramp0{5,3}_500states.txt`. Agent refs: infer greedy
50ep s42 @λ1.5 (infer KHÔNG restore λ_D → mọi run chấm @params.py λ=1.5).

### 6.1 SINR tổng quát (mở rộng §1 khỏi equal-power)

Fractions $x_k = w_{p,k}/P_S$, $y_g = w_c^g/P_S$, $\sum x + \sum y \le 1$:

$$
\mathrm{SINR}_{p,k} = \frac{x_k\,\Gamma_k}{\Gamma_k\big(1 - x_k - y_{g(k)}\big) + 1},
\qquad
\mathrm{SINR}_{c,k} = \frac{y_{g(k)}\,\Gamma_k}{\Gamma_k\big(1 - y_{g(k)}\big) + 1}
$$

(§1 = trường hợp $x_k{=}b, y_g{=}a$. Mọi công thức dưới đây suy từ 6.1 + §0.3.)

### 6.2 Endpoint rate-max — SINGLE-USER ESCAPE (biên phải frontier)

$x = e_{k^*},\ y=0$ → interference nội-hệ $=0$ → SINR $= \Gamma$ (thoát interference-limit;
**cách duy nhất**: mọi cấu hình ≥2 stream bị chặn bởi 6.3):

$$
\boxed{\;V^*_{\text{rate}} = \mathbb{E}\big[\log_2(1+\Gamma_{\max})\big],\qquad
\Gamma_{\max} = \max_k \max_{\text{link}} |h_k|^2 P_S/\sigma^2\;}
$$

**Validated EXACT**: formula 10.934 (C1) / 11.712 (C2) vs probe λ=0 row **10.9345 / 11.7122**
(analysis/verify_vstar_formulas.py; pre-WTA-patch probe chỉ đạt 7.2/6.8 = grid artifact β≤16).

### 6.3 Concentration identity (saturated) — mức frontier phẳng

$\Gamma\to\infty$, private pool, own-common SIC'd: $R_{p,k} = \log_2\frac{1}{1-\tilde x_k}$ →

$$
\sum_k R_{p,k} = -\log_2 \prod_k (1-\tilde x_k)
\quad\xrightarrow{\text{equal-split } m}\quad
R(m) = m\log_2\frac{m}{m-1}\ \downarrow\ \frac{1}{\ln 2} \approx 1.4427
$$

- $m{=}2$: 2.0 · $m{=}5$: 1.610 · $m{=}10$: **1.520** — khớp C2 serve-all frontier
  1.6325 = 1.520 + $\sum_g R_c \approx 0.11$ ✓.
- Skew 2-user: $-\log_2 x(1-x)$ không chặn khi $x\to1$ nhưng hội tụ về 6.2 (finite-Γ cắt).

### 6.4 V*(λ) piecewise + λ_cliff (park-hard-zero)

Hard-park 1 user = penalty $\hat p = (D_k/(D_k{+}\epsilon))^2 \approx 0.9804$ mỗi user:

$$
V^*(\lambda) = \max_{m\in\{1..K\}} \big\{ R^*(m) - \lambda\,(K{-}m)\,\hat p \big\},
\qquad
\boxed{\;\lambda_{\text{cliff}} = \frac{V^*_{\text{rate}} - R^*(K)}{(K-1)\,\hat p}\;}
$$

với $R^*(1) = V^*_{\text{rate}}$ (6.2), $R^*(K)$ = serve-all frontier (6.6).

| | $\lambda_{\text{cliff}}$ formula | probe transition | @λ=1.5 argmax |
|---|---|---|---|
| C1 (K5, $R^*(K){=}3.05$)  | **2.01** | giữa λ2 (mixed 7.35@41%) và λ3 (serve) ✓ | **WTA-vùng: V*(1.5)=5.095** @21.7% QoS |
| C2 (K10, $R^*(K){=}1.63$) | **1.14** | giữa λ1 (WTA 2.90) và λ1.5 (serve) ✓ | **serve-vùng: V*(1.5)=1.6486** @81.5% |

⟹ **λ=1.5 nằm DƯỚI cliff ở C1 nhưng TRÊN cliff ở C2** — cùng λ, 2 case ở 2 phía ranh
giới degenerate↔serve. %V* của agent (C1 34.7% vs C2 86-89%) phản ánh vị trí cliff,
KHÔNG phản ánh chất lượng agent — so sánh vận hành phải dùng frontier matched-QoS (6.6).

### 6.5 Soft-QoS parking (waterfill) — vì sao argmax không bao giờ 100% QoS

Penalty biên của shortfall $t$: $2\lambda t/(D_k{+}\epsilon)^2 \to 0$ khi $t\to0$ →
**near-miss ~ FREE**. Ck reward-optimal = waterfill (exact cho penalty bậc 2):
mực nước $t^*$: $\sum_i (ς_i - t^*)^+ = R_c^g$, $\text{pen}_{\min} = \sum_i \min(ς_i,t^*)^2/(D_k{+}\epsilon)^2$.
Nghiệm optimal "đỗ" user hụt sát dưới $D_k$ → QoS-count fail nhưng pen ≈ 0
(C1 λ∞ argmax vẫn chỉ 92% noisy-QoS; quadratic không ép được 100%).

### 6.6 Frontier dichotomy — hệ quả trực tiếp của $n^*_{\text{mix}}$

Serve-all + concentration đòi common cõng các user bị rút private:
$(n_g - m_g)\,D_k \le R_c(\Gamma_{\min}^g)$ per group — chính là **general Σ-shortfall bound §1**
(trường hợp full-shortfall members $ς_k = D_k$; bound general §1 thống nhất 6.6 với QoS bound).

- **C1** ($R_p(\Gamma_{\text{wk}}) \ge D_k$, regime cap$=K$ §1): private-floor tự cõng QoS
  → private TỰ DO concentrate (6.3 skew) → **frontier CAO**: serve-all $R^*(K){=}3.05$;
  robust-δ: (99.7%, 2.54) · (99.8%, 2.01) · (100%, 1.78); $\delta_{\max} \ge 0.10$.
- **C2** (knife-edge $n^*_{\text{mix}}{\approx}3.8 \approx$ group load): không slack →
  $x$ buộc ~equal → **frontier PHẲNG** = 6.3: 70%→99.8% QoS chỉ 1.77→1.63;
  $\delta_{\max} < 0.10$ (probe δ0.10 attain 0%) — margin ceiling = knife-edge bằng số.

### 6.7 Agent gaps (@λ1.5 · infer greedy 50ep s42 · per-step)

| | V*(1.5) | VQC | DNN | %V* VQC/DNN | frontier matched-QoS VQC/DNN |
|---|---|---|---|---|---|
| C1 r0.5 (r45/r106) | 5.095 | 1.7667 | 1.7677 | 34.7% / 34.7% (gap TRÙNG) | **88%** (1.77 vs 2.01@99.8) / **61%** (1.81 vs 2.97@94.9) |
| C2 r0.3 (r99/r98)  | 1.6486 | 1.4743 | 1.4214 | **89.4% / 86.2%** | ~93% (1.53 vs ~1.64@96.8) / ~93% (1.52 vs ~1.64@95.8) |

- Gap C1 sống ở trục POWER-concentration (6.3 — α-blend đè về equal); gap C2 sống ở
  **Ck demand-fill + QoS-tail** (recipe oracle-asg+equal+wf-Ck = 1.5349@99.8% dominate cả 2 agent).
- ⚠ V* = achievable lower-bound (power grid + local-search); 6.2 chứng minh endpoint đã TIGHT
  (formula=probe 4 chữ số); serve-region còn có thể cao hơn chút → gap thật ≥ số báo.
