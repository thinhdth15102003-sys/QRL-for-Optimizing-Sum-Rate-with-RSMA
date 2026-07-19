# IRS GROUP-CAPACITY MATH ABLATION — worksheet (2026-07-03 · RESTRUCTURED 07-10)

> Goal: derive the group-composition threshold — given input $(N, P_S, \text{position}, \sigma^2)$,
> at what point does a group with $A$ blocked + $B$ non-blocked users see its $R_{\text{tot}}$
> start to DECREASE. Extends formula [N] (Common-Knowledge) along 2 axes:
> (i) finite-SNR (bring $\sigma^2$ back → $N, P_S$, position appear explicitly);
> (ii) mixed group $(A,B)$ instead of a homogeneous $n$.
> All §0 formulas mirror the code EXACTLY (CSI/rate.py + istn/channel.py) — validate before deriving.
>
> **Structure [update 07-19]**: §0 model → **§1 THEORY** (assumptions ledger + toàn bộ lý thuyết) →
> **§2 EXPERIMENTS** (validation chain, kết bằng controlled rig) → **§3 VARIABLE-B SEAT LAW** (07-18)
> → **§4 B* THEO HYPER** (07-19: 2 dạng công thức + hyper map + verdict C_k) → §6 V*(λ)+Pareto
> **ĐÃ TÁCH → docs/Pareto-Frontier.md** (pointer cuối file). Shortfall ký hiệu ς_k (khớp paper).

---

## §0. SYSTEM MODEL (exact — matches the code)

### 0.1 Channel gains (istn/channel.py)

Sat→IRS (scalar, far-field), sat→user, IRS→user:

$$
g_{SR}^{(m)} = \frac{c\sqrt{G_S\,\varepsilon_{\text{rain}}}}{4\pi f\, d_{SR}}\,e^{j\phi},
\qquad
g_{SU}^{(k)} = \frac{c\sqrt{G_S G_U\,\varepsilon}}{4\pi f\, d_{SU,k}}\,e^{j\phi}
\;\times\;
\underbrace{\beta_{\text{blk}}}_{\text{if blocked, }=0.1}
$$

$$
g_{RU}^{(m,k)} = \sqrt{G_U\, g_{sf}\,\Big(\tfrac{d_0}{d_{mk}}\Big)^{\alpha}\,\varepsilon}\;\cdot\;
\mathcal{CN}(0,1), \qquad \alpha = 3.5
$$

CSI error (all links): $\hat g = g + \kappa\,|g|\,\delta,\ \delta\sim\mathcal{CN}(0,1),\ \kappa=0.05$.

Noise per step: $\sigma^2 = 10^{\mathcal{N}(-30,\,10)/10}$ (dBW). Power: $W_p + W_c \le P_S$.

### 0.2 Effective channel (rate.py::effective_channels_all)

$$
h_k =
\begin{cases}
\beta_m\,\overline{\hat g_{SR}^{(m)}}\,\Big(\sum_{n=1}^{N}\varphi_n\Big)\,\hat g_{RU}^{(m,k)}
  & \text{user } k \text{ on IRS } m\\[4pt]
\hat g_{SU}^{(k)} & \text{user } k \text{ direct}
\end{cases}
$$

Since the cascade phase is **common to all elements** ($g_{SR}$ scalar) and the SINR uses only
$|h_k|^2$ (the phase of $\sum\varphi_n$ is irrelevant): **oracle = all elements at ANY common index**
→ $\big|\sum_n \varphi_n\big| = N$ **EXACTLY, no quantization loss** [verified probe 07-03]:

$$
|h_k^{\text{IRS}}|^2 = \beta_m^2\,|\hat g_{SR}|^2\,N^2\,|\hat g_{RU}^{(m,k)}|^2
\qquad\text{(oracle)};\qquad
|\Sigma\varphi|_{\text{random}} \approx \sqrt{N}\ \ (\text{random-walk — source of servable-undercount})
$$

⭐ **N enters the model ONLY through the $N^2$ array gain** — there is NO phase-sharing penalty
among users on the same IRS (probe_irs_overassign confirms $|\Sigma\varphi|=N$ for ALL users at once).
(The quantization factor $q=\sin(\pi/L)/(\pi/L)\approx 0.90$ applies ONLY to a vector-channel model
with per-user phase mismatch — NOT to this scalar model.)

### 0.3 SINR + rates (rate.py::_sinr_all, per-group SIC)

Group $g(k)$, group common power $w_c^{g}$, $W_p=\sum_k w_{p,k}$, $W_c=\sum_g w_c^g$:

$$
\mathrm{SINR}_{p,k} = \frac{|h_k|^2\, w_{p,k}}
{|h_k|^2\big(W_p - w_{p,k} + W_c - w_c^{g(k)}\big) + \sigma^2},
\qquad
\mathrm{SINR}_{c,k} = \frac{|h_k|^2\, w_c^{g(k)}}
{|h_k|^2\big(W_p + W_c - w_c^{g(k)}\big) + \sigma^2}
$$

$$
R_{p,k} = \log_2(1+\mathrm{SINR}_{p,k}),
\qquad
\boxed{\;R_c^{g} = \log_2\Big(1 + \min_{k\in g}\ \mathrm{SINR}_{c,k}\Big)\;}
\qquad(\text{MIN = the main drag mechanism})
$$

$$
R_{\text{tot}} = \sum_k \big(R_{p,k} + C_k\big),
\qquad \sum_{k\in g} C_k = R_c^{g}\ \ \forall g,
\qquad \text{QoS}_k:\ R_{p,k}+C_k \ge D_k
$$

Grouping: direct users = group 0; each active IRS = 1 group → $G+1$ common streams.

---

## §1. THEORY — GROUP-COMPOSITION THRESHOLD $B^*$ [gộp §1+§2+§2c+§5.1 cũ, 07-10]

> **⚑ ASSUMPTIONS LEDGER [thêm 07-19 — áp cho TOÀN BỘ §1-§4 trừ khi nói khác]:**
> (A1) weights $(\alpha_k, \beta_g, \rho)$ CỐ ĐỊNH qua mọi re-assignment (isolation; equal-power
> = numerical reference). (A2) fading = Rayleigh cascade thuần → $X\sim$Exp(1) iid; shadowing/κ
> fold vào $\bar\Gamma$ ($\Lambda$ đổi, machinery giữ). (A3) scalar cascade $|\Sigma\varphi|=N$
> EXACT (không phase-sharing penalty — §0.2, verified). (A4) shortfall ký hiệu $ς_k$ (khớp paper
> App A; đổi từ s_k 07-19). (A5) $D_k$ đồng nhất 0.1. Mục nào validated có đánh dấu probe;
> mục chưa chạy = hypothesis.

Reparametrize the §0.3 model with power fractions and a per-user effective SNR, so that
$N$, $P_S$, position, and blockage appear explicitly (in §0.3 they are hidden inside $|h_k|^2$).

**Power allocation** (common-split $\rho{=}0.2$ between the common and private budgets;
per-user private weights $\alpha_k$, per-group common weights $\beta_g$):

$$
w_{p,k} = \alpha_k (1-\rho)\,P_S,\ \ \sum_{k=1}^{K}\alpha_k = 1;
\qquad
w_c^{g} = \beta_g\,\rho\,P_S,\ \ \sum_{g=0}^{G}\beta_g = 1
$$

so the private fraction of user $k$ is $\alpha_k(1-\rho)$ and the common fraction of group $g$
is $\beta_g\rho$ (equal power $\Rightarrow \alpha_k{=}1/K,\ \beta_g{=}1/(G{+}1)$).

**Per-user effective SNR** (folds $N$, $P_S$, position, blockage into one scalar):

$$
\Gamma_k = \frac{|h_k|^2\,P_S}{\sigma^2}
$$

**Per-user SINR** (divide the §0.3 SINR by $|h_k|^2 P_S$, so $\sigma^2/|h_k|^2P_S = 1/\Gamma_k$):

$$
\gamma_{k,p,g} = \frac{\alpha_k(1-\rho)}{1 - \alpha_k(1-\rho) - \beta_g \rho + \frac{1}{\Gamma_k}},
\qquad
\gamma_{k,c,g} = \frac{\beta_g \rho}{1 - \beta_g \rho + \frac{1}{\Gamma_k}}
$$

**Per-user rates:**

$$
R_{k,p,g} = \log_2(1+\gamma_{k,p,g}),
\qquad
R_{k,c,g} = \log_2(1+\gamma_{k,c,g})
$$

**Group common (min), sum-rate, QoS** (as in §0.3): the group common rate is set by its worst
member and shared as $C_{k,g}$; each user's total is private plus its common share:

$$
R_c^{g} = \min_{k\in g} R_{k,c,g},
\qquad
\sum_{k\in g} C_{k,g} = R_c^{g},
\qquad
R_k = R_{k,p,g} + C_{k,g}
$$

$$
R_{\text{tot}} = \sum_{g} R_c^{g} + \sum_{k} R_{k,p,g},
\qquad
\text{QoS}_k:\ R_k \ge D_k
$$

**Notation (function form — used throughout).** When the indices are clear from context we
drop them and write

$$
R_p(\Gamma) \equiv R_{k,p,g}\big|_{\Gamma_k=\Gamma},
\qquad
R_c(\Gamma) \equiv R_{k,c,g}\big|_{\Gamma_k=\Gamma},
$$

i.e. the per-user rates viewed as functions of the effective SNR $\Gamma$ alone, with the weights
$(\alpha_k, \beta_g, \rho)$ FIXED (per the power setting below). Min/two-point arguments plug the
appropriate $\Gamma$ (e.g. $R_c(\min_k \Gamma_k)$ = the group common rate).

**Setup.** One IRS m with group G = {0, 1} (0 is direct, 1 is IRS-link), we got A users currently in direct group 0 and B users in IRS group 1 with K = A + B, the re-assign user from A to B called user a. We will calculate how much sum-rate will contribute to total archievable sum-rate $R_{tot}$ when adding free-users to IRS group and find the threshold when it becomes disavantage.

**Power setting (generalization — no power-fairness).** The parametrization above is kept as-is:
the weights $(\alpha_k, \beta_g, \rho)$ are GIVEN and held **fixed** across the move — power is never
re-optimized and no power-fairness blending is applied. Re-assignment changes only user $a$'s route
($\Gamma_a^{\text{dir}} \to \Gamma_a^{\text{IRS}}$) and its common budget
($\beta_0\rho P_S \to \beta_1\rho P_S$); the totals $W_p, W_c$ are unchanged (all $\alpha_k,\beta_g$
kept), so $\Delta R_{\text{tot}}$ isolates the **pure re-grouping effect**. Equal power
($\alpha_k{=}1/K$, $\beta_g{=}1/(G{+}1)$) is only the numerical reference point used in the
experiments (§2).

**Sum-rate of entire system before re-assignment**:

$$
R_{\text{tot}}(A,B) = \sum_{i\in A} R_p(\Gamma_i^{dir}) + \sum_{j\in B} R_p(\Gamma_j^{IRS}) + R_{c,A}^{old} + R_{c,B}^{old}
$$

**Sum-rate of entire system after re-assignment**:

$$
R_{\text{tot}}(A-1,B+1) = \sum_{i\in A/a} R_p(\Gamma_i^{dir}) + \sum_{j\in B} R_p(\Gamma_j^{IRS}) + R_{a,p,B}(\Gamma_a^{IRS})  + R_{c,A}^{new} + \min(R_{c,B}^{old}, R_{a,c,B})
$$


**Threshold condition** (formal definition, stated under the general power parametrization —
$(\alpha_k,\beta_g,\rho)$ fixed, per the power setting above):

$$
\Delta R_{\text{tot}}(a) \;=\;
\underbrace{R_{a,p,B}(\Gamma_a^{IRS}) - R_{a,p,A}(\Gamma_a^{dir})}_{\text{(I) private re-route of }a}
\;+\;
\underbrace{R^{new}_{c,A} - R_{c,A}^{old}}_{\text{(II) direct-group common relief}}
\;+\;
\underbrace{\min\big(R_{c,B}^{old},\, R_{a,c,B}(\Gamma_a^{IRS})\big) - R_{c,B}^{old}}_{\text{(III) IRS-group min-drag}}
$$

Convention: $\Delta R_{\text{tot}} = R_{\text{tot}}(\text{after}) - R_{\text{tot}}(\text{before})$ = the sum-rate
**GAIN** by moving user $a$ (direct $A$ → IRS $B$). The move HELPS iff $\Delta R_{\text{tot}} > 0$; it turns
into a **disadvantage** once $\Delta R_{\text{tot}} < 0$. The three terms:

**(I) Private re-route of user $a$:** $ R_{a,p,B}(\Gamma_a^{\text{IRS}}) - R_{a,p,A}(\Gamma_a^{\text{dir}}) $ —
$a$'s private rate as a IRS user (after) minus as an direct user (before). Two things change with the route:
its SNR — $\Gamma_a^{\text{dir}}$ (direct link) vs $\Gamma_a^{\text{IRS}}$ (cascade, $N^2$ gain) — and, through
the $\gamma_{k,p,g}$ denominator, its own-group common weight $\beta_0 \to \beta_1$ (SIC of the own-group
common stream).

- $> 0$: IRS private higher (cascade wins) → the move GAINS private → favors re-assignment.
- $< 0$: direct private higher → the move LOSES private → against re-assignment.
- $= 0$: private-neutral.

**(II) Direct-group common change ($A$ = group 0):** $R^{new}_{c,A} - R_{c,A}$ — direct-group common
after minus before $a$ leaves. Removing a member can only keep or RAISE a $\min$-SINR common ($\min$ over a
smaller set).

- $= 0$: $a$ was not the worst common member of $A$ → the min is unchanged.
- $> 0$: $a$ WAS the worst → $A$'s common rises to the next-worst → a gain.
- ⚠ **Cliff:** if $a$ is the LAST direct user, group $A$ vanishes. Term (II) = $- R_{c,A}$.

**(III) IRS-group common change ($B$ = group 1) — the min-drag:** $\min(R_{c,B}^{old}, R_{a,c,B}(\Gamma_a^{IRS})) - R_{c,B}^{old}$
IRS-group common after minus before $a$ joins. Adding a member can only keep or LOWER a $\min$-SINR common →
this term $\le 0$ = **drag** (against the move).

- $= 0$: $R_{a,c,B} \ge R_{c,B}^{old}$ → $a$ is not the worst in the IRS group → no drag.
- $< 0$: $R_{a,c,B} < R_{c,B}^{old}$ → $a$ becomes the new worst → IRS common drops to $R_{a,c,B}$, hurting ALL
  $B$ members at once.

**Definition (sum-rate carrying capacity).** Admit free users one by one while the marginal move
still pays; $N$, $P_S$, and positions enter ONLY through the $\Gamma$'s:

$$
\boxed{\;B^*_{\text{rate}}\big(A;\ \bar\Gamma_A,\, \bar\Gamma_B,\, \Gamma_{\text{dir}}\big)
= \max\big\{B \,:\, \Delta_{B-1\to B} \ge 0\big\}\;}
$$

Evaluated at TWO levels, derived together right below — **(i) two-point skeleton** (deterministic)
and **(ii) expectation under fading** (order statistics) — plus the **QoS bound**
$B^*_{\text{QoS}}$ (general, no share assumption). The binding threshold
$B^* = \min(B^*_{\text{rate}}, B^*_{\text{QoS}})$ is stated ONCE at the end, when both pieces exist.

**Fading model of the group min (order statistics).** The cascade SNR factorizes (from §0.1–0.2:
$|g_{RU}|^2 \sim \mathrm{Exp}$, Rayleigh power):

$$
\Gamma_j = \bar\Gamma_j\, X_j,\qquad X_j \sim \mathrm{Exp}(1)\ \text{i.i.d.},
$$

where $\bar\Gamma_j$ = mean SNR folding $N^2$, $P_S$, $d_{mk}^{-\alpha}$ (deterministic given
positions). ⭐ Blockage does NOT enter the cascade ($\beta_{\text{blk}}$ hits only $g_{SU}$) → on
the IRS route, blocked vs free users differ ONLY through $\bar\Gamma$ (spawn geometry: blocked
in/near the IRS footprint → nearer → $\bar\Gamma_A \ge \bar\Gamma_B$ typically) and the draw $X_j$.
Group = $A$ blocked (means $\bar\Gamma_{A,i}$) + $B$ free (means $\bar\Gamma_{B,j}$); $m = \min$
over members. **Exact exponential facts (heterogeneous, no approximation):**

$$
m \sim \mathrm{Exp}(\Lambda),\quad
\Lambda = \sum_{i\in A}\frac{1}{\bar\Gamma_{A,i}} + \sum_{j\in B}\frac{1}{\bar\Gamma_{B,j}},
\qquad
E[m] = \frac{1}{\Lambda},
\qquad
P(m = R_{k,c,g}) = \frac{1/\bar\Gamma_k}{\Lambda}.
$$

**Marginal of adding one free user — both levels at once** (equal-power reference point; the
general-$(\alpha,\beta)$ threshold above stays as defined). Two-point $R_{\text{tot}}$ snapshot
(each group contributes its common exactly once, regardless of $n$):

$$
R_{\text{tot}}(A,B) = A R_p(\Gamma_A) + B R_p(\Gamma_B) + (K{-}A{-}B) R_p(\Gamma_{\text{dir}})
+ R_c\big(\min(\Gamma_A,\Gamma_B)\big) + R_c(\Gamma_{\text{dir}})
$$

*(i) two-point skeleton* — representative $\Gamma_A, \Gamma_B$ deterministic (giả sử
$\Gamma_B < \Gamma_A$): the first admit carries a one-time min-drag, from the 2nd on it is constant:

$$
\Delta_{0\to1} = \big[R_p(\Gamma_B) - R_p(\Gamma_{\text{dir}})\big]
+ \underbrace{\big[R_c(\Gamma_B) - R_c(\Gamma_A)\big]}_{\text{min-drag, once}},
\qquad
\Delta_{B\to B+1}\big|_{B\ge1} = R_p(\Gamma_B) - R_p(\Gamma_{\text{dir}})\ \ (\text{constant in } B)
$$

*(ii) expectation under fading* — per-realization NEITHER the one-time drag NOR the zero later
drag is certain (order-statistics events of the min): the drag fires w.p.
$(1/\bar\Gamma_B)/\Lambda_{B+1}$ with magnitude decaying in $B$, and every term is closed-form
via $E_1$ (machinery right below):

$$
E[\Delta_{B\to B+1}] =
\underbrace{E\big[R_p(\Gamma_{B+1})\big] - R_p(\Gamma_{\text{dir}})}_{\text{private (own draw)}}
+ \underbrace{E\big[R_c(m_{B+1})\big] - E\big[R_c(m_B)\big]}_{<0,\ \downarrow 0\ \text{(drag)}}
\;\;\Longrightarrow\;\; \textbf{INCREASING in } B.
$$

**Drag-fire probabilities (replaces "once" / "never").** Two-point the MEANS
($\bar\Gamma_{A,i} \equiv \bar\Gamma_A$, $\bar\Gamma_{B,j} \equiv \bar\Gamma_B$).
First time adding a direct-link user to IRS group:

$$
P\big(\Gamma_b < \min_{i\in A}\Gamma_i\big)
= \frac{1/\bar\Gamma_B}{1/\bar\Gamma_B + A/\bar\Gamma_A}
= \frac{\bar\Gamma_A}{\bar\Gamma_A + A\,\bar\Gamma_B} \;<\; 1
$$

Second time and so on adding a direct-link user to IRS group:

$$
P(\Gamma_b < \min_{i\in A,B}\Gamma_i\big)
= \frac{1/\bar\Gamma_B}{(B{+}1)/\bar\Gamma_B + A/\bar\Gamma_A}
\;\xrightarrow{\ \bar\Gamma_A \gg \bar\Gamma_B\ }\; \frac{1}{B+1}
\qquad\big(= \tfrac{1}{A+B+1}\ \text{if same-distributed}\big)
$$

Sanity: $\bar\Gamma_A\!\to\!\infty$ → first free always the min ✓; more members → lower fire-prob ✓.

**Expected min & the $1/B$ decay.**

$$
E[m_B] = \frac{1}{A/\bar\Gamma_A + B/\bar\Gamma_B}
\;\xrightarrow{\ \text{free-dominated}\ }\; \frac{\bar\Gamma_B}{B}
$$

→ the group min DECAYS $\propto 1/B$: the drag ACCUMULATES (each extra user pushes the expected
min down), far stronger than the two-point "one-time drag".

**Exact expectation machinery (everything reduces to $E_1$).** Both rate functions are differences
of $\log_2(1+c\,\Gamma)$ (Möbius decomposition of the SINRs):

$$
R_c(\Gamma) = \log_2\!\big(1+\Gamma\big) - \log_2\!\big(1+(1-\beta_g\rho)\Gamma\big),
\qquad
R_p(\Gamma) = \log_2\!\big(1+(1-\beta_g\rho)\Gamma\big) - \log_2\!\big(1+c_p\Gamma\big),
$$
$$
c_p = 1-\alpha_k(1-\rho)-\beta_g\rho .
$$

For any $\mathrm{Exp}$-rate-$\lambda$ variable: $E[\ln(1+cX)] = e^{\lambda/c}E_1(\lambda/c)$
(exponential integral) → **every term of $E[\Delta_{B\to B+1}]$ has a closed form**:

$$
E[R_c(m)] = \frac{1}{\ln 2}\Big[e^{\Lambda}E_1(\Lambda) - e^{\Lambda/c_2}E_1(\Lambda/c_2)\Big],
\quad c_2 = 1-\beta_g\rho,\qquad m\sim\mathrm{Exp}(\Lambda)
$$

⚠ [07-10, validated §2] The drag term is negative but SHRINKS with $B$ → the increasing-marginal
conclusion of level (ii); the cumulative $E[\Delta R_{\text{tot}}(B)]$ is CONVEX. (Greedy
stop-at-first-negative can under-admit when the marginal is increasing — use the cumulative max,
not the greedy rule.)

**Closed-form $B^*_{\text{rate}}$ — both levels agree on CORNERS.** Level (i): the marginal is
constant for $B\ge1$ and $R_p$ is monotone in $\Gamma$, so the sum-rate step depends only on
$\text{sign}(\Gamma_B - \Gamma_{\text{dir}})$:

$$
B^*_{\text{rate}} =
\begin{cases}
(K{-}A) - 1 & \Gamma_B \ge \Gamma_{\text{dir}}\ \ (\text{fill to the group-0 cliff})\\[3pt]
0\ \ (\text{or } 1 \text{ iff } \Delta_{0\to1}\ge0) & \Gamma_B < \Gamma_{\text{dir}}
\end{cases}
$$

⭐ Level (i): **step function** → $B^*_{\text{rate}}$ degenerate (0 or fill-to-cliff), NO interior.
@P50 saturated $R_p(\Gamma_B)\approx R_p(\Gamma_{\text{dir}})$ → $\Delta\approx0$ → fill;
@P40 cascade $\Gamma_B^{\text{IRS}} > \Gamma_{\text{dir}}$ ($N^2$ gain) → $\Delta>0$ → also fill.
⭐ Level (ii) **agrees**: $E[\Delta]$ increasing in $B$ ⇒ cumulative CONVEX ⇒ corner survives in
expectation (machinery ⚠ above); i.i.d. fading only GRADES magnitudes (first admit carries the
largest expected drag). An interior $B^*_{\text{rate}}$ requires **heterogeneous candidates
admitted best-first** (each next candidate has a worse $\bar\Gamma$ → its private term falls with
$B$ → decreasing marginal — validated §2); otherwise the interior lives in QoS (next).

**QoS bound — GENERAL, no share assumption [REWRITTEN 07-10].** Shares $C_k$ là BIẾN QUYẾT ĐỊNH
(Ck head): $C_k \ge 0$, $\sum_{k\in g} C_k = R_c^g = R_c(\Gamma_{\min})$. Đặt shortfall
$ς_k = \big(D_k - R_{k,p,g}(\Gamma_k)\big)^+$. Group QoS-feasible ⟺ common budget phủ hết tổng
shortfall (linear feasibility — **EXACT by construction**, không cần equal-power hay equal-share):

$$
\boxed{\;\sum_{k\in g} ς_k \;\le\; R_c(\Gamma_{\min})\;}
\qquad\Longrightarrow\qquad
B^*_{\text{QoS}}(A) = \max\Big\{B :\ \textstyle\sum_{k}^{A+B} ς_k \le R_c\big(\Gamma_{\min}(B)\big)\Big\}
$$

(chú ý DOUBLE effect theo $B$: shortfall mass TĂNG và budget $R_c(\Gamma_{\min}(B))$ CO — min của
nhiều draw hơn.) **Two-point corollary** ($\Gamma_B < \Gamma_A$, blocked đủ khỏe → $s_A = 0$):

$$
B^*_{\text{QoS}} = \Big\lfloor \frac{R_c(\Gamma_B)}{ς_B} \Big\rfloor,
\qquad ς_B = \big(D_k - R_p(\Gamma_B)\big)^+
\qquad(\text{KHÔNG còn } -A)
$$

**Special case — equal-share floor** ($C_k = R_c/n$ chia đều): bind tại max-shortfall member →
$n^*_{\text{mix}} = R_c(\Gamma_{\min})/(D_k - R_p(\Gamma_{\text{wk}}))$,
$B^*_{\text{QoS}} = \lfloor n^*_{\text{mix}}\rfloor - A$. Quan hệ: **floor ≤ general LUÔN**
($\sum ς_k \le n\,ς_{\max}$), TRÙNG khi shortfall đồng nhất (two-point homogeneous) — heatmap
validation (§2) 100% = validate floor này dưới reference của nó.

- Regime split: $R_p(\Gamma_{\text{wk}}) \ge D_k$ → $ς = 0$ → không có ngưỡng QoS từ group-size
  (cap $=K$); $\Gamma\to\infty$ thu về công thức [N]. ✓
- **α-free note**: $\gamma_{k,c,g}$ KHÔNG phụ thuộc $\alpha$ → budget $R_c(\Gamma_{\min})$
  bất biến theo power-share; $\alpha$ chỉ vào qua $ς_k$ → bound α-optimal = allocation problem
  ($\min_\alpha \sum_k ς_k$, $\sum\alpha{=}1$) = lãnh địa PowerMLP + waterfill-Ck (§6.5) —
  deployed agent xấp xỉ bound này (giải thích agent ≥ floor).

**FINAL RESULT — binding composition threshold.**

$$
\boxed{\;B^*(A) \;=\; \min\big(B^*_{\text{rate}},\ B^*_{\text{QoS}}\big)\;}
$$

**Mechanism map (ai tạo threshold nào):**

| Level | $B^*_{\text{rate}}$ behaviour | nguồn |
|---|---|---|
| Two-point deterministic | **step**: 0 or fill-to-cliff, no interior | marginal constant in $B$ |
| i.i.d. fading (level (ii)) | **corner survives** (marginal ↑ về private const); chỉ grade magnitude, cú admit đầu chịu drag lớn nhất | min-drag decays |
| Heterogeneous best-first | **INTERIOR** xuất hiện (marginal ↓: mỗi candidate sau tệ hơn) | private term giảm theo $B$ |
| QoS | **GENERAL: $\sum ς_k \le R_c(\Gamma_{\min})$** (exact by construction) — interior operative bound; equal-share floor $\lfloor n^*_{\text{mix}}\rfloor - A$ validated 100% | common-budget vs shortfall mass |

**Takeaway.** Sum-rate gives NO interior $B^*$ under two-point NOR under i.i.d. fading (both
corner-optimal) → the operative interior threshold is the **QoS** bound. An interior sum-rate peak
(e.g. $B{=}2$ @P40, spot-check §2) comes from candidate **HETEROGENEITY** (best-first admission),
NOT from fading spread per se — fading grades magnitudes but preserves corners. Both branches
$\Gamma_B \gtrless \Gamma_{\text{dir}}$ fill to the cliff, re-confirming that @saturated it is QoS,
not sum-rate, that binds.

**Paper mapping:** ablation appendix subsection "Group-Composition Capacity" (đặt ĐẦU ablation
section) = Observation 2 phần (sum-rate side: safe-blocked + corner-vs-interior mechanism + cliff;
QoS side: **general Σ-shortfall bound** + equal-share floor $n^*_{\text{mix}}$ validated) +
limitation (scalar cascade $|\Sigma\varphi|{=}N$, Rayleigh-only, fixed weights — agent heads
soften cliff). NO-OVERCLAIM: mechanism claims có closed-form + MC đỡ. ⬜ TODO caveat: theory
assumes Rayleigh-only cascade fading ($g_{SR}$ scalar deterministic per-draw); thêm shadowing/CSI-κ
vào $\bar\Gamma$ thì $\Lambda$ đổi nhưng machinery giữ nguyên.

---

## §2. EXPERIMENTS — VALIDATION CHAIN [gộp §3+§4+§5.2+§7 cũ; theory → code → rig]

> Chuỗi validation theo thứ tự: (1) Γ-form ≡ code · (2) regime check (đâu là vùng toán sống) ·
> (3) two-point spot-check · (4) heatmap floor 100% · (5) order-statistics MC 27/27 ·
> (6) agent-vs-theory · (7) **controlled rig — final confirm 100%**.

**(1) Γ-form ≡ code (exact).** scratchpad/verify_gamma_form.py — 4 assert (A1 decomposition $|h|$
· A2/A3 SINR Γ-form · A4 $R_c$ min-form) **PASS rel-err ~1e-16** trên 9 config $(N,P_S)$.
⭐ Bonus: worst-case user lệch saturated tới **49% ngay @P50** — min-user của common stream
chính là worst-case → PHẢI dùng finite-Γ cho phần min (không saturated-approx).

**(2) Regime check [07-03]** — probe_regime_check (oracle routing/phase, equal power, Case-2,
80 draws/config):

| $P_S$ | $R_{\text{tot}}$ theo $N$ (6→12→24→48) | med $\Gamma_{\text{irs}}$@N24 | med $\Gamma_{\text{dir}}$ | %$\Gamma{<}10$ |
|---|---|---|---|---|
| 50 dBm | 1.475→1.523→1.541→1.553 (**FLAT**, 24→48 +0.8%) | 703 | 14 | 1.4% |
| 40 dBm | 1.052→1.221→1.303→1.354 (transition) | 70 | **1.4** | 32-52% |
| 30 dBm | 0.391→0.647→0.868→1.051 (**N lever mạnh** ×2.7; QoS 3.6→66.6%) | 7 | 0.14 | 44-100% |

⟹ (a) $P_S{=}50$ = interference-limited SÂU → tại operating point, ngưỡng $(A,B)$ quy về công
thức đếm; $N$/vị trí ~inert. (b) **Vùng toán sống = $P_S$ 30-40** ($\Gamma_{\text{dir}}\sim O(1)$
@P40 — đúng vùng "free lên IRS hay ở direct" non-trivial). (c) $\Gamma_{\text{irs}}\propto N^2$
verified (×4 mỗi lần gấp đôi). (d) ⚠ N-sweep training ablation @P50 sẽ KHÔNG thấy gì — chạy @P40
hoặc reframe thành Observation "N inert ở saturated regime".

**(3) Two-point spot-check** (scratchpad/spotcheck_B_marginal.py, N=24, 30 draws):

| $P_S$ | $\Delta_{0\to1}$ | $\Delta_{1\to2}$ | $\Delta_{2\to3}$ (cliff) | QoS(B=0→3) |
|---|---|---|---|---|
| 50 | −0.007 | −0.004 | −0.137 | 99.7→95.3 (đơn điệu giảm → $B^*_{\text{QoS}}{=}0$) |
| 40 | **+0.001** | −0.004 | −0.118 | 68→**86→91**→81 ($B^*_{\text{QoS}}{=}2$ interior!) |
| 30 | **+0.006** | −0.009 | −0.041 | 60→62→61→58 |

Hệ quả (đều khớp probe cũ/mới): **1.** $A$ KHÔNG có ngưỡng sum-rate — blocked→IRS luôn tăng
($R_p(\Gamma_A) \gg R_p(\beta_{\text{blk}}^2\Gamma)\approx 0$; khớp crossover_blocked monotonic↑).
**2.** @P50 saturated: marginal ≈ 0 → over-assign ~FREE về sum-rate (khớp overassign-cost ~1pp);
ngưỡng thật = QoS (equal-share = dilution knife-edge; general-share two-point $s\approx0$ → bind
qua fading tails). **3.** @P30-40 marginal đổi dấu dương → $B^*>0$ thật, QoS-optimal interior.
**4.** ⭐ GROUP-0-EMPTY CLIFF (spot-check phát hiện): direct cuối rời → slice common LÃNG PHÍ +
mất $R_c(\Gamma_{\text{dir}})$ → rơi ~0.12-0.14; ngưỡng cứng $B_{\max} = \#\text{free}-1$.
⚠ scheme-specific (fixed weights); agent (Ck/power head) re-allocate → cliff mềm hơn deployed.

**(4) Heatmap — equal-share floor exact** (analysis/probe_group_mix_threshold.py — K10 M1 N24,
40 draws): **classification accuracy $n^*_{\text{mix}}$ (floor): 100.0% (1066/1066 cells) @ CẢ
P50 & P40** — exact dưới reference equal-power/equal-share của nó. Cấu trúc heatmap: $A$ tăng →
$R_{\text{tot}}$ tăng tuyến tính (~+0.12/blocked) + QoS-overall +10pp/user — monotone-safe;
$B$ tăng → group-QoS% giảm đơn điệu (@P50 92.5→76.3; @P40 sập 70→24.6), $R_{\text{tot}}$ giảm nhẹ
(magnitude phụ thuộc geometry/G); **@P40 QoS-overall có ĐỈNH INTERIOR theo B** (A=3:
29.2→34.2→35.8→35.3, đỉnh B=2) — re-confirm $B^*>0$ ở noise-limited.

**(5) Order-statistics MC — 27/27** (`analysis/probe_orderstat_marginal.py`):
- **18/18 closed-form ≡ MC within 3σ** (K5 A2, equal-power, Γ̄ = regime-check medians, MC 400k,
  ratio $\bar\Gamma_B/\bar\Gamma_A \in \{1, 0.05\}$): marginals khớp 4–5 decimals; drag-probs exact
  0.909/0.476/0.323 = $(1/\bar\Gamma_B)/\Lambda$; $E[m_B]$ exact. KEY: $E[\Delta]$ **increasing
  in $B$ mọi regime** (r=1 @P50 +0.0190→+0.0191; r=0.05 @P40 −0.024→+0.002→+0.009) →
  corner-optimum survives — đúng ⚠-correction §1.
- **Heterogeneous-sorted PART 3** (pool 0.3/0.05/0.01 best-first, A=2, MC 200k, incl. cliff):
  9/9 thêm (tổng **27/27**). Marginal **DECREASING** (@P40: +0.078 → −0.015 → −0.198) → interior
  $B^*_{\text{rate}}$ sống. **Cliff cross-check: $R_c(\Gamma_{\text{dir}})=0.141$ model vs 0.137
  env @P50 (3%)**. Per-draw $B^*$ modes = **2/2/1 @P50/40/30**, khớp heatmap (4). General-share
  $B^*_{\text{QoS}}$ ≥ floor MỌI regime (mean 2.93/2.38/1.03 vs 2.88/2.14/0.82; @P40 mode 3 vs 2)
  nhưng **$B^* = \min$ KHÔNG đổi** (rate binds) → FINAL RESULT robust theo share assumption.
- Evidence re-read: pattern giảm trong spot-check (3) là **heterogeneity giữa candidates**, KHÔNG
  phải i.i.d. min-drag (i.i.d. dự đoán tăng). @P50 both terms ≈ 0 → QoS binds.

**(6) Agent-vs-theory [07-03]** — probe_assignment_quality @ Case-2 ramp0.4 (VQC r103/ep900 ·
DNN r100/ep2000): Σirs 7.00/6.96 ≈ Blk 6.77 · blocked→đúng-IRS 99.6%/99.0% · UNDER 0.0%/0.1% ·
OVER 7.1%/6.1% (≈0, đúng vùng marginal≈0 "benign"). ⟹ 2 actor khác kiến trúc hội tụ về CÙNG
nghiệm = closed-form optimum → **task-parity là hệ quả cấu trúc, không phải trùng hợp** (evidence
mạnh cho A1). Load=4 groups chạm $n^*_{\text{mix}}$ ~3.8-4 = knife-edge cells → nghi nguồn
QoS-miss tail. ⚠ Dòng "QoS agent" của probe (47.1/41.0%) lệch deploy thật (~96%) = artifact
probe-eval (nghi sampled/thiếu α-restore) — FIX trước khi dùng số đó.

**(7) CONTROLLED RIG — final confirm** (`analysis/probe_bstar_confirm.py`, 50 draws × {P50, P40}).
Rig: Case-2 topology K=10 M=2 N=24, D_k=0.1, κ=0.05, σ² nominal, static (reset-only), oracle
phase; ⭐ **IRS 2 TẮT HẲN** (không assign/group/slice, `active_irs_ids=[1]` mọi call — zero nhiễu);
weights PIN 2-group ($x = 0.08$, $y = 0.1$); $A=3$ blocked gần IRS-1 nhất pin lên IRS 1 (blocked
thừa nằm chết group 0 = background hằng số); free admit best-first $B = 0 \to \min(n_{\text{free}}{-}1, 5)$
(giữ ≥1 direct — không đo cliff, đã có (5)); paired CRN (cùng draw cho mọi $B$); theory
per-realization trên CÙNG $\hat h$ (extract $\Gamma_{dir}/\Gamma_{irs}$ bằng 2 call all-direct /
all-IRS1 — h_eff không phụ thuộc power).
**Power-accounting guarantee (isolation, verified code+số):** (i) private đi theo user (rate.py:128)
— không redistribute; (ii) mapping slice→group CỐ ĐỊNH VỊ TRÍ (`_make_wc_map` {0:0, 1:1}) — slice
group-0 KHÔNG BAO GIỜ dồn sang IRS; (iii) `total_wc = sum(w_c as-passed)` (rate.py:119) → group
rỗng thì slice CHÁY như interference (wasted = theory cliff convention), KHÔNG freed — ⚠ đúng vì
probe PIN vector w_c; code-path auto-derive w_c sẽ ra convention khác. $W_p{+}W_c = P_S$ hằng số.

| Metric | P50 | P40 |
|---|---|---|
| wiring $\|R_{\text{code}} - R_{\text{theory}}\|_{\max}$ | **6.7e-16** | **8.9e-16** |
| mean marginal $\Delta_1, \Delta_2$ | −0.061, −0.108 (**decreasing** ✓) | −0.090, −0.104 |
| $B^*_{\text{rate}}$ (mode/mean) | 0 / 0.34 | 0 / 0.32 |
| $B^*_{\text{QoS}}$ floor · general | 1 / 1.04 · **2 / 1.44** (gen>floor ✓) | 0 / 0.42 · 0 / 0.54 |
| ⭐ $B^* = \min$ (mode/mean) | **0** / 0.34 | **0** / 0.32 |
| **theory-vs-code classification (cả 4 ngưỡng)** | **100%** | **100%** |

**Đọc kết quả:**
1. **CONFIRM TRỌN VẸN**: công thức §1 dự đoán đúng per-draw **100%** cả 4 threshold × 2 regime;
   wiring exact tới machine-precision (Γ-form ≡ code end-to-end qua rig).
2. **Marginal giảm dần** ✓ (heterogeneity best-first) và **general ≥ floor in-rig** ✓.
3. ⚠ **$B^*$ CỦA RIG NÀY = 0**: với A=3 pinned + spawn geometry THẬT, cascade của free user
   THUA direct của chính nó → admit đầu tiên đã lỗ (−0.06). Kỳ vọng cũ (2/2/1) dựa trên pool
   synthetic (5) + heatmap QoS-overall (4) — khác reference. ⟹ **threshold là reference-specific**
   (đúng thiết kế §1: $B^*(\bar\Gamma\text{'s})$); cái bất biến = FORMULA, confirm 100%.
4. Cliff không đo trong rig (giữ ≥1 direct); đã validated riêng 0.141-vs-0.137 ở (5).

**Assets tái dùng:** probe_irs_group_capacity.py (validate n*) · probe_crossover_blocked.py
(0d/5i monotonic) · probe_irs_overassign.py (n_irs=3 sweet-spot, over-assign cost ~1pp,
$|\Sigma\varphi|=N$ no-phase-tradeoff) · Common-Knowledge [N] ·
probe: scratchpad/probe_regime_check.py (chuyển vào analysis/ khi chốt).

---

## §3. VARIABLE-B ADMISSION CAPACITY — SEAT LAW [2026-07-18, theo yêu cầu user]

> User setup: $A$ free-users (direct) + $B_0$ users ĐANG ở IRS group; assignment shift TỪNG
> free-user sang IRS. Câu hỏi: **$B_0$ dao động ảnh hưởng gì đến số user thêm được** $n^*_{\text{add}}(B_0)$?
> Assumptions user chốt: (i) IRS-element $= N^*$ tại thời điểm admit → $\bar\Gamma_B = \bar\Gamma_B(N^*) \propto N^{*2}$;
> (ii) free-users còn lại max $R_{\text{tot}}$; (iii) weights fixed (isolation convention §1).
> §2 rig trước đây PIN $A{=}3$ / $B$ sweep ngắn — §3 tổng quát theo occupancy $B_0$.

### 3.1 Seat law (two-point / homogeneous — EXACT)

QoS-feasibility của group chỉ phụ thuộc **TẬP thành viên** (qua $\Sigma\varsigma$ và $\Gamma_{\min}$),
KHÔNG phụ thuộc thứ tự vào. Với thành viên exchangeable (cùng $\bar\Gamma_B$):

$$
n^* = \max\Big\{n:\ \textstyle\sum_{i\le n}\varsigma_i \le R_c\big(\Gamma_{\min}(n)\big)\Big\}
\qquad\Longrightarrow\qquad
\boxed{\;n^*_{\text{add}}(B_0) = n^* - B_0,\qquad \frac{\partial n^*_{\text{add}}}{\partial B_0} = -1\;}
$$

**"B dao động" = trừ ghế TUYẾN TÍNH 1-1**: group có tổng số ghế $n^*$ cố định (theo draw/geometry);
occupancy chỉ TIÊU ghế, không đổi tổng. Two-point homogeneous: $n^* = \lfloor R_c(\Gamma_B)/\varsigma_B \rfloor$
(= $n^*_{\text{mix}}$ §1 với $s_A{=}0$). Fading: cả 2 phía co theo $n$ (double effect §1) nhưng
exchangeability GIỮ seat law per-draw — validated §3.4 mục 1 (mode $n^*$ trừ đúng 1 mỗi $B_0$).

### 3.2 Khi nào seat law VỠ — heterogeneity

Nếu candidates KHÔNG exchangeable (best-first: mỗi candidate sau $\bar\Gamma$ tệ hơn — đúng thực tế
spawn geometry), feasibility phụ thuộc AI đang chiếm ghế → $n^*_{\text{add}}(B_0)$ giảm **NHANH hơn** $-1$:
probe (§3.4 mục 3, means $\bar\Gamma_B\cdot[1,.5,.25,...]$):

| regime | $n^*_{\text{add}}(0)$ het vs hom-law | $n^*_{\text{add}}(2)$ het vs hom-law |
|---|---|---|
| P50 | 8 vs 8 (saturation che) | 6 vs 6 |
| P40 | **6 vs 8** | **4 vs 6** |
| P30 | **3 vs 8** | **0 vs 6** |

⟹ đo lệch $n^*_{\text{add}}(B_0) - (n^*_{\text{hom}} - B_0)$ = thước đo heterogeneity của candidate pool.

### 3.3 Vai trò $N^*$ — lever CHỈ Ở NOISE-LIMITED + free-ride threshold

$\bar\Gamma_B(N^*) = \beta_m^2 |g_{SR}|^2 N^{*2} |g_{RU}|^2 P_S/\sigma^2$ vào $n^*$ qua 2 kênh:
budget $R_c(\Gamma_{\min})\uparrow$ và $\varsigma_B\downarrow$ — nhưng CẢ HAI saturate khi
$\Gamma \gg 1$ (interference-limited ceiling: $R_c \to \log_2(1+\frac{y}{1-y})$, $R_p \to \log_2(1+\frac{x}{1-x-y})$).
Probe (§3.4 mục 1, $N^* \in \{12,24,48\}$ = $\bar\Gamma \times\{0.25,1,4\}$):

- **P50**: mean $n^*$ 7.87→7.97→7.99 — $N^*$ **inert** (saturated sâu, khớp §2(2)).
- **P40**: 6.74→7.68→7.92 — lever nhẹ.
- **P30**: **1.53→5.01→7.23** (mode 0→8!) — $N^*$ là lever THẬT ở noise-limited.

**Free-ride threshold** (cô lập yếu tố quyết): $\varsigma_B = 0 \Leftrightarrow R_p(\Gamma(N^*)) \ge D_k$
→ QoS không bind → fill-to-cliff. Giải $R_p(\Gamma_D) = D_k$ cho $\Gamma_D$:

$$
N^*_{\min} = \sqrt{\frac{\Gamma_D\,\sigma^2}{\beta_m^2\,|g_{SR}|^2\,|g_{RU}|^2\,P_S}}
$$

### 3.4 Probe validation (`analysis/probe_variable_b_seats.py`, numpy MC 20k, seed 42)

1. **Homogeneous seat-law EXACT** per-draw (mode $n^*{=}8{=}n_{\max}$ @P50/P40 mọi $N^*$;
   $n^*_{\text{add}}(B_0)$ = 8/7/6/5 đúng trừ-1). Interior thật @P30 $N^*{=}12$: mode 0.
2. **Rate-side sign** trên regime-medians: frac($R_p(\Gamma_B X) > R_p(\Gamma_{\text{dir}} X')$) ≈ 0.98
   mọi regime → fill-to-cliff. ⚠ NGƯỢC rig §2(7) ($B^*_{\text{rate}}{=}0$ với spawn geometry thật) —
   reference-specific đúng thiết kế: medians synthetic ≠ per-draw geometry; formula là bất biến.
3. **Heterogeneous**: bảng §3.2. 4. **$E[m_n]$ ≡ $\bar\Gamma_B/n$** exact (4 chữ số).

### 3.5 PARAMETER MAP — cái gì ảnh hưởng capacity TRỰC TIẾP, assume được hay không (mục c user hỏi)

| Para | Vào capacity qua | Assume được? | Options |
|---|---|---|---|
| $\bar\Gamma$ (fold $N, P_S$, position, $\sigma^2$) | MỌI công thức (qua $\Gamma_k = \bar\Gamma X$) | GIỮ THAM SỐ (assumption map 07-17) | two-point medians / per-draw geometry / sweep |
| $P_S$ | scale $\Gamma$ + CHỌN REGIME (interference vs noise-limited) | **CHƯA assume được** (user note đúng) | (a) fix 50 dBm (saturated → mọi thứ thành counting, §6); (b) high-power LIMIT $P_S\to\infty$ ($\Gamma$ cancel — clean nhất để cô lập); (c) sweep 30-50 (vùng toán sống P30-40) |
| $C_k$ policy | quyết bound QoS nào áp dụng | **CHƯA assume được** — nhưng CÓ 3 mức đã chứng minh | (a) **general LP $\Sigma\varsigma \le R_c$ = ASSUME-optimal, EXACT by construction** (khuyến nghị — không cần biết policy!); (b) equal-share floor $n^*_{\text{mix}}$ (conservative, validated 100%); (c) waterfill (reward-optimal, park-below-D §6.5) |
| $N$ / $N^*$ | CHỈ qua $N^2$ trong $\bar\Gamma$ (§0.2, no phase-sharing penalty) | ✅ assume $N^*$ tại admit (user đã chốt) | inert @P50; lever @P30-40; free-ride $N^*_{\min}$ §3.3 |
| $K$ | tổng user; cap $n^* \le$ #free$-1$ (group-0 cliff) | ✅ | per-case 5/10/15 |
| $M, G$ | slice common $\beta_g\rho/(G{+}1)$ → budget mỗi group; group-0-empty cliff | ✅ (fix topology per case) | 1 IRS isolation (rig) / full M |
| power-split $(\alpha_k, \beta_g, \rho)$ | $x, y$ trong MỌI rate; ⭐ α-free note §1: budget $R_c$ BẤT BIẾN theo $\alpha$ — $\alpha$ chỉ vào $\varsigma_k$ | GIỮ THAM SỐ (map 07-17); equal-power = reference | fixed (isolation) / oracle-grid (frontier §6) |
| $D_k = R_k^{\min}$ | scale shortfall $\varsigma$ → $n^* \approx R_c/\varsigma$ gần tuyến tính nghịch | ✅ (0.1 nominal) | sweep nếu cần capacity-vs-demand curve |
| fading law | Exp(1) → TOÀN BỘ order-stat machinery exact ($E_1$) | ✅ assume (Rayleigh cascade = code) | shadowing/κ fold vào $\bar\Gamma$ — $\Lambda$ đổi, machinery GIỮ (caveat §1 cuối) |
| blockage $\beta_{\text{blk}}$, d | $\Gamma^{\text{dir}} \approx 0$ → blocked luôn route (no threshold); $d$ = drop-budget trần $R^*(d)$ | ✅ | ties §6 capacity band [R*(0), R*(d)] user đang làm |
| $\sigma^2$ | trong $\bar\Gamma$ + chọn regime | GIỮ THAM SỐ | nominal / per-step lognormal |
| $\kappa$ CSI | 2nd-order (design-vs-reality gap; ≈null @0.05) | ✅ bỏ qua bậc 1 | Math-Hypo H2 nếu muốn bound |

---

## §4. B* THEO HYPERPARAMETERS — BỨC TRANH TỔNG THỂ [2026-07-19, theo yêu cầu user]

> Câu hỏi user: B*_rate/B*_QoS như HÀM của hyper — quan trọng nhất **P_S, C_k-policy, N**; cần 2 dạng
> công thức (QoS-serve-all + drop-percent); conjecture "C_k là lever quan trọng nhất vì là var cuối
> và RSMA group càng lớn càng extreme". Probe validate: `analysis/probe_bstar_hyper.py` (MC 20k,
> n_max 16, seed 42; anchor Γ̄=703 @P50/N24, Γ_dir=14; equal-power reference x=0.08, y=0.1, D=0.1).

### 4.1 HAI DẠNG CÔNG THỨC CHỐT

**Dạng 1 — QoS-SERVE-ALL** (mọi thành viên đạt D_k):

$$
n^*_{\text{serve}} = \max\Big\{n:\ \sum_{i\le n} ς_i \le R_c\big(\Gamma_{\min}(n)\big)\Big\},
\qquad ς_i = \big(D_k - R_p(\bar\Gamma X_i)\big)^+,
\qquad \bar\Gamma = c_{\text{geo}}\,\frac{N^2 P_S}{\sigma^2}
$$

- Two-point (deterministic): $n^*_{\text{tp}} = \lfloor R_c(\bar\Gamma)/ς(\bar\Gamma)\rfloor$.
- Fading: $\Gamma_{\min}(n)\sim$ Exp-min, $E[\Gamma_{\min}] = \bar\Gamma/n$; mọi kỳ vọng closed-form qua $E_1$ (§1).
- Equal-share: thay $\sum ς_i$ bằng $n\,ς_{\max}$ (bind tại worst member).
- ⚠ Two-point **misleading ở fading regime**: P30/N24 cho $n^*_{\text{tp}}{=}16$ nhưng mean thật 6.36 —
  min-decay $1/n$ ăn budget nhanh hơn deterministic. LUÔN dùng bản fading khi $\bar\Gamma \lesssim 100$.

**Dạng 2 — DROP-PERCENT** (được phép bỏ $d$% user tệ nhất — corner rule, nối band $[R^*(0), R^*(d)]$):

$$
n^*(d) = \max\Big\{n:\ \sum_{i>j} ς_{(i)} \le R_c\big(\bar\Gamma X_{(j+1)}\big)\Big\},\quad j=\lceil d\,n\rceil,
\qquad
E\big[\Gamma_{\min}^{\text{served}}\big] = \bar\Gamma \sum_{i=0}^{j} \frac{1}{n-i}
$$

($X_{(j)}$ = order statistic thứ $j$; công thức partial-sum **validated MC 3 chữ số** — probe mục 1.)
$B^*(A; d) = n^*(d) - B_0$ theo seat law §3.1 (homogeneous; heterogeneous giảm nhanh hơn §3.2).
$d_{\min} = \max(0, (K-K^*)/K)$ = drop bắt buộc khi serve-all infeasible (C3 dự đoán ~13%).

### 4.2 HYPER MAP — vào đâu, chiều nào, saturate không, regime nào sống

| Hyper | Vào qua | Chiều | Saturate? | Regime sống | Probe evidence |
|---|---|---|---|---|---|
| **P_S** | $\bar\Gamma \propto P_S$ (linear W) | extend | ✅ $R_c(\infty)=\log_2\frac{1}{1-y}=0.152$; $ς\to ς_\infty$ | chọn REGIME: bind chỉ khi P≲40 @N24 | P50: MỌI cell = cap 16 (inert); P30/N24: mean 6.36 |
| **N** | $\bar\Gamma \propto N^2$ (cùng kênh P_S, bậc 2) | extend | ✅ như trên | P30-40 | P30 d0 general: N12→48 = 1.56→12.95; P50 inert |
| **C_k policy** | $\sum ς$ (general-LP) vs $n\,ς_{\max}$ (equal) | general ≥ equal LUÔN | không saturate — gap TĂNG theo n | mọi regime chưa-saturated | P[feas] gap 0.011(n2)→**0.309(n16)**; $E[ς_{\max}]/E[\bar ς]$ ≈ 2→13.4 |
| **d (drop)** | order-stat jump $+\bar\Gamma/(n{-}j)$ mỗi nấc + bỏ $ς_{(n)}$ | extend KÉP | — | mạnh nhất khi bind | P30/N24 general: 6.36→**13.42** chỉ với d=10% |
| **D_k** | scale ς tuyến tính | shrink khi tăng | — | mọi | $n^*_{\text{tp}} \propto 1/(D_k - R_p)^+$ |
| **ρ, β_g** (slice $y$) | budget trần $\log_2\frac{1}{1-y}$ NHƯNG ăn private ($c_p{=}1{-}x{-}y$ giảm → ς tăng) | two-sided | — | — | GIỮ THAM SỐ (map 07-17) |
| **α_k** | KHÔNG vào budget (α-free §1) — chỉ vào ς | policy-lever phía PowerMLP | — | — | bound α-optimal = $\min_\alpha \sum ς$ |
| **K, M/G** | cap $n \le$ #free−1 (cliff) + slice $/(G{+}1)$ | cap cứng | — | — | §2(3) cliff 0.141 |
| **σ²** | $\bar\Gamma \propto 1/\sigma^2$ | cùng trục P_S | ✅ | regime knob | — |

### 4.3 VERDICT CONJECTURE C_k ("var cuối, group càng lớn càng extreme") — ✅ ĐÚNG, CÓ ĐIỀU KIỆN

1. **Đúng và định lượng được**: equal-share bind tại $ς_{\max}$ của order statistics; $E[ς_{\max}]/E[\bar ς]$
   tăng ~tuyến tính theo $n$ (probe: 1.98 @n2 → 13.41 @n16) ⟹ phần capacity equal-share BỎ PHÍ tăng
   theo group size — "group càng lớn càng extreme" chính xác là hiện tượng order-statistics này.
   General-LP (= Ck-head tự do) thu hồi toàn bộ: P[feasible] hơn tới **+31pp @n16 @P40/N12-24**.
2. **Điều kiện**: (a) @saturated (P50/N24 operating point) mọi policy đều đủ — C_k inert như mọi lever
   khác; C_k chỉ là "lever quan trọng nhất" trong vùng capacity-bind (noise-limited / N nhỏ / D_k cao);
   (b) drop-budget $d$ per-unit còn mạnh hơn (×2 capacity mỗi 10% drop @P30) — nhưng $d$ đổi METRIC
   (chấp nhận hy sinh QoS-count), còn C_k-policy là lever KHÔNG-mất-gì duy nhất. Xếp hạng trong nhóm
   "free lever": **C_k-policy > drop-budget (nếu được phép) ≫ α (chỉ qua ς)**; nhóm hardware
   (N, P_S): chỉ đáng giá dưới saturation.
3. Khớp chain-position: C_k là var CUỐI được quyết ⟹ nó thấy toàn bộ realization (Γ, ς của từng user)
   → LP per-draw đạt exact bound; các var trước (α, phase, routing) chỉ dịch phân bố ς. Đây là lý do
   cấu trúc khiến ck-fix + waterfill-Ck từng dominate cả 2 agent (§6.7 recipe oracle-asg+equal+wf-Ck).

### 4.4 Probe log (probe_bstar_hyper.py, 07-19)

Grid n*(mode/mean) đầy đủ 18 config × 4 d — xem output script; điểm nhấn:
P50 mọi cell = 16 (cap) · P40/N12: general 11.20 vs equal 6.75 (d0) · P30/N12: 1.56/1.17 (bind sâu) ·
P30/N24 general: 6.36 → 13.42 (d10%) → 15.52 (d20%) · sanity order-stat partial-sum khớp 3 chữ số.

---
---

## §6. V*(λ) + PARETO FRONTIER — ĐÃ TÁCH [2026-07-19]

> Toàn bộ §6 (V*_rate, concentration identity, λ_cliff, waterfill parking, frontier dichotomy,
> agent gaps 6.7) đã chuyển sang **docs/Pareto-Frontier.md** theo yêu cầu user 07-19.
> Số liệu và nội dung giữ nguyên; refs bên ngoài trỏ "worksheet §6" → đọc file mới.
