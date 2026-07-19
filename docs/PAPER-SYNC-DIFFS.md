# Paper ↔ Code sync — edit list (paper theo code/reality)

Nguyên tắc: **reality (code) là chuẩn**, paper sửa theo code. Mỗi mục có OLD → NEW copy-paste được.
Sắp theo thứ tự dòng trong `HQC-HAC for Optimizing Sum-Rate with IRS-Assisted.tex` để sửa 1 lượt từ trên xuống.
Số dòng là tham chiếu lúc soạn (có thể lệch sau khi sửa — tìm theo nội dung/label).

Quyết định đã chốt: **① = B** (state ≡ affinity, bỏ raw eq:state) · **② = R1** (readout structured) ·
**④ = giữ 120/144** (tính enc-norm γ,β) · **⑥ = P_S 50 uniform**.

> **GLOBAL NOTE cho ①(B):** thống nhất **state MDP `s_t` ≡ affinity `a_t`**. Bỏ vector raw
> `[Re/Im, cosΘ, sinΘ, φ, β]`. Giữ ký hiệu `V(s_t)`, `π(·|s_t)` (giờ `s_t` = affinity). Kiểm tra
> và bỏ mọi `\eqref{eq:state}`/`\ref{eq:state}` còn sót.

---

## ⑦ CSI prose — agent nhận ĝ, reward trên g thật  (System Model, ~tex:603)

**Reality:** sau code-change, policy quyết định trên CSI ước lượng ĝ; môi trường chấm rate trên g thật.
`eq:affinity` (đang để hat ĝ) → **GIỜ ĐÚNG, giữ nguyên.**

OLD:
```
The RL agent, however, observes the true channel magnitudes |g| as part of its state,
reflecting the assumption that the environment itself operates on the true physical channels
while the designed policy must cope with the estimation error present in the system's decision inputs.
```
NEW:
```latex
The policy observes only the estimated channel magnitudes $|\hat{g}|$ and designs the assignment and
precoding on them, whereas the environment scores the achieved rate on the true channels $g$.
Imperfect CSI therefore manifests as a design-versus-reality gap: the $\hat{g}$-optimal decisions
are evaluated against the physical channel $g$.
```

---

## ① state ≡ affinity — bỏ raw eq:state  (Overall Framework, ~tex:795–814)

**Reality:** actor + critic đều nhận affinity `a_t` (eq:affinity). Vector raw không được lắp ở đâu.
Lý do (user): raw Re/Im phản ánh kém quan hệ địa lý user–IRS → dùng affinity → rồi AE nén còn ít qubit.

OLD (cả đoạn + `eq:state` + câu d_s + câu circular-encoding):
```
Fig.~\ref{fig:framework} illustrates the full architecture. At each decision step~$t$, the global
state vector is constructed from the raw channel coefficients ... [eq:state] ... where $\Theta$ ...
the total state dimension is $d_s = M(3+2K+2N)+3K$. The circular encoding ... $2\pi$ phase boundary.
```
NEW (thay cả đoạn, KHÔNG còn eq:state):
```latex
Fig.~\ref{fig:framework} illustrates the full architecture. At each decision step~$t$, the raw
channel observation is condensed into a compact \emph{affinity feature vector}
$\mathbf{a}_t\in\mathbb{R}^{d_{\mathrm{aff}}}$ (Eq.~\eqref{eq:affinity}) that encodes the end-to-end
user--IRS path qualities, the per-user demands, and per-IRS summary statistics --- the quantities
that determine the assignment. A flat real/imaginary channel vector is a poor policy input: its
dimension grows with the number of reflecting elements $N$ and it does not expose the user--IRS
spatial relationships directly, whereas $\mathbf{a}_t$ makes them explicit while discarding the
$N$-dependent bulk. The affinity $\mathbf{a}_t$ is the state on which both the quantum actor
(through the AE and VQC) and the shared critic operate; the autoencoder further compresses it to the
latent $\mathbf{z}_t\in\mathbb{R}^{2n_q}$ that fits the near-term qubit register.
```
> `d_s = M(3+2K+2N)+3K` vẫn được nhắc trong Sec.~\ref{sec:ae} như "raw baseline nếu không dùng
> affinity" — không cần eq:state riêng.

---

## ① critic dùng affinity  (Overall Framework, ~tex:816)

OLD:
```
A single shared critic evaluates the state $\mathbf{s}_t$ and supplies the advantage signal used to update all four actors.
```
NEW:
```latex
A single shared critic evaluates the same affinity state $\mathbf{a}_t$ (Eq.~\eqref{eq:affinity}) that the quantum actor consumes, and supplies the advantage signal used to update all four actors.
```

---

## ④ eq:ln_z — thêm affine học được γ,β  (VQC Data Encoding, ~tex:895–899)

**Reality:** encoding LayerNorm có affine học được γ,β ∈ R^{2n_q} (train cùng quantum params).
Cần thêm để khớp với param-count 120/144 (mục ④ ở eq:vqc_params bên dưới).

OLD:
```
\tilde{\mathbf{z}}_t = \frac{\mathbf{z}_t - \mu_{z}}{\sigma_{z} + \epsilon},
```
NEW:
```latex
\tilde{\mathbf{z}}_t = \boldsymbol{\gamma}\odot\frac{\mathbf{z}_t - \mu_{z}}{\sigma_{z} + \epsilon} + \boldsymbol{\beta},
```
Câu ngay sau (`where $\mu_z$ and $\sigma_z$ ...`) thêm:
```latex
, and $\boldsymbol{\gamma},\boldsymbol{\beta}\in\mathbb{R}^{2n_q}$ are a learnable affine trained
jointly with the quantum parameters, restoring the DC component that a plain zero-mean LayerNorm
would annihilate.
```

---

## ② + ⑨ eq:observables → R1 + train-shots  (VQC Measurement, ~tex:944–953)

**Reality (②):** deploy dùng R1 structured readout, N_Q=(n_q−M)(M+2)+M = 34/42 (C1/C2), KHÔNG phải
generic Z+NN-ZZ (23). Các readout khác chỉ là ablation chứng minh readout-design quan trọng.
Sửa cái này làm `tab:abl_readout` ("R1 = Eq.~observables") trở nên đúng.

OLD (eq:observables):
```
\hat{\mathbf{o}}_t = [⟨Z_0⟩,…,⟨Z_{n_q-1}⟩, ⟨Z_0 Z_1⟩,…,⟨Z_{n_q-2} Z_{n_q-1}⟩] ∈ [-1,1]^{N_Q}, N_Q = 2n_q-1.
```
NEW:
```latex
\hat{\mathbf{o}}_t = \bigl[\,
\underbrace{\langle Z_u\rangle}_{u\in\mathcal{U}},\;
\underbrace{\langle Z_u Z_r\rangle}_{u\in\mathcal{U},\,r\in\mathcal{R}},\;
\underbrace{\tfrac{1}{|\mathcal{U}|-1}\!\!\sum_{u'\neq u}\!\langle Z_u Z_{u'}\rangle}_{u\in\mathcal{U}},\;
\underbrace{\langle Z_r\rangle}_{r\in\mathcal{R}}\,\bigr]\in[-1,1]^{N_Q},
\quad N_Q=(n_q{-}M)(M{+}2)+M,
```
với $\mathcal{U}=\{0,\dots,n_q{-}M{-}1\}$ (user qubits) và $\mathcal{R}=\{n_q{-}M,\dots,n_q{-}1\}$
(IRS qubits). Bốn khối: per-user $\langle Z_u\rangle$, user–IRS $\langle Z_uZ_r\rangle$, user-cluster
mean-$\langle Z_uZ_{u'}\rangle$, per-IRS $\langle Z_r\rangle$ — các quan sát **aligned với action
user→IRS**. Câu "at a cost of only $2n_q-1$ features" → "$N_Q$ action-aligned features".

**⑨ shots** — câu tiếp theo (analytic-vs-shots):
OLD:
```
These expectations are computed analytically from the statevector during training, for low-variance
gradients, and estimated from S measurement shots at inference.
```
NEW:
```latex
During training the behaviour policy samples actions from $S_{\mathrm{tr}}{=}1500$-shot estimates,
while the gradient Jacobian and the PPO importance ratio use the exact statevector expectations
(low variance); at inference the policy uses $S_{\mathrm{ev}}{=}3000$-shot estimates.
```

---

## ③ head = SoftmaxPQC (linear + β), không phải MLP-ReLU  (VQC Post-processing, ~tex:955–963)

**Reality:** `vqc_softmax_head=True` → logits = β·(W·h_t + b). Đây là lý do head ~2K param (param-eff),
không phải MLP ~50K.

OLD (câu head):
```
A classical post-processing head $\xi$ (an MLP with ReLU activations) therefore maps the quantum
features to the action space.
```
NEW:
```latex
A lightweight SoftmaxPQC head $\xi$ --- a single linear map followed by a trainable
inverse-temperature $\beta$, after~\cite{jerbi2021parametrized} --- therefore maps the quantum
features to the action space (its $\sim\!2$K parameters, versus a $\sim\!50$K ReLU-MLP, underpin the
parameter-efficiency result of Section~\ref{sec:res_qvsc}).
```

OLD (eq:pi_k):
```
\boldsymbol{\pi}_k = \mathrm{softmax}(ξ(h_t)[k(M+1):(k+1)(M+1)]) ∈ Δ^M
```
NEW:
```latex
\boldsymbol{\pi}_k = \mathrm{softmax}\!\Bigl(\beta\,\bigl[\mathbf{W}\mathbf{h}_t+\mathbf{b}\bigr]_{k(M{+}1):(k{+}1)(M{+}1)}\Bigr)\in\Delta^{M}, \quad k\in\mathcal{K},
```

---

## ① MDP state  (Training / MDP, ~tex:973)

OLD:
```
The \emph{state} $\mathbf{s}_t$ is the system observation defined in~\eqref{eq:state}, comprising the
imperfect channel coefficients, the reflection efficiencies, the current IRS phase configuration, and
the current assignment.
```
NEW:
```latex
The \emph{state} $\mathbf{s}_t\equiv\mathbf{a}_t$ is the affinity feature vector of
Eq.~\eqref{eq:affinity}, built from the imperfect channel estimates $\hat{g}$.
```

---

## ⑩ reward/QoS trên g thật  (thêm sau eq:reward, ~tex:987)

```latex
Here $R_{\mathrm{tot}}$ and the per-user rates $R_k$ entering the reward and the QoS check are the
\emph{achieved} rates on the true channel $g$, whereas all four sub-policies are conditioned on the
estimated channel $\hat{g}$; the penalty thus charges the policy for the design-versus-reality gap.
```

---

## ⑧ eq:spsa — SPSA của Jacobian observables, không phải finite-diff của J_φ  (~tex:1043–1051)

**Reality:** SPSA ước lượng Jacobian `∂ô/∂η` (1 cặp circuit-eval / hướng Rademacher), rồi chain-rule
giải tích qua `∂J_φ/∂ô`. Head/entropy/AE-recon đều exact backprop.

OLD:
```
\hat{g}_i = [ \mathcal{J}_\varphi(η+c_sΔ) − \mathcal{J}_\varphi(η−c_sΔ) ] / (2 c_s Δ_i)
```
NEW:
```latex
\hat{J}_{o,i}=\frac{\hat{o}_o(\boldsymbol{\eta}+c_{\mathrm s}\boldsymbol{\Delta})-\hat{o}_o(\boldsymbol{\eta}-c_{\mathrm s}\boldsymbol{\Delta})}{2\,c_{\mathrm s}}\,\Delta_i,
\qquad
\frac{\partial\mathcal{J}_\varphi}{\partial\eta_i}=\sum_o\frac{\partial\mathcal{J}_\varphi}{\partial\hat{o}_o}\,\hat{J}_{o,i},
```
Sửa prose quanh đó:
```latex
SPSA estimates the Jacobian of the circuit readout $\partial\hat{\mathbf{o}}/\partial\boldsymbol{\eta}$
from a single pair of circuit evaluations per Rademacher direction $\boldsymbol{\Delta}$ (cost
independent of $\dim\boldsymbol{\eta}$); the head, entropy, and autoencoder-reconstruction gradients
are then obtained by exact back-propagation through $\partial\mathcal{J}_\varphi/\partial\hat{\mathbf{o}}$.
The estimate is averaged over a few repetitions to reduce variance.
```

---

## ⑥ P_S Case-3: 60 → 50  (Sim Setup ~tex:1062  +  tab:hyper ~tex:1078)

**Sim setup:**
```latex
% OLD: ...$(10,2,50~\mathrm{dBm})$, and $(15,3,60~\mathrm{dBm})$, with VQC depth...
% NEW:
...$(10,2,50~\mathrm{dBm})$, and $(15,3,50~\mathrm{dBm})$, with VQC depth $L=2/3/5$...
```
**tab:hyper:**
```latex
% OLD: Satellite power $P_S$ (Case 1/2/3) & $50$ / $50$ / $60$ dBm & PPO clip ...
% NEW:
Satellite power $P_S$ (Case 1/2/3) & $50$ / $50$ / $50$ dBm & PPO clip $\epsilon_{\mathrm c}$ / epochs & $0.2$ / $6$ \\
```

---

## ⑤ LR sub-actor: tách lr_ck = 5e-5  (tab:hyper ~tex:1084)

```latex
% OLD: LR (sub-actors / critic) & $10^{-4}$ / $3\times10^{-4}$ \\
% NEW:
LR (phase,\,power / $C_k$ / critic) & $10^{-4}$ / $5\times10^{-5}$ / $3\times10^{-4}$ \\
```

---

## ④ eq:vqc_params — thêm enc-norm γ,β (giữ 120/144)  (Prop 1, ~tex:1702–1708)

**Reality:** circuit-core 120/144 = θ(2n_qL) + λ(2n_q) + enc-norm γ,β(4n_q). Ratio 338×/300× & bảng
`tab:param_eff` **giữ nguyên**.

OLD:
```
|θ_VQC| = under{2 n_q L}_{R_y,R_z per layer} + under{2 n_q}_{λ_y,λ_z} = 2 n_q (L+1) = O(n_q L)
```
NEW:
```latex
|\boldsymbol{\theta}_{\mathrm{VQC}}|
= \underbrace{2 n_q L}_{\boldsymbol{\theta}^y,\boldsymbol{\theta}^z\ \mathrm{per\ layer}}
+ \underbrace{2 n_q}_{\boldsymbol{\lambda}_y,\boldsymbol{\lambda}_z}
+ \underbrace{4 n_q}_{\boldsymbol{\gamma},\boldsymbol{\beta}\ (\mathrm{enc.\ LN\ affine})}
= 2 n_q (L{+}3) = \mathcal{O}(n_q L),
```
Kiểm: C1 nq12/L2 = 48+24+48 = **120** ✓ · C2 nq12/L3 = 72+24+48 = **144** ✓.

---

## ⑪ A6 power-fairness: Case-2 main = α0.90, KHÔNG phải 0.95  (Appendix D A6, ~tex:1811+1815+1832) [added 07-17 review]

**Reality:** run main Case-2 (r90→r110 chain = tab:case2_main) chạy `--power-fairness 0.9`
(hyperparameters.json + infer banner "α=0.90 restored"). α0.95 là recipe DNN r53-era, KHÔNG phải
operating point của bảng main.

- ~1811 Setup: `\alpha_{\mathrm{pf}}\in\{0.5,0.7,0.95\}` on Case~2 → `\{0.5,0.7,0.9\}`
- ~1815 Observations: "Case-2 ($\alpha_{\mathrm{pf}}{=}0.95$, Table~\ref{tab:case2_main})" → `0.9`;
  cuối đoạn "($0.5$ vs.\ $0.95$)" → "($0.5$ vs.\ $0.9$)"
- ~1832 tab:abl_power hàng "C2, $0.95$" → "C2, $0.9$"

---

## ⑫ [REVISED 07-18, user chốt] AFFINITY $\mathbf{a}_t \to \mathbf{s}_t$; ACTION GIỮ $\mathbf{a}_t$

**Quyết định mới:** thay vì đổi action→u_t, đổi ký hiệu AFFINITY thành $\mathbf{s}_t$ — giữ convention
RL chuẩn (state s, action a): π(a_t|s_t), transition (s_t,a_t,r_t,V_t), Lemma 1 KHÔNG đụng.
- eq:affinity giờ định nghĩa thẳng $\mathbf{s}_t$; AE compress $\mathbf{s}_t$→z_t; decoder $\hat{s}_t$,
  $s_{\text{norm},t}$; caption fig:ae_process đổi theo.
- Raw observation KHÔNG còn mang ký hiệu s_t (prose gọi "the raw environment observation").
- Entries affinity giữ $a_{k,m}$ (plain, subscript k,m — phân biệt với action bold $\mathbf{a}_t$).
- Figures không phải vẽ lại (đã check: không hình nào in vector a_t).

---

## ⑬ Appendix A: shortfall $s_k \to \varsigma_k$  (~tex:1401-1416) [added 07-17 review]

**Lý do:** $s_k$ = user message (toàn paper + notation table). Appendix A dùng lại $s_k$ cho shortfall
$(R_k^{\min}-R_p(\Gamma_k))^+$ → clash. Đổi shortfall thành $\varsigma_k$ tại ~1401 (định nghĩa),
~1405 ($\sum_{k=1}^{A+B}\varsigma_k$), và prose "shortfalls are homogeneous" quanh 1418 nếu có ký hiệu.

---

## Checklist (tick khi sửa xong)

| # | Mục | Vị trí | Loại |
|---|---|---|---|
| ⑦ | CSI prose → ĝ observe, g thật achieve | ~603 | prose ✅ 07-18 |
| ① | bỏ raw eq:state, state ≡ affinity (s_t) | ~795–814 | đoạn+eq ✅ 07-18 |
| ① | critic dùng affinity s_t | ~816 | prose ✅ 07-18 |
| ④ | eq:ln_z thêm γ,β | ~895 | eq ✅ 07-18 |
| ② | eq:observables → R1 | ~944 | eq ✅ 07-18 |
| ⑨ | train=shots / grad=analytic | ~953 | prose ✅ 07-18 |
| ③ | head SoftmaxPQC + β; eq:pi_k | ~955–963 | prose+eq ✅ 07-18 |
| ① | MDP state = affinity s_t | ~973 | prose ✅ 07-18 |
| ⑩ | reward/QoS trên g thật | sau ~987 | prose ✅ 07-18 |
| ⑧ | eq:spsa → Jacobian-form | ~1043 | eq+prose ✅ 07-18 |
| ⑥ | P_S Case-3 60→50 (sim setup) | ~1062 | số ✅ 07-18 |
| ⑥ | P_S Case-3 60→50 (tab:hyper) | ~1078 | số ✅ 07-18 |
| ⑤ | lr_ck tách 5e-5 | ~1084 | số ✅ 07-18 |
| ④ | eq:vqc_params → 2n_q(L+3) | ~1702 | eq ✅ 07-18 |
| ⑪ | A6 α Case-2 0.95→0.9 (3 chỗ) | ~1811/1815/1832 | số ✅ 07-18 |
| ⑫ | [REVISED] affinity a_t→s_t, action giữ a_t | eq:affinity + AE §, MDP | ký hiệu ✅ 07-18 |
| ⑬ | App-A shortfall s_k→ς_k | ~1401-1416 | ký hiệu ✅ 07-18 |

**Không đổi (giờ đã đúng):** `eq:affinity` (ĝ) · `tab:param_eff` (120/144, 338×/300×) ·
`tab:abl_readout` counts (12/23/33-41/34-42) · system-model (SINR/rate/channel/noise) · GAE/PopArt critic.

**Chưa đụng (domain user):** Related Works, Intro.
