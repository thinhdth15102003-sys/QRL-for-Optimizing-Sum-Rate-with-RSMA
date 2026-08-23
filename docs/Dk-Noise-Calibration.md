# Noise + D_k calibration against Tan (chốt 2026-07-24)

Ghi lại vì sao `params.py` mang đúng hai giá trị này, để không phải đo lại.
Reference: `docs/Flexible_IRS-Assisted_Grouped_Rate-Splitting_...pdf` (Tan et al.,
IEEE TCCN vol. 12, 2026, pp. 3719-3730).

---

## 1. Noise — GIỮ `noise_mean_dBW = -30`, `noise_var_dBW = 10`

Tan Sec. V-A (p. 3728) viết nguyên văn: *the Gaussian white noise for each user ... is
set to* `n_0 ~ N(0, 10)[dB]`. Fig. 8 (p. 3729) quét **"Variance of Gaussian white noise
σ₁²"** từ 8→20 dB, mặc định 10 dB. Hai chỗ này khớp nhau: **σ₁² là VARIANCE**, mean = 0 dB
(= 0 dBm = −30 dBW). Đó chính là `params.py`.

Đã test và LOẠI cách đọc σ₁² = noise power:

| σ₁² (dB) | (a) đọc là POWER (mean = x−30) | (b) đọc là VARIANCE (mean cố định −30) | Tan Fig. 8 |
|---|---|---|---|
| 8 | 1.582 | 1.856 | 1.57 |
| 10 | 1.455 | 1.847 | 1.52 |
| 14 | 1.191 | 1.831 | 1.42 |
| 20 | 0.755 | 1.809 | 1.33 |

(a) khớp **số** ở đầu trái (1.582 vs 1.57, +0.8%) nhưng dốc hơn Tan nhiều và **trái với
text**. (b) khớp text nhưng gần như phẳng. → chọn (b), coi trùng khớp của (a) là ngẫu nhiên.

⚠ `params.py` đã có lúc bị đổi sang −22 dBW theo cách đọc (a) rồi **revert**. Mọi số đo
mang nhãn "@ −22 dBW" / "@ Tan noise 8 dBm" trong `analysis/data/dk_sweep_tan*.log` là ở
setting ĐÃ BỎ — cần đo lại trước khi trích vào paper.

### Vì sao AO của mình (1.837) > Tan (1.52) dù cùng tham số

**Trước hết: khoảng cách nhỏ hơn vẻ ngoài rất nhiều.** Đo bằng
`analysis/probe_link_budget.py --run results/result_234` (2026-08-03):

| | R_tot | per-user | γ = 2^(R/K)−1 |
|---|---|---|---|
| Tan | 1.5200 | 0.3040 | **−6.30 dB** |
| mình (AO) | 1.8366 | 0.3673 | **−5.38 dB** |

→ chênh **+20.8% rate nhưng chỉ +0.92 dB SINR hiệu dụng**. Ở điểm vận hành low-SINR này
(direct-link median −5.67 dB, capacity direct-only 0.346 bps/Hz ≈ đúng per-user AO 0.367)
`log₂(1+γ)` gần tuyến tính, nên sai khác dưới 1 dB của bất kỳ khâu mô hình nào cũng hiện
ra thành ~20% rate. **Đây không phải dấu hiệu chuẩn hoá hụt.**

⚠ **ĐÃ SỬA 2026-08-03 — luận điểm `g_RU` trước đây ghi SAI CHIỀU.**
Bản cũ viết "`g_RU` của Tan không có path loss" như một lý do khiến số của mình cao hơn.
Đo trực tiếp `_irs_user_path_loss`: term đó là **median −17.4 dB** (min −66.1, max −13.2).
Bỏ nó đi làm nhánh IRS của **họ MẠNH HƠN của mình 17.4 dB**, tức đẩy số của họ **LÊN**,
không xuống. Luận điểm đó không giải thích được chiều của khoảng cách.

Cái còn đứng vững:

1. **Solver của họ yếu hơn.** Alg. 1-4 = SDR + rank-1 extraction (relaxation, mất tối ưu)
   + RMCG + simulated annealing. AO của mình là coordinate-ascent + grid, đo được là mạnh
   hơn. → baseline của mình **khó hơn** baseline của họ, vượt nó thì thuyết phục hơn.
2. **AO của mình chạy DƯỚI ràng buộc QoS** `D_k = 0.05` (đạt 100%), còn Tan thực tế
   `R^min = 0` (xem §2). Ràng buộc thêm chỉ có thể làm bài toán của mình **khó hơn**.

Giả thuyết CHƯA ĐO cho chiều còn lại: trong grouped-RSMA, common stream phải decode được
bởi **mọi** user trong nhóm nên bị ghim bởi user tệ nhất; `g_RU` mạnh hơn 17.4 dB đồng thời
làm tăng nhiễu liên-user, nên kênh mạnh hơn không nhất thiết cho sum-rate cao hơn. Muốn
dùng lập luận này trong paper thì phải đo, đừng viết như đã biết.

Các tham số còn lại đã khớp sẵn: K=5, N=24, P_S=50 dBm, f=1.58 GHz, h_SR=800 km, góc
60°-90°, d_SU=d_SR, G_S=G_U=60 dBi, ε_SR=ε_SU ~ N(4,0.1), ε₁..ε₅=0.001.

⚠ Cả hai bên đều dùng noise **cao hơn thermal thực tế ~100 dB** (thermal @20 MHz, NF 3 dB
≈ −98 dBm). Nó là tham số hiệu chỉnh để SINR về vùng hợp lý, không phải đại lượng vật lý —
đừng claim "setting của mình thực tế hơn" ở trục noise.

---

## 2. D_k = 0.05 bps/Hz

**Tan KHÔNG đặt giá trị nào.** Eq. (15g) có ràng buộc `log₂(1+γ_{k,p}) ≥ R_{k,p}^min`,
nhưng Sec. V-A liệt kê đủ mọi tham số mà không nêu `R^min`, và không figure nào quét nó →
thực tế `R^min = 0`, (15h) `R_{k,p} ≥ 0` đã đủ. **QoS là đóng góp riêng của paper mình.**

⚠ Khác định nghĩa: ràng buộc của Tan bó **private rate `R_{k,p}`**; `D_k` của mình bó
**tổng `R_private + C_k`**. Phải nói rõ trong paper, nếu không reviewer so nhầm.

### Vì sao 0.05

D_k tính theo % rate mỗi user mà AO chia được (`R_tot^AO / K`):

| D_k | Case 1 | Case 2 | Case 3 | best QoS đạt được (C1/C2/C3) |
|---|---|---|---|---|
| **0.05** | 17% | 36% | 55% | **100 / 100 / 100%** |
| 0.06 | 20% | 43% | 66% | 100 / 100 / 97.7% |
| 0.08 | 27% | 58% | 89% | 100 / 100 / 68% |
| 0.10 | 34% | 72% | 111% | 100 / 91.5 / 67% |
| 0.15 | 51% | 108% | 167% | 100 / 75 / — |
| 0.20 | 67% | 144% | 222% | 100 / **59** / — |

Ba điều kiện gặp nhau ở 0.05: nằm trong dải thông lệ 10-40% của min-rate; **cả ba case
cùng khả thi 100%**; và AO vẫn khoẻ ở C1/C2 (100% QoS) nên baseline không bị bóp.

**D_k ≥ 0.15 là trần cứng, không phải lựa chọn phong cách**: ở Case 2 nó đòi mỗi user
>100% lượng rate trung bình cả hệ chia được → `best QoS` tụt còn 59-75%, cả agent lẫn AO
đều vi phạm QoS, so sánh mất nghĩa.

Ngược lại D_k cao **co dư địa của chính mình** ở Case 1: dư địa (trần − AO) rơi từ +1.383
@0.05 xuống +0.100 @0.20. D_k cao siết cả hai đầu.

⚠ Bảng % ở trên tính từ AO đo @ −22 dBW. Ở −30 dBW per-user rate cao hơn (C1 0.369,
C2 0.165) nên cùng D_k sẽ **nhẹ hơn** — thứ tự và kết luận không đổi, nhưng con số % cần
đo lại nếu trích vào paper.
