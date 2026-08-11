# Tối ưu hoá log-loss — Santander Product Recommendation

Cài đặt từ đầu (chỉ dùng `numpy`) **4 thuật toán** tối ưu cho hồi quy logistic:

$$\min_w \; F(w) = \underbrace{\frac1n\sum_i \big[\log(1+e^{x_i^\top w}) - y_i x_i^\top w\big] + \frac{\lambda_2}{2}\|w\|^2}_{f:\ \text{trơn}} \;+\; \underbrace{\lambda_1\|w\|_1}_{g:\ \text{không trơn}}$$

| | Bài toán giải được | Ý tưởng |
|---|---|---|
| **GD** | $f$ trơn | $w \leftarrow w - t\nabla f(w)$ |
| **SGD** | $f$ trơn | như GD nhưng gradient tính trên mini-batch |
| **Subgradient** | $F$ có L1 | thay $\nabla f$ bằng một phần tử của $\partial F$ |
| **Proximal** | $F$ có L1 | bước gradient trên $f$ rồi `prox` cho $g$ (ISTA/FISTA) |

## Các tệp

| Tệp | Nội dung |
|---|---|
| `logloss_optim.py` | 4 thuật toán + hàm `subgradient` và `prox_l1` |
| `santander_data.py` | Đọc `train_ver2.csv` (2.3 GB) theo chunk → ma trận X, nhãn Y |
| `run_optim.py` | Thực nghiệm 2 vòng + vẽ hình + nhận xét + MAP@7 |
| `optimization_logloss.ipynb` | Notebook giải thích lý thuyết kèm code chạy được |
| `figures/`, `cache/` | Hình và cache, tự sinh |

## Chạy

```bash
python santander_data.py            # dựng dữ liệu (lần đầu ~40 giây, sau đó có cache)
python run_optim.py                 # thực nghiệm đầy đủ
python run_optim.py --fast          # lấy mẫu 100k dòng cho nhanh
python run_optim.py --map7          # + huấn luyện 24 sản phẩm, đo MAP@7
python run_optim.py --product ind_cco_fin_ult1 --vong 100
```

## Bài toán

Nhãn `y = 1` nếu khách hàng **thêm mới** sản phẩm ở tháng *t* (có ở *t*, không có ở
*t−1*) — đúng định nghĩa của cuộc thi.

- **Đặc trưng (93 cột)**: 1 intercept + 7 cột số (age, antiguedad, log renta, ind_nuevo,
  indrel, ind_actividad_cliente, alta_year) + 61 cột one-hot (giới tính, phân khúc, kênh,
  tỉnh, quan hệ…) + **24 cờ sở hữu ở tháng t−1** (tín hiệu mạnh nhất).
  Tất cả chuẩn hoá z-score theo thống kê tập train; intercept không bị phạt.
- **Train**: 2015-05 → 2015-06, 628.603 khách · **Validation**: 2016-04 → 2016-05, 926.663 khách.
- Rất mất cân bằng: sản phẩm phổ biến nhất `ind_recibo_ult1` cũng chỉ 1,435% khách thêm mới.

## Hai hàm được yêu cầu định nghĩa

```python
subgradient(w, X, y, lam1, lam2, idx=None, cach_chon="min_norm")
```
Trả về **một** phần tử của $\partial F(w) = \nabla f(w) + \lambda_1\partial\|w\|_1$, trong đó
$\partial|w_j| = \{\mathrm{sign}(w_j)\}$ nếu $w_j\neq0$ và $=[-1,1]$ nếu $w_j=0$.
Tại $w_j=0$ phải *chọn*; `cach_chon="min_norm"` lấy phần tử chuẩn nhỏ nhất
$g_j = \nabla_j f - \mathrm{clip}(\nabla_j f, -\lambda_1, \lambda_1)$, nên $\|g\|=0$ **khi và chỉ khi**
$w$ tối ưu → dùng luôn làm kiểm tra KKT. `idx` = chỉ số mini-batch → dưới đạo hàm ngẫu nhiên.

```python
prox_l1(v, t, lam1)          # soft-thresholding
prox_elastic_net(v, t, lam1, lam2)
```
$\mathrm{prox}_{tg}(v)=\arg\min_u\{g(u)+\|u-v\|^2/2t\}$; với $g=\lambda_1\|\cdot\|_1$ có nghiệm dạng đóng
$\mathrm{sign}(v_j)\max(|v_j|-t\lambda_1,0)$ — **đặt hẳn** hệ số nhỏ về 0.

## Bố cục thực nghiệm

**Vòng 1 — cả 4 thuật toán dùng chung một bộ tham số**
($w_0=0$, bước $1/L$, 60 lần duyệt dữ liệu, $B=1024$, không bật cải tiến nào):

| thuật toán | log-loss va | nnz | giây |
|---|---|---|---|
| GD | 0,075648 | 85 | 7,3 |
| SGD | **0,044315** | 85 | 65,4 |
| Subgradient | 0,076268 | 29 | 12,1 |
| Proximal (ISTA) | 0,076268 | **23** | 9,6 |

Bước $1/L$ là bước *của GD* — quá nhỏ với SGD (mỗi epoch SGD đi 614 bước) và sai kiểu
với subgradient (lý thuyết đòi bước giảm dần). Dùng chung tham số **không** đồng nghĩa
với công bằng.

**Vòng 2 — mỗi thuật toán dò tham số riêng**, giữ nguyên ngân sách 60 lần duyệt dữ liệu,
chọn theo log-loss validation. $\lambda_1$ được coi là tham số riêng của
Subgradient/Proximal (GD/SGD không dùng được L1). Bảng kết quả và nhận xét đầy đủ do
`run_optim.py` in ra.

**Hình sinh ra trong `figures/`**

| Hình | Nội dung |
|---|---|
| `01_vong1.png`, `02_vong2.png` | $F(w_k)-F^\star$ theo số lần duyệt dữ liệu và theo thời gian |
| `01b`, `02b` | log-loss validation — chỉ số so sánh được giữa cả 4 thuật toán |
| `03_do_thua.png` | số hệ số khác 0 |
| `04_vong1_vs_vong2.png` | tinh chỉnh tham số giúp được bao nhiêu |

## Ghi chú khi đọc kết quả

- GD/SGD tối ưu $f$ (không có L1), Subgradient/Proximal tối ưu $F=f+\lambda_1\|w\|_1$ →
  cột `F` là hai hàm khác nhau. Cột **log-loss trần** mới so chéo được.
- Mỗi đường trong hình hội tụ được đo với $F^\star$ của **đúng bài toán nó giải**.
- Cột `giây` có tính cả thời gian tính log-loss để vẽ đồ thị (giống nhau cho cả 4 thuật
  toán vì đều ghi 60 lần).
- 9 trong 93 cột là hằng số trên tập train (mức hạng mục không xuất hiện) nên hệ số của
  chúng luôn bằng 0 → `nnz` tối đa thực tế là 84.
- MAP@7 báo cáo 2 kiểu: trên khách **có** thêm mới, và trên **toàn bộ** khách (cách
  Kaggle chấm, nhỏ hơn khoảng 25 lần vì đa số khách không thêm gì).
