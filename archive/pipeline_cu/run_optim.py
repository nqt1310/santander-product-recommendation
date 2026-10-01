"""
Thuc nghiem: toi uu log-loss cho Santander bang 4 thuat toan
    GD, SGD, Subgradient, Proximal

Bo cuc:
    0. Mo ta bo dac trung
    1. VONG 1 - ca 4 thuat toan dung CHUNG mot bo tham so   -> ket qua
    2. VONG 2 - moi thuat toan do tham so RIENG             -> ket qua
    3. Nhan xet

Chay:
    python run_optim.py
    python run_optim.py --fast        # lay mau 100k dong cho nhanh
    python run_optim.py --map7        # + huan luyen 24 san pham, do MAP@7
"""

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
if __name__ == "__main__":
    matplotlib.use("Agg")          # chay dang script thi khong can cua so do hoa
import matplotlib.pyplot as plt

from logloss_optim import (logloss, objective, subgradient, lipschitz,
                           kiem_tra_grad, du_doan, gradient_descent, sgd,
                           subgradient_method, proximal_gradient)
from santander_data import (build_dataset, TARGET_COLS, NUM_COLS, CAT_COLS,
                            TRAIN_PAIR, VALID_PAIR)

# in duoc tieng Viet tren console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MAU = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # 4 mau cho 4 thuat toan
TEN_4 = ["GD", "SGD", "Subgradient", "Proximal"]


# ------------------------------------------------------------------ do luong
def auc(y, p):
    """AUC = xac suat mau duong duoc xep hang cao hon mau am."""
    thu_tu = np.argsort(p)
    hang = np.empty(len(p))
    hang[thu_tu] = np.arange(1, len(p) + 1)
    n_duong = y.sum()
    n_am = len(y) - n_duong
    if n_duong == 0 or n_am == 0:
        return np.nan
    return (hang[y == 1].sum() - n_duong * (n_duong + 1) / 2) / (n_duong * n_am)


def apk(that, du_doan_top, k=7):
    """Average precision @ k cho 1 khach hang."""
    if len(that) == 0:
        return 0.0
    diem, trung = 0.0, 0
    for i, sp in enumerate(du_doan_top[:k]):
        if sp in that:
            trung += 1
            diem += trung / (i + 1)
    return diem / min(len(that), k)


def map_at_7(P, Y, da_co, chi_khach_mua=True):
    """
    MAP@7. San pham da so huu o thang t-1 thi khong the "them moi" -> bo ra.
    chi_khach_mua=True : trung binh tren khach CO them moi
                 =False: trung binh tren TOAN BO khach (cach Kaggle cham, ~0.03)
    """
    diem = np.where(da_co > 0, -np.inf, P)
    top = np.argsort(-diem, axis=1)[:, :7]
    if chi_khach_mua:
        hang = np.where(Y.sum(1) > 0)[0]
    else:
        hang = np.arange(len(Y))
    tong = sum(apk(set(np.where(Y[i])[0]), list(top[i])) for i in hang)
    return tong / len(hang)


def danh_gia(w, ls, X_tr, y_tr, X_va, y_va, lam1, lam2):
    """Bo chi so dung chung de so sanh 4 thuat toan."""
    if not np.all(np.isfinite(w)):          # cau hinh phan ky
        return {"F": np.inf, "ll_tr": np.inf, "ll_va": np.inf, "auc": np.nan,
                "nnz": int(np.sum(w != 0)), "giay": ls["time"][-1]}
    return {"F": objective(w, X_tr, y_tr, lam1, lam2),
            "ll_tr": logloss(w, X_tr, y_tr),        # log-loss tran (khong phat)
            "ll_va": logloss(w, X_va, y_va),
            "auc": auc(y_va, du_doan(X_va, w)),
            "nnz": int(np.sum(w != 0)),
            "giay": ls["time"][-1]}


def in_bang(dong, tieu_de):
    print("\n" + tieu_de)
    print("%-13s%-34s%12s%13s%13s%8s%6s%8s" %
          ("thuật toán", "cấu hình", "F (train)", "log-loss tr", "log-loss va",
           "AUC", "nnz", "giây"))
    print("-" * 107)
    for ten, cfg, e in dong:
        print("%-13s%-34s%12.8f%13.6f%13.6f%8.4f%6d%8.2f" %
              (ten, cfg, e["F"], e["ll_tr"], e["ll_va"], e["auc"], e["nnz"], e["giay"]))


# --------------------------------------------------------------------- ve hinh
def dep(ax, xlabel, ylabel, title=None):
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11.5, loc="left")


def tach_nhan(y_cuoi, y_min, y_max, khoang_cach=0.055):
    """
    Tinh do lech doc (don vi point) de cac nhan cuoi duong khong de len nhau.
    Duyet cac duong tu duoi len, cai nao qua sat cai truoc thi day len mot chut.
    """
    lech = np.zeros(len(y_cuoi))
    truoc = -np.inf
    for i in np.argsort(y_cuoi):
        vi_tri = (y_cuoi[i] - y_min) / (y_max - y_min)      # doi ve ti le 0..1
        if vi_tri - truoc < khoang_cach:
            lech[i] = (truoc + khoang_cach - vi_tri) * 300  # ~300 point cho ca truc
            truoc = truoc + khoang_cach
        else:
            truoc = vi_tri
    return lech


def ve_hoi_tu(ket_qua, F_sao, duong_dan, tieu_de):
    """
    2 khung: hoi tu theo so lan duyet du lieu va theo thoi gian thuc.
    F_sao la list: moi thuat toan so voi F* cua BAI TOAN NO GIAI
    (GD/SGD giai bai toan tron, Subgradient/Proximal giai bai toan co L1).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, cot, nhan in zip(axes, ["vong", "time"],
                             ["số lần duyệt dữ liệu", "thời gian chạy (giây)"]):
        gap = [np.maximum(np.array(ls["F"]) - F_sao[i], 1e-16)
               for i, (_, ls) in enumerate(ket_qua)]
        # truc y la thang log nen tinh do lech nhan tren log10
        cuoi = np.log10([g[-1] for g in gap])
        tat_ca = np.log10(np.concatenate(gap))
        lech = tach_nhan(cuoi, tat_ca.min(), tat_ca.max())

        for i, (ten, ls) in enumerate(ket_qua):
            x = ls[cot]
            ax.plot(x, gap[i], color=MAU[i], linewidth=2, label=ten)
            ax.annotate(ten, (x[-1], gap[i][-1]), color=MAU[i], fontsize=8.5,
                        fontweight="bold", xytext=(4, lech[i]),
                        textcoords="offset points", va="center")
        ax.set_yscale("log")
        dep(ax, nhan, "F(w) - F*" if cot == "vong" else "")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(tieu_de, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)
    print("  -> " + duong_dan)


def ve_val(ket_qua, duong_dan, tieu_de, moc=None):
    """Log-loss validation - chi so SO SANH DUOC giua ca 4 thuat toan."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    tat_ca = np.concatenate([ls["val"] for _, ls in ket_qua])
    lech = tach_nhan([ls["val"][-1] for _, ls in ket_qua], tat_ca.min(), tat_ca.max())

    for i, (ten, ls) in enumerate(ket_qua):
        ax.plot(ls["vong"], ls["val"], color=MAU[i], linewidth=2, label=ten)
        ax.annotate(ten, (ls["vong"][-1], ls["val"][-1]), color=MAU[i],
                    fontsize=8.5, fontweight="bold", xytext=(5, lech[i]),
                    textcoords="offset points", va="center")
    if moc is not None:
        ax.axhline(moc, color="#898781", linewidth=1, linestyle=":")
        ax.annotate("mô hình hằng số", (0, moc), color="#898781", fontsize=8,
                    xytext=(2, 3), textcoords="offset points")
    dep(ax, "số lần duyệt dữ liệu", "log-loss trên validation", tieu_de)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)
    print("  -> " + duong_dan)


def ve_do_thua(ket_qua, duong_dan, d, tieu_de):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    lech = tach_nhan([ls["nnz"][-1] for _, ls in ket_qua], 0, d * 1.12)

    for i, (ten, ls) in enumerate(ket_qua):
        ax.plot(ls["vong"], ls["nnz"], color=MAU[i], linewidth=2, label=ten)
        ax.annotate("%s: %d" % (ten, ls["nnz"][-1]), (ls["vong"][-1], ls["nnz"][-1]),
                    color=MAU[i], fontsize=9, fontweight="bold", xytext=(5, lech[i]),
                    textcoords="offset points", va="center")
    ax.axhline(d, color="#898781", linewidth=1, linestyle=":")
    ax.annotate("tổng %d hệ số" % d, (0, d), color="#898781", fontsize=8,
                xytext=(2, 3), textcoords="offset points")
    ax.set_ylim(-2, d * 1.12)
    dep(ax, "số lần duyệt dữ liệu", "số hệ số khác 0", tieu_de)
    ax.legend(frameon=False, fontsize=9, loc="center left")
    fig.tight_layout(rect=(0, 0, 0.84, 1))
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)
    print("  -> " + duong_dan)


def ve_2_vong(ev1, ev2, duong_dan):
    """So sanh log-loss validation: vong 1 (chung tham so) vs vong 2 (tinh chinh)."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(4)
    v1 = [e["ll_va"] for e in ev1]
    v2 = [e["ll_va"] for e in ev2]
    ax.bar(x - 0.19, v1, 0.36, color=MAU[0], label="Vòng 1 - chung tham số")
    ax.bar(x + 0.19, v2, 0.36, color=MAU[1], label="Vòng 2 - tinh chỉnh riêng")
    for i in range(4):
        ax.annotate("%.5f" % v1[i], (x[i] - 0.19, v1[i]), ha="center",
                    fontsize=8.5, xytext=(0, 3), textcoords="offset points")
        ax.annotate("%.5f" % v2[i], (x[i] + 0.19, v2[i]), ha="center",
                    fontsize=8.5, xytext=(0, 3), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(TEN_4)
    ax.set_ylim(0, max(v1 + v2) * 1.22)
    dep(ax, "", "log-loss validation (thấp hơn = tốt hơn)",
        "Tinh chỉnh tham số giúp được bao nhiêu?")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(duong_dan, dpi=150)
    plt.close(fig)
    print("  -> " + duong_dan)


# ------------------------------------------------------------ 0. mo ta features
def mo_ta_features(ten_cot, X_tr, X_va, y_tr, y_va):
    so_num = len(NUM_COLS)
    so_lag = len(TARGET_COLS)
    so_cat = len(ten_cot) - 1 - so_num - so_lag

    print("=" * 96)
    print("0. BỘ ĐẶC TRƯNG")
    print("=" * 96)
    print("Ma trận thiết kế: train %s x %d | validation %s x %d" %
          ("{:,}".format(X_tr.shape[0]), X_tr.shape[1],
           "{:,}".format(X_va.shape[0]), X_va.shape[1]))
    print("Nhãn y = 1 nếu khách hàng THÊM MỚI sản phẩm ở tháng t (có ở t, không có ở t-1)")
    print("  train: %s -> %s, tỉ lệ dương %.3f%%" % (TRAIN_PAIR[0], TRAIN_PAIR[1],
                                                     y_tr.mean() * 100))
    print("  valid: %s -> %s, tỉ lệ dương %.3f%%" % (VALID_PAIR[0], VALID_PAIR[1],
                                                     y_va.mean() * 100))

    print("\n%-26s%8s   %s" % ("nhóm", "số cột", "mô tả"))
    print("-" * 96)
    print("%-26s%8d   %s" % ("intercept", 1, "hằng số 1, KHÔNG bị phạt"))
    print("%-26s%8d   %s" % ("số (đã chuẩn hoá)", so_num, ", ".join(NUM_COLS)))
    print("%-26s%8d   %s" % ("hạng mục (one-hot)", so_cat, ", ".join(CAT_COLS)))
    print("%-26s%8d   %s" % ("sở hữu tháng t-1 (lag)", so_lag,
                             "24 cờ sở hữu sản phẩm tháng trước - tín hiệu mạnh nhất"))
    print("-" * 96)
    print("%-26s%8d" % ("TỔNG", len(ten_cot)))

    chet = int(np.sum(X_tr[:, 1:].std(axis=0) < 1e-12))
    if chet > 0:
        print("\n(%d cột one-hot là hằng số trên tập train - mức hạng mục không xuất "
              "hiện.\n Hệ số của chúng luôn bằng 0 với mọi thuật toán, nên nnz tối đa "
              "thực tế là %d.)" % (chet, len(ten_cot) - chet))

    print("\nGhi chú tiền xử lý:")
    print("  - age cắt về [18, 95]; antiguedad cờ -999999 -> NaN rồi cắt [0, 260]")
    print("  - renta lấy log1p (thu nhập lệch phải rất mạnh), NaN điền bằng trung vị")
    print("  - biến hạng mục nhiều mức gộp đuôi thành OTHER")
    print("  - mọi cột chuẩn hoá z-score theo thống kê của TẬP TRAIN")
    print("    -> điều kiện số tốt hơn, GD/SGD hội tụ nhanh hơn nhiều")


# --------------------------------------------------------------- 2. do tham so
def do_tham_so(ten, luoi, chay, X_tr, y_tr, X_va, y_va, lam1_chung, lam2):
    """Chay het luoi tham so, in top 5, tra ve cau hinh tot nhat theo val log-loss."""
    kq = []
    for cfg, tham_so in luoi:
        w, ls, _ = chay(tham_so)
        # lam1_chung=None nghia la moi cau hinh tu mang lam1 cua no
        l1 = tham_so.get("lam1", 0.0) if lam1_chung is None else lam1_chung
        kq.append((cfg, danh_gia(w, ls, X_tr, y_tr, X_va, y_va, l1, lam2), w, ls, tham_so))
    kq.sort(key=lambda x: x[1]["ll_va"])

    print("\n  %s - dò %d cấu hình, xếp theo log-loss validation:" % (ten, len(luoi)))
    for cfg, e, _, _, _ in kq[:5]:
        print("    %-34s val=%.6f  F=%.8f  nnz=%3d  %.1fs" %
              (cfg, e["ll_va"], e["F"], e["nnz"], e["giay"]))
    if len(kq) > 5:
        cfg, e = kq[-1][0], kq[-1][1]
        te = "%.6f" % e["ll_va"] if np.isfinite(e["ll_va"]) else "phân kỳ"
        print("    %-34s val=%s" % ("(tệ nhất) " + cfg, te))
    return kq[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", default="ind_recibo_ult1")
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--lam2", type=float, default=1e-6)
    ap.add_argument("--vong", type=int, default=60, help="so lan duyet du lieu")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--map7", action="store_true")
    args = ap.parse_args()

    du_lieu = build_dataset()
    X_tr = du_lieu["X_tr"].astype(float)
    Y_tr = du_lieu["Y_tr"]
    X_va = du_lieu["X_va"].astype(float)
    Y_va = du_lieu["Y_va"]
    ten_cot = du_lieu["feature_names"]

    if args.fast:
        rs = np.random.RandomState(0)
        i = rs.choice(len(X_tr), 100000, replace=False)
        j = rs.choice(len(X_va), 100000, replace=False)
        X_tr, Y_tr, X_va, Y_va = X_tr[i], Y_tr[i], X_va[j], Y_va[j]

    k = TARGET_COLS.index(args.product)
    y_tr = Y_tr[:, k].astype(float)
    y_va = Y_va[:, k].astype(float)
    n, d = X_tr.shape
    lam1, lam2, VONG, B = args.lam1, args.lam2, args.vong, args.batch
    val = (X_va, y_va)

    # ---------------------------------------------------------- 0. features
    mo_ta_features(ten_cot, X_tr, X_va, y_tr, y_va)

    print("\n" + "=" * 96)
    print("BÀI TOÁN: dự đoán khách hàng THÊM MỚI '%s'" % args.product)
    print("=" * 96)
    L = lipschitz(X_tr, lam2)
    eta = 1 / L
    w_thu = np.random.RandomState(0).randn(d) * 0.05
    print("kiểm tra gradient (sai phân hữu hạn): %.2e  (phải rất nhỏ)" %
          kiem_tra_grad(w_thu, X_tr, y_tr, lam2))
    print("hằng số Lipschitz L = %.4f  ->  bước an toàn 1/L = %.4f" % (L, eta))
    p0 = y_tr.mean()
    moc = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
    print("log-loss của mô hình hằng số = %.6f   <-- mốc cần vượt" % moc)
    print("lam1 = %g | lam2 = %g | ngân sách = %d lần duyệt dữ liệu" % (lam1, lam2, VONG))

    # F* tham chieu: chay FISTA that lau
    bo_nho_F = {}

    def F_sao_cua(l1):
        if l1 not in bo_nho_F:
            w, _, _ = proximal_gradient(X_tr, y_tr, l1, lam2, so_vong=1500,
                                        gia_toc=True,
                                        kieu_prox="l1" if l1 > 0 else "khong")
            bo_nho_F[l1] = (objective(w, X_tr, y_tr, l1, lam2), w)
        return bo_nho_F[l1][0]

    t0 = time.time()
    F_sao, F_sao_tron = F_sao_cua(lam1), F_sao_cua(0.0)
    print("F* (lam1=%g) = %.10f | F* (chỉ phần trơn) = %.10f   [%.1fs]" %
          (lam1, F_sao, F_sao_tron, time.time() - t0))
    print("\nLưu ý khi đọc bảng: GD/SGD chỉ tối ưu phần TRƠN f (không xử lý được L1),")
    print("Subgradient/Proximal tối ưu F = f + lam1*||w||_1. Cột log-loss là log-loss")
    print("TRẦN (không cộng phạt) nên so sánh chéo được giữa cả 4 thuật toán.")

    # ------------------------------------------------------------ 1. vong 1
    print("\n" + "=" * 96)
    print("1. VÒNG 1 - CẢ 4 THUẬT TOÁN DÙNG CHUNG MỘT BỘ THAM SỐ")
    print("=" * 96)
    print("w0 = 0 | bước học = 1/L = %.4f (cố định) | %d lần duyệt dữ liệu | B = %d"
          % (eta, VONG, B))
    print("không bật line search / momentum / trung bình / gia tốc")

    kq1 = [
        ("bước 1/L", gradient_descent(X_tr, y_tr, lam2, buoc=eta, so_vong=VONG, val=val)),
        ("bước 1/L, B=%d" % B, sgd(X_tr, y_tr, lam2, batch=B, so_epoch=VONG,
                                   eta0=eta, kieu_buoc="hang_so", val=val)),
        ("bước 1/L", subgradient_method(X_tr, y_tr, lam1, lam2, eta0=eta,
                                        so_vong=VONG, kieu_buoc="hang_so", val=val)),
        ("ISTA, bước 1/L", proximal_gradient(X_tr, y_tr, lam1, lam2, buoc=eta,
                                             so_vong=VONG, val=val)),
    ]
    lam1_theo_tt = [0.0, 0.0, lam1, lam1]          # GD/SGD khong co L1
    ev1 = [danh_gia(w, ls, X_tr, y_tr, X_va, y_va, l1, lam2)
           for (_, (w, ls, _)), l1 in zip(kq1, lam1_theo_tt)]

    in_bang([(TEN_4[i], kq1[i][0], ev1[i]) for i in range(4)], "KẾT QUẢ VÒNG 1")

    ve1 = [(TEN_4[i], kq1[i][1][1]) for i in range(4)]
    ve_hoi_tu(ve1, [F_sao_tron, F_sao_tron, F_sao, F_sao],
              os.path.join(FIG_DIR, "01_vong1.png"),
              "Vòng 1 - chung tham số (bước 1/L) - " + args.product)
    ve_val(ve1, os.path.join(FIG_DIR, "01b_vong1_val.png"),
           "Vòng 1 - log-loss validation", moc)

    # ------------------------------------------------------------ 2. vong 2
    print("\n" + "=" * 96)
    print("2. VÒNG 2 - DÒ THAM SỐ RIÊNG CHO TỪNG THUẬT TOÁN")
    print("=" * 96)
    print("Giữ nguyên ngân sách %d lần duyệt dữ liệu; chọn theo log-loss validation."
          % VONG)

    # --- luoi tham so cua tung thuat toan ---
    luoi_gd = [("bước %g/L" % c, {"buoc": eta * c}) for c in (1, 2, 5, 10, 20, 50, 100, 200)]
    luoi_gd += [("line search Armijo", {"line_search": True})]

    luoi_sgd = []
    for b in (256, 1024, 8192):
        for kb in ("hang_so", "1/k", "1/sqrt(k)"):
            luoi_sgd.append(("B=%d, bước %s" % (b, kb), {"batch": b, "eta0": eta,
                                                         "kieu_buoc": kb}))
    for e0 in (0.5, 2.0):
        luoi_sgd.append(("B=%d, eta0=%g, bước 1/k" % (B, e0),
                         {"batch": B, "eta0": e0, "kieu_buoc": "1/k"}))
    luoi_sgd.append(("B=%d, momentum 0.9" % B,
                     {"batch": B, "eta0": eta, "kieu_buoc": "1/k", "momentum": 0.9}))
    luoi_sgd.append(("B=%d, trung bình Polyak" % B,
                     {"batch": B, "eta0": eta, "kieu_buoc": "1/sqrt(k)",
                      "trung_binh": True}))

    # lam1 la tham so cua RIENG subgradient/proximal (GD/SGD khong dung duoc L1)
    luoi_sub, luoi_prox = [], []
    for l1 in (lam1, lam1 / 10):
        for c in (1, 5, 20, 50, 200):
            for kb in ("hang_so", "1/sqrt(k)", "1/k"):
                luoi_sub.append(("eta0=%g/L, %s, lam1=%g" % (c, kb, l1),
                                 {"eta0": eta * c, "kieu_buoc": kb, "lam1": l1}))
        for gt, ten_gt in ((False, "ISTA"), (True, "FISTA")):
            for lsr, ten_ls in ((False, "bước 1/L"), (True, "line search")):
                luoi_prox.append(("%s, %s, lam1=%g" % (ten_gt, ten_ls, l1),
                                  {"gia_toc": gt, "line_search": lsr, "lam1": l1}))
        luoi_prox.append(("FISTA, prox Elastic-Net, lam1=%g" % l1,
                          {"gia_toc": True, "kieu_prox": "elastic_net", "lam1": l1}))

    t0 = time.time()
    tot_gd = do_tham_so("GD", luoi_gd,
                        lambda ts: gradient_descent(X_tr, y_tr, lam2, so_vong=VONG,
                                                    val=val, **ts),
                        X_tr, y_tr, X_va, y_va, 0.0, lam2)
    tot_sgd = do_tham_so("SGD", luoi_sgd,
                         lambda ts: sgd(X_tr, y_tr, lam2, so_epoch=VONG, val=val, **ts),
                         X_tr, y_tr, X_va, y_va, 0.0, lam2)
    tot_sub = do_tham_so("Subgradient", luoi_sub,
                         lambda ts: subgradient_method(X_tr, y_tr, lam2=lam2,
                                                       so_vong=VONG, val=val, **ts),
                         X_tr, y_tr, X_va, y_va, None, lam2)
    tot_prox = do_tham_so("Proximal", luoi_prox,
                          lambda ts: proximal_gradient(X_tr, y_tr, lam2=lam2,
                                                       so_vong=VONG, val=val, **ts),
                          X_tr, y_tr, X_va, y_va, None, lam2)
    print("\n  (tổng thời gian dò tham số: %.1fs)" % (time.time() - t0))

    tot = [tot_gd, tot_sgd, tot_sub, tot_prox]
    ev2 = [t[1] for t in tot]
    in_bang([(TEN_4[i], tot[i][0], ev2[i]) for i in range(4)],
            "KẾT QUẢ VÒNG 2 (cấu hình tốt nhất của mỗi thuật toán)")

    ve2 = [(TEN_4[i], tot[i][3]) for i in range(4)]
    ve_hoi_tu(ve2, [F_sao_tron, F_sao_tron,
                    F_sao_cua(tot_sub[4]["lam1"]), F_sao_cua(tot_prox[4]["lam1"])],
              os.path.join(FIG_DIR, "02_vong2.png"),
              "Vòng 2 - tham số riêng - " + args.product)
    ve_val(ve2, os.path.join(FIG_DIR, "02b_vong2_val.png"),
           "Vòng 2 - log-loss validation", moc)
    ve_do_thua(ve2, os.path.join(FIG_DIR, "03_do_thua.png"), d,
               "Độ thưa của nghiệm sau vòng 2")
    ve_2_vong(ev1, ev2, os.path.join(FIG_DIR, "04_vong1_vs_vong2.png"))

    # ----------------------------------------------------------- 3. nhan xet
    print("\n" + "=" * 96)
    print("3. NHẬN XÉT")
    print("=" * 96)
    print("%-13s%12s%12s%11s   %s" %
          ("thuật toán", "val vòng 1", "val vòng 2", "cải thiện", "cấu hình thắng"))
    print("-" * 96)
    for i in range(4):
        cai_thien = (ev1[i]["ll_va"] - ev2[i]["ll_va"]) / ev1[i]["ll_va"] * 100
        print("%-13s%12.6f%12.6f%10.2f%%   %s" %
              (TEN_4[i], ev1[i]["ll_va"], ev2[i]["ll_va"], cai_thien, tot[i][0]))

    kkt = np.linalg.norm(subgradient(bo_nho_F[lam1][1], X_tr, y_tr, lam1, lam2))
    thang = TEN_4[int(np.argmin([e["ll_va"] for e in ev2]))]

    print("""
(a) VÒNG 1 cho thấy "cùng một bộ tham số" KHÔNG có nghĩa là công bằng. Bước 1/L suy ra
    từ chặn Lipschitz, nhưng đó là bước "của GD":
      - với SGD nó quá nhỏ một cách vô lý, vì mỗi epoch SGD đi %d bước chứ không phải 1
        -> SGD vẫn về đích sớm nhất dù dùng đúng bước đó;
      - với Subgradient thì sai KIỂU: lý thuyết đòi bước giảm dần, bước hằng chỉ đưa
        nghiệm tới một LÂN CẬN của w* rồi dao động quanh đó;
      - với GD/ISTA thì 1/L đúng nhưng rất bi quan (chặn cho trường hợp xấu nhất).

(b) VÒNG 2, sau khi mỗi thuật toán được dò tham số riêng với CÙNG ngân sách %d lần
    duyệt dữ liệu, thuật toán tốt nhất theo log-loss validation là %s.
    Mỗi thuật toán "thích" một kiểu tham số khác nhau:
      - GD          -> bước lớn hơn 1/L nhiều lần, hoặc line search tự dò;
      - SGD         -> lịch giảm dần, batch vừa phải;
      - Subgradient -> bắt buộc giảm bước, eta0 lớn hơn 1/L nhiều lần;
      - Proximal    -> chỉ cần bật gia tốc Nesterov (FISTA) là đủ.
    Chênh lệch giữa cấu hình tốt nhất và tệ nhất TRONG CÙNG một thuật toán lớn hơn
    nhiều so với chênh lệch GIỮA các thuật toán -> chọn tham số quan trọng hơn chọn
    thuật toán.

(c) So sánh chéo phải cẩn thận: GD/SGD tối ưu f (không có L1), Subgradient/Proximal
    tối ưu F = f + lam1*||w||_1. Với lam1 = %g cố định ở vòng 1, hai thuật toán sau bị
    thiệt trên log-loss vì L1 làm co hệ số (đánh đổi lấy độ thưa). Vì vậy vòng 2 cho
    chúng tự chọn lam1: Subgradient chọn %g, Proximal chọn %g.

(d) Về ĐỘ THƯA, so ở VÒNG 1 vì lúc đó cả hai dùng chung lam1 = %g:
    Proximal nnz = %d/%d, Subgradient nnz = %d/%d, GD/SGD nnz = %d và %d.
    Chỉ Proximal cho nghiệm thưa theo CẤU TRÚC: soft-thresholding đặt HẲN hệ số về 0 ở
    mọi vòng lặp. Subgradient chỉ thưa nhờ quy ước chọn phần tử chuẩn nhỏ nhất trong dF
    (khi |grad_j| <= lam1 thì subgradient bằng đúng 0 nên hệ số "dính" lại 0); đổi sang
    cach_chon="random" là đặc hoàn toàn. GD/SGD không có cơ chế nào tạo ra số 0.
    (Ở vòng 2 cả hai tự chọn lam1 nhỏ hơn nên nghiệm đặc hơn - đó là cái giá của việc
    ưu tiên log-loss thay vì độ thưa.)

(e) Kiểm chứng điều kiện tối ưu KKT: ||dưới đạo hàm chuẩn nhỏ nhất|| tại nghiệm tham
    chiếu = %.2e, xấp xỉ 0, tức 0 thuộc dF(w*) -> nghiệm tham chiếu đúng là nghiệm tối ưu.

(f) Về SỐ LẦN DUYỆT DỮ LIỆU thì SGD tiến nhanh nhất (moi epoch no cap nhat %d lan).
    Nhưng về THỜI GIAN THỰC nó lại CHẬM hơn: mỗi epoch SGD gọi %d phép nhân ma trận
    nhỏ, trong khi GD/ISTA chỉ gọi 1 phép nhân lớn trên cả %s dòng - numpy/BLAS chạy
    phép lớn hiệu quả hơn nhiều (nhất là khi d = %d khá nhỏ). Bài học: uu the ly thuyet
    O(B*d) cua SGD chi thanh uu the thuc te khi d lon hoac du lieu khong vua bo nho.
    (Cột "giây" có tính cả thời gian tính log-loss để vẽ đồ thị, giống nhau cho cả 4
    thuật toán vì đều ghi %d lần.)
""" % (n // B, VONG, thang, lam1, tot_sub[4]["lam1"], tot_prox[4]["lam1"],
       lam1, ev1[3]["nnz"], d, ev1[2]["nnz"], d, ev1[0]["nnz"], ev1[1]["nnz"],
       kkt, n // B, n // B, "{:,}".format(n), d, VONG))

    # -------------------------------------------------- 4. MAP@7 (tuy chon)
    if args.map7:
        print("=" * 96)
        print("4. HUẤN LUYỆN 24 SẢN PHẨM BẰNG CẤU HÌNH PROXIMAL TỐT NHẤT -> MAP@7")
        print("=" * 96)
        cot_lag = [ten_cot.index(c + "_lag") for c in TARGET_COLS]
        da_co = (X_va[:, cot_lag] > X_va[:, cot_lag].min(axis=0) + 1e-9).astype(int)
        tham_so = dict(tot_prox[4])

        P = np.zeros((len(X_va), 24))
        t0 = time.time()
        for j, sp in enumerate(TARGET_COLS):
            yj = Y_tr[:, j].astype(float)
            if yj.sum() < 10:                  # qua hiem thi lay ti le nen
                P[:, j] = yj.mean()
                continue
            wj, _, _ = proximal_gradient(X_tr, yj, lam2=lam2, so_vong=VONG, **tham_so)
            P[:, j] = du_doan(X_va, wj)
            print("  %-22s nnz=%3d  val logloss=%.6f" %
                  (sp, int(np.sum(wj != 0)), logloss(wj, X_va, Y_va[:, j].astype(float))))

        ti_le = (Y_va.sum(1) > 0).mean()
        print("\nMAP@7 (validation %s), %.1fs:" % (VALID_PAIR[1], time.time() - t0))
        print("  trên khách CÓ thêm mới (%.2f%% số khách) = %.6f" %
              (ti_le * 100, map_at_7(P, Y_va, da_co, True)))
        print("  trên TOÀN BỘ khách (cách Kaggle chấm)   = %.6f" %
              map_at_7(P, Y_va, da_co, False))

    print("\nXong. Hình đã lưu trong thư mục figures/")


if __name__ == "__main__":
    main()
