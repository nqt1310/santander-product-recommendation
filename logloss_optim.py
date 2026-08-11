"""
4 thuat toan toi uu log-loss cho hoi quy logistic:
    GD, SGD, Subgradient, Proximal (ISTA/FISTA)

Ham muc tieu:
    F(w) = logloss(w) + lam2/2 * ||w||^2  +  lam1 * ||w||_1
           |_____________ tron ___________|    |_ khong tron _|

Quy uoc: cot 0 cua X la intercept (toan so 1) va KHONG bi phat.
"""

import time
import numpy as np


# ---------------------------------------------------------------- ham co ban
def sigmoid(z):
    # tach 2 truong hop cho khoi tran so khi |z| lon
    out = np.zeros_like(z, dtype=float)
    duong = z >= 0
    out[duong] = 1 / (1 + np.exp(-z[duong]))
    e = np.exp(z[~duong])
    out[~duong] = e / (1 + e)
    return out


def logloss(w, X, y, lam2=0.0):
    """Phan TRON: log-loss trung binh + phat L2."""
    z = X @ w
    # -[y*log(p) + (1-y)*log(1-p)]  =  log(1+e^z) - y*z
    # dung np.logaddexp(0, z) thay vi log(1+exp(z)) cho khoi tran
    loss = np.mean(np.logaddexp(0, z) - y * z)
    if lam2 > 0:
        loss = loss + 0.5 * lam2 * np.sum(w[1:] ** 2)   # bo intercept
    return loss


def grad(w, X, y, lam2=0.0):
    """Gradient cua phan tron: X'(sigmoid(Xw) - y)/n + lam2*w"""
    n = X.shape[0]
    g = X.T @ (sigmoid(X @ w) - y) / n
    if lam2 > 0:
        g = g.copy()
        g[1:] = g[1:] + lam2 * w[1:]
    return g


def objective(w, X, y, lam1=0.0, lam2=0.0):
    """Ham muc tieu day du F = phan tron + lam1*||w||_1"""
    F = logloss(w, X, y, lam2)
    if lam1 > 0:
        F = F + lam1 * np.sum(np.abs(w[1:]))
    return F


def lipschitz(X, lam2=0.0, so_vong=50):
    """
    Hang so Lipschitz cua gradient: L = sigma_max(X)^2 / (4n) + lam2
    (vi dao ham sigmoid <= 1/4). Tim sigma_max bang power iteration.
    Buoc 1/L luon dam bao GD va ISTA hoi tu.
    """
    n, d = X.shape
    v = np.random.RandomState(0).rand(d)
    v = v / np.linalg.norm(v)
    for _ in range(so_vong):
        v = X.T @ (X @ v)
        v = v / np.linalg.norm(v)
    sigma_max_binh = np.sum((X @ v) ** 2)      # = sigma_max^2 vi ||v||=1
    return sigma_max_binh / (4 * n) + lam2


def kiem_tra_grad(w, X, y, lam2=0.0, eps=1e-6, so_cot=15):
    """Kiem tra gradient bang sai phan huu han. Sai so phai rat nho (~1e-8)."""
    g = grad(w, X, y, lam2)
    rs = np.random.RandomState(0)
    sai_so = 0.0
    for j in rs.choice(len(w), min(so_cot, len(w)), replace=False):
        w1, w2 = w.copy(), w.copy()
        w1[j] += eps
        w2[j] -= eps
        so = (logloss(w1, X, y, lam2) - logloss(w2, X, y, lam2)) / (2 * eps)
        sai_so = max(sai_so, abs(so - g[j]) / max(1.0, abs(g[j])))
    return sai_so


# --------------------------------------------------- HAM DUOI DAO HAM (bai yeu cau)
def subgradient(w, X, y, lam1=0.0, lam2=0.0, idx=None, cach_chon="min_norm"):
    """
    Tra ve MOT phan tu g thuoc duoi vi phan dF(w).

    Vi phan tron f kha vi nen:  dF(w) = grad_f(w) + lam1 * d||w||_1
    trong do   d|w_j| = sign(w_j)   neu w_j != 0
               d|w_j| = [-1, 1]     neu w_j == 0   <-- ca mot doan, phai CHON

    cach_chon:
      "min_norm" : chon phan tu co chuan nho nhat
                   g_j = grad_j - clip(grad_j, -lam1, lam1)
                   -> ||g|| = 0 dung bang 0 khi w toi uu, nen dung lam tieu chuan dung
      "zero"     : lay 0 trong doan [-lam1, lam1] (quy uoc don gian)
      "random"   : lay ngau nhien trong [-lam1, lam1] (cho thay tinh da tri)

    idx: neu truyen chi so mini-batch thi chi tinh tren cac mau do
         -> duoi dao ham NGAU NHIEN (stochastic subgradient)
    """
    if idx is not None:
        g = grad(w, X[idx], y[idx], lam2)
    else:
        g = grad(w, X, y, lam2)

    if lam1 == 0:
        return g

    g = g.copy()
    khac_0 = w != 0
    bang_0 = w == 0
    khac_0[0] = bang_0[0] = False          # bo intercept (khong bi phat)

    g[khac_0] += lam1 * np.sign(w[khac_0])
    if cach_chon == "min_norm":
        g[bang_0] -= np.clip(g[bang_0], -lam1, lam1)
    elif cach_chon == "random":
        g[bang_0] += np.random.uniform(-lam1, lam1, bang_0.sum())
    # cach_chon == "zero" thi cong them 0, khong lam gi ca
    return g


# ------------------------------------------------- TOAN TU PROXIMAL (bai yeu cau)
def prox_l1(v, t, lam1):
    """
    prox cua g(w) = lam1*||w||_1 voi buoc t:

        prox(v)_j = argmin_u  lam1*|u| + (u - v_j)^2 / (2t)
                  = sign(v_j) * max(|v_j| - t*lam1, 0)      <- soft-thresholding

    Day la ly do proximal manh hon subgradient voi L1: no dat HAN he so ve 0.
    """
    u = v.copy()
    u[1:] = np.sign(v[1:]) * np.maximum(np.abs(v[1:]) - t * lam1, 0)
    return u                                # u[0] giu nguyen: khong phat intercept


def prox_elastic_net(v, t, lam1, lam2):
    """prox cua lam1*||w||_1 + lam2/2*||w||^2 = soft-threshold roi co lai."""
    u = prox_l1(v, t, lam1)
    u[1:] = u[1:] / (1 + t * lam2)
    return u


# ------------------------------------------------------------------ ghi lich su
def ghi_lich_su(ls, w, X, y, lam1, lam2, val, t0, vong):
    """Ghi lai F, thoi gian, so he so khac 0 va log-loss validation."""
    ls["vong"].append(vong)
    ls["F"].append(objective(w, X, y, lam1, lam2))
    ls["time"].append(time.time() - t0)
    ls["nnz"].append(int(np.sum(w != 0)))
    if val is not None:
        ls["val"].append(logloss(w, val[0], val[1]))


def lich_su_moi():
    return {"vong": [], "F": [], "time": [], "nnz": [], "val": []}


def buoc_hoc(kieu, eta0, k, gamma=1e-3):
    """Lich giam buoc hoc cho SGD / subgradient."""
    if kieu == "hang_so":
        return eta0
    if kieu == "1/k":                 # eta0 / (1 + gamma*k)
        return eta0 / (1 + gamma * k)
    if kieu == "1/sqrt(k)":
        return eta0 / np.sqrt(1 + k)
    raise ValueError("khong biet kieu buoc hoc: " + kieu)


# ==================================================================== 1. GD
def gradient_descent(X, y, lam2=0.0, buoc=None, so_vong=100,
                     line_search=False, val=None):
    """
    GD:  w = w - t * grad(w)

    buoc=None    -> dung t = 1/L (an toan, dam bao giam don dieu)
    line_search  -> Armijo: thu buoc gap doi lan truoc roi co dan lai den khi
                    f(w - t*g) <= f(w) - c*t*||g||^2
    Chi dung duoc khi lam1 = 0 vi L1 khong kha vi.
    """
    n, d = X.shape
    w = np.zeros(d)
    t = 1 / lipschitz(X, lam2) if buoc is None else buoc

    ls = lich_su_moi()
    t0 = time.time()
    ghi_lich_su(ls, w, X, y, 0, lam2, val, t0, 0)

    for k in range(1, so_vong + 1):
        g = grad(w, X, y, lam2)
        if line_search:
            f0 = logloss(w, X, y, lam2)
            chuan_g = np.sum(g ** 2)
            t = t * 2                       # thu buoc lon hon lan truoc
            for _ in range(60):
                w_moi = w - t * g
                if logloss(w_moi, X, y, lam2) <= f0 - 1e-4 * t * chuan_g:
                    break
                t = t / 2
            w = w_moi
        else:
            w = w - t * g
        ghi_lich_su(ls, w, X, y, 0, lam2, val, t0, k)

    ten = "GD (Armijo)" if line_search else "GD"
    return w, ls, ten


# =================================================================== 2. SGD
def sgd(X, y, lam2=0.0, batch=1024, so_epoch=60, eta0=0.1, kieu_buoc="1/k",
        gamma=1e-3, momentum=0.0, trung_binh=False, val=None, seed=0):
    """
    SGD mini-batch:  w = w - eta_k * grad_B(w)

    grad_B la uoc luong khong chech cua grad, moi buoc re hon n/batch lan
    nhung co nhieu -> can buoc giam dan (dieu kien Robbins-Monro).

    momentum    : w = w - eta*v  voi  v = momentum*v + g
    trung_binh  : lay trung binh cac w di qua (Polyak averaging)
    """
    n, d = X.shape
    rs = np.random.RandomState(seed)
    w = np.zeros(d)
    v = np.zeros(d)
    w_tb = w.copy()
    dem = 1

    ls = lich_su_moi()
    t0 = time.time()
    ghi_lich_su(ls, w, X, y, 0, lam2, val, t0, 0)

    k = 0
    for epoch in range(1, so_epoch + 1):
        thu_tu = rs.permutation(n)
        for i in range(0, n, batch):
            idx = thu_tu[i:i + batch]
            Xb, yb = X[idx], y[idx]
            g = Xb.T @ (sigmoid(Xb @ w) - yb) / len(idx)
            if lam2 > 0:
                g[1:] = g[1:] + lam2 * w[1:]

            eta = buoc_hoc(kieu_buoc, eta0, k, gamma)
            if momentum > 0:
                v = momentum * v + g
                w = w - eta * v
            else:
                w = w - eta * g
            k += 1

            if trung_binh:
                dem += 1
                w_tb = w_tb + (w - w_tb) / dem

        w_ghi = w_tb if trung_binh else w
        ghi_lich_su(ls, w_ghi, X, y, 0, lam2, val, t0, epoch)

    ten = "SGD"
    if momentum > 0:
        ten += " + momentum"
    if trung_binh:
        ten += " + trung binh"
    return (w_tb if trung_binh else w), ls, ten


# =========================================================== 3. SUBGRADIENT
def subgradient_method(X, y, lam1=1e-3, lam2=0.0, eta0=None, so_vong=100,
                       kieu_buoc="1/sqrt(k)", cach_chon="min_norm",
                       ngau_nhien=False, batch=1024, val=None, seed=0):
    """
    Phuong phap duoi dao ham:  w = w - eta_k * g_k  voi  g_k thuoc dF(w_k)

    Khac GD o 3 diem:
      - F(w_k) KHONG giam don dieu -> phai giu lai nghiem tot nhat da gap
      - buoc phai giam dan, toc do chi O(1/sqrt(k)) (ISTA la O(1/k))
      - do thua phu thuoc cach_chon chu khong phai ban chat thuat toan
    """
    n, d = X.shape
    rs = np.random.RandomState(seed)
    w = np.zeros(d)
    if eta0 is None:
        eta0 = 1 / lipschitz(X, lam2)

    ls = lich_su_moi()
    t0 = time.time()
    ghi_lich_su(ls, w, X, y, lam1, lam2, val, t0, 0)

    w_tot = w.copy()
    F_tot = objective(w, X, y, lam1, lam2)

    for k in range(1, so_vong + 1):
        idx = rs.randint(0, n, batch) if ngau_nhien else None
        g = subgradient(w, X, y, lam1, lam2, idx=idx, cach_chon=cach_chon)
        w = w - buoc_hoc(kieu_buoc, eta0, k) * g

        F = objective(w, X, y, lam1, lam2)
        if F < F_tot:                       # nho lai nghiem tot nhat
            F_tot, w_tot = F, w.copy()
        ghi_lich_su(ls, w, X, y, lam1, lam2, val, t0, k)

    return w_tot, ls, "Subgradient"


# ============================================================= 4. PROXIMAL
def proximal_gradient(X, y, lam1=1e-3, lam2=0.0, buoc=None, so_vong=100,
                      gia_toc=False, line_search=False, kieu_prox="l1", val=None):
    """
    ISTA:   w = prox( w - t*grad(w) )
            di mot buoc gradient tren phan TRON roi "chieu mem" bang prox
            cua phan KHONG TRON -> nghiem thua that su.

    gia_toc=True  -> FISTA, them quan tinh Nesterov:
            v = w + (k-1)/(k+2) * (w - w_truoc)
            w = prox( v - t*grad(v) )
            Toc do: ISTA O(1/k)  ->  FISTA O(1/k^2), chi phi moi vong nhu nhau.

    line_search=True -> tu do buoc t (thu buoc lon hon roi co dan lai)
    """
    n, d = X.shape
    w = np.zeros(d)
    t = 1 / lipschitz(X, lam2) if buoc is None else buoc

    # neu prox lo phan L2 roi thi gradient khong tinh L2 nua (tranh phat 2 lan)
    lam2_tron = 0.0 if kieu_prox == "elastic_net" else lam2

    def lam_prox(u, buoc_t):
        if kieu_prox == "l1":
            return prox_l1(u, buoc_t, lam1)
        if kieu_prox == "elastic_net":
            return prox_elastic_net(u, buoc_t, lam1, lam2)
        if kieu_prox == "khong":            # prox = anh xa dong nhat -> ISTA == GD
            return u
        raise ValueError("khong biet kieu prox: " + kieu_prox)

    ls = lich_su_moi()
    t0 = time.time()
    ghi_lich_su(ls, w, X, y, lam1, lam2, val, t0, 0)

    w_truoc = w.copy()
    for k in range(1, so_vong + 1):
        if gia_toc:
            v = w + (k - 1) / (k + 2) * (w - w_truoc)     # quan tinh Nesterov
        else:
            v = w
        g = grad(v, X, y, lam2_tron)

        if line_search:
            fv = logloss(v, X, y, lam2_tron)
            t = t * 1.5                     # thu buoc lon hon truoc
            for _ in range(60):
                w_moi = lam_prox(v - t * g, t)
                hieu = w_moi - v
                if logloss(w_moi, X, y, lam2_tron) <= fv + g @ hieu + hieu @ hieu / (2 * t):
                    break
                t = t / 2
        else:
            w_moi = lam_prox(v - t * g, t)

        w_truoc, w = w, w_moi
        ghi_lich_su(ls, w, X, y, lam1, lam2, val, t0, k)

    ten = "FISTA" if gia_toc else "ISTA"
    if line_search:
        ten += " + line search"
    return w, ls, ten


# ------------------------------------------------------------------ tien ich
def du_doan(X, w):
    return sigmoid(X @ w)
