"""
Chuan bi du lieu Santander cho bai toan hoi quy logistic.

Nhan: y = 1 neu khach hang THEM MOI san pham o thang t
      (co o thang t nhung KHONG co o thang t-1)
Dac trung: thong tin khach hang o thang t + 24 co so huu o thang t-1

Lan chay dau quet file train_ver2.csv (2.3 GB) mat khoang 40 giay,
sau do luu cache vao thu muc cache/ nen cac lan sau chi mat vai giay.
"""

import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

THU_MUC = os.path.dirname(os.path.abspath(__file__))
FILE_TRAIN = os.path.join(THU_MUC, "train_ver2.csv")
CACHE = os.path.join(THU_MUC, "cache")

# cap thang: (thang t-1 lay lag, thang t lay nhan)
TRAIN_PAIR = ("2015-05-28", "2015-06-28")
VALID_PAIR = ("2016-04-28", "2016-05-28")

TARGET_COLS = [
    "ind_ahor_fin_ult1", "ind_aval_fin_ult1", "ind_cco_fin_ult1",
    "ind_cder_fin_ult1", "ind_cno_fin_ult1", "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1", "ind_ctop_fin_ult1", "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1", "ind_deme_fin_ult1", "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1", "ind_fond_fin_ult1", "ind_hip_fin_ult1",
    "ind_plan_fin_ult1", "ind_pres_fin_ult1", "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1", "ind_valo_fin_ult1", "ind_viv_fin_ult1",
    "ind_nomina_ult1", "ind_nom_pens_ult1", "ind_recibo_ult1",
]

NUM_COLS = ["age", "antiguedad", "renta", "ind_nuevo", "indrel",
            "ind_actividad_cliente", "alta_year"]

CAT_COLS = ["sexo", "ind_empleado", "pais_residencia", "tiprel_1mes",
            "indresi", "indext", "indfall", "canal_entrada", "segmento",
            "indrel_1mes", "cod_prov"]

# bien hang muc nhieu muc thi chi giu top-k, con lai gop thanh OTHER
TOP_K = {"pais_residencia": 4, "canal_entrada": 10, "cod_prov": 20}

COT_CAN_DOC = (["fecha_dato", "ncodpers", "ind_empleado", "pais_residencia",
                "sexo", "age", "fecha_alta", "ind_nuevo", "antiguedad", "indrel",
                "indrel_1mes", "tiprel_1mes", "indresi", "indext", "canal_entrada",
                "indfall", "cod_prov", "ind_actividad_cliente", "renta", "segmento"]
               + TARGET_COLS)


def doc_cac_thang(cac_thang):
    """Doc file 2.3GB theo tung chunk, chi giu lai cac dong thuoc cac thang can."""
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    ten_cache = os.path.join(CACHE, "months_%s.pkl" %
                             "_".join(m.replace("-", "") for m in sorted(cac_thang)))
    if os.path.exists(ten_cache):
        print("[cache] doc " + ten_cache)
        return pd.read_pickle(ten_cache)

    print("[doc  ] quet train_ver2.csv de lay %d thang ..." % len(cac_thang))
    t0 = time.time()
    giu = []
    da_doc = 0
    for i, chunk in enumerate(pd.read_csv(FILE_TRAIN, usecols=COT_CAN_DOC,
                                          dtype=str, chunksize=1000000)):
        da_doc += len(chunk)
        phan_can = chunk[chunk["fecha_dato"].isin(cac_thang)]
        if len(phan_can) > 0:
            giu.append(phan_can)
        print("  chunk %3d | da doc %10s dong | giu %9s" %
              (i, "{:,}".format(da_doc),
               "{:,}".format(sum(len(g) for g in giu))))

    df = pd.concat(giu, ignore_index=True)
    df.to_pickle(ten_cache)
    print("[doc  ] xong sau %.1fs -> %s dong" % (time.time() - t0,
                                                 "{:,}".format(len(df))))
    return df


def lam_sach(df):
    """Doi kieu du lieu, xu ly gia tri thieu va cac gia tri bat thuong."""
    df = df.copy()

    for c in TARGET_COLS:                     # cot san pham: 0/1, co vai o trong
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int8)

    df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(18, 95)

    df["antiguedad"] = pd.to_numeric(df["antiguedad"], errors="coerce")
    df.loc[df["antiguedad"] < 0, "antiguedad"] = np.nan      # co -999999 = thieu
    df["antiguedad"] = df["antiguedad"].clip(0, 260)

    # thu nhap lech phai rat manh -> lay log cho gon
    df["renta"] = pd.to_numeric(df["renta"], errors="coerce")
    df["renta"] = np.log1p(df["renta"].clip(0, 1500000))

    df["alta_year"] = pd.to_datetime(df["fecha_alta"], errors="coerce").dt.year

    for c in ["ind_nuevo", "indrel", "ind_actividad_cliente"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    for c in NUM_COLS:                        # con thieu thi dien trung vi
        df[c] = df[c].astype(float)
        df[c] = df[c].fillna(df[c].median())

    for c in CAT_COLS:
        s = df[c].fillna("NA").astype(str).str.strip().replace("", "NA")
        if c == "indrel_1mes":                # cot nay co ca '1' va '1.0'
            s = s.str.replace(r"\.0$", "", regex=True)
        df[c] = s

    df["ncodpers"] = df["ncodpers"].astype(np.int64)
    return df


def ghep_2_thang(df, thang_lag, thang_dich):
    """Ghep thang t voi thang t-1 de biet khach hang THEM MOI san pham nao."""
    hien_tai = df[df["fecha_dato"] == thang_dich]
    truoc_do = df[df["fecha_dato"] == thang_lag][["ncodpers"] + TARGET_COLS]
    truoc_do = truoc_do.rename(columns={c: c + "_lag" for c in TARGET_COLS})

    m = hien_tai.merge(truoc_do, on="ncodpers")    # chi giu khach co ca 2 thang
    co_bay_gio = m[TARGET_COLS].values
    co_thang_truoc = m[[c + "_lag" for c in TARGET_COLS]].values
    Y = ((co_bay_gio == 1) & (co_thang_truoc == 0)).astype(np.int8)
    return m, Y


def tao_ma_tran(df, cac_muc):
    """Tao ma tran X: cot so + one-hot bien hang muc + 24 co so huu thang truoc."""
    khoi = [df[NUM_COLS].values.astype(float)]
    ten = list(NUM_COLS)

    for c in CAT_COLS:
        muc = cac_muc[c]
        cot = df[c].where(df[c].isin(muc), "OTHER")
        for mm in muc[1:] + ["OTHER"]:        # bo muc dau tien de tranh trung lap
            khoi.append((cot == mm).values.astype(float).reshape(-1, 1))
            ten.append("%s=%s" % (c, mm))

    khoi.append(df[[c + "_lag" for c in TARGET_COLS]].values.astype(float))
    ten += [c + "_lag" for c in TARGET_COLS]

    return np.hstack(khoi), ten


def build_dataset(dung_cache=True):
    """
    Tra ve dict: X_tr, Y_tr, X_va, Y_va, feature_names, prod_names
    Cot 0 cua X la intercept (toan so 1) va khong bi phat.
    """
    if not os.path.exists(CACHE):
        os.makedirs(CACHE)
    ten_npz = os.path.join(CACHE, "design.npz")
    ten_meta = os.path.join(CACHE, "design_names.pkl")

    if dung_cache and os.path.exists(ten_npz) and os.path.exists(ten_meta):
        print("[cache] doc " + ten_npz)
        z = np.load(ten_npz)
        with open(ten_meta, "rb") as f:
            ten = pickle.load(f)
        return {"X_tr": z["X_tr"], "Y_tr": z["Y_tr"], "X_va": z["X_va"],
                "Y_va": z["Y_va"], "feature_names": ten, "prod_names": TARGET_COLS}

    cac_thang = sorted(set(TRAIN_PAIR) | set(VALID_PAIR))
    df = lam_sach(doc_cac_thang(cac_thang))

    df_tr, Y_tr = ghep_2_thang(df, TRAIN_PAIR[0], TRAIN_PAIR[1])
    df_va, Y_va = ghep_2_thang(df, VALID_PAIR[0], VALID_PAIR[1])
    print("[tao  ] train %s khach | valid %s khach" %
          ("{:,}".format(len(df_tr)), "{:,}".format(len(df_va))))

    # chot danh sach muc TREN TAP TRAIN (khong duoc dung tap validation)
    cac_muc = {}
    for c in CAT_COLS:
        dem = df_tr[c].value_counts()
        k = TOP_K.get(c)
        cac_muc[c] = list(dem.index[:k]) if k else list(dem.index)

    X_tr, ten = tao_ma_tran(df_tr, cac_muc)
    X_va, _ = tao_ma_tran(df_va, cac_muc)

    # chuan hoa z-score theo thong ke cua TAP TRAIN -> GD/SGD hoi tu nhanh hon nhieu
    tb = X_tr.mean(axis=0)
    do_lech = X_tr.std(axis=0)
    do_lech[do_lech < 1e-8] = 1.0
    X_tr = (X_tr - tb) / do_lech
    X_va = (X_va - tb) / do_lech

    # them cot intercept toan so 1 vao dau
    X_tr = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    X_va = np.hstack([np.ones((len(X_va), 1)), X_va])
    ten = ["intercept"] + ten

    np.savez(ten_npz, X_tr=X_tr.astype(np.float32), Y_tr=Y_tr,
             X_va=X_va.astype(np.float32), Y_va=Y_va)
    with open(ten_meta, "wb") as f:
        pickle.dump(ten, f)
    print("[cache] luu " + ten_npz)

    return {"X_tr": X_tr, "Y_tr": Y_tr, "X_va": X_va, "Y_va": Y_va,
            "feature_names": ten, "prod_names": TARGET_COLS}


if __name__ == "__main__":
    d = build_dataset()
    print("X_tr:", d["X_tr"].shape, "| X_va:", d["X_va"].shape)
    print("\nTi le 'them moi' theo san pham (train):")
    ti_le = d["Y_tr"].mean(axis=0)
    for i in np.argsort(-ti_le)[:10]:
        print("  %-22s %6.3f%%" % (TARGET_COLS[i], ti_le[i] * 100))
