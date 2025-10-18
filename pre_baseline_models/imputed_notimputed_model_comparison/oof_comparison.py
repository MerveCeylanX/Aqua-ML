"""
oof_comparison.py
--------------------------------------------------------
RF (imputed) ve XGBoost (not-imputed) için OOF yüzde hata analizi
ve tek grafikte karşılaştırmalı yatay bar chart.

Bu script şunları yapar:
1) RF için imputed veriyi yükler
2) XGBoost için ham veriyi yükler ve enrich eder
3) Her iki model için HPO ile bulunmuş en iyi parametreleri kullanır
4) 5-fold cross_val_predict ile OOF tahminleri alır
5) Yüzde mutlak hataları şu aralıklara böler:
   0–5, 5–10, 10–15, 15–20, 20–30, 30–50, 50–100, 100%+
6) Her iki modelin dağılımını tek grafikte karşılaştırır (yatay bar, yan yana)
7) Konsola OOF metriklerini (R2, RMSE, MAE) ve dağılım tablolarını yazar
--------------------------------------------------------
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# --- Paths & I/O ---
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Veri yolları
RF_DATA_PATH  = r"D:\Aqua_ML\pre_baseline_models\imputed_models\data\Raw_data_imputed.xlsx"
XGB_DATA_PATH = r"D:\Aqua_ML\data\Raw_data.xlsx"  # Ham veri - enrich edilecek

# --- Config ---
RANDOM_STATE = 42
TEST_SIZE = 0.2
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# --- Feature candidates ---
ALL_NUM_FEATS = [
    "Agent/Sample(g/g)", "Soaking_Time(min)", "Soaking_Temp(K)",
    "Activation_Time(min)", "Activation_Temp(K)", "Activation_Heating_Rate (K/min)",
    "BET_Surface_Area(m2/g)", "Total_Pore_Volume(cm3/g)", "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)", "pHpzc",
    "C_molar","H_C_molar","O_C_molar","N_C_molar","S_C_molar",
    "Initial_Concentration(mg/L)", "Solution_pH", "Temperature(K)", "Agitation_speed(rpm)",
    "Dosage(g/L)", "Contact_Time(min)", "E","S","A","B","V"
]
CAT_CANDIDATES = ["Activation_Atmosphere"]
TARGET_COL = "qe(mg/g)"

# --- Veri yükleme ---
print(">>> Veri yükleniyor...")
df_rf  = pd.read_excel(RF_DATA_PATH)   # RF için imputed veri
df_xgb = pd.read_excel(XGB_DATA_PATH)  # XGB için ham veri

# --- XGBoost için enrich işlemi ---
print(">>> XGBoost verisi enrich ediliyor...")

# 1) PHARM ÖZELLİKLERİNİ EKLE
code_candidates = ["Pharmaceutical_code", "Pharm", "pharm", "Pharmaceutical", "Target_Phar"]
code_col = next((c for c in code_candidates if c in df_xgb.columns), None)
if code_col is None:
    raise KeyError("İlaç kodu kolonu bulunamadı. Aday isimler: " + ", ".join(code_candidates))

pharm_data = [
    ("PHE",  0.85, 0.95, 0.30, 0.78, 1.1156),
    ("APAP", 1.06, 1.63, 1.04, 0.86, 1.1724),
    ("ASA",  0.78, 1.69, 0.71, 0.67, 1.2879),
    ("BENZ", 1.03, 1.31, 0.31, 0.69, 1.3133),
    ("CAF",  1.50, 1.72, 0.05, 1.28, 1.3632),
    ("CIP",  2.20, 2.34, 0.70, 2.52, 2.3040),
    ("CIT",  1.83, 1.99, 0.00, 1.53, 2.5328),
    ("DCF",  1.81, 1.85, 0.55, 0.77, 2.0250),
    ("FLX",  1.23, 1.30, 0.12, 1.03, 2.2403),
    ("IBU",  0.73, 0.70, 0.56, 0.79, 1.7771),
    ("MTZ",  1.12, 1.79, 0.37, 1.04, 1.1919),
    ("NPX",  1.51, 2.02, 0.60, 0.67, 1.7821),
    ("NOR",  1.98, 2.50, 0.05, 2.39, 2.2724),
    ("OTC",  3.60, 3.05, 1.65, 3.50, 3.1579),
    ("SA",   0.90, 0.85, 0.73, 0.37, 0.9904),
    ("SDZ",  2.08, 2.55, 0.65, 1.37, 1.7225),
    ("SMR",  2.10, 2.65, 0.65, 1.42, 1.8634),
    ("SMT",  2.13, 2.53, 0.59, 1.53, 2.0043),
    ("SMX",  1.89, 2.23, 0.58, 1.29, 1.7244),
    ("TC",   3.50, 3.60, 1.35, 3.29, 3.0992),
    ("CBZ",  2.15, 1.90, 0.50, 1.15, 1.8106),
]
pharm_df = pd.DataFrame(pharm_data, columns=["Pharmaceutical_code","E","S","A","B","V"])
pharm_df["pharm_code_norm"] = pharm_df["Pharmaceutical_code"].str.strip().str.upper()

df_xgb["pharm_code_norm"] = df_xgb[code_col].astype(str).str.strip().str.upper()
map_cols = ["E","S","A","B","V"]
tmp_map = {c: f"__map_{c}" for c in map_cols}
pharm_merge = pharm_df[["pharm_code_norm"] + map_cols].rename(columns=tmp_map)
df_xgb = df_xgb.merge(pharm_merge, on="pharm_code_norm", how="left")
for c in map_cols:
    tmpc = f"__map_{c}"
    if c in df_xgb.columns:
        df_xgb[c] = pd.to_numeric(df_xgb[c], errors="coerce").fillna(df_xgb[tmpc])
    else:
        df_xgb[c] = df_xgb[tmpc]
    df_xgb.drop(columns=[tmpc], inplace=True)
df_xgb.drop(columns=["pharm_code_norm"], inplace=True)
for c in map_cols:
    df_xgb[c] = pd.to_numeric(df_xgb[c], errors="coerce")

# 2) ELEMENTAL ORANLAR
C, O, H, N, S = "C_percent", "O_percent", "H_percent", "N_percent", "S_percent"
aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}

for col in [C, O, H, N, S]:
    if col in df_xgb.columns:
        df_xgb[col] = pd.to_numeric(df_xgb[col], errors="coerce")
    else:
        print(f"[Uyarı] {col} kolonu bulunamadı; ilgili molar hesaplar NaN kalabilir.")

if C in df_xgb.columns:
    maskC = df_xgb[C].notna() & (df_xgb[C] > 0)
    badC = (~maskC).sum()
    if badC:
        print(f"[Not] {badC} satırda {C} yok veya ≤0; H/C, O/C, N/C, S/C ve C_molar hesaplanmadı (NaN).")

    nC = df_xgb.loc[maskC, C] / aw["C"]
    df_xgb.loc[maskC, "C_molar"] = nC

    if H in df_xgb.columns:
        df_xgb.loc[maskC, "H_C_molar"] = (df_xgb.loc[maskC, H] / aw["H"]) / nC
    if O in df_xgb.columns:
        df_xgb.loc[maskC, "O_C_molar"] = (df_xgb.loc[maskC, O] / aw["O"]) / nC
    if N in df_xgb.columns:
        df_xgb.loc[maskC, "N_C_molar"] = (df_xgb.loc[maskC, N] / aw["N"]) / nC
    if S in df_xgb.columns:
        df_xgb.loc[maskC, "S_C_molar"] = (df_xgb.loc[maskC, S] / aw["S"]) / nC

    for r in ["C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar"]:
        if r in df_xgb.columns:
            df_xgb[r] = pd.to_numeric(df_xgb[r], errors="coerce").clip(lower=0)

print(">>> XGBoost verisi enrich edildi!")

# --- RF: kolon filtreleri (imputed) ---
num_feats_rf = [c for c in ALL_NUM_FEATS if c in df_rf.columns]
cat_feats_rf = [c for c in CAT_CANDIDATES if c in df_rf.columns]
if TARGET_COL not in df_rf.columns:
    raise KeyError(f"[RF] Hedef kolon yok: {TARGET_COL}")

# --- XGB: kolon filtreleri (not-imputed enriched) ---
num_feats_xgb = [c for c in ALL_NUM_FEATS if c in df_xgb.columns]
cat_feats_xgb = [c for c in CAT_CANDIDATES if c in df_xgb.columns]
if TARGET_COL not in df_xgb.columns:
    raise KeyError(f"[XGB] Hedef kolon yok: {TARGET_COL}")

# Numerik tip güvenliği
for c in num_feats_rf + [TARGET_COL]:
    df_rf[c] = pd.to_numeric(df_rf[c], errors="coerce")
for c in num_feats_xgb + [TARGET_COL]:
    df_xgb[c] = pd.to_numeric(df_xgb[c], errors="coerce")

# X, y setleri
X_rf  = df_rf[num_feats_rf + cat_feats_rf].copy()
y_rf  = df_rf[TARGET_COL].copy()

# XGBoost için one-hot encoding
src_col = "Activation_Atmosphere"
dummy_cols = []
if src_col in df_xgb.columns:
    df_xgb[src_col] = df_xgb[src_col].astype(str).str.strip()
    df_xgb[src_col] = df_xgb[src_col].replace({"": np.nan})

    dummies = pd.get_dummies(
        df_xgb[src_col],
        prefix="activation_atmosphere",
        prefix_sep="_",
        dtype=np.float32,
        dummy_na=False
    )
    # isimleri normalize et
    dummies.columns = (
        dummies.columns
        .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.lower()
    )
    df_xgb = pd.concat([df_xgb, dummies], axis=1)
    dummy_cols = list(dummies.columns)

    for c in dummy_cols:
        if c not in num_feats_xgb:
            num_feats_xgb.append(c)

    print(f"[OK] One-hot eklendi: {', '.join(dummy_cols)}")
else:
    print("[Uyarı] 'Activation_Atmosphere' bulunamadı; one-hot atlandı.")

# Artık kategorik liste yok; tüm girdiler numerik/dummy
present_num_xgb = [c for c in num_feats_xgb if c in df_xgb.columns]
X_xgb = df_xgb[present_num_xgb].copy()
y_xgb = pd.to_numeric(df_xgb[TARGET_COL], errors="coerce")

# --- Train/Test split (her model kendi verisiyle) ---
X_rf_tr, X_rf_te, y_rf_tr, y_rf_te = train_test_split(
    X_rf, y_rf, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_xgb_tr, X_xgb_te, y_xgb_tr, y_xgb_te = train_test_split(
    X_xgb, y_xgb, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- Preprocessors ---
def make_preprocessor(num_feats, cat_feats):
    num_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_tr = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        transformers=[("num", num_tr, num_feats),
                      ("cat", cat_tr, cat_feats)],
        remainder="drop"
    )

pre_rf  = make_preprocessor(num_feats_rf,  cat_feats_rf)

# --- Models (HPO ile bulunmuş en iyi parametreler) ---
# RF için en iyi parametreler (impured_randomforest_hyper_and_SHAP.py'den)
rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_estimators=200,  # HPO sonucu
    max_depth=None,    # HPO sonucu
    min_samples_split=2,  # HPO sonucu
    min_samples_leaf=1,   # HPO sonucu
    max_features=None     # HPO sonucu
)
pipe_rf = Pipeline([("pre", pre_rf), ("reg", rf)])

# XGBoost için en iyi parametreler (not_imputed_xgboost_hyper_SHAP.py'den)
xgb = XGBRegressor(
    random_state=RANDOM_STATE,
    n_estimators=1754,
    max_depth=7,
    learning_rate=float(0.09790236510178592),
    subsample=float(0.7425191352307899),
    colsample_bytree=float(0.8533615026041694),
    gamma=float(1.6730409962757933),
    min_child_weight=4,
    reg_alpha=float(0.0019832619693868135),
    reg_lambda=float(0.0030785897480259308),
    objective="reg:squarederror",
    n_jobs=-1,
    tree_method="hist"
)

# --- Yardımcılar ---
def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mae (y_true, y_pred): return float(mean_absolute_error(y_true, y_pred))

def oof_pct_error_buckets(pipe, X_tr, y_tr, cv, is_xgb=False):
    """OOF tahminleri al, yüzde mutlak hatayı hesapla, aralıklara böl ve (counts, shares) döndür."""
    if is_xgb:
        # XGBoost için direkt model
        oof_pred = cross_val_predict(pipe, X_tr, y_tr, cv=cv, n_jobs=-1, method="predict")
    else:
        # RF için pipeline
        oof_pred = cross_val_predict(pipe, X_tr, y_tr, cv=cv, n_jobs=-1, method="predict")
    
    y_true = y_tr.to_numpy()
    eps = 1e-12
    pct_err = np.abs(y_true - oof_pred) / np.maximum(np.abs(y_true), eps) * 100.0

    bin_edges  = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
    bin_labels = ["0–5%", "5–10%", "10–15%", "15–20%", "20–30%", "30–50%", "50–100%", "100%+"]

    bucket = pd.cut(pct_err, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)
    counts = bucket.value_counts().sort_index()
    shares = (counts / counts.sum())

    # Ek metrikler
    oof_r2   = r2_score(y_true, oof_pred)
    oof_rmse = rmse(y_true, oof_pred)
    oof_mae  = mae(y_true, oof_pred)

    return counts, shares, (oof_r2, oof_rmse, oof_mae)

# --- OOF hesapla: RF (imputed) ---
print("\n>>> RF (imputed) için OOF hesaplanıyor...")
rf_counts, rf_shares, (rf_r2, rf_rmse, rf_mae) = oof_pct_error_buckets(
    pipe_rf, X_rf_tr, y_rf_tr, cv, is_xgb=False
)

# --- OOF hesapla: XGB (NOT-imputed enriched) ---
print("\n>>> XGB (not-imputed enriched) için OOF hesaplanıyor...")
xgb_counts, xgb_shares, (xgb_r2, xgb_rmse, xgb_mae) = oof_pct_error_buckets(
    xgb, X_xgb_tr, y_xgb_tr, cv, is_xgb=True
)

# =================== KONSOL ÇIKTILARI ===================
rf_table  = pd.DataFrame({
    "Aralık": rf_counts.index,
    "RF_Adet": rf_counts.values,
    "RF_Yüzde(%)": (rf_shares.values*100).round(2)
})
xgb_table = pd.DataFrame({
    "Aralık": xgb_counts.index,
    "XGB_Adet": xgb_counts.values,
    "XGB_Yüzde(%)": (xgb_shares.values*100).round(2)
})

# Kategorik tip ve sıralama sabitle (fillna hatasını önlemek için)
bins = ["0–5%", "5–10%", "10–15%", "15–20%", "20–30%", "30–50%", "50–100%", "100%+"]
cat_type = pd.api.types.CategoricalDtype(categories=bins, ordered=True)
rf_table["Aralık"]  = rf_table["Aralık"].astype(cat_type)
xgb_table["Aralık"] = xgb_table["Aralık"].astype(cat_type)

# Dış birleştirme + doğru sıraya göre sırala
merged = rf_table.merge(xgb_table, on="Aralık", how="outer").sort_values("Aralık")

# Sadece sayısal sütunlarda NaN -> 0
num_cols = merged.select_dtypes(include=[np.number]).columns
merged[num_cols] = merged[num_cols].fillna(0)

print("\n=== OOF Yüzde Hata Dağılımı — Karşılaştırma (Train folds) ===")
print(merged.to_string(index=False))

print("\n=== OOF Metrikler ===")
print(f"RF  (imputed)           : R2={rf_r2:.4f} | RMSE={rf_rmse:.4f} | MAE={rf_mae:.4f}")
print(f"XGB (not-imputed enrich): R2={xgb_r2:.4f} | RMSE={xgb_rmse:.4f} | MAE={xgb_mae:.4f}")

# --- Tek grafikte karşılaştırma (yatay bar, yan yana) ---
print("\n>>> OOF karşılaştırma grafiği oluşturuluyor...")

# Not: farklı veri setlerinden geldiği için adetler toplamı farklı olabilir. Karşılaştırma görselidir.
labels = rf_counts.index.tolist()
ypos = np.arange(len(labels))
height = 0.38  # bar kalınlığı

plt.figure(figsize=(9.5, 6.0))
plt.barh(ypos + height/2, rf_counts.values, height=height,
         label="RF (Best Params, Imputed)", alpha=0.85, color="#FECB04")
plt.barh(ypos - height/2, xgb_counts.values, height=height,
         label="XGBoost (Best Params, Not-Imputed)", alpha=0.85, color="#0014E7")

plt.yticks(ypos, labels)
plt.xlabel("Adet")
plt.title("OOF Yüzde Hata Dağılımı — RF (Imputed) vs XGBoost (Not-Imputed) — 5-fold")
plt.legend(loc="upper right")

# x-limit ayarı (grafiği biraz sağa uzatalım)
xmax = max(rf_counts.max() if len(rf_counts) else 0,
           xgb_counts.max() if len(xgb_counts) else 0)
plt.xlim(0, xmax * 1.10 if xmax > 0 else 1.0)

# anotasyon (yüzde olarak yazalım)
rf_pct  = (rf_shares.values * 100.0).round(1)
xgb_pct = (xgb_shares.values * 100.0).round(1)
for i, (rv, xv) in enumerate(zip(rf_counts.values, xgb_counts.values)):
    plt.text(rv + xmax*0.015, i + height/2,  f"{rf_pct[i]}%", va="center")
    plt.text(xv + xmax*0.015, i - height/2, f"{xgb_pct[i]}%", va="center")

plt.tight_layout()
fig_path = FIG_DIR / "oof_pct_error_buckets_RF_vs_XGB.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
print(f"[OK] Karşılaştırma grafiği kaydedildi: {fig_path}")
plt.show(); plt.close()

print("\n[OK] OOF comparison completed!")
