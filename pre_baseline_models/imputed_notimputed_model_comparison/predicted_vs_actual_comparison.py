"""
predicted_vs_actual_comparison.py
--------------------------------------------------------
RF (imputed) ve XGBoost (not-imputed) için predicted vs actual grafikleri
Yan yana iki alt grafikte gerçek vs tahmin saçılımı

Bu script şunları yapar:
1) RF için imputed veriyi yükler
2) XGBoost için ham veriyi yükler ve enrich eder
3) Her iki model için HPO ile bulunmuş en iyi parametreleri kullanır
4) Train/Test split yapar
5) Modelleri eğitir ve tahminler yapar
6) Yan yana predicted vs actual grafikleri çizer
7) Her grafikte metrik kutusu: Train/Test R2, RMSE, MAE
--------------------------------------------------------
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate
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

# --- Helpers ---
def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mae (y_true, y_pred): return float(mean_absolute_error(y_true, y_pred))

def fit_predict_and_metrics(pipe, X_tr, y_tr, X_te, y_te, cv, is_xgb=False):
    # Fit
    if is_xgb:
        # XGBoost için direkt fit (pipeline yok)
        pipe.fit(X_tr, y_tr)
        yhat_tr = pipe.predict(X_tr)
        yhat_te = pipe.predict(X_te)
    else:
        # RF için pipeline
        pipe.fit(X_tr, y_tr)
        yhat_tr = pipe.predict(X_tr)
        yhat_te = pipe.predict(X_te)

    # Point metrics
    m = {}
    m["train_r2"]   = r2_score(y_tr, yhat_tr)
    m["train_rmse"] = rmse(y_tr, yhat_tr)
    m["train_mae"]  = mae (y_tr, yhat_tr)

    m["test_r2"]    = r2_score(y_te, yhat_te)
    m["test_rmse"]  = rmse(y_te, yhat_te)
    m["test_mae"]   = mae (y_te, yhat_te)

    # 5-fold CV (Test) metrics: R2, RMSE, MAE (ortalamalar)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    cvres = cross_validate(
        pipe, X_tr, y_tr, cv=cv, scoring=scoring,
        return_train_score=False, n_jobs=-1
    )
    m["cv_test_r2_mean"]   = float(np.mean(cvres["test_r2"]))
    m["cv_test_rmse_mean"] = float(np.mean(-cvres["test_neg_rmse"]))  # negatiften pozitif
    m["cv_test_mae_mean"]  = float(np.mean(-cvres["test_neg_mae"]))

    return yhat_tr, yhat_te, m

# --- Run both models on their own datasets ---
print(">>> Modeller eğitiliyor...")
yhat_tr_rf, yhat_te_rf, metr_rf = fit_predict_and_metrics(pipe_rf,  X_rf_tr,  y_rf_tr,  X_rf_te,  y_rf_te,  cv, is_xgb=False)
yhat_tr_xgb, yhat_te_xgb, metr_xgb = fit_predict_and_metrics(xgb, X_xgb_tr, y_xgb_tr, X_xgb_te, y_xgb_te, cv, is_xgb=True)

# --- Plot (side-by-side) ---
print(">>> Grafikler oluşturuluyor...")

# renkler/işaretler
c_test  = "#03C2FB"
c_train = "#001583"
ms_train = 28
ms_test  = 34

# ortak eksen limitleri (iki modelin verileri birlikte)
all_real = np.concatenate([y_rf_tr.values, y_rf_te.values, y_xgb_tr.values, y_xgb_te.values])
all_pred = np.concatenate([yhat_tr_rf, yhat_te_rf, yhat_tr_xgb, yhat_te_xgb])
vmin = float(np.nanmin([all_real.min(), all_pred.min()]))
vmax = float(np.nanmax([all_real.max(), all_pred.max()]))
pad  = 0.05*(vmax - vmin) if np.isfinite(vmax - vmin) else 1.0
lo, hi = vmin - pad, vmax + pad

fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)

def draw_panel(ax, y_tr, yhat_tr, y_te, yhat_te, metrics, title):
    ax.scatter(y_tr, yhat_tr, s=ms_train, alpha=0.75, edgecolors="none", label="Train", marker="o", c=c_train)
    ax.scatter(y_te, yhat_te, s=ms_test,  alpha=0.85, edgecolors="none", label="Test",  marker="^", c=c_test)

    # y = x
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.3, alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("Gerçek (y)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="lower right", frameon=True)

    # ---- İSTEDİĞİN FORMATTA METİN KUTUSU ----
    tr_line = f"TR  R2={metrics['train_r2']:.2f}, RMSE={metrics['train_rmse']:.2f}, MAE={metrics['train_mae']:.2f}"
    te_line = f"TE  R2={metrics['test_r2']:.2f},  RMSE={metrics['test_rmse']:.2f},  MAE={metrics['test_mae']:.2f}"
    cv_line = f"CV R2={metrics['cv_test_r2_mean']:.2f}, RMSE={metrics['cv_test_rmse_mean']:.2f}, MAE={metrics['cv_test_mae_mean']:.2f}"

    ax.text(
        0.02, 0.98,
        f"{tr_line}\n{te_line}\n{cv_line}",
        transform=ax.transAxes, fontsize=8, family="monospace",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85, edgecolor="#e2e8f0")
    )

draw_panel(axes[0], y_rf_tr, yhat_tr_rf, y_rf_te, yhat_te_rf, metr_rf,
           "RF (Best Params, Imputed) — Gerçek vs Tahmin")
draw_panel(axes[1], y_xgb_tr, yhat_tr_xgb, y_xgb_te, yhat_te_xgb, metr_xgb,
           "XGBoost (Best Params, Not-Imputed Enriched) — Gerçek vs Tahmin")

fig.suptitle("RF vs XGBoost — Gerçek vs Tahmin (Train/Test) | CV: Test ort. R2/RMSE/MAE", fontsize=13)
plt.tight_layout(rect=[0, 0.00, 1, 0.96])

out_path = FIG_DIR / "predicted_vs_actual_comparison.png"
plt.savefig(out_path, dpi=220, bbox_inches="tight")
print(f"[OK] Grafik kaydedildi: {out_path}")
plt.show(); plt.close()

# --- Summary ---
print("\n=== Model Performance Summary ===")
print(f"RF  (imputed)           : Test R2={metr_rf['test_r2']:.3f} | RMSE={metr_rf['test_rmse']:.3f} | MAE={metr_rf['test_mae']:.3f}")
print(f"XGB (not-imputed enrich): Test R2={metr_xgb['test_r2']:.3f} | RMSE={metr_xgb['test_rmse']:.3f} | MAE={metr_xgb['test_mae']:.3f}")

print("\n[OK] Predicted vs Actual comparison completed!")
