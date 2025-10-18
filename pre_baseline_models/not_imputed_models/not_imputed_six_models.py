"""
not_imputed_six_models.py
------------------
Amaç:
- Ham veriyi yükle
- İlaç (pharmaceutical) özelliklerini ekle (E, S, A, B, V)
- Elemental oranlardan molar oranları hesapla (C_molar, H/C, O/C, N/C, S/C)
- Zenginleştirilmiş veriyi 'data/' klasörüne kaydet
- 6 farklı ML modeli (CatBoost, LightGBM-GBDT, LightGBM-DART, XGBoost, HistGBR, EBM) ile
  5-fold CV ve Train/Test değerlendirmesi yap
- Sonuç grafiklerini 'figures/' klasörüne kaydet
- Tüm modellerin özet metriklerini ve en iyi modelin fold detaylarını Excel'e kaydet
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import parallel_backend
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# -------------------- KULLANICI AYARLARI --------------------
IN_PATH   = r"D:\Aqua_ML\data\Raw_data.xlsx"       # ham dosya
BASE_DIR  = r"D:\Aqua_ML\pre_baseline_models\not_imputed_models"               # ana çıktı klasörü
OUT_DATA  = os.path.join(BASE_DIR, "data", "Raw_data_enriched.xlsx")
OUT_FIG   = os.path.join(BASE_DIR, "figures")      # figure klasörü
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# Klasörleri oluştur
os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
os.makedirs(OUT_FIG, exist_ok=True)

# -------------------- PAKET KONTROL (opsiyonel modeller) --------------------
have = {}
try:
    from catboost import CatBoostRegressor
    have["catboost"] = True
except Exception:
    have["catboost"] = False

try:
    from lightgbm import LGBMRegressor
    have["lightgbm"] = True
except Exception:
    have["lightgbm"] = False

try:
    from xgboost import XGBRegressor
    have["xgboost"] = True
except Exception:
    have["xgboost"] = False

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    have["ebm"] = True
except Exception:
    have["ebm"] = False

# -------------------- CPU / loky fix --------------------
N_JOBS = max(1, (os.cpu_count() or 1) - 1)
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)

# -------------------- 0) VERİYİ YÜKLE --------------------
df_ml = pd.read_excel(IN_PATH)
df = df_ml.copy()

# -------------------- 1) PHARM ÖZELLİKLERİNİ EKLE --------------------
# Farmasötik özellikler tablosu (kısaltma, E, S, A, B, V)
# Veriler UFZ LSER-Database'den alınmıştır.
# https://web.app.ufz.de/compbc/lserd/public/start/#searchresult

code_candidates = ["Pharmaceutical_code", "Pharm", "pharm", "Pharmaceutical", "Target_Phar"]
code_col = next((c for c in code_candidates if c in df.columns), None)
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

df["pharm_code_norm"] = df[code_col].astype(str).str.strip().str.upper()
map_cols = ["E","S","A","B","V"]
tmp_map = {c: f"__map_{c}" for c in map_cols}
pharm_merge = pharm_df[["pharm_code_norm"] + map_cols].rename(columns=tmp_map)
df = df.merge(pharm_merge, on="pharm_code_norm", how="left")

for c in map_cols:
    tmpc = f"__map_{c}"
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[tmpc])
    else:
        df[c] = df[tmpc]
    df.drop(columns=[tmpc], inplace=True)

df.drop(columns=["pharm_code_norm"], inplace=True)

# Tür güvenliği
for c in map_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------- 2) ELEMENTAL ORANLAR (imputation yok) --------------------
C, O, H, N, S = "C_percent", "O_percent", "H_percent", "N_percent", "S_percent"
aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}

# yüzdeleri güvenli sayısala çevir
for col in [C, O, H, N, S]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"[Uyarı] {col} kolonu bulunamadı; ilgili molar hesaplar NaN kalabilir.")

if C in df.columns:
    # C>0 olan satırlar: hem oranlar hem de C_molar hesaplanabilir
    maskC = df[C].notna() & (df[C] > 0)
    badC = (~maskC).sum()
    if badC:
        print(f"[Not] {badC} satırda {C} yok veya ≤0; H/C, O/C, N/C, S/C ve C_molar hesaplanmadı (NaN).")

    # C'nin mol karşılığı (ör: 100 g bazında, %C/12.011)
    nC = df.loc[maskC, C] / aw["C"]
    df.loc[maskC, "C_molar"] = nC  # <<< EKLENEN FEATURE

    # H/C, O/C, N/C, S/C oranları (mevcut mantık)
    if H in df.columns:
        df.loc[maskC, "H_C_molar"] = (df.loc[maskC, H] / aw["H"]) / nC
    if O in df.columns:
        df.loc[maskC, "O_C_molar"] = (df.loc[maskC, O] / aw["O"]) / nC
    if N in df.columns:
        df.loc[maskC, "N_C_molar"] = (df.loc[maskC, N] / aw["N"]) / nC
    if S in df.columns:
        df.loc[maskC, "S_C_molar"] = (df.loc[maskC, S] / aw["S"]) / nC

    # olası negatif/bozuk değerleri kırp
    for r in ["C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar"]:
        if r in df.columns:
            df[r] = pd.to_numeric(df[r], errors="coerce").clip(lower=0)

# -------------------- 3) ZENGİNLEŞTİRİLMİŞ VERİYİ KAYDET -------------------- 
try:
    df.to_excel(OUT_DATA, index=False)
    print(f"[OK] Zenginleştirilmiş veri kaydedildi: {OUT_DATA}")
except Exception as e:
    print(f"[Uyarı] Excel kaydında sorun: {e}")
    
# -------------------- 4) ML HAZIRLIK --------------------
target_col = "qe(mg/g)"
if target_col not in df.columns:
    raise KeyError(f"Hedef kolon yok: {target_col}")

num_feats = [
    "Agent/Sample(g/g)",
    "Soaking_Time(min)",
    "Soaking_Temp(K)",
    "Activation_Time(min)",
    "Activation_Temp(K)",
    "Activation_Heating_Rate (K/min)",
    "BET_Surface_Area(m2/g)",
    "Total_Pore_Volume(cm3/g)",
    "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)",
    "pHpzc",
    "C_molar", "H_C_molar","O_C_molar","N_C_molar","S_C_molar",
    "Initial_Concentration(mg/L)",
    "Solution_pH",
    "Temperature(K)",
    "Agitation_speed(rpm)",
    "Dosage(g/L)",
    "Contact_Time(min)",
    "E","S","A","B","V",
]
num_feats = [c for c in num_feats if c in df.columns]

# Sadece Activation_Atmosphere kategorik olsun
cat_feats = ["Activation_Atmosphere"] if "Activation_Atmosphere" in df.columns else []

# dtype 'category' olarak ayarla
for c in cat_feats:
    df[c] = df[c].astype("category")


for c in num_feats + [target_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

for c in cat_feats:
    df[c] = df[c].astype("category")

X = df[num_feats + cat_feats].copy()
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

pre_passthrough = "passthrough"

# OneHotEncoder uyumluluğu
try:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

pre_ohe = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_feats),
        ("cat", ohe, cat_feats)
    ],
    remainder="drop"
)

# CatBoost için kategorik kolon indeksleri (passthrough düzeninde)
cat_idx = [X.columns.get_loc(c) for c in cat_feats]

# -------------------- 5) MODEL SARMA (Cat/LGBM/XGB) --------------------
from sklearn.base import BaseEstimator, RegressorMixin

class CatBoostSk(BaseEstimator, RegressorMixin):
    def __init__(self, depth=8, learning_rate=0.05, n_estimators=1200, random_state=42, verbose=False, allow_writing_files=False):
        self.depth = depth; self.learning_rate = learning_rate
        self.n_estimators = n_estimators; self.random_state = random_state
        self.verbose = verbose; self.allow_writing_files = allow_writing_files
        if not have.get("catboost", False):
            self.model_ = None
        else:
            self.model_ = CatBoostRegressor(
                depth=self.depth, learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                loss_function="RMSE", random_state=self.random_state, verbose=self.verbose,
                allow_writing_files=self.allow_writing_files
            )
    def fit(self, X, y):
        if self.model_ is None: raise RuntimeError("CatBoost yüklü değil.")
        self.model_.fit(X, y, cat_features=cat_idx if len(cat_idx) else None)
        return self
    def predict(self, X): return self.model_.predict(X)
    

class LGBMSk(BaseEstimator, RegressorMixin):
    def __init__(self, boosting_type="gbdt", num_leaves=63, learning_rate=0.05, n_estimators=1500, random_state=42):
        self.boosting_type = boosting_type; self.num_leaves = num_leaves
        self.learning_rate = learning_rate; self.n_estimators = n_estimators
        self.random_state = random_state
        if not have.get("lightgbm", False):
            self.model_ = None
        else:
            self.model_ = LGBMRegressor(
                boosting_type=self.boosting_type, num_leaves=self.num_leaves,
                learning_rate=self.learning_rate, n_estimators=self.n_estimators,
                random_state=self.random_state, n_jobs=1
            )
    def fit(self, X, y):
        if self.model_ is None: raise RuntimeError("LightGBM yüklü değil.")
        self.model_.fit(X, y, categorical_feature=cat_feats if len(cat_feats) else None)
        return self
    def predict(self, X): return self.model_.predict(X)

class XGBSk(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=1500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=1.0, booster="gbtree", random_state=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.learning_rate=learning_rate
        self.subsample=subsample; self.colsample_bytree=colsample_bytree
        self.reg_lambda=reg_lambda; self.booster=booster; self.random_state=random_state
        if not have.get("xgboost", False):
            self.model_ = None
        else:
            self.model_ = XGBRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=self.learning_rate, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, reg_lambda=self.reg_lambda,
                booster=self.booster, tree_method="hist", enable_categorical=True,
                random_state=self.random_state, n_jobs=1
            )
    def fit(self, X, y):
        if self.model_ is None: raise RuntimeError("XGBoost yüklü değil.")
        self.model_.fit(X, y)
        return self
    def predict(self, X): return self.model_.predict(X)

# -------------------- 6) MODELLER --------------------
models = []
if have["catboost"]:
    models.append(("CatBoost", Pipeline([("pre", pre_passthrough), ("reg", CatBoostSk())])))
else:
    models.append(("CatBoost (skipped)", None))

if have["lightgbm"]:
    models.append(("LightGBM-GBDT", Pipeline([("pre", pre_passthrough), ("reg", LGBMSk(boosting_type="gbdt"))])))
else:
    models.append(("LightGBM-GBDT (skipped)", None))

if have["xgboost"]:
    models.append(("XGBoost-GBTree", Pipeline([("pre", pre_passthrough), ("reg", XGBSk())])))
else:
    models.append(("XGBoost-GBTree (skipped)", None))

models.append(("HistGBR", Pipeline([("pre", pre_ohe), ("reg", HistGradientBoostingRegressor(
    max_depth=None, learning_rate=0.08, max_iter=500, random_state=RANDOM_STATE
))])))

if have["ebm"]:
    models.append(("EBM", Pipeline([("pre", pre_ohe), ("reg", ExplainableBoostingRegressor(
        interactions=0, max_leaves=3, learning_rate=0.05, max_bins=256, random_state=RANDOM_STATE
    ))])))
else:
    models.append(("EBM (skipped)", None))

if have["lightgbm"]:
    models.append(("LightGBM-DART", Pipeline([("pre", pre_passthrough), ("reg", LGBMSk(boosting_type="dart"))])))
else:
    models.append(("LightGBM-DART (skipped)", None))

# -------------------- 7) CV & METRİKLER --------------------
def rmse_val(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"r2":"r2","rmse":"neg_mean_squared_error","mae":"neg_mean_absolute_error"}

# Plot aralığı
y_all = pd.concat([y_train, y_test]).values
y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
pad = 0.05*(y_max - y_min) if (y_max > y_min) else 1.0
lo, hi = y_min - pad, y_max + pad
BLUE = "#1f77b4"; RED = "#d62728"

fig, axes = plt.subplots(2, 3, figsize=(15, 9)); axes = axes.ravel()
results, fold_store = [], {}

with parallel_backend("threading", n_jobs=N_JOBS):
    for ax, (name, pipe) in zip(axes, models):
        if pipe is None:
            ax.axis("off"); ax.set_title(name); continue

        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring,
                               n_jobs=N_JOBS, return_train_score=False)
        fold_r2   = cvres["test_r2"]
        fold_rmse = np.sqrt(-cvres["test_rmse"])
        fold_mae  = -cvres["test_mae"]

        cv_r2, cv_rmse, cv_mae = fold_r2.mean(), fold_rmse.mean(), fold_mae.mean()
        print(f"[CV] {name:14s} | R2={cv_r2:.3f} | RMSE={cv_rmse:.3f} | MAE={cv_mae:.3f}")

        pipe.fit(X_train, y_train)
        yhat_tr = pipe.predict(X_train)
        yhat_te = pipe.predict(X_test)

        tr_r2, te_r2 = r2_score(y_train, yhat_tr), r2_score(y_test, yhat_te)
        tr_rmse, te_rmse = rmse_val(y_train, yhat_tr), rmse_val(y_test, yhat_te)
        tr_mae, te_mae = mean_absolute_error(y_train, yhat_tr), mean_absolute_error(y_test, yhat_te)

        results.append({
            "model": name,
            "cv_r2": cv_r2, "cv_rmse": cv_rmse, "cv_mae": cv_mae,
            "train_r2": tr_r2, "train_rmse": tr_rmse, "train_mae": tr_mae,
            "test_r2": te_r2, "test_rmse": te_rmse, "test_mae": te_mae,
        })
        fold_store[name] = (fold_r2, fold_rmse, fold_mae, pipe)

        ax.scatter(y_train, yhat_tr, s=18, alpha=0.65, color="#03C2FB",
                   label="Train" if name == models[0][0] else None, edgecolors="none")
        ax.scatter(y_test,  yhat_te, s=30, alpha=0.85, color="#001583", marker="^",
                   label="Test"  if name == models[0][0] else None, edgecolors="none")
        ax.plot([lo,hi],[lo,hi],"--",lw=1.0)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_title(name); ax.set_xlabel("Gerçek qe (mg/g)"); ax.set_ylabel("Tahmin qe (mg/g)")
        tr_line = f"Train R2={tr_r2:.2f} | RMSE={tr_rmse:.1f} | MAE={tr_mae:.1f}"
        te_line = f"Test  R2={te_r2:.2f} | RMSE={te_rmse:.1f} | MAE={te_mae:.1f}"
        ax.text(0.02, 0.98, f"{tr_line}\n{te_line}\nCV R2={cv_r2:.2f}, RMSE={cv_rmse:.1f}, MAE={cv_mae:.1f}",
                transform=ax.transAxes, fontsize=8, family="monospace", va="top", ha="left")

axes[0].legend(loc="lower right")

# --- GRAFİĞİ KAYDET ---
os.makedirs(OUT_FIG, exist_ok=True)   # figures klasörünü oluştur

out_fig = os.path.join(OUT_FIG, "ml_results.png")
plt.tight_layout()
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
print(f"[OK] Grafik kaydedildi: {out_fig}")
plt.show()


# -------------------- Özet & En iyi model --------------------
res_df = pd.DataFrame(results)
if len(res_df):
    res_df_sorted = res_df.sort_values(["test_r2","cv_r2"], ascending=False)
    print("\n=== Özet (CV & Test, Test R2'ye göre sıralı) ===")
    print(res_df_sorted.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    best_name = res_df_sorted.iloc[0]["model"]
    fold_r2, fold_rmse, fold_mae, best_pipe = fold_store[best_name]

    # Fold validasyon örnek sayıları
    fold_sizes = []
    for _, val_idx in cv.split(X_train, y_train):
        fold_sizes.append(len(val_idx))

    best_fold_df = pd.DataFrame({
        "Fold": np.arange(1, len(fold_r2) + 1),
        "n_val": fold_sizes,
        "R2": fold_r2,
        "RMSE": fold_rmse,
        "MAE": fold_mae
    })

    print(f"\n===== En iyi model: {best_name} — 5-Fold Detay =====")
    print(best_fold_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Mean±SD: R2={fold_r2.mean():.3f}±{fold_r2.std():.3f} | "
          f"RMSE={fold_rmse.mean():.3f}±{fold_rmse.std():.3f} | "
          f"MAE={fold_mae.mean():.3f}±{fold_mae.std():.3f}")

    with parallel_backend("threading", n_jobs=N_JOBS):
        oof_pred = cross_val_predict(best_pipe, X_train, y_train, cv=cv, n_jobs=N_JOBS)

    oof_r2   = r2_score(y_train, oof_pred)
    oof_rmse = float(np.sqrt(mean_squared_error(y_train, oof_pred)))
    oof_mae  = mean_absolute_error(y_train, oof_pred)
    print(f"OOF      : R2={oof_r2:.3f} | RMSE={oof_rmse:.3f} | MAE={oof_mae:.3f}")

    # Metrikleri Excel'e yaz
    try:
        with pd.ExcelWriter(OUT_DATA, mode="a", if_sheet_exists="replace") as xw:
            res_df_sorted.to_excel(xw, sheet_name="ML_Summary", index=False)
            best_fold_df.to_excel(xw, sheet_name="BestModel_Folds", index=False)
        print(f"[OK] Metrikler Excel sayfalarına eklendi: {OUT_DATA}")
    except Exception as e:
        print(f"[Uyarı] Metrikleri Excel'e yazarken sorun: {e}")

    