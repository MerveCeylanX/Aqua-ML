"""
not_imputed_best_two_SHAP.py
----------------------------------------------------
ML Workflow: XGBoost + CatBoost + SHAP  (One-Hot for Activation_Atmosphere)
----------------------------------------------------
- Ham veri yüklenir, farmasötik özellikler ve molar oranlar eklenir
- Activation_Atmosphere --> one-hot dummies (activation_atmosphere_air, ... )
- XGBoost ve CatBoost modelleri ile eğitim yapılır
- CV, Train/Test performansı raporlanır
- SHAP importance ve summary grafikleri çıkarılır
- Grafikler 'figures/' klasörüne, OOF sonuçları 'results/' klasörüne kaydedilir
"""

# === ML (NaN-friendly): Ham veri + PHARM merge + molar oranlar + XGBoost-GBTree + SHAP ===
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import parallel_backend

warnings.filterwarnings("ignore")

# -------------------- KULLANICI AYARLARI --------------------
IN_PATH      = r"D:\Aqua_ML\data\Raw_data.xlsx"   # Ham veri girişi
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# -------------------- DİZİN/ÇIKTI KURULUMU --------------------
try:
    BASE_DIR = Path(__file__).resolve().parent   # .py dosyasının klasörü
except NameError:
    BASE_DIR = Path.cwd()                        # notebook vb. için alternatif

DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR     = BASE_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Zenginleştirilmiş veri yeni konum (timestamp YOK)
OUT_DATA = DATA_DIR / "Raw_data_enriched.xlsx"
# Sonuç Excel (timestamp YOK — her çalıştırmada üzerine yazar)
RESULTS_XLSX = RESULTS_DIR / "ml_results.xlsx"

# -------------------- CPU / loky fix --------------------
N_JOBS = max(1, (os.cpu_count() or 1) - 1)
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)

# -------------------- 0) VERİYİ YÜKLE --------------------
df_ml = pd.read_excel(IN_PATH)
df = df_ml.copy()

# -------------------- 1) PHARM ÖZELLİKLERİNİ EKLE --------------------
# Farmasötik özellikler tablosu (kısaltma, E, S, A, B, V)
# Veriler UFZ LSER-Database'den alınmıştır.

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
for c in map_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------- 2) ELEMENTAL ORANLAR --------------------
C, O, H, N, S = "C_percent", "O_percent", "H_percent", "N_percent", "S_percent"
aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}

for col in [C, O, H, N, S]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        print(f"[Uyarı] {col} kolonu bulunamadı; ilgili molar hesaplar NaN kalabilir.")

if C in df.columns:
    maskC = df[C].notna() & (df[C] > 0)
    badC = (~maskC).sum()
    if badC:
        print(f"[Not] {badC} satırda {C} yok veya ≤0; H/C, O/C, N/C, S/C ve C_molar hesaplanmadı (NaN).")

    nC = df.loc[maskC, C] / aw["C"]
    df.loc[maskC, "C_molar"] = nC

    if H in df.columns:
        df.loc[maskC, "H_C_molar"] = (df.loc[maskC, H] / aw["H"]) / nC
    if O in df.columns:
        df.loc[maskC, "O_C_molar"] = (df.loc[maskC, O] / aw["O"]) / nC
    if N in df.columns:
        df.loc[maskC, "N_C_molar"] = (df.loc[maskC, N] / aw["N"]) / nC
    if S in df.columns:
        df.loc[maskC, "S_C_molar"] = (df.loc[maskC, S] / aw["S"]) / nC

    for r in ["C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar"]:
        if r in df.columns:
            df[r] = pd.to_numeric(df[r], errors="coerce").clip(lower=0)

# -------------------- 3) ZENGİNLEŞTİRİLMİŞ VERİ KAYDI -> ./data/ --------------------
try:
    df.to_excel(OUT_DATA, index=False)
    print(f"[OK] Zenginleştirilmiş veri kaydedildi: {OUT_DATA}")
except Exception as e:
    print(f"[Uyarı] Excel kaydında sorun: {e}")

# -------------------- 4) ML HAZIRLIK --------------------
target_col = "qe(mg/g)"
if target_col not in df.columns:
    raise KeyError(f"Hedef kolon yok: {target_col}")

# Sayısal özellikler (temel + türetilenler)
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

# --- One-Hot: Activation_Atmosphere --> activation_atmosphere_* dummies ---
src_col = "Activation_Atmosphere"  # düzeltilmiş isim
if src_col in df.columns:
    df[src_col] = df[src_col].astype(str).str.strip()
    df[src_col] = df[src_col].replace({"": np.nan})

    dummies = pd.get_dummies(
        df[src_col],
        prefix="activation_atmosphere",
        prefix_sep="_",
        dtype=np.float32,
        dummy_na=False
    )

    dummies.columns = (
        dummies.columns
        .str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
        .str.lower()
    )

    df = pd.concat([df, dummies], axis=1)
    dummy_cols = list(dummies.columns)

    for c in dummy_cols:
        if c not in num_feats:
            num_feats.append(c)

    print(f"[OK] One-hot eklendi: {', '.join(dummy_cols)}")
else:
    print("[Uyarı] 'Activation_Atmosphere' bulunamadı; one-hot atlandı.")
    dummy_cols = []

# kategorik liste artık boş (hepsi numerik)
cat_feats = []

# tip dönüşümleri
for c in num_feats + [target_col]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

X = df[[c for c in num_feats if c in df.columns]].copy()
y = pd.to_numeric(df[target_col], errors="coerce")

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# -------------------- 5) XGBoost-GBTree (categorical kapalı) --------------------
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=1500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    booster="gbtree", tree_method="hist",
    enable_categorical=False,     # one-hot sonrası kapalı
    random_state=RANDOM_STATE, n_jobs=1
)

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {"r2":"r2","rmse":"neg_mean_squared_error","mae":"neg_mean_absolute_error"}
with parallel_backend("threading", n_jobs=N_JOBS):
    cvres = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring,
                           n_jobs=N_JOBS, return_train_score=False)

fold_r2   = cvres["test_r2"]
fold_rmse = np.sqrt(-cvres["test_rmse"])
fold_mae  = -cvres["test_mae"]

cv_r2, cv_rmse, cv_mae = fold_r2.mean(), fold_rmse.mean(), fold_mae.mean()
print(f"[CV] XGBoost-GBTree | R2={cv_r2:.3f} | RMSE={cv_rmse:.3f} | MAE={cv_mae:.3f}")

# Fit & Test
model.fit(X_train, y_train)
yhat_tr = model.predict(X_train)
yhat_te = model.predict(X_test)

def rmse_val(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
tr_r2, te_r2 = r2_score(y_train, yhat_tr), r2_score(y_test, yhat_te)
tr_rmse, te_rmse = rmse_val(y_train, yhat_tr), rmse_val(y_test, yhat_te)
tr_mae, te_mae = mean_absolute_error(y_train, yhat_tr), mean_absolute_error(y_test, yhat_te)

print(f"[Train] R2={tr_r2:.3f} | RMSE={tr_rmse:.3f} | MAE={tr_mae:.3f}")
print(f"[Test ] R2={te_r2:.3f} | RMSE={te_rmse:.3f} | MAE={te_mae:.3f}")

# --- SCATTER GRAFİĞİ -> ./figures/ ---
y_all = pd.concat([y_train, y_test]).values
y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
pad = 0.05*(y_max - y_min) if (y_max > y_min) else 1.0
lo, hi = y_min - pad, y_max + pad

plt.figure(figsize=(7.5, 6.0))
plt.scatter(y_train, yhat_tr, s=18, alpha=0.65, color="#03C2FB", label="Train", edgecolors="none")
plt.scatter(y_test,  yhat_te, s=30, alpha=0.85, color="#001583", marker="^", label="Test", edgecolors="none")
plt.plot([lo,hi],[lo,hi],"--",lw=1.0)
plt.xlim(lo,hi); plt.ylim(lo,hi)
plt.title("XGBoost-GBTree")
plt.xlabel("Gerçek qe (mg/g)"); plt.ylabel("Tahmin qe (mg/g)")
tr_line = f"Train R2={tr_r2:.2f} | RMSE={tr_rmse:.1f} | MAE={tr_mae:.1f}"
te_line = f"Test  R2={te_r2:.2f} | RMSE={te_rmse:.1f} | MAE={te_mae:.1f}"
plt.text(0.02, 0.98, f"{tr_line}\n{te_line}\nCV R2={cv_r2:.2f}, RMSE={cv_rmse:.1f}, MAE={cv_mae:.1f}",
         transform=plt.gca().transAxes, fontsize=9, family="monospace", va="top", ha="left")
out_fig = FIG_DIR / "ml_results_xgboost.png"
plt.tight_layout(); plt.savefig(out_fig, dpi=200, bbox_inches="tight"); plt.close()
print(f"[OK] Grafik kaydedildi: {out_fig}")

# --- OOF ---
with parallel_backend("threading", n_jobs=N_JOBS):
    oof_pred = cross_val_predict(model, X_train, y_train, cv=cv, n_jobs=N_JOBS)
oof_r2   = r2_score(y_train, oof_pred)
oof_rmse = float(np.sqrt(mean_squared_error(y_train, oof_pred)))
oof_mae  = mean_absolute_error(y_train, oof_pred)
print(f"OOF      : R2={oof_r2:.3f} | RMSE={oof_rmse:.3f} | MAE={oof_mae:.3f}")

# --- METRİKLER -> ./results/ml_results.xlsx ---
try:
    res_row = pd.DataFrame([{
        "model": "XGBoost-GBTree (one-hot Activation_Atmosphere)",
        "cv_r2": cv_r2, "cv_rmse": cv_rmse, "cv_mae": cv_mae,
        "train_r2": tr_r2, "train_rmse": tr_rmse, "train_mae": tr_mae,
        "test_r2": te_r2, "test_rmse": te_rmse, "test_mae": te_mae,
        "oof_r2": oof_r2, "oof_rmse": oof_rmse, "oof_mae": oof_mae
    }])
    with pd.ExcelWriter(RESULTS_XLSX, mode="w", engine="openpyxl") as xw:
        res_row.to_excel(xw, sheet_name="ML_Summary", index=False)
    print(f"[OK] Metrikler yazıldı: {RESULTS_XLSX}")
except Exception as e:
    print(f"[Uyarı] Sonuç Excel yazımında sorun: {e}")

# -------------------- 6) SHAP (interventional, NaN OK, sağlam fallback) --------------------
try:
    import shap
    import scipy.sparse as sp
    from catboost import CatBoostRegressor

    def to_numeric_array(df_like):
        if isinstance(df_like, pd.DataFrame):
            df_num = df_like.copy()
            for c in df_num.columns:
                if pd.api.types.is_numeric_dtype(df_num[c]):
                    continue
                if pd.api.types.is_categorical_dtype(df_num[c]):
                    df_num[c] = df_num[c].cat.codes
                else:
                    df_num[c] = pd.factorize(df_num[c], sort=True)[0]
            arr = df_num.to_numpy(dtype=np.float64)
        else:
            arr = np.asarray(df_like)
            if sp.issparse(arr): arr = arr.toarray()
            if arr.dtype.kind != "f":
                arr = arr.astype(np.float64, copy=False)
        return arr

    feat_names = list(X.columns)

    # -------- XGBoost için SHAP --------
    X_bkg_full = to_numeric_array(X_train)
    X_te_full  = to_numeric_array(X_test)

    rng = np.random.RandomState(RANDOM_STATE)
    n = X_bkg_full.shape[0]
    k = min(1000, n)
    sel = rng.choice(n, size=k, replace=False)
    X_bkg = X_bkg_full[sel]

    try:
        explainer = shap.TreeExplainer(model, X_bkg, feature_perturbation="interventional")
        shap_values = explainer(X_te_full)
    except Exception:
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        shap_values = explainer(X_te_full)

    out_shap = FIG_DIR / "shap_summary_xgboost.png"
    shap.summary_plot(shap_values.values if hasattr(shap_values,"values") else shap_values,
                      features=X_te_full, feature_names=feat_names,
                      show=False, max_display=30)
    plt.tight_layout(); plt.savefig(out_shap, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[OK] XGBoost SHAP summary kaydedildi: {out_shap}")

    # --- XGBoost SHAP individual bar (custom format) ---
    shap_abs = np.abs(shap_values.values if hasattr(shap_values,"values") else shap_values)
    mean_abs = shap_abs.mean(axis=0)
    
    imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    
    print("\n=== XGBoost SHAP (bireysel, |SHAP| ort.) — İlk 30 ===")
    print(imp_df.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    # --- Individual SHAP barh (ilk 30) ---
    topn = min(30, len(imp_df))
    top_df = imp_df.head(topn).iloc[::-1]  # barh için ters sırala
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"], alpha=0.85, color="#0014E7")
    plt.xlabel("Ortalama |SHAP| (XGBoost, TreeExplainer)")
    plt.title("Özellik Önemi (SHAP, kategorik seviyeler ayrı)")
    plt.tight_layout()
    
    out_shap_bar = FIG_DIR / "shap_importance_xgboost.png"
    plt.savefig(out_shap_bar, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] XGBoost SHAP importance bar kaydedildi: {out_shap_bar}")

    # -------- CatBoost için SHAP --------
    # One-hot yaptığımız için artık kategorik indeks yok (hepsi numerik)
    cat_model = CatBoostRegressor(
        depth=8, learning_rate=0.05, n_estimators=1200,
        loss_function="RMSE", random_state=RANDOM_STATE,
        verbose=False, allow_writing_files=False
    )
    cat_model.fit(X_train, y_train)

    cat_explainer = shap.TreeExplainer(cat_model)
    cat_shap_values = cat_explainer.shap_values(X_test)
    if isinstance(cat_shap_values, list):
        cat_shap_values = cat_shap_values[0]

    out_shap_cat = FIG_DIR / "shap_summary_catboost.png"
    shap.summary_plot(cat_shap_values, features=X_test, feature_names=feat_names,
                      show=False, max_display=30)
    plt.tight_layout(); plt.savefig(out_shap_cat, dpi=200, bbox_inches="tight"); plt.close()
    print(f"[OK] CatBoost SHAP summary kaydedildi: {out_shap_cat}")

    # --- CatBoost SHAP individual bar (custom format) ---
    cat_shap_abs = np.abs(cat_shap_values)
    cat_mean_abs = cat_shap_abs.mean(axis=0)
    
    cat_imp_df = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": cat_mean_abs
    }).sort_values("mean_abs_shap", ascending=False)
    
    print("\n=== CatBoost SHAP (bireysel, |SHAP| ort.) — İlk 30 ===")
    print(cat_imp_df.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    
    # --- Individual SHAP barh (ilk 30) ---
    topn = min(30, len(cat_imp_df))
    top_df = cat_imp_df.head(topn).iloc[::-1]  # barh için ters sırala
    
    plt.figure(figsize=(10, 8))
    plt.barh(top_df["feature"], top_df["mean_abs_shap"], alpha=0.85, color="#0014E7")
    plt.xlabel("Ortalama |SHAP| (CatBoost, TreeExplainer)")
    plt.title("Özellik Önemi (SHAP, kategorik seviyeler ayrı)")
    plt.tight_layout()
    
    out_shap_bar_cat = FIG_DIR / "shap_importance_catboost.png"
    plt.savefig(out_shap_bar_cat, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] CatBoost SHAP importance bar kaydedildi: {out_shap_bar_cat}")

except Exception as e:
    print(f"[Uyarı] SHAP hesaplanamadı: {e}")
