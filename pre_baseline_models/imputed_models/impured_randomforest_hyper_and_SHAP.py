"""
impured_randomforest_hyper_and_SHAP.py
--------------------------------------------------------
RandomForest için Hyperparameter Optimization + SHAP
(Activation_Atmosphere -> One-Hot; SHAP summary seviyeleri ayrı,
 grouped bar ana sütunda toplanmış)
--------------------------------------------------------

Bu script şunları yapar:
1) İmpute edilmiş veriyi Excel'den yükler.
2) Özellikleri (num + cat) ve hedef "qe(mg/g)" seçer.
3) Train/Test böl.
4) RF için RandomizedSearchCV (5-fold, R2).
5) En iyi RF'yi fit et, Train/Test metriklerini yazdır.
6) SHAP TreeExplainer:
   - (A) Gruplanmış bar: OHE seviyeleri ana sütuna toplanır
   - (B) Transformed summary: OHE seviyeleri ayrı ayrı gösterilir (örn. Activation_Atmosphere=N2)
7) Çıktılar:
   - figures/rf_shap_grouped_bar.png
   - figures/rf_shap_summary_transformed.png
--------------------------------------------------------
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

warnings.filterwarnings("ignore")

# ---------------------- Yol / klasörler ----------------------
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# İmpute edilmiş veri
OUT_PATH  = r"D:\Aqua_ML\pre_baseline_models\imputed_models\data\Raw_data_imputed.xlsx"
df_ml = pd.read_excel(OUT_PATH)

# ---------------------- Ayarlar ----------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------- Özellik listeleri ----------------------
num_feats = [
    "Agent/Sample(g/g)", "Soaking_Time(min)", "Soaking_Temp(K)",
    "Activation_Time(min)", "Activation_Temp(K)", "Activation_Heating_Rate (K/min)",
    "BET_Surface_Area(m2/g)", "Total_Pore_Volume(cm3/g)", "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)", "pHpzc",
    "C_molar","H_C_molar","O_C_molar","N_C_molar","S_C_molar",
    "Initial_Concentration(mg/L)", "Solution_pH", "Temperature(K)", "Agitation_speed(rpm)",
    "Dosage(g/L)", "Contact_Time(min)", "E","S","A","B","V"
]
cat_feats = []
if "Activation_Atmosphere" in df_ml.columns:
    cat_feats.append("Activation_Atmosphere")

# Mevcut olanları filtrele
num_feats = [c for c in num_feats if c in df_ml.columns]
cat_feats = [c for c in cat_feats if c in df_ml.columns]

target_col = "qe(mg/g)"
if target_col not in df_ml.columns:
    raise KeyError(f"Hedef kolon yok: {target_col}")

# Sayısallaştır (hedef dahil)
for c in num_feats + [target_col]:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

X = df_ml[num_feats + cat_feats].copy()
y = df_ml[target_col]

# ---------------------- Train/Test ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# ---------------------- OHE kategorilerini tüm veri üzerinden sabitle ----------------------
ohe_categories = None
if "Activation_Atmosphere" in cat_feats:
    vals = (
        df_ml["Activation_Atmosphere"]
        .astype(str).str.strip().replace({"": np.nan}).dropna().unique().tolist()
    )
    vals = sorted(vals)
    ohe_categories = [vals]  # cat_feats sırasına göre

# ---------------------- Ön işleme ----------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# sklearn>=1.2: sparse_output; daha eski: sparse
try:
    cat_encoder = OneHotEncoder(
        handle_unknown="ignore", drop=None, sparse_output=False,
        categories=ohe_categories
    )
except TypeError:
    cat_encoder = OneHotEncoder(
        handle_unknown="ignore", drop=None, sparse=False,
        categories=ohe_categories
    )

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  cat_encoder),
])

pre = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_feats),
        ("cat", cat_transformer, cat_feats)
    ],
    remainder="drop"
)

# ---------------------- RF Pipeline ----------------------
pipe_rf = Pipeline([
    ("pre", pre),
    ("reg", RandomForestRegressor(random_state=RANDOM_STATE))
])

# ---------------------- HPO grid ----------------------
param_grid = {
    "reg__n_estimators": [200, 400, 600, 800, 1000],
    "reg__max_depth": [None, 10, 20, 30, 50],
    "reg__min_samples_split": [2, 5, 10],
    "reg__min_samples_leaf": [1, 2, 4],
    "reg__max_features": ["sqrt", "log2", None],
}

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = RandomizedSearchCV(
    pipe_rf,
    param_distributions=param_grid,
    n_iter=30,
    cv=cv,
    scoring="r2",
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=2,
    refit=True
)

print("\n>>> RandomForest Hyperparameter Optimization başlıyor...")
search.fit(X_train, y_train)

print("\n=== En iyi RF Parametreleri ===")
print(search.best_params_)
print("Best CV R2:", f"{search.best_score_:.4f}")

# ---------------------- En iyi model ve metrikler ----------------------
best_rf = search.best_estimator_
best_rf.fit(X_train, y_train)

def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
def mae (y_true, y_pred): return float(mean_absolute_error(y_true, y_pred))

yhat_tr = best_rf.predict(X_train)
yhat_te = best_rf.predict(X_test)
print(f"\nTrain: R2={r2_score(y_train, yhat_tr):.3f} | RMSE={rmse(y_train, yhat_tr):.3f} | MAE={mae(y_train, yhat_tr):.3f}")
print(f"Test : R2={r2_score(y_test,  yhat_te):.3f} | RMSE={rmse(y_test,  yhat_te):.3f} | MAE={mae(y_test,  yhat_te):.3f}")

# ---------------------- SHAP Analizi ----------------------
print("\n>>> SHAP Analizi (RF) başlıyor...")

pre_fitted = best_rf.named_steps["pre"]
Xt_train   = pre_fitted.transform(X_train)
Xt_test    = pre_fitted.transform(X_test)

# Dense'e çevir (bazı sürümlerde sparse dönebilir)
def _to_dense(Xm):
    try:
        import scipy.sparse as sp
        if sp.issparse(Xm):
            return Xm.toarray()
    except Exception:
        pass
    return Xm

Xt_train = _to_dense(Xt_train)
Xt_test  = _to_dense(Xt_test)

# ---- Çıktı feature isimleri: genelde 'num__...','cat__onehot__Activation_Atmosphere_N2' ----
raw_feat_out = pre_fitted.get_feature_names_out().tolist()

# ---- İsim temizleme & gruplama yardımcıları (SAĞDAN AYIR) ----
def clean_name(name: str) -> str:
    """'cat__onehot__Activation_Atmosphere_N2' -> 'Activation_Atmosphere=N2'
       'num__scaler__pHpzc'                   -> 'pHpzc'
    """
    parts = name.split("__")
    last  = parts[-1]  # 'Activation_Atmosphere_N2' veya 'pHpzc'
    if name.startswith("cat__"):
        if "_" in last:
            base, level = last.rsplit("_", 1)   # <- kritik düzeltme
            return f"{base}={level}"
        return last
    else:
        return last

def base_group(name: str) -> str:
    """Gruplama için temel sütun ('Activation_Atmosphere' / 'pHpzc')."""
    parts = name.split("__")
    last  = parts[-1]
    if name.startswith("cat__"):
        return last.rsplit("_", 1)[0] if "_" in last else last  # <- kritik düzeltme
    else:
        return last

feat_out_clean = [clean_name(n) for n in raw_feat_out]
group_keys     = [base_group(n) for n in raw_feat_out]

# Teşhis: kategorik seviyeler
if "Activation_Atmosphere" in cat_feats:
    cat_levels = [n for n in feat_out_clean if n.startswith("Activation_Atmosphere=")]
    print(f"[Teşhis] Activation_Atmosphere seviye sayısı: {len(cat_levels)}")
    print(f"[Teşhis] İlk 10 seviye: {cat_levels[:10]}")

# === SHAP: TreeExplainer (RF) ===
explainer   = shap.TreeExplainer(best_rf.named_steps["reg"])
# Performans için test setinden örnekleme
rng   = np.random.RandomState(RANDOM_STATE)
n_exp = min(400, Xt_test.shape[0])
idx   = rng.choice(Xt_test.shape[0], size=n_exp, replace=False)
Xexp  = Xt_test[idx]

shap_values = explainer.shap_values(Xexp)   # (n_samples_exp, n_features_out)

# ---------- (A) Bireysel SHAP: OHE seviyeleri ayrı ayrı göster ----------
shap_abs = np.abs(shap_values)
mean_abs = shap_abs.mean(axis=0)

imp_df = pd.DataFrame({
    "feature_clean": feat_out_clean,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== SHAP (bireysel, |SHAP| ort.) — İlk 30 ===")
print(imp_df.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# --- Bireysel SHAP barh (ilk 30) ---
topn   = min(30, len(imp_df))
top_df = imp_df.head(topn).iloc[::-1]  # barh için ters sırala

plt.figure(figsize=(10, 8))
plt.barh(top_df["feature_clean"], top_df["mean_abs_shap"], alpha=0.85, color="#0014E7")
plt.xlabel("Ortalama |SHAP| (RF, TreeExplainer)")
plt.title("Özellik Önemi (SHAP, kategorik seviyeler ayrı)")
plt.tight_layout()

out_bar = FIG_DIR / "rf_shap_individual_bar.png"
plt.savefig(out_bar, dpi=200, bbox_inches="tight")
plt.close()

# ---------- (B) Transformed-space summary: OHE seviyeleri ayrı göster ----------
shap.summary_plot(shap_values, Xexp, feature_names=feat_out_clean, show=False, max_display=30)
plt.tight_layout()

out_sum = FIG_DIR / "rf_shap_summary_transformed.png"
plt.savefig(out_sum, dpi=200, bbox_inches="tight")
plt.close()

print("\n[Teşhis] Çıktı feature sayısı:", len(raw_feat_out))
