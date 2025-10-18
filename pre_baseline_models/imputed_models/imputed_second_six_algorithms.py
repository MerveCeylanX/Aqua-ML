"""
imputed_new_six_algorithms.py
--------------------------------------------------------
ML: 6 model (yeni set), 5-fold CV, tek figürde 6 subplot (RMSE fix)
--------------------------------------------------------

Bu script şunları yapar:

1) Daha önce imputasyon uygulanmış (imputed) veriyi Excel'den yükler.
   - Veri zaten temizlendiği için burada tekrar imputasyon yapılmaz.
   - Ancak pipeline içinde imputasyon adımları korunmuştur;
     böylece ileride eksik veri içeren yeni datasetler geldiğinde
     güvenle çalıştırılabilir.
2) Numerik ve kategorik feature listelerini hazırlar.
3) Train/Test setlerine böler.
4) Ön işleme pipeline kurar:
   - Numerik: median imputasyon (gerekirse) + standardizasyon
   - Kategorik: most_frequent imputasyon (gerekirse) + OneHotEncoder
5) 6 farklı regresyon modeli kurar:
   - ExtraTrees, HistGradientBoosting, KNN, AdaBoost, MLP, KernelRidge
6) Her model için:
   - 5-fold CV (R², RMSE, MAE) skorlarını hesaplar.
   - Train ve Test metriklerini yazdırır.
   - Tahmin vs Gerçek scatter grafiklerini tek figürde (2x3 subplot) çizer.
7) Tüm modellerin sonuçlarını özet tablo halinde gösterir.
8) Kullanılan final feature adlarını (ön işleme sonrası, OHE genişlemeli) yazdırır.
9) CV R²’si en yüksek 2 model için fold bazlı CV metriklerini (R², RMSE, MAE) detaylı olarak raporlar.
10) Çıktılar:
    - Grafik → ./figures/imputed_second_six_algorithms.png
    - Konsol → özet tablo, en iyi 2 model fold sonuçları, kullanılan feature listesi
--------------------------------------------------------
"""

import os
import pandas as pd
import warnings, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Yeni modeller
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge

warnings.filterwarnings("ignore")

OUT_PATH  = r"D:\Aqua_ML\pre_baseline_models\imputed_models\data\Raw_data_imputed.xlsx"
ML_PATH = OUT_PATH
df_ml = pd.read_excel(ML_PATH)

# --- Config ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Özellikler / hedef ---
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
    "C_molar",
    "H_C_molar",
    "O_C_molar",
    "N_C_molar",
    "S_C_molar",
    "Initial_Concentration(mg/L)",
    "Solution_pH",
    "Temperature(K)",
    "Agitation_speed(rpm)",
    "Dosage(g/L)",
    "Contact_Time(min)",
    "E","S","A","B","V",
]
cat_feats = []
if "Activation_Atmosphere" in df_ml.columns:
    cat_feats.append("Activation_Atmosphere")

num_feats = [c for c in num_feats if c in df_ml.columns]
cat_feats = [c for c in cat_feats if c in df_ml.columns]

target_col = "qe(mg/g)"
if target_col not in df_ml.columns:
    raise KeyError(f"Hedef kolon yok: {target_col}")

for c in num_feats + [target_col]:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

X = df_ml[num_feats + cat_feats].copy()
y = df_ml[target_col]

# --- Train/Test böl ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# --- Ön işleme ---
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
pre = ColumnTransformer(
    transformers=[("num", num_transformer, num_feats),
                  ("cat", cat_transformer, cat_feats)],
    remainder="drop"
)

# --- Modeller ---
models = {
    "ExtraTrees": ExtraTreesRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=1
    ),
    "HistGBR": HistGradientBoostingRegressor(
        max_depth=None, learning_rate=0.08, max_iter=500, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsRegressor(
        n_neighbors=15, weights="distance"
    ),
    "AdaBoost": AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=4, random_state=RANDOM_STATE),
        n_estimators=400, learning_rate=0.05, random_state=RANDOM_STATE
    ),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(128, 64), activation="relu", alpha=1e-3,
        learning_rate_init=1e-3, max_iter=2000, random_state=RANDOM_STATE
    ),
    "KernelRidge": KernelRidge(
        alpha=1.0, kernel="rbf", gamma=None
    ),
}

def rmse_val(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
scoring = {
    "r2": "r2",
    "rmse": "neg_mean_squared_error",
    "mae": "neg_mean_absolute_error"
}

def fmt_metrics(y_true, y_pred, prefix=""):
    r2  = r2_score(y_true, y_pred)
    rmse_ = rmse_val(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    return f"{prefix}R2={r2:.3f} | RMSE={rmse_:.3f} | MAE={mae:.3f}"

# --- Tek figür ve ortak eksen limitleri ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.ravel()
y_all = np.concatenate([y_train.values, y_test.values])
y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
pad = 0.05 * (y_max - y_min) if (y_max > y_min) else 1.0
lo, hi = y_min - pad, y_max + pad

results = []
for ax, (name, reg) in zip(axes, models.items()):
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    cvres = cross_validate(pipe, X_train, y_train, cv=cv,
                           scoring=scoring, n_jobs=-1,
                           return_train_score=False)
    fold_r2   = cvres["test_r2"]
    fold_rmse = np.sqrt(-cvres["test_rmse"])
    fold_mae  = -cvres["test_mae"]
    cv_r2, cv_rmse, cv_mae = fold_r2.mean(), fold_rmse.mean(), fold_mae.mean()

    # Fit ve tahmin
    pipe.fit(X_train, y_train)
    yhat_tr, yhat_te = pipe.predict(X_train), pipe.predict(X_test)

    tr_line = fmt_metrics(y_train, yhat_tr, prefix="Train ")
    te_line = fmt_metrics(y_test,  yhat_te, prefix="Test  ")

    results.append({
        "model": name,"cv_r2": cv_r2,"cv_rmse": cv_rmse,"cv_mae": cv_mae,
        "train_r2": r2_score(y_train, yhat_tr),"train_rmse": rmse_val(y_train, yhat_tr),"train_mae": mean_absolute_error(y_train, yhat_tr),
        "test_r2": r2_score(y_test, yhat_te),"test_rmse": rmse_val(y_test, yhat_te),"test_mae": mean_absolute_error(y_test, yhat_te),
    })

    
    # --- Subplot çizim ---
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0)
    ax.scatter(y_train, yhat_tr, s=18, alpha=0.65, color="#03C2FB",
               label="Train" if name=="ExtraTrees" else None, edgecolors="none")
    ax.scatter(y_test,  yhat_te, s=30, alpha=0.85, color="#FECB04",
               marker="^", label="Test" if name=="ExtraTrees" else None, edgecolors="none")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_title(name)
    ax.set_xlabel("Gerçek qe (mg/g)"); ax.set_ylabel("Tahmin qe (mg/g)")
    ax.text(0.02, 0.98,
            f"{tr_line}\n{te_line}\nCV R2={cv_r2:.2f}, RMSE={cv_rmse:.2f}, MAE={cv_mae:.2f}",
            transform=ax.transAxes, fontsize=8, family="monospace", va="top", ha="left")

axes[0].legend(loc="lower right")
plt.tight_layout()

# --- figures klasörüne kaydet ---
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
out_fig = FIG_DIR / "imputed_second_six_algorithms.png"
plt.savefig(out_fig, dpi=200, bbox_inches="tight")
print(f"[OK] Grafik kaydedildi: {out_fig}")

plt.show()

# --- Özet tablo ---
res_df = pd.DataFrame(results)
print("\n=== Özet (CV ve Test) ===")
print(res_df.sort_values("test_r2", ascending=False).to_string(index=False,
      float_format=lambda x: f"{x:.3f}"))

# === Ön işleme sonrası feature isimleri ===
from sklearn.linear_model import Ridge
pipe = Pipeline([("pre", pre), ("reg", Ridge())])
pipe.fit(X_train, y_train)
feature_names = pipe.named_steps["pre"].get_feature_names_out()
print("\nModelin gördüğü featurelar:")
print(feature_names)
print("Toplam:", len(feature_names))

# === En iyi 2 model için fold bazında CV metrikleri ===
top2 = res_df.sort_values("cv_r2", ascending=False).head(2)["model"].tolist()
print("\n=== En İyi 2 Model — Fold Bazında CV Sonuçları ===")
for name in top2:
    print(f"\n>>> {name} <<<")
    reg = models[name]
    pipe = Pipeline([("pre", pre), ("reg", reg)])
    cvres = cross_validate(pipe, X_train, y_train, cv=cv,
                           scoring=scoring, n_jobs=-1,
                           return_train_score=False)
    fold_r2   = cvres["test_r2"]
    fold_rmse = np.sqrt(-cvres["test_rmse"])
    fold_mae  = -cvres["test_mae"]
    for i, (r2i, ri, mi) in enumerate(zip(fold_r2, fold_rmse, fold_mae), start=1):
        print(f"  Fold {i}: R2={r2i:.3f} | RMSE={ri:.3f} | MAE={mi:.3f}")
    print(f"  Mean±SD: R2={fold_r2.mean():.3f}±{fold_r2.std():.3f} | "
          f"RMSE={fold_rmse.mean():.3f}±{fold_rmse.std():.3f} | "
          f"MAE={fold_mae.mean():.3f}±{fold_mae.std():.3f}")
