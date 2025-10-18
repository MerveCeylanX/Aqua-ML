"""
rf_oof_analysis.py
--------------------------------------------------------
RandomForest (best parametrelerle) OOF yüzde hata analizi
--------------------------------------------------------

Bu script şunları yapar:
1) İmputed veriyi Excel'den yükler.
2) Önceden belirlenmiş en iyi RF parametreleriyle model kurar.
3) 5-fold cross_val_predict ile OOF tahminleri alır.
4) OOF yüzde hatalarını aralıklar halinde gruplar:
   - 0–5%, 5–10%, 10–15%, 15–20%, 20–30%, 30–50%, 50–100%, 100%+
5) Yatay bar grafiği çizer ve ./figures/rf_oof_pct_error_buckets.png olarak kaydeder.
6) Konsolda OOF dağılım tablosunu ve metrikleri (R2, RMSE, MAE) yazdırır.
--------------------------------------------------------
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# --- Figures klasörü ---
BASE_DIR = Path(__file__).resolve().parent
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- Veri ---
OUT_PATH  = r"D:\Aqua_ML\pre_baseline_models\imputed_models\data\Raw_data_imputed.xlsx"
df_ml = pd.read_excel(OUT_PATH)

# Config
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Özellik listeleri
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

# Sahnede olanları filtrele
num_feats = [c for c in num_feats if c in df_ml.columns]
cat_feats = [c for c in cat_feats if c in df_ml.columns]

target_col = "qe(mg/g)"
if target_col not in df_ml.columns:
    raise KeyError(f"Hedef kolon yok: {target_col}")

# Sayısallaştır
for c in num_feats + [target_col]:
    df_ml[c] = pd.to_numeric(df_ml[c], errors="coerce")

X = df_ml[num_feats + cat_feats].copy()
y = df_ml[target_col]

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Ön işleme
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

# --- Best RF parametreleri ile model ---
# Hyperparametere optimizasyon sonucu elde edilen en iyi parametreler
# (grid search ile bulunmuş olabilir)
rf = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=None,
    max_depth=None
)

pipe = Pipeline([("pre", pre), ("reg", rf)])

# --- OOF hesaplama ---
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof_pred = cross_val_predict(pipe, X_train, y_train, cv=cv, n_jobs=-1, method="predict")

# Yüzde mutlak hata
y_true = y_train.to_numpy()
eps = 1e-12
pct_err = np.abs(y_true - oof_pred) / np.maximum(np.abs(y_true), eps) * 100.0

# Aralıklar
bin_edges  = [0, 5, 10, 15, 20, 30, 50, 100, np.inf]
bin_labels = ["0–5%", "5–10%", "10–15%", "15–20%", "20–30%", "30–50%", "50–100%", "100%+"]

bucket = pd.cut(pct_err, bins=bin_edges, labels=bin_labels, right=False, include_lowest=True)

# pandas yeni sürüm uyumu: sort argümanı yok → .sort_index()
counts = bucket.value_counts().sort_index()
shares = (counts / counts.sum() * 100.0).round(2)

print("\n=== OOF Yüzde Hata Dağılımı (Train folds) ===")
oof_summary = pd.DataFrame({"Aralık": counts.index, "Adet": counts.values, "Yüzde(%)": shares.values})
print(oof_summary.to_string(index=False))

# --- Grafik ---
plt.figure(figsize=(8, 5.5))
ypos = np.arange(len(bin_labels))
plt.barh(ypos, counts.values, alpha=0.85, color="#FECB04")
plt.yticks(ypos, bin_labels)
plt.xlabel("Adet")
plt.title("OOF Yüzde Hata Dağılımı (RF, 5-fold)")
# çubuk üzerine yüzde yaz
for i, v in enumerate(counts.values):
    plt.text(v + max(counts.values)*0.01, i, f"{shares.values[i]}%", va="center")
    
xmax = counts.max()                
plt.xlim(0, xmax * 1.1)    

plt.tight_layout()
oof_fig = FIG_DIR / "rf_oof_pct_error_buckets.png"
plt.savefig(oof_fig, dpi=200, bbox_inches="tight")
print(f"[OK] OOF yüzde hata bar grafiği kaydedildi: {oof_fig}")
plt.show(); plt.close()

# --- Ek metrikler ---
oof_r2   = r2_score(y_train, oof_pred)
oof_rmse = float(np.sqrt(mean_squared_error(y_train, oof_pred)))
oof_mae  = float(mean_absolute_error(y_train, oof_pred))
print(f"\nOOF Metrikler — R2={oof_r2:.4f} | RMSE={oof_rmse:.4f} | MAE={oof_mae:.4f}")
