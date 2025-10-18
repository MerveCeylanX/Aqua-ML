"""
oof_error_bins.py

- 0–5, 5–10, ... % aralıkları için hata bin tablosu hesaplanır
- Her bin için hem adet (count) hem de yüzde (%) değerleri üretilir
- Çıktı ./results/oof_error_bins.xlsx dosyasına kaydedilir
- Eğer aynı isimli dosya varsa üzerine yazılır
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import parallel_backend

warnings.filterwarnings("ignore")

# -------------------- YOLLAR / DİZİNLER --------------------
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR     = BASE_DIR / "figures"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = DATA_DIR / "Raw_data_enriched.xlsx"  # önceki scriptin ürettiği dosya

# -------------------- AYARLAR --------------------
RANDOM_STATE = 42
N_SPLITS     = 5
N_JOBS       = max(1, (os.cpu_count() or 1) - 1)
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)

TARGET_COL = "qe(mg/g)"
CAT_COL    = "Activation_Atmosphere"  # varsa tek kategorik
NUM_FEATS = [
    "Agent/Sample(g/g)","Soaking_Time(min)","Soaking_Temp(K)",
    "Activation_Time(min)","Activation_Temp(K)","Activation_Heating_Rate (K/min)",
    "BET_Surface_Area(m2/g)","Total_Pore_Volume(cm3/g)","Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)","pHpzc",
    "C_molar","H_C_molar","O_C_molar","N_C_molar","S_C_molar",
    "Initial_Concentration(mg/L)","Solution_pH","Temperature(K)",
    "Agitation_speed(rpm)","Dosage(g/L)","Contact_Time(min)",
    "E","S","A","B","V",
]

# -------------------- VERİ --------------------
if not IN_PATH.exists():
    raise FileNotFoundError(f"Girdi bulunamadı: {IN_PATH}\nÖnce zenginleştirilmiş veriyi üretin.")

df = pd.read_excel(IN_PATH)

# tipler
for c in NUM_FEATS + [TARGET_COL]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# kategorik -> kod (NaN'ı koruyarak)
cat_used = CAT_COL in df.columns
if cat_used:
    cat = df[CAT_COL].astype("category")
    codes = cat.cat.codes.replace(-1, np.nan).astype("float64")  # -1 (NaN) geri NaN olsun
    df["__cat_code"] = codes
    FEATS = NUM_FEATS + ["__cat_code"]
else:
    FEATS = NUM_FEATS

# X, y
X = df[FEATS].copy()
y = df[TARGET_COL].copy()

# SADECE hedef (y) NaN olan satırları at
mask_valid = y.notna()
X = X.loc[mask_valid].reset_index(drop=True)
y = y.loc[mask_valid].reset_index(drop=True)

if len(X) < N_SPLITS:
    raise ValueError("Yeterli gözlem yok: OOF için satır sayısı fold sayısından büyük olmalı.")

# -------------------- MODELLER --------------------
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

xgb = XGBRegressor(
    n_estimators=1500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    booster="gbtree", tree_method="hist",
    enable_categorical=False,  # cat'ı önceden kodladık
    random_state=RANDOM_STATE, n_jobs=1
)

# CatBoost'u da önceden kodlanmış X ile kullanıyoruz (cat_features vermeye gerek yok)
cat = CatBoostRegressor(
    depth=8, learning_rate=0.05, n_estimators=1200,
    loss_function="RMSE", random_state=RANDOM_STATE,
    verbose=False, allow_writing_files=False
)

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# -------------------- OOF TAHMİNLER --------------------
with parallel_backend("threading", n_jobs=N_JOBS):
    oof_xgb = cross_val_predict(xgb, X, y, cv=cv, n_jobs=N_JOBS)
with parallel_backend("threading", n_jobs=N_JOBS):
    oof_cat = cross_val_predict(cat, X, y, cv=cv, n_jobs=N_JOBS)

# -------------------- METRİKLER --------------------
def rmse(a,b): return float(np.sqrt(mean_squared_error(a,b)))
def mae(a,b):  return float(mean_absolute_error(a,b))
def r2(a,b):   return float(r2_score(a,b))

print("[OOF] XGB  -> R2={:.3f}  RMSE={:.3f}  MAE={:.3f}".format(r2(y,oof_xgb), rmse(y,oof_xgb), mae(y,oof_xgb)))
print("[OOF] CAT  -> R2={:.3f}  RMSE={:.3f}  MAE={:.3f}".format(r2(y,oof_cat), rmse(y,oof_cat), mae(y,oof_cat)))

# -------------------- % HATA (MAPE) VE BİNLER --------------------
EPS = 1e-8
den = np.maximum(np.abs(y.values), EPS)
mape_xgb = 100.0*np.abs((y.values - oof_xgb)/den)
mape_cat = 100.0*np.abs((y.values - oof_cat)/den)

bin_edges = np.array([0,5,10,15,20,25,30,35,40,45,50, np.inf], dtype=float)
bin_labels = [f"{int(bin_edges[i])}–{int(bin_edges[i+1])}%" if np.isfinite(bin_edges[i+1])
              else f"{int(bin_edges[i])}+%" for i in range(len(bin_edges)-1)]

def bin_percentages(values, edges):
    idx = np.digitize(values, edges, right=False) - 1
    counts = np.bincount(np.clip(idx, 0, len(edges)-2), minlength=len(edges)-1)
    pct = 100.0 * counts / counts.sum()
    return counts, pct

counts_xgb, pct_xgb = bin_percentages(mape_xgb, bin_edges)
counts_cat, pct_cat = bin_percentages(mape_cat, bin_edges)

# ---- tabloyu Excel olarak kaydet ----
table = pd.DataFrame({
    "Bin": bin_labels,
    "XGB_count": counts_xgb, "XGB_percent": np.round(pct_xgb, 2),
    "CAT_count": counts_cat, "CAT_percent": np.round(pct_cat, 2),
})
table_path = RESULTS_DIR / "oof_error_bins.xlsx"
with pd.ExcelWriter(table_path, engine="openpyxl") as writer:
    table.to_excel(writer, sheet_name="OOF_Error_Bins", index=False)
print(f"[OK] Bin tablosu Excel olarak kaydedildi: {table_path}")

# -------------------- GRAFİKLER (yatay, iki model) --------------------
y_pos = np.arange(len(bin_labels))
bar_h = 0.35

plt.figure(figsize=(9.2, 6.6))
bars1 = plt.barh(y_pos - bar_h/2, pct_xgb, height=bar_h,
                 label="XGBoost", alpha=0.9, color="#1E5AF9")  # koyu mavi
bars2 = plt.barh(y_pos + bar_h/2, pct_cat, height=bar_h,
                 label="CatBoost", alpha=0.9, color="#03C2FB")  # kırmızı

plt.xlabel("Örnek yüzdesi (%)")
plt.ylabel("Mutlak yüzde hata binleri")
plt.yticks(y_pos, bin_labels)
plt.title("OOF Mutlak Yüzde Hata Dağılımı (0–5, 5–10, …, 50+ %)")
plt.legend(loc="best")
plt.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.5)

# >>> x-limit ayarı: mevcut maksimumdan %20 fazla <<<
xmax = max(max(pct_xgb), max(pct_cat))
plt.xlim(0, xmax * 1.1)

# barların sonuna yüzde anotasyonu (0.5% üstü için)
def annotate_bars(bars, values):
    for rect, val in zip(bars, values):
        if val >= 0.5:
            x = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            plt.text(x + 0.6, y, f"{val:.1f}%", va="center", fontsize=8)

annotate_bars(bars1, pct_xgb)
annotate_bars(bars2, pct_cat)

plt.tight_layout()
fig_path = FIG_DIR / "oof_pct_error_bins.png"
plt.savefig(fig_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"[OK] Grafik kaydedildi: {fig_path}")

# -------------------- (Opsiyonel) Ayrı grafikler --------------------
def save_single_model_plot(pcts, title, filename):
    plt.figure(figsize=(8.5, 6.2))

    bars = plt.barh(y_pos, pcts, height=0.55, alpha=0.95,
                color="#FECB04")  # veya "#d62728"

    for rect, val in zip(bars, pcts):
        if val >= 0.5:
            x = rect.get_width()
            y = rect.get_y() + rect.get_height()/2
            plt.text(x + 0.6, y, f"{val:.1f}%", va="center", fontsize=8)
    plt.xlabel("Örnek yüzdesi (%)")
    plt.ylabel("Mutlak yüzde hata binleri")
    plt.yticks(y_pos, bin_labels)
    plt.title(title)
    plt.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.5)

    # >>> x-limit ayarı: mevcut maksimumdan %10 fazla <<<
    xmax = max(pcts)
    plt.xlim(0, xmax * 1.1)

    plt.tight_layout()
    outp = FIG_DIR / filename
    plt.savefig(outp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grafik kaydedildi: {outp}")

save_single_model_plot(pct_xgb, "OOF MAPE Dağılımı — XGBoost", "oof_pct_error_bins_xgb.png")
save_single_model_plot(pct_cat, "OOF MAPE Dağılımı — CatBoost", "oof_pct_error_bins_cat.png")
