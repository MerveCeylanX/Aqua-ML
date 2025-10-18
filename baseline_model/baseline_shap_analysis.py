"""
baseline_shap_analysis.py
--------------------------------------------------------
Baseline model için SHAP analizi (shap_comparison.py XGBoost kısmından uyarlandı)
En iyi modeli joblib'den yükler ve aynı veri işleme ile SHAP analizi yapar

Bu script şunları yapar:
1) En iyi modeli joblib'den yükler
2) Ham veriyi yükler ve enrich eder (shap_comparison.py ile aynı)
3) SHAP TreeExplainer ile importance değerlerini hesaplar
4) Individual bar formatında SHAP grafiği çizer
5) Kategorik seviyeler ayrı gösterilir
6) Feature isimleri temizlenir (prefix'ler ve birimler kaldırılır)
--------------------------------------------------------
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import json
import scipy.sparse as sp

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import shap

warnings.filterwarnings("ignore")

# --- Paths & I/O ---
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Veri yolu (ham veri - enrich edilecek, shap_comparison.py ile aynı)
DATA_PATH = r"D:\Aqua_ML\data\Raw_data.xlsx"

# --- Config ---
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Feature candidates (shap_comparison.py ile aynı) ---
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

# --- En iyi model bilgilerini yükle ---
print(">>> En iyi model bilgileri yükleniyor...")

# En son model dosyasını bul
model_files = list(MODELS_DIR.glob("best_model__*.joblib"))
if not model_files:
    raise FileNotFoundError("Model dosyası bulunamadı!")

latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
print(f"Model dosyası: {latest_model}")

# Sadece metadata'yı yükle
meta_file = latest_model.with_suffix('.meta.json')
with open(meta_file, 'r') as f:
    metadata = json.load(f)

print(f"En iyi model: {metadata['best_name']}")
print(f"En iyi parametreler: {metadata['best_params']}")

# --- Veri yükleme ve enrich işlemi (shap_comparison.py XGBoost kısmı ile aynı) ---
print(">>> Veri yükleniyor ve enrich ediliyor...")
df = pd.read_excel(DATA_PATH)

# 1) PHARM ÖZELLİKLERİNİ EKLE (shap_comparison.py ile aynı)
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

# 2) ELEMENTAL ORANLAR (shap_comparison.py ile aynı)
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

print(">>> Veri enrich edildi!")

# --- Kolon filtreleri (shap_comparison.py ile aynı) ---
num_feats = [c for c in ALL_NUM_FEATS if c in df.columns]
cat_feats = [c for c in CAT_CANDIDATES if c in df.columns]
if TARGET_COL not in df.columns:
    raise KeyError(f"Hedef kolon yok: {TARGET_COL}")

# Numerik tip güvenliği
for c in num_feats + [TARGET_COL]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# One-hot encoding (shap_comparison.py ile aynı)
src_col = "Activation_Atmosphere"
dummy_cols = []
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
    # isimleri normalize et
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

# Artık kategorik liste yok; tüm girdiler numerik/dummy
present_num = [c for c in num_feats if c in df.columns]
X = df[present_num].copy()
y = pd.to_numeric(df[TARGET_COL], errors="coerce")

print(f"Final feature sayısı: {len(present_num)}")
print(f"Feature'lar: {present_num}")

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]} satır")
print(f"Test: {X_test.shape[0]} satır")

# --- Model oluştur ve eğit ---
print(">>> Model oluşturuluyor ve eğitiliyor...")

# Model tipine göre model oluştur
model_name = metadata['best_name']
best_params = metadata['best_params']

if model_name == "CatBoost":
    from catboost import CatBoostRegressor
    # Parametreleri temizle (reg__ prefix'ini kaldır)
    clean_params = {}
    for key, value in best_params.items():
        if key.startswith('reg__'):
            clean_key = key.replace('reg__', '')
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    
    model = CatBoostRegressor(
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        **clean_params
    )
    
elif model_name == "XGBoost":
    from xgboost import XGBRegressor
    # Parametreleri temizle (reg__ prefix'ini kaldır)
    clean_params = {}
    for key, value in best_params.items():
        if key.startswith('reg__'):
            clean_key = key.replace('reg__', '')
            clean_params[clean_key] = value
        else:
            clean_params[key] = value
    
    model = XGBRegressor(
        random_state=RANDOM_STATE,
        **clean_params
    )
    
else:
    raise ValueError(f"Desteklenmeyen model tipi: {model_name}")

print(f"Model oluşturuldu: {model_name}")
print(f"Parametreler: {clean_params}")

# Modeli eğit
model.fit(X_train, y_train)

# Test performansı
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"Test R2: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# --- SHAP hesaplama (shap_comparison.py ile aynı) ---
print(">>> SHAP değerleri hesaplanıyor...")

def to_numeric_array(df_like):
    """Convert DataFrame or array to numeric array (shap_comparison.py ile aynı)"""
    if isinstance(df_like, pd.DataFrame):
        arr = df_like.values
    else:
        arr = df_like
    if sp.issparse(arr): 
        arr = arr.toarray()
    if arr.dtype.kind != "f":
        arr = arr.astype(np.float64, copy=False)
    return arr

def get_shap_values_and_features(model, X_train, X_test, model_name):
    """Calculate SHAP values and return feature names and values (shap_comparison.py ile aynı)"""
    
    # Direkt model - pipeline yok
    X_train_full = to_numeric_array(X_train)
    X_test_full = to_numeric_array(X_test)
    feature_names = list(X_train.columns)
    regressor = model
    
    # Debug: Model tipini kontrol et
    print(f"Regressor tipi: {type(regressor)}")
    print(f"Regressor sınıfı: {regressor.__class__.__name__}")
    
    # Background sampling (shap_comparison.py ile aynı)
    rng = np.random.RandomState(RANDOM_STATE)
    n = X_train_full.shape[0]
    k = min(1000, n)
    sel = rng.choice(n, size=k, replace=False)
    X_bkg = X_train_full[sel]
    
    # Calculate SHAP values (shap_comparison.py ile aynı)
    try:
        print("SHAP TreeExplainer oluşturuluyor (background ile)...")
        explainer = shap.TreeExplainer(regressor, X_bkg, feature_perturbation="interventional")
        shap_values = explainer(X_test_full)
    except Exception as e:
        print(f"Background ile hata: {e}")
        try:
            print("SHAP TreeExplainer oluşturuluyor (background olmadan)...")
            explainer = shap.TreeExplainer(regressor, feature_perturbation="interventional")
            shap_values = explainer(X_test_full)
        except Exception as e2:
            print(f"Background olmadan da hata: {e2}")
            raise e2
    
    return shap_values, feature_names

# SHAP değerlerini hesapla
shap_values, feature_names = get_shap_values_and_features(model, X_train, X_test, metadata['best_name'])

print(f"SHAP hesaplama sonrası feature sayısı: {len(feature_names)}")
print(f"İlk 5 feature: {feature_names[:5]}")
print(f"Son 5 feature: {feature_names[-5:]}")

# --- Feature isimlerini temizle (shap_comparison.py ile aynı) ---
def clean_feature_names(feature_names):
    """Clean feature names - remove prefixes and shorten names (shap_comparison.py ile aynı)"""
    cleaned_names = []
    for name in feature_names:
        # Remove prefixes
        if name.startswith("cat__onehot__"):
            # Extract categorical feature name - remove cat__onehot__ prefix
            parts = name.split("__")
            if len(parts) >= 3:
                cat_name = parts[2]  # This is the actual feature name after cat__onehot__
                if "_" in cat_name:
                    base, level = cat_name.rsplit("_", 1)
                    cleaned_names.append(f"{base}={level}")
                else:
                    cleaned_names.append(cat_name)
            else:
                cleaned_names.append(name)
        elif name.startswith("cat__"):
            # Remove cat__ prefix (fallback)
            cleaned_names.append(name.split("__", 1)[-1])
        elif name.startswith("num__scaler__"):
            # Remove num__scaler__ prefix
            cleaned_names.append(name.split("__")[-1])
        elif name.startswith("num__"):
            # Remove num__ prefix (fallback)
            cleaned_names.append(name.split("__", 1)[-1])
        else:
            cleaned_names.append(name)
    
    # Shorten feature names for better readability
    shortened_names = []
    for name in cleaned_names:
        # Common abbreviations and shortenings
        name = name.replace("(mg/g)", "")
        name = name.replace("(g/g)", "")
        name = name.replace("(min)", "")
        name = name.replace("(K)", "")
        name = name.replace("(K/min)", "")
        name = name.replace("(m2/g)", "")
        name = name.replace("(cm3/g)", "")
        name = name.replace("(nm)", "")
        name = name.replace("(mg/L)", "")
        name = name.replace("(rpm)", "")
        name = name.replace("(L)", "")
        name = name.replace("Agent/Sample", "Agent/Sample")
        name = name.replace("Soaking_Time", "Soak_Time")
        name = name.replace("Soaking_Temp", "Soak_Temp")
        name = name.replace("Activation_Time", "Act_Time")
        name = name.replace("Activation_Temp", "Act_Temp")
        name = name.replace("Activation_Heating_Rate", "Act_Heat_Rate")
        name = name.replace("BET_Surface_Area", "BET_SA")
        name = name.replace("Total_Pore_Volume", "Total_PV")
        name = name.replace("Micropore_Volume", "Micro_PV")
        name = name.replace("Average_Pore_Diameter", "Avg_Pore_Dia")
        name = name.replace("Initial_Concentration", "Init_Conc")
        name = name.replace("Solution_pH", "pH")
        name = name.replace("Agitation_speed", "Agit_Speed")
        name = name.replace("Contact_Time", "Contact_Time")
        name = name.replace("Activation_Atmosphere", "Act_Atm")
        name = name.replace("activation_atmosphere", "Act_Atm")
        shortened_names.append(name)
    
    return shortened_names

# Feature isimlerini temizle
clean_feature_names_list = clean_feature_names(feature_names)

# --- SHAP importance hesapla ---
print(">>> SHAP importance değerleri hesaplanıyor...")

shap_abs = np.abs(shap_values.values if hasattr(shap_values, "values") else shap_values)
mean_abs = shap_abs.mean(axis=0)

imp_df = pd.DataFrame({
    "feature": clean_feature_names_list,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False)

print(f"\n=== {metadata['best_name']} SHAP (bireysel, |SHAP| ort.) — İlk 30 ===")
print(imp_df.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

# --- SHAP grafiği çiz ---
print(">>> SHAP importance grafiği oluşturuluyor...")

# Individual bar format (shap_comparison.py ile aynı)
topn = min(30, len(imp_df))
top_df = imp_df.head(topn).iloc[::-1]  # barh için ters sırala

plt.figure(figsize=(12, 10))
plt.barh(top_df["feature"], top_df["mean_abs_shap"], alpha=0.85, color="#0014E7")

plt.xlabel("Ortalama |SHAP| (TreeExplainer)")
plt.title(f"{metadata['best_name']} (Baseline Model) — Özellik Önemi (SHAP, kategorik seviyeler ayrı)")
plt.grid(True, alpha=0.3)

plt.tight_layout()

out_path = FIG_DIR / f"shap_importance_{metadata['best_name'].lower()}_baseline_final.png"
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"[OK] SHAP importance grafiği kaydedildi: {out_path}")
plt.show(); plt.close()

# --- Summary ---
print(f"\n=== {metadata['best_name']} Baseline Model SHAP Summary ===")
print(f"Model: {metadata['best_name']}")
print(f"Test R2: {r2:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"En önemli 5 özellik: {', '.join(imp_df.head(5)['feature'].tolist())}")

print(f"\n[OK] Baseline SHAP analysis completed!")