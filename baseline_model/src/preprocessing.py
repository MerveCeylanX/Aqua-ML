"""
preprocessing.py
----------------

Veriyi ML için hazırlar ve inference'ta tüm ön işlemleri Pipeline içine gömmek için
kullanılabilecek bir transformer (DomainFE) sağlar.

Sağlananlar:
- Hedef kolon kontrolü
- Sayısal/kategorik feature listeleri ve tip dönüşümleri
- Train/Test ayrımı
- OneHotEncoder (sklearn sürüm uyumluluğu) + ColumnTransformer (preprocessor)
- CatBoost için ham X'te kategorik kolon indeksleri (cat_idx_raw)
- DomainFE: add_pharm_features → clean_pharm_features → add_elemental_ratios zinciri + tip güvenliği

"""

from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from src.config import RANDOM_STATE, TEST_SIZE
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from src.features import add_pharm_features, clean_pharm_features, add_elemental_ratios


class DomainFE(BaseEstimator, TransformerMixin):
    """
    Tüm domain ön işlemlerini tek adımda uygular.
    - Kolon adlarını temizler (strip)
    - Farmasötik özellikleri ekler (E,S, A, B, V)
    - Element oranlarını ekler (C_molar, H/C, O/C, N/C, S/C)
    - Tip güvenliği: numerikler → float, kategorikler → category
    - Sadece istenen feature kolonlarını döndürür
    """
    def __init__(self, num_feats: List[str], cat_feats: List[str], target_col: str = "qe(mg/g)"):
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.target_col = target_col

    def fit(self, X, y=None):
        # Öğrenilecek parametre yok; sklearn uyumu için döndürülür.
        return self

    def transform(self, X):
        df = X.copy()
        df.columns = df.columns.astype(str).str.strip()

        # Domain FE zinciri
        df = add_pharm_features(df)
        df = clean_pharm_features(df)
        df = add_elemental_ratios(df)

        # Tip güvenliği
        for c in self.num_feats:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in self.cat_feats:
            if c in df.columns:
                df[c] = df[c].astype("category")

        # Sadece istenen kolonları döndür
        use_cols = [c for c in (self.num_feats + self.cat_feats) if c in df.columns]
        return df[use_cols]

def prepare_ml_data(
    df: pd.DataFrame,
    target_col: str = "qe(mg/g)"
) -> Dict[str, object]:
    """
    Verilen DataFrame'den ML için gerekli X/y, train/test bölünmeleri ve
    OHE'li ön-işleme pipeline'ını hazırlar.

    Dönüş:
        {
          "X_train", "X_test", "y_train", "y_test",
          "preprocessor", "num_feats", "cat_feats", "cat_idx_raw"
        }
    """
    # --- 0) Defansif kopya + kolon adlarını normalize et (strip) ---
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # 1) Hedef kolon kontrolü ---
    if target_col not in df.columns:
        raise KeyError(f"Hedef kolon yok: {target_col}")

    # 2) Feature listeleri (var olanları filtrele) ---
    num_feats_all = [
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
        "C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar",
        "Initial_Concentration(mg/L)",
        "Solution_pH",
        "Temperature(K)",
        "Agitation_speed(rpm)",
        "Dosage(g/L)",
        "Contact_Time(min)",
        "E", "S", "A", "B", "V",
    ]
    # Mevcut kolonları sırayı koruyarak seç
    num_feats = [c for c in num_feats_all if c in df.columns]

    # Bilgi amaçlı eksikler (opsiyonel)
    missing = [c for c in num_feats_all if c not in df.columns]
    if missing:
        print(f"[Bilgi] Veride olmayan num_feats: {missing}")

    # Kategorik feature: sadece 'Target_Phar'
    cat_feats: List[str] = ["Activation_Atmosphere"] if "Activation_Atmosphere" in df.columns else []

    # En az bir feature olmalı
    if not num_feats and not cat_feats:
        raise ValueError("Hiç bir özellik bulunamadı (num_feats ve cat_feats boş).")

    # 3) Tip güvenliği / dönüştürmeler ---
    for c in num_feats + [target_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in cat_feats:
        df[c] = df[c].astype("category")

    # 4) X/y, train/test ---
    X = df[num_feats + cat_feats].copy()
    y = df[target_col]

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )    

    # 5) OneHotEncoder (sklearn sürüm uyumluluğu) ---
    # sklearn>=1.2: sparse_output; daha eski sürümlerde: sparse
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # eski sklearn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_feats),
            ("cat", ohe, cat_feats),
        ],
        remainder="drop"
    )

    # 6) CatBoost için kategorik kolon indeksleri (ham X'te) ---
    cat_idx_raw = [X.columns.get_loc(c) for c in cat_feats]

    return {
        "X_train": X_train,
        "X_test": X_test,   
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "num_feats": num_feats,
        "cat_feats": cat_feats,
        "cat_idx_raw": cat_idx_raw,
    }
