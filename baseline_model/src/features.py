"""
features.py
------------

Özellik mühendisliği (feature engineering) fonksiyonları.
- Farmasötik (E, S, A, B, V) özelliklerini ekler
- Merge sonrası geçici kolonları temizler
- Element yüzdelerinden C_molar ve H/C, O/C, N/C, S/C molar oranlarını hesaplar

"""

import pandas as pd
import numpy as np

# --- 1) Referans farmasötik özellikler tablosu (E, S, A, B, V) ---
# Not: Bu tablo sabit olduğundan, fonksiyon dışında tek sefer tanımlanır.
# Veriler UFZ LSER-Database'den alınmıştır.
# https://web.app.ufz.de/compbc/lserd/public/start/#searchresult

_pharm_data = [
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
_pharm_df = pd.DataFrame(_pharm_data, columns=["Pharmaceutical_code","E","S","A","B","V"])
_pharm_df["pharm_code_norm"] = _pharm_df["Pharmaceutical_code"].str.strip().str.upper()


def add_pharm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Target_Phar kolonuna göre E, S, A, B, V farmasötik özelliklerini ana tabloya ekler.

    Adımlar:
    - Target_Phar'ı normalize et (boşlukları sil, büyük harf)
    - Referans tablo (_pharm_df) ile 'pharm_code_norm' üzerinden left-merge yap
    - Çakışmayı önlemek için map kolon adlarını geçici '__map_*' ile taşı
    """
    df = df.copy()
    if "Target_Phar" not in df.columns:
        raise KeyError("Target_Phar kolonu bulunamadı.")

    # İlaç kodunu normalize et → eşleşme hatalarını önler
    df["pharm_code_norm"] = df["Target_Phar"].astype(str).str.strip().str.upper()

    map_cols = ["E", "S", "A", "B", "V"]
    tmp_map = {c: f"__map_{c}" for c in map_cols}

    # Sadece gerekli kolonları seç, geçici isimlerle hazırla
    pharm_merge = _pharm_df[["pharm_code_norm"] + map_cols].rename(columns=tmp_map)

    # Left-merge: df'yi koru, eşleşenlere özellikleri ekle
    df = df.merge(pharm_merge, on="pharm_code_norm", how="left")

    return df


def clean_pharm_features(df: pd.DataFrame, map_cols=["E","S","A","B","V"]) -> pd.DataFrame:
    """
    add_pharm_features sonrası oluşan geçici '__map_*' kolonlarını
    asıl kolonlara taşır; 'pharm_code_norm' ve '__map_*' kolonlarını temizler.
    """
    df = df.copy()

    for c in map_cols:
        tmpc = f"__map_{c}"
        if tmpc not in df.columns:
            # merge gerçekleşmemişse sessizce devam et
            continue

        if c in df.columns:
            # Var olan (muhtemelen kirli tip/missing) kolonu sayısala çevir,
            # eksikleri geçici kolonla doldur
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(df[tmpc])
        else:
            # Asıl kolon yoksa doğrudan geçiciden oluştur
            df[c] = df[tmpc]

        # Geçici kolonu temizle
        df.drop(columns=[tmpc], inplace=True)

    # Merge sırasında eklenmiş yardımcı kolon
    df.drop(columns=["pharm_code_norm"], inplace=True, errors="ignore")

    # Tür güvenliği: E,S,A,B,V kesin sayısal olsun
    for c in map_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_elemental_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Element yüzdelerinden (wt%) H/C, O/C, N/C, S/C molar oranlarını hesaplar.
    Gereken kolonlar: 'C_percent', 'O_percent', 'H_percent', 'N_percent', 'S_percent'

    Not:
    - C_percent <= 0 veya NaN ise ilgili satır için oranlar hesaplanmaz (NaN kalır).
    - Negatif değerler güvenlik için 0'a kırpılır.
    """
    df = df.copy()

    # Girdi kolon adları ve atomik ağırlıklar (g/mol)
    C, O, H, N, S = "C_percent", "O_percent", "H_percent", "N_percent", "S_percent"
    aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}

    # Tip temizliği / uyarılar
    for col in [C, O, H, N, S]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"[Uyarı] {col} kolonu bulunamadı; ilgili molar oranlar NaN kalabilir.")

    if C in df.columns:
        # Hesaplama sadece C > 0 olan satırlarda yapılır
        maskC = df[C].notna() & (df[C] > 0)
        badC = (~maskC).sum()
        if badC:
            print(f"[Not] {badC} satırda {C} yok veya ≤0; H/C, O/C, N/C, S/C hesaplanmadı (NaN).")

        # Karbon yüzdesinden mol sayısı hesapla (C_molar); diğer oranlar bunun üzerinden normalize edilir
        denom = df.loc[maskC, C] / aw["C"]
        df.loc[maskC, "C_molar"] = denom

        if H in df.columns:
            df.loc[maskC, "H_C_molar"] = (df.loc[maskC, H] / aw["H"]) / denom
        if O in df.columns:
            df.loc[maskC, "O_C_molar"] = (df.loc[maskC, O] / aw["O"]) / denom
        if N in df.columns:
            df.loc[maskC, "N_C_molar"] = (df.loc[maskC, N] / aw["N"]) / denom
        if S in df.columns:
            df.loc[maskC, "S_C_molar"] = (df.loc[maskC, S] / aw["S"]) / denom

        # Güvenlik: Negatifleri NaN yap
        for r in ["C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar"]:
            if r in df.columns:
                df[r] = pd.to_numeric(df[r], errors="coerce")
                df.loc[df[r] < 0, r] = 0

    return df
