"""
impute_and_enrich.py
-------------------

Bu dosya, ham verideki eksik gözlemleri tamamlamak ve ek özellikler kazandırmak için kullanılmaktadır.

- Ham veri yüklenir (varsa güncel imputed dosya, yoksa ham dosya).
- Kritik kolonlarda (ör. gözenek hacmi, pHpzc) eksik veri oranı raporlanır.
- Activation_Heating_Rate (K/min): Atmosfer ve sıcaklığa göre doldurulur.
- Activation_Time (min): Medyan değer ile doldurulur.
- Average_Pore_Diameter (nm): 4000·V / A formülü ile hesaplanır.
- pHpzc: Aktivasyon ajanına göre literatür değerleri atanır (+0.3 düzeltme ≥800 K).
- Elementel oranlar (H/C, O/C, N/C, S/C): wt% → molar oranlara dönüştürülür.
- Solution pH: Eksikler 7.0 ile doldurulur.
- Contact Time (min): Eksikler 1440 dk (24 saat) ile doldurulur.
- Agitation Speed (rpm): Önce grup medyanı, yoksa global medyan ile doldurulur.
- Farmasötik özellikler (E, S, A, B, V): Kodlara göre dış veri tabanından eklenir.
- Güncellenmiş veri, ./data/Raw_data_imputed.xlsx dosyasına kaydedilir.

"""
import os
import pandas as pd
from pathlib import Path

# --- Setup ---
# Script'in bulunduğu klasörün yolu
BASE_DIR = Path(__file__).resolve().parent

# Data klasörü oluştur (yoksa)
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dosya yolları
IN_PATH      = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET    = 0
OUT_PATH = DATA_DIR / "Raw_data_imputed.xlsx"

# --- Veri yükleme ---
# OUT_PATH varsa güncel dosya yüklenir,
# yoksa ham dosyadan okunur.
if os.path.exists(OUT_PATH):
    df = pd.read_excel(OUT_PATH)
    print(f"Yüklendi (güncel): {OUT_PATH}")
else:
    df = pd.read_excel(IN_PATH, sheet_name=SHEET)
    print(f"Yüklendi (ilk): {IN_PATH}")

# --- Eksik veri kontrolü ---
# Belirli kritik kolonlarda boşluk oranı raporlanır.
cols = [
    "Total_Pore_Volume(cm3/g)",
    "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)",
    "Yield(%)",
    "Activation_Heating_Rate (K/min)",
    "pHpzc"
]

# --- Eksik veri oranı raporu ---
for c in cols:
    if c in df.columns:
        ratio = df[c].isna().mean() * 100
        print(f"{c}: %{ratio:.2f} boş veri")
    else:
        print(f"{c}: kolon bulunamadı")    

# --- Güvenlik: sıcaklığı sayısala çevir ---
if "Activation_Temp(K)" in df.columns:
    df["Activation_Temp(K)"] = pd.to_numeric(df["Activation_Temp(K)"], errors="coerce")

# --- Activation_Heating_Rate (K/min) doldurma ---
# Bu kolon kritik, yoksa hata ver.
col = "Activation_Heating_Rate (K/min)"
if col not in df.columns:
    raise KeyError(f"Beklenen kolon yok: {col}")

# Imputation fonksiyonu
def impute_heating_rate(row):
    val = row[col]
    if pd.notna(val):
        return val

    atm = row.get("Activation_Atmosphere", None)
    atm = (str(atm).strip().lower()) if pd.notna(atm) else ""

    temp = row.get("Activation_Temp(K)", None)
    # temp NaN olabilir; kıyas yapmadan önce kontrol et

    if "n2" in atm:  # inert
        return 10
    elif ("air" in atm) or ("sg" in atm):  # oxidative: air veya self-generated (sg)
        # temp NaN ise eşik değerlendir ve 10 veya 15 döndür
        if pd.isna(temp):
            return 10
        return 10 if temp < 773 else 15
    else:
        # Belirsiz atmosfer: dokunma
        return val

# --- Sadece boş olanları doldur ---
missing_before = df[col].isna().sum()
df[col] = df.apply(impute_heating_rate, axis=1)
missing_after = df[col].isna().sum()
filled = missing_before - missing_after

# --- Raporla ---
print(f"[{col}] Boş (önce): {missing_before} | Doldurulan: {filled} | Boş (sonra): {missing_after}")

# --- Activation_Time(min) doldurma (medyan ile) ---
# Bu kolon kritik değil, varsa doldur.
if "Activation_Time(min)" in df.columns:
    before = df["Activation_Time(min)"].isna().sum()
    med = df["Activation_Time(min)"].median()
    df["Activation_Time(min)"] = df["Activation_Time(min)"].fillna(med)
    after = df["Activation_Time(min)"].isna().sum()
    print(f"[Activation_Time(min)] Boş (önce): {before} | Medyan ile doldurulan: {before - after} | Boş (sonra): {after}")

print(f"[{col}] Boş (önce): {missing_before} | Doldurulan: {filled} | Boş (sonra): {missing_after}")


# --- Average pore diameter doldur (D[nm] = 4000 * V(cm3/g) / BET(m2/g)) ---
apd = "Average_Pore_Diameter(nm)"
V   = "Total_Pore_Volume(cm3/g)"
A   = "BET_Surface_Area(m2/g)"

if all(c in df.columns for c in [apd, V, A]):
    before = df[apd].isna().sum()
    m = df[apd].isna() & df[V].notna() & df[A].notna() & (pd.to_numeric(df[A], errors="coerce") > 0)
    df.loc[m, apd] = 4000.0 * pd.to_numeric(df.loc[m, V], errors="coerce") / pd.to_numeric(df.loc[m, A], errors="coerce")
    after = df[apd].isna().sum()
    print(f"[{apd}] Boş (önce): {before} | 4V/BET ile doldurulan: {before - after} | Boş (sonra): {after}")
else:
    print("Average_Pore_Diameter doldurma atlandı (gerekli kolon(lar) eksik).")

# --- pHPZC imputasyonu: activation_agent → ajan önceliği (+T düzeltmesi) ---
pzc_col   = "pHpzc"                      
agent_col = "Activation_Agent"
temp_col  = "Activation_Temp(K)"         # varsa +0.3 düzeltme

if agent_col in df.columns:
    # normalize agent (trim + lowercase)
    agent_norm = df[agent_col].astype(str).str.strip().str.lower()

    # ajan → öncel pHPZC (cmn. öneri)
    agent_prior = {
        "koh":                 8.5,
        "koh and ca(oh)2":     8.8,   
        "h3po4":               4.5,
        "h2so4":               3.5,
        "zncl2":               6.5,
        "k2co3":               8.3,
        "k2c2o4":              6.7,
        "(nh4)2hpo4":          5.0,
    }

    # prior değerlerini çıkar
    prior = agent_norm.map(agent_prior)

    # sıcaklık düzeltmesi (>=800 K → +0.3), yoksa dokunma
    T = pd.to_numeric(df[temp_col], errors="coerce") if temp_col in df.columns else pd.Series(pd.NA, index=df.index)
    prior_adj = prior.where(~(T >= 800), prior + 0.3)
    prior_adj = prior_adj.clip(lower=1, upper=12)

    # sadece boş olanları doldur
    before = df[pzc_col].isna().sum()
    mask_fill = df[pzc_col].isna() & prior_adj.notna()
    df.loc[mask_fill, pzc_col] = prior_adj[mask_fill]

    # flag + kural etiketi
    rule = ["agent_prior+T" if (pd.notna(T.iloc[i]) and T.iloc[i] >= 800) else "agent_prior" for i in df.index]
    df.loc[mask_fill, "pHPZC_imputed_flag"] = 1
    df.loc[~mask_fill, "pHPZC_imputed_flag"] = df.get("pHPZC_imputed_flag", 0)
    df["pHPZC_imputed_flag"] = df["pHPZC_imputed_flag"].fillna(0).astype(int)
    df.loc[mask_fill, "pHPZC_impute_rule"] = pd.Series(rule, index=df.index)[mask_fill]

    after  = df[pzc_col].isna().sum()
    filled = before - after
    print(f"[{pzc_col}] Boş (önce): {before} | agent-prior ile doldurulan: {filled} | Boş (sonra): {after}")
else:
    print("Kolon bulunamadı:", agent_col)

# --- Element oranları (wt% → molar: H/C, O/C, N/C, S/C) ---
C, O, H, N, S = "C_percent", "O_percent", "H_percent", "N_percent", "S_percent"
aw = {"C": 12.011, "H": 1.008, "O": 15.999, "N": 14.007, "S": 32.06}

# sayısala çevir
for c in [C, O, H, N, S]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    else:
        print(f"{c} kolon bulunamadı")

# eksikleri ve SIFIRLARI küçük epsilon ile doldur (wt%)
EPS_WTPCT = 0.001  # 0.001 wt%
for c in [O, H, N, S]:           # C'ye dokunma
    if c in df.columns:
        n_na   = df[c].isna().sum()
        n_zero = (df[c] == 0).sum()
        if n_na or n_zero:
            df.loc[df[c].isna() | (df[c] == 0), c] = EPS_WTPCT
            print(f"[{c}] NaN: {n_na}, 0: {n_zero} → {EPS_WTPCT} ile değiştirildi")

# C>0 olanlarda molar oranları hesapla
if C in df.columns:
    maskC = df[C].notna() & (df[C] > 0)
    badC = (~maskC).sum()
    if badC:
        print(f"[Uyarı] {badC} satırda {C} yok veya ≤0; molar oran hesaplanmadı.")

    denom = df.loc[maskC, C] / aw["C"]    # mol C / 100 g (normalize)
    df.loc[maskC, "C_molar"] = denom      # yeni kolon eklendi
    df.loc[maskC, "H_C_molar"] = (df.loc[maskC, H] / aw["H"]) / denom
    df.loc[maskC, "O_C_molar"] = (df.loc[maskC, O] / aw["O"]) / denom
    df.loc[maskC, "N_C_molar"] = (df.loc[maskC, N] / aw["N"]) / denom
    df.loc[maskC, "S_C_molar"] = (df.loc[maskC, S] / aw["S"]) / denom

    # güvenlik: negatifleri 0'a kırp
    for r in ["C_molar","H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar"]:
        df[r] = df[r].clip(lower=0)

# --- Solution_pH imputation: boşları 7.0 ile doldur + flag ---
ph_col = "Solution_pH"
if ph_col in df.columns:
    df[ph_col] = pd.to_numeric(df[ph_col], errors="coerce")
    m = df[ph_col].isna()
    n = int(m.sum())
    if n:
        df.loc[m, ph_col] = 7.0
    df["Solution_pH_imputed_flag"] = 0
    df.loc[m, "Solution_pH_imputed_flag"] = 1
    print(f"[{ph_col}] NaN (önce): {n} | 7.0 ile doldurulan: {n} | NaN (sonra): {int(df[ph_col].isna().sum())}")
else:
    print(f"{ph_col}: kolon bulunamadı")

# --- Contact_Time(min) imputation: boşları 1440 ile doldur + flag ---
ct_col = "Contact_Time(min)"
if ct_col in df.columns:
    df[ct_col] = pd.to_numeric(df[ct_col], errors="coerce")
    m = df[ct_col].isna()
    n = int(m.sum())
    if n:
        df.loc[m, ct_col] = 1440.0
    df["Contact_Time_imputed_flag"] = 0
    df.loc[m, "Contact_Time_imputed_flag"] = 1
    print(f"[{ct_col}] NaN (önce): {n} | 1440 ile doldurulan: {n} | NaN (sonra): {int(df[ct_col].isna().sum())}")
else:
    print(f"{ct_col}: kolon bulunamadı")

# --- Agitation_speed(rpm) imputation: grup (Target_Phar) medyanı → yoksa global medyan + flag/etiket ---
COL = "Agitation_speed(rpm)"
PHAR_candidates = ["Target_Phar", "target_pH", "target_ph", "Solution_pH"]  # elindeki isme göre ilk bulunan kullanılır
PHAR = next((c for c in PHAR_candidates if c in df.columns), None)

if COL in df.columns:
    df[COL] = pd.to_numeric(df[COL], errors="coerce")
    df["Agitation_missing"] = df[COL].isna().astype(int)

    if PHAR is not None:
        # grup ve global medyan
        phar_median = df.groupby(PHAR, dropna=False)[COL].median()
        global_median = df[COL].median()

        def _impute(row):
            if pd.notna(row[COL]):
                return row[COL], None
            key = row[PHAR]
            if (key in phar_median.index) and pd.notna(phar_median.loc[key]):
                return phar_median.loc[key], "median_by_Target_Phar"
            return global_median, "median_global"

        imputed = df.apply(_impute, axis=1, result_type="expand")
        df[COL] = imputed[0]
        df["Agitation_impute_method"] = imputed[1]
    else:
        # sadece global medyan
        global_median = df[COL].median()
        m = df[COL].isna()
        df["Agitation_impute_method"] = None
        df.loc[m, "Agitation_impute_method"] = "median_global"
        df.loc[m, COL] = global_median

    # özet
    n_by_group = int((df.get("Agitation_impute_method") == "median_by_Target_Phar").sum()) if "Agitation_impute_method" in df.columns else 0
    n_by_glob  = int((df.get("Agitation_impute_method") == "median_global").sum()) if "Agitation_impute_method" in df.columns else 0
    print(f"[{COL}] grup-medyan ile: {n_by_group} | global-medyan ile: {n_by_glob} | toplam doldurulan: {n_by_group + n_by_glob}")
else:
    print(f"{COL}: kolon bulunamadı")


# --- Farmasötik özellikleri (E, S, A, B, V) ekle: kısaltmadan eşleştir ---
code_candidates = ["Pharmaceutical_code", "Pharm", "pharm", "Pharmaceutical", "Target_Phar"]
code_col = next((c for c in code_candidates if c in df.columns), None)
if code_col is None:
    raise KeyError("İlaç kodu kolonu bulunamadı. Aday isimler: " + ", ".join(code_candidates))

# Farmasötik özellikler tablosu (kısaltma, E, S, A, B, V)
# Veriler UFZ LSER-Database'den alınmıştır.
# https://web.app.ufz.de/compbc/lserd/public/start/#searchresult

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

# DataFrame oluştur
pharm_df = pd.DataFrame(pharm_data, columns=["Pharmaceutical_code","E","S","A","B","V"])
pharm_df["pharm_code_norm"] = pharm_df["Pharmaceutical_code"].str.strip().str.upper()

# df'deki kodları normalize et  
df["pharm_code_norm"] = df[code_col].astype(str).str.strip().str.upper()

# E, S, A, B, V kolonlarını ekle (varsa NaN'leri doldur)
map_cols = ["E","S","A","B","V"]
tmp_map = {c: f"__map_{c}" for c in map_cols}
pharm_merge = pharm_df[["pharm_code_norm"] + map_cols].rename(columns=tmp_map)

# Merge (soldaki df'yi bozmadan, farm koduna göre)
df = df.merge(pharm_merge, on="pharm_code_norm", how="left")

# Var olan kolonları ezmeden doldur (yoksa oluştur, varsa sadece NaN'leri doldur)
for c in map_cols:
    tmpc = f"__map_{c}"
    if c in df.columns:
        df[c] = df[c].fillna(df[tmpc])
    else:
        df[c] = df[tmpc]
    df.drop(columns=[tmpc], inplace=True)

# Temizlik: yardımcı kolon
df.drop(columns=["pharm_code_norm"], inplace=True)

# Raporla
matched = df[map_cols].notna().all(axis=1).sum()
print(f"[PHARM özellikleri] Eklendi. Tam eşleşen satır sayısı: {matched}, eşleşmeyen (NaN kalan) örnek sayısı: {len(df)-matched}")

# --- Kaydet ---
df.to_excel(OUT_PATH, index=False)
print(f"Kaydedildi → {OUT_PATH}")


cols = ["C_percent", "O_percent", "H_percent", "N_percent", "S_percent"]

for c in cols:
    if c in df.columns:
        s = df[c].replace(r"^\s*$", pd.NA, regex=True)  # boş/whitespace'i NaN say
        n_missing = s.isna().sum()
        pct_missing = 100 * n_missing / len(s)
        print(f"{c}: %{pct_missing:.2f} boş veri (n={n_missing})")
    else:
        print(f"{c}: kolon bulunamadı")

