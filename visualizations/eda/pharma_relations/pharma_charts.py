"""
pharma_charts.py
----------------

Bu kod, farmasötik–adsorbent–feedstock–referans ilişkilerini görselleştirir.  

Çıktılar:
- Görseller   : ./visualizations/eda/pharma_relations/figures/
- (Opsiyonel) Excel raporu: ./reports/   [burada eklenmedi, istersen export eklenebilir]

Üretilen görseller:
1. Total rows per pharmaceutical
2. Top N adsorbents per pharmaceutical
3. Top N feedstocks per pharmaceutical
4. Unique references per pharmaceutical
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0
TOP_N = 5
INCLUDE_UNKNOWN = True  # NaN'leri "UNKNOWN" olarak dahil etmek istersen True

COLS = {
    "pharma": "Target_Phar",
    "adsorbent": "Adsorbent_Name",
    "feedstock": "Source_Material",
    "ref": "Reference",
}

# === OUTPUT KLASÖRLERİ ===
figures_dir = Path("visualizations/eda/pharma_relations/figures")
figures_dir.mkdir(exist_ok=True, parents=True)

# === LOAD ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# Kolon kontrolü
missing_cols = [COLS[k] for k in COLS if COLS[k] not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# UNKNOWN dahil etme
if INCLUDE_UNKNOWN:
    for k in COLS.values():
        df[k] = df[k].fillna("UNKNOWN")

ph = COLS["pharma"]; ad = COLS["adsorbent"]; fs = COLS["feedstock"]; rf = COLS["ref"]

# -------------------------------
# 1) Toplam satır (ilaç başına)
# -------------------------------
totals = df.groupby(ph).size().sort_values(ascending=True)  # hbar için küçük->büyük
plt.figure(figsize=(10, max(4, 0.4 * len(totals))))
plt.barh(totals.index, totals.values, color="#001583")
plt.xlabel("Rows")
plt.title("Total Rows per Pharmaceutical")
plt.tight_layout()
plt.savefig(figures_dir / "01_total_rows_per_pharma.png", dpi=200)
plt.close()

# ------------------------------------------------
# 2) Her ilaç için en çok kullanılan ilk N adsorbent
# ------------------------------------------------
for pharma, sub in df.groupby(ph):
    top_ads = (
        sub.groupby(ad).size()
        .sort_values(ascending=False)
        .head(TOP_N)
        .sort_values(ascending=True)  # hbar için küçük->büyük
    )
    plt.figure(figsize=(8, max(3, 0.6 * len(top_ads))))
    plt.barh(top_ads.index, top_ads.values, color="red")
    plt.xlabel("Rows")
    plt.title(f"Top {TOP_N} Adsorbents — {pharma}")
    plt.tight_layout()
    fname = figures_dir / f"02_adsorbents_top{TOP_N}_{pharma}.png"
    # Dosya adında yasak karakter olursa temizle
    fname = figures_dir / str(fname.name).replace("/", "_").replace("\\", "_").replace(":", "_")
    plt.savefig(fname, dpi=200)
    plt.close()

# -----------------------------------------------
# 3) Her ilaç için en çok kullanılan ilk N feedstock
# -----------------------------------------------
for pharma, sub in df.groupby(ph):
    top_fs = (
        sub.groupby(fs).size()
        .sort_values(ascending=False)
        .head(TOP_N)
        .sort_values(ascending=True)
    )
    plt.figure(figsize=(8, max(3, 0.6 * len(top_fs))))
    plt.barh(top_fs.index, top_fs.values, color="red")
    plt.xlabel("Rows")
    plt.title(f"Top {TOP_N} Feedstocks — {pharma}")
    plt.tight_layout()
    fname = figures_dir / f"03_feedstocks_top{TOP_N}_{pharma}.png"
    fname = figures_dir / str(fname.name).replace("/", "_").replace("\\", "_").replace(":", "_")
    plt.savefig(fname, dpi=200)
    plt.close()

# ---------------------------------------------------
# 4) Benzersiz Reference sayısı (ilaç başına)
# ---------------------------------------------------
ref_counts = (
    df.groupby(ph)[rf].nunique(dropna=not INCLUDE_UNKNOWN)
    .sort_values(ascending=True)
)
plt.figure(figsize=(10, max(4, 0.4 * len(ref_counts))))
plt.barh(ref_counts.index, ref_counts.values, color="red")
plt.xlabel("Unique References")
plt.title("Unique References per Pharmaceutical")
plt.tight_layout()
plt.savefig(figures_dir / "04_unique_references_per_pharma.png", dpi=200)
plt.close()

print(f"Done. Charts saved under: {figures_dir.resolve()}")
