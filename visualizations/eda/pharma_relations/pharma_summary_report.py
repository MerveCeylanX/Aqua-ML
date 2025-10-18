"""
pharma_summary_report.py
------------------------

Bu kod, veri setindeki farmasötik–adsorbent–feedstock–referans ilişkilerini
özetler ve hem tablo (Excel) hem de görselleştirme çıktıları üretir.

Çıktılar:
- Excel raporu: ./reports/pharma_summary_report.xlsx
- Görseller   : ./visualizations/eda/pharma_relations/figures/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === AYARLAR ===
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0
OUT_XLSX = "pharma_summary_report.xlsx"
INCLUDE_UNKNOWN = True

COLS = {
    "pharma": "Target_Phar",
    "adsorbent": "Adsorbent_Name",
    "feedstock": "Source_Material",
    "ref": "Reference",
}

# === VERİYİ YÜKLE ===
df = pd.read_excel(PATH, sheet_name=SHEET)
missing_cols = [COLS[k] for k in COLS if COLS[k] not in df.columns]
if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

if INCLUDE_UNKNOWN:
    for k in COLS.values():
        df[k] = df[k].fillna("UNKNOWN")

ph = COLS["pharma"]; ad = COLS["adsorbent"]; fs = COLS["feedstock"]; rf = COLS["ref"]

# === 1) SUMMARY ===
summary = pd.DataFrame({
    "Total_Rows": df.groupby(ph).size(),
    "Unique_Adsorbents": df.groupby(ph)[ad].nunique(dropna=not INCLUDE_UNKNOWN),
    "Unique_Feedstocks": df.groupby(ph)[fs].nunique(dropna=not INCLUDE_UNKNOWN),
}).reset_index().sort_values(["Total_Rows", ph], ascending=[False, True])

# === 2) ADSORBENT BREAKDOWN ===
adsorbent_breakdown = (
    df.groupby([ph, ad]).size().reset_index(name="Rows")
      .sort_values([ph, "Rows"], ascending=[True, False])
)

# === 3) FEEDSTOCK BREAKDOWN ===
feedstock_breakdown = (
    df.groupby([ph, fs]).size().reset_index(name="Rows")
      .sort_values([ph, "Rows"], ascending=[True, False])
)

# === 4) REFERENCE MAP ===
N = 10
ref_map = (
    df.groupby(ph)[rf]
      .agg(Unique_References=lambda s: s.dropna().nunique(),
           References_Sample=lambda s: "; ".join(pd.unique(s.dropna())[:N]))
      .reset_index()
      .sort_values(["Unique_References", ph], ascending=[False, True])
)

# === OUTPUT KLASÖRLERİ ===
# raporlar → proje kökünde reports/
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

# görseller → visualizations/eda/pharma_relations/figures/
figures_dir = Path("visualizations/eda/pharma_relations/figures")
figures_dir.mkdir(exist_ok=True, parents=True)

# === EXCEL'E YAZ ===
excel_path = reports_dir / OUT_XLSX
with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    summary.to_excel(writer, sheet_name="00_Summary", index=False)
    adsorbent_breakdown.to_excel(writer, sheet_name="01_Adsorbents", index=False)
    feedstock_breakdown.to_excel(writer, sheet_name="02_Feedstocks", index=False)
    ref_map.to_excel(writer, sheet_name="03_References", index=False)

print("Excel kaydedildi:", excel_path.resolve())

# === GÖRSELLER ===

# 1. İlaç bazında özet
fig, ax = plt.subplots(figsize=(10, 6))
summary.set_index(ph)[["Total_Rows","Unique_Adsorbents","Unique_Feedstocks"]].plot(
    kind="bar", ax=ax, edgecolor="black"
)
plt.title("İlaç Bazında Özet")
plt.ylabel("Sayı")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(figures_dir / "summary_per_drug.png", dpi=400)
plt.close(fig)

# 2. İlaç × Adsorbent heatmap
pivot_ads = adsorbent_breakdown.pivot(index=ph, columns=ad, values="Rows").fillna(0)
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_ads, cmap="Reds", cbar_kws={"label": "Satır Sayısı"})
plt.title("İlaç × Adsorbent Frekansı")
plt.ylabel("İlaçlar")
plt.xlabel("Adsorbentler")
plt.tight_layout()
plt.savefig(figures_dir / "heatmap_pharma_adsorbents.png", dpi=400)
plt.close()

# 3. Referans sayısı
fig, ax = plt.subplots(figsize=(10, 6))
ref_map.plot(x=ph, y="Unique_References", kind="bar", ax=ax, color="teal", edgecolor="black")
plt.title("İlaç Başına Benzersiz Referans Sayısı")
plt.ylabel("Referans Sayısı")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
fig.savefig(figures_dir / "references_per_drug.png", dpi=400)
plt.close(fig)

print("Görseller kaydedildi:", figures_dir.resolve())
