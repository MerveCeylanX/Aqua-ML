"""
missing_data_summary.py
-----------------------

Bu kod, veri setindeki eksik değerleri özetler ve görselleştirir.
- Kolon bazında eksik veri sayısı ve yüzdesi hesaplanır.
- Tüm verisetindeki genel eksik oran hesaplanır.
- Sonuçlar tablo (Excel) ve iki grafik olarak kaydedilir.

Çıktılar:
- Excel : ./figures/missing_data_summary.xlsx
- Grafik1 (kolon bazında): ./figures/missing_percent_per_column.png
- Grafik2 (genel oran)   : ./figures/missing_percent_overall.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# === AYARLAR ===
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0

# === VERİYİ YÜKLE ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# === HARİÇ TUTULACAK KOLONLAR ===
exclude_cols = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name",
    "Activation_Method", "%sum", "Target_Phar", "Reference"
]

# === EKSİK ÖZETİ (kolon bazında) ===
missing_counts  = df.isnull().sum()
missing_percent = (missing_counts / len(df)) * 100

missing_summary = (
    pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_Percent": missing_percent
    })
    .drop(exclude_cols, errors="ignore")             # hariç tutulan kolonları çıkar
    .sort_values("Missing_Percent", ascending=False) # çoktan aza sırala
)

print("Veri boyutu:", df.shape)
print(missing_summary)

# === ÇIKTI KLASÖRÜ ===
current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
save_dir = current_dir / "figures"
save_dir.mkdir(parents=True, exist_ok=True)

# === TABLOYU KAYDET ===
excel_path = save_dir / "missing_data_summary.xlsx"
missing_summary.to_excel(excel_path)
print(f"Excel kaydedildi: {excel_path}")

# === GRAFİK 1: Kolon bazında eksik % (barh, her bar farklı renk) ===
fig1 = plt.figure(figsize=(10, max(4, 0.25 * len(missing_summary))))
ax1 = fig1.add_subplot(111)

# Her bar için farklı renk üret (N renk)
N = len(missing_summary)
# 0..1 arası eşit aralıklı N nokta al, tab20 paletinden renkleri çek
colors = plt.cm.tab20(np.linspace(0, 1, max(2, min(20, N))))  # tab20 en fazla 20 farklı ton
# Eğer N>20 ise renkleri döngüsel kullan
bar_colors = [colors[i % len(colors)] for i in range(N)]

missing_summary["Missing_Percent"].plot(
    kind="barh",
    color=bar_colors,
    edgecolor="black",
    ax=ax1
)

ax1.set_title("Kolon Bazında Eksik Veri Oranı (%)")
ax1.set_xlabel("Eksik Veri (%)")
ax1.set_ylabel("Kolonlar")
ax1.invert_yaxis()  # en çok eksik olan yukarıda
fig1.tight_layout()

fig1_path = save_dir / "missing_percent_per_column.png"
fig1.savefig(fig1_path, bbox_inches="tight", dpi=500)
plt.show()
print(f"Grafik kaydedildi: {fig1_path}")

# === GRAFİK 2: Tüm verisetinde genel eksik oran (%) ===
overall_missing_percent = (
    df.drop(exclude_cols, axis=1, errors="ignore")
      .isnull().sum().sum()
    / (df.shape[0] * (df.shape[1] - len(exclude_cols)))
) * 100

fig2 = plt.figure(figsize=(4, 4))
ax2 = fig2.add_subplot(111)
ax2.bar(["Genel"], [overall_missing_percent], color="blue", edgecolor="black")
ax2.set_ylim(0, 100)  # y-ekseni 0–100 sabit
ax2.set_ylabel("Eksik Veri (%)")
ax2.set_title("Tüm Verisetinde Genel Eksik Veri Oranı (%)")
ax2.text(0, overall_missing_percent, f"{overall_missing_percent:.2f}%",
         ha="center", va="bottom")
fig2.tight_layout()

fig2_path = save_dir / "missing_percent_overall.png"
fig2.savefig(fig2_path, bbox_inches="tight", dpi=500)
plt.show()
print(f"Grafik kaydedildi: {fig2_path}")
