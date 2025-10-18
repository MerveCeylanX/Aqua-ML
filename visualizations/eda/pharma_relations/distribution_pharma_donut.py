"""
distribution_pharma_donut.py
----------------------------

Bu kod, veri setindeki farmasötiklerin (Target_Phar) dağılımını
daire/donut grafik ile görselleştirir.

Çıktılar:
- Görsel: ./visualizations/eda/pharma_relations/figures/05_distribution_pharma_donut.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === AYARLAR ===
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0

# Çıktı klasörü (figures)
OUT_DIR = Path("visualizations/eda/pharma_relations/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FIG = OUT_DIR / "distribution_pharma_donut.png"

COLS = {
    "pharma": "Target_Phar",
}

# === VERİYİ YÜKLE ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# Eksik değerleri UNKNOWN olarak ekle
df[COLS["pharma"]] = df[COLS["pharma"]].fillna("UNKNOWN")

# Yüzde dağılımı
dist = df[COLS["pharma"]].value_counts(normalize=True) * 100

# === DONUT CHART ===
plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(
    dist.values,
    labels=dist.index,
    autopct="%1.1f%%",
    startangle=140,
    pctdistance=0.85,     # yüzde yazılarının konumu
    colors=plt.cm.tab20.colors
)

# Donut görünümü için ortayı boşalt
centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Yazı stilini güzelleştir
for t in autotexts:
    t.set_color("black")
    t.set_fontsize(10)
for t in texts:
    t.set_fontsize(9)

plt.title("Farmasötiklerin Veri Setindeki Dağılımı (%)", fontsize=14, weight="bold")
plt.tight_layout()

# Kaydet
plt.savefig(OUT_FIG, dpi=200)
plt.show()

print("Görsel kaydedildi:", OUT_FIG.resolve())
