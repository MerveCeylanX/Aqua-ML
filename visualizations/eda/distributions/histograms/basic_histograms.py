"""
basic_histograms.py
-------------------
Bu dosya Excel verisindeki sayısal kolonlar için temel istatistiksel grafikler üretir:
1) Tüm sayısal kolonların histogram + KDE grafikleri tek bir figür halinde (all_histograms.png)
2) 2x2 grid halinde gruplar (her grup 4 özellik için ayrı bir PNG dosyası)
3) BET yüzey alanı (m2/g) için ECDF grafiği (ecdf_bet.png)

Tüm görseller script ile aynı dizinde oluşturulan 'figures/' klasörüne kaydedilir.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET = 0

# Çıktı klasörü
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hariç tutulan kolonlar
exclude_cols = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name",
    "Activation_Method", "%sum", "Target_Phar", "Reference",
    "qe (mg/g)", "% Removal"
]

# === LOAD ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# Sayısal kolonları seç ve hariç tut
num_cols = [c for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]

if not num_cols:
    raise ValueError("DataFrame içinde uygun sayısal kolon bulunamadı.")

# === RENK PALETİ ===
colors = ["#FECB04", "#0014E7",  "#03C2FB","#1E5AF9","#001583"]  # koyu kırmızı, koyu mavi, pembe, yeşil

# === 1) TEK BÜYÜK GRAFİK (tüm histogramlar) ===
ncols = 5
nrows = int(np.ceil(len(num_cols) / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    ax = axes[i]
    color = colors[i % len(colors)]
    
    sns.histplot(
        df[col].dropna(),
        bins=30,
        kde=True,
        stat="density",
        edgecolor="black",
        color=color,
        ax=ax,
        alpha=0.6
    )
    ax.set_title(col, fontsize=11, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.subplots_adjust(hspace=0.6, wspace=0.5)
fig.suptitle("Histogram + KDE for Numerical Features", fontsize=18, weight="bold", y=1.02)
plt.savefig(OUT_DIR / "all_histograms.png", dpi=300, bbox_inches="tight")
plt.close()
print("All histograms saved to:", OUT_DIR / "all_histograms.png")

# === 2) GRUP GRUP (2x2 grid = 4 grafik) ===
group_size = 4
for g in range(0, len(num_cols), group_size):
    cols_group = num_cols[g:g+group_size]
    n = len(cols_group)
    
    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_group):
        ax = axes[i]
        color = colors[(g+i) % len(colors)]
        
        sns.histplot(
            df[col].dropna(),
            bins=30,
            kde=True,
            stat="density",
            edgecolor="black",
            color=color,
            ax=ax,
            alpha=0.6
        )
        ax.set_title(col, fontsize=11, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
    
    # Boş eksenleri sil
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    fname = OUT_DIR / f"hist_group_{g//group_size+1}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print("Group saved to:", fname)

# === 3) ÖRNEK ECDF ===
if "BET_Surface_Area(m2/g)" in df.columns:
    data = df['BET_Surface_Area(m2/g)'].dropna()
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)

    plt.figure(figsize=(8,5))
    plt.plot(x, y, marker='.', linestyle='none', color="#1f77b4")
    plt.xlabel("BET_Surface_Area (m2/g)")
    plt.ylabel("ECDF")
    plt.title("ECDF Plot", fontsize=14, weight="bold")
    plt.grid(False)
    plt.savefig(OUT_DIR / "ecdf_bet.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("ECDF plot saved to:", OUT_DIR / "ecdf_bet.png")
