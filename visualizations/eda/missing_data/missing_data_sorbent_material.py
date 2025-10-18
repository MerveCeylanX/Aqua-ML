"""
missing_data_sorbent_material.py
-----------------------------

"Sorbent Material Characteristics" (malzeme karakterizasyonu) kolonları için
eksik değer analizi ve yatay bar grafik.

Kapsam (veride varsa otomatik seçilir):
- Textural: BET, Total/Micropore Volume, Average Pore Diameter
- Surface chemistry: pHpzc
- Composition: C/H/O/N/S (molar oran/frac), Ash(%), Yield(%)

Çıktı:
- Grafik başlığı: "Sorbent Material Characteristics için eksik veri (%)"
- Kaydedilen dosya: ./figures/missing_data_sorbent_material.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================
# Veri Yükleme
# =====================
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx"  # veri dosyan
SHEET = 0

# Bu gruba ait olası kolon isimleri (senin veri şemanla uyumlu)
CANDIDATE_COLS = [
    # Textural
    "BET_Surface_Area(m2/g)",
    "Total_Pore_Volume(cm3/g)",
    "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)",
    # Surface chemistry
    "pHpzc",
    # Composition
    "C_percent", "H_percent", "O_percent", "N_percent", "S_percent",
    "Ash(%)", "Yield(%)", "VM(Volatile_Matter)", "FC(Fixed_Carbon)",
]

df = pd.read_excel(PATH, sheet_name=SHEET)

# Veri dosyanda gerçekten bulunanları filtrele
TARGET_COLS = [c for c in CANDIDATE_COLS if c in df.columns]

# =====================
# Kolon yoksa bilgilendir
# =====================
if not TARGET_COLS:
    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=200, constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5,
            "Seçili 'Sorbent Material Characteristics' kolonları bulunamadı.",
            ha="center", va="center")
    # Kaydetme
    current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    save_dir = current_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "missing_data_sorbent_material.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    print(f"Uyarı: Uygun kolon bulunamadı. Bilgilendirme görseli kaydedildi: {fig_path}")
else:
    # =====================
    # Eksik Veri Analizi
    # =====================
    n = len(df)
    missing_counts = df[TARGET_COLS].isna().sum()
    missing_pct = (missing_counts / n * 100).round(1)

    # En çok eksik olandan az eksik olana sırala
    order = missing_pct.sort_values(ascending=False).index
    missing_pct = missing_pct.loc[order]
    missing_counts = missing_counts.loc[order]

    # =====================
    # Grafik Çizimi
    # =====================
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, constrained_layout=True)
    y = np.arange(len(order))
    bars = ax.barh(y, missing_pct.values, color="#0014E7")

    ax.set_title("Sorbent Material Characteristics için eksik veri (%)", fontsize=11)
    ax.set_xlabel("Eksik Veri (%)", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=9)

    # Maksimum değerin biraz ötesine kadar eksen (etiketler rahat sığsın)
    xmax = float(missing_pct.max()) if len(missing_pct) else 0.0
    ax.set_xlim(0, xmax * 1.2 if xmax > 0 else 1.0)
    ax.margins(x=0.08)

    # Çubukların yanına yüzde + n etiketi
    for i, (p, c) in enumerate(zip(missing_pct.values, missing_counts.values)):
        ax.annotate(f"{p:.1f}% (n={c})", xy=(p, i), xytext=(3, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=9)

    # En çok eksik olan en üstte gözüksün
    ax.invert_yaxis()

    # =====================
    # Görselleştirme & Kaydetme
    # =====================
    # plt.show()  # istersen göster

    current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
    save_dir = current_dir / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)

    fig_path = save_dir / "missing_data_sorbent_material.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=500)
    print(f"Grafik kaydedildi: {fig_path}")
