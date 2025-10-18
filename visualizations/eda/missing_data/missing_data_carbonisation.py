"""
missing_data_carbonisation.py
-----------------------------

Bu kod, aktif karbon sentezi koşullarına ait verilerdeki eksik değerleri
analiz edip görselleştirir. Excel dosyasından seçilen sütunlar okunur,
her sütundaki eksik gözlem sayısı ve yüzdesi hesaplanır, ardından
yatay bar grafiği şeklinde görselleştirilir. 

Çıktı:
- Grafik: "Aktif karbon sentezi koşulları için eksik veri (%)"
- Kaydedilen dosya: ./figures/missing_data_carbonisation.png
- Eğer 'figures' klasörü yoksa otomatik oluşturulur.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================
# Veri Yükleme
# =====================
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx"     # Veri dosyasının yolu
SHEET = 0                               # Kullanılacak sayfa (ilk sheet)
TARGET_COLS = [
    "Activation_Agent", "Agent/Sample(g/g)", "Soaking_Time(min)",
    "Soaking_Temp(K)", "Activation_Temp(K)", "Activation_Time(min)",
    "Activation_Heating_Rate (K/min)", "Activation_Atmosphere"
]  # 'Ce' sütunu çıkarıldı

df = pd.read_excel(PATH, sheet_name=SHEET)

# =====================
# Eksik Veri Analizi
# =====================
n = len(df)                                   # Toplam satır sayısı
missing_counts = df[TARGET_COLS].isna().sum() # Eksik gözlem sayısı
missing_pct = (missing_counts / n * 100).round(1)  # Yüzdelik eksik oran

# En çok eksik olanı yukarıda göstermek için sıralama
order = missing_pct.sort_values(ascending=False).index
missing_pct = missing_pct.loc[order]
missing_counts = missing_counts.loc[order]

# =====================
# Grafik Çizimi
# =====================
fig, ax = plt.subplots(figsize=(9, 5), dpi=150, constrained_layout=True)
y = np.arange(len(order))
bars = ax.barh(y, missing_pct.values, color="#03C2FB")

# Başlık ve eksen etiketleri
ax.set_title("Aktif karbon sentezi koşulları için eksik veri (%)", fontsize=11)
ax.set_xlabel("Eksik Veri (%)", fontsize=10)
ax.set_yticks(y)
ax.set_yticklabels(order, fontsize=9)
ax.set_xlim(0, missing_pct.max() * 1.2)  # Maksimum değerin biraz ötesine kadar eksen
ax.margins(x=0.08)

# =====================
# Barların üzerine yüzde + n etiketleri ekleme
# =====================
for i, (p, c) in enumerate(zip(missing_pct.values, missing_counts.values)):
    ax.annotate(f"{p:.1f}% (n={c})", xy=(p, i), xytext=(3, 0),
                textcoords="offset points", va="center", ha="left", fontsize=9)

# En çok eksik olan değişken yukarıda gözüksün
ax.invert_yaxis()

# =====================
# Görselleştirme & Kaydetme
# =====================
plt.show()

# Bu dosyanın bulunduğu klasörü bul
current_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Alt klasörü oluştur
save_dir = current_dir / "figures"
save_dir.mkdir(parents=True, exist_ok=True)

# Dosyayı kaydet
fig_path = save_dir / "missing_data_carbonisation.png"
fig.savefig(fig_path, bbox_inches="tight", dpi=500)

print(f"Grafik kaydedildi: {fig_path}")
