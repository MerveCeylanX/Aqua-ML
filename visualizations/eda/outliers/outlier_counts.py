"""
outlier_counts.py
-----------------
Bu dosya her sayısal feature için IQR metoduna göre outlier sayısını hesaplar
ve bar grafikte görselleştirir.

Çıktı: figures/outlier_counts.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET = 0

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

exclude_cols = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name",
    "Activation_Method", "%sum", "Target_Phar", "Reference", "% Removal"
]

# === LOAD ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# Sadece sayısal kolonlar
num_cols = [c for c in df.columns 
            if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude_cols]

outlier_counts = {}

for col in num_cols:
    data = df[col].dropna()
    if data.empty:
        outlier_counts[col] = 0
        continue
    
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = data[(data < lower) | (data > upper)]
    outlier_counts[col] = len(outliers)

# DataFrame'e çevir
outlier_df = pd.DataFrame(list(outlier_counts.items()), columns=["Feature", "OutlierCount"])
outlier_df = outlier_df.sort_values("OutlierCount", ascending=False)

print(outlier_df)

# === GRAFİK ===
plt.figure(figsize=(12,6))
sns.barplot(x="Feature", y="OutlierCount", data=outlier_df, palette="viridis")
plt.xticks(rotation=90)
plt.title("Outlier Counts per Feature (IQR Method)", fontsize=14, weight="bold")
plt.xlabel("Feature")
plt.ylabel("Outlier Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "outlier_counts.png", dpi=300, bbox_inches="tight")
plt.close()

print("Outlier counts plot saved to:", OUT_DIR / "outlier_counts.png")
