"""
violin_plot_analysis.py
-------------------------------

Bu dosya, veri setindeki sayısal özelliklerin dağılımlarını **violin plot** ile,
box'ta yaptığımız gibi **8 gruba** ayrılmış biçimde görselleştirir.
Ayrıca `qe(mg/g)` için Source_Material ve Target_Phar kırılımlarında
violin + swarm / violin + strip grafikleri üretir.

Çıktılar (otomatik oluşturulur):
- ./visualizations/eda/distributions/box_violin/violin_plot_figures/overall/
    - violin_group1.png ... violin_group8.png  (gruplara göre toplu violin, yatay)
- ./visualizations/eda/distributions/box_violin/violin_plot_figures/by_source/
    - qe_by_source_material__violin_swarm.png
    - qe_by_source_material__violin_strip.png
- ./visualizations/eda/distributions/box_violin/violin_plot_figures/by_pharma/
    - qe_by_pharma__violin_swarm.png
    - qe_by_pharma__violin_strip.png
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET = 0

# Box/violin dışında tutulan kolonlar (kimliksel/kat. kırılım kolonları)
EXCLUDE = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name",
    "Activation_Method", "%sum", "Target_Phar", "Reference"
]

# === OUTPUT ROOT (istenen yapı) ===
ROOT = Path("visualizations/eda/distributions/box_violin") / "violin_plot_figures"
DIR_OVERALL   = ROOT / "overall"
DIR_BY_SOURCE = ROOT / "by_source"
DIR_BY_PHARMA = ROOT / "by_pharma"
for d in (DIR_OVERALL, DIR_BY_SOURCE, DIR_BY_PHARMA):
    d.mkdir(parents=True, exist_ok=True)

# === VERİ ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# === GRUP TANIMLARI (box'takiyle aynı mantık) ===
# Kolon isimleri yoksa otomatik filtreler; olanları alır.
group1 = [c for c in ["Agent/Sample(g/g)", "Activation_Heating_Rate (K/min)"] if c in df.columns]
group2 = [c for c in ["Soaking_Temp(K)", "Soaking_Time(min)", "Activation_Temp(K)", "Activation_Time(min)"] if c in df.columns]
group3 = [c for c in ["Total_Pore_Volume(cm3/g)","Micropore_Volume(cm3/g)","Average_Pore_Diameter(nm)","pHpzc"] if c in df.columns]

# group4: C_percent ... VM(Volatile_Matter) aralığı (varsa, sırayı koruyarak)
cols_all = [c for c in df.columns if c not in EXCLUDE]
try:
    i1 = cols_all.index("C_percent")
    i2 = cols_all.index("VM(Volatile_Matter)")
    group4 = cols_all[i1:i2] if i1 < i2 else []
except ValueError:
    group4 = []

group5 = [c for c in ["Yield(%)", "VM(Volatile_Matter)", "Ash", "FC(Fixed_Carbon)"] if c in df.columns]
group6 = [c for c in ["Solution_pH", "Dosage(g/L)"] if c in df.columns]
group7 = [c for c in ["Temperature(K)", "Agitation_speed(rpm)", "Contact_Time(min)"] if c in df.columns]
group8 = [c for c in ["BET_Surface_Area(m2/g)", "Initial_Concentration(mg/L)", "qe(mg/g)"] if c in df.columns]

GROUPS = {
    "Group 1 — Agent/Sample & Heating Rate": group1,
    "Group 2 — Soaking & Activation": group2,
    "Group 3 — Pore & pHpzc": group3,
    "Group 4 — Other Numerical (C%..VM)": group4,
    "Group 5 — Yield, VM, Ash, FC": group5,
    "Group 6 — Solution_pH & Dosage": group6,
    "Group 7 — Temp, Agitation, Contact": group7,
    "Group 8 — BET, Initial, qe": group8,
}

# === STİL ===
sns.set_style("whitegrid")
PALETTE = "Set3"

# === 1) GRUP BAZLI VIOLIN (yatay) ===
for idx, (gname, gcols) in enumerate(GROUPS.items(), start=1):
    num_cols = [c for c in gcols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        continue

    dlong = (
        df[num_cols]
        .melt(var_name="Feature", value_name="Value")
        .dropna(subset=["Value"])
    )

    plt.figure(figsize=(12, max(6, 0.45 * len(num_cols))))
    sns.violinplot(
        data=dlong, y="Feature", x="Value",
        inner="box", palette=PALETTE, cut=0
    )
    plt.title(f"{gname} — Violin", fontsize=14, weight="bold")
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    out = DIR_OVERALL / f"violin_group{idx}.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out.resolve())

# === 2) qe(mg/g) dağılımları (by_source / by_pharma) ===
has_qe = "qe(mg/g)" in df.columns

# 2a) Source_Material kırılımı (violin + swarm)
if has_qe and "Source_Material" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Source_Material", y="qe(mg/g)", data=df, palette="Set2",
                   inner="quartile", cut=0)
    sns.swarmplot(x="Source_Material", y="qe(mg/g)", data=df, color="black", size=3)
    plt.xticks(rotation=45, ha="right")
    plt.title("qe Distribution by Source Material (Violin + Swarm)", fontsize=14, weight="bold")
    plt.xlabel("Source Material")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out = DIR_BY_SOURCE / "qe_by_source_material__violin_swarm.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out.resolve())

# 2b) Source_Material (violin + strip)
if has_qe and "Source_Material" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Source_Material", y="qe(mg/g)", data=df, palette="Set2",
                   inner="quartile", cut=0)
    sns.stripplot(x="Source_Material", y="qe(mg/g)", data=df, color="black", size=3, jitter=0.2)
    plt.xticks(rotation=45, ha="right")
    plt.title("qe Distribution by Source Material (Violin + Strip)", fontsize=14, weight="bold")
    plt.xlabel("Source Material")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out = DIR_BY_SOURCE / "qe_by_source_material__violin_strip.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out.resolve())

# 2c) Target_Phar kırılımı (violin + swarm)
if has_qe and "Target_Phar" in df.columns:
    plt.figure(figsize=(14, 6))
    sns.violinplot(x="Target_Phar", y="qe(mg/g)", data=df, palette="Set3",
                   inner="quartile", cut=0)
    sns.swarmplot(x="Target_Phar", y="qe(mg/g)", data=df, color="black", size=3)
    plt.xticks(rotation=90)
    plt.title("qe Distribution by Pharmaceutical (Violin + Swarm)", fontsize=14, weight="bold")
    plt.xlabel("Pharmaceutical")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out = DIR_BY_PHARMA / "qe_by_pharma__violin_swarm.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out.resolve())

# 2d) Target_Phar (violin + strip)
if has_qe and "Target_Phar" in df.columns:
    plt.figure(figsize=(14, 6))
    sns.violinplot(x="Target_Phar", y="qe(mg/g)", data=df, palette="Set3",
                   inner="quartile", cut=0)
    sns.stripplot(x="Target_Phar", y="qe(mg/g)", data=df, color="black", size=3, jitter=0.2)
    plt.xticks(rotation=90)
    plt.title("qe Distribution by Pharmaceutical (Violin + Strip)", fontsize=14, weight="bold")
    plt.xlabel("Pharmaceutical")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out = DIR_BY_PHARMA / "qe_by_pharma__violin_strip.png"
    plt.savefig(out, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out.resolve())
