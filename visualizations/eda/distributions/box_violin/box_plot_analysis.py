"""
box_plot_analysis.py
----------------------

Bu dosya; veri setindeki seçili özellik grupları için **box plot** (yatay),
ayrıca `qe(mg/g)` dağılımlarını **Source_Material** ve **Target_Phar**
kırılımlarında box plot olarak; son olarak da ilaç ve feedstock bazında
**qmax (max qe)** bar grafiklerini üretir.

Çıktılar (bu dosyanın klasöründe otomatik oluşan ./figures/ içine kaydedilir):
- boxplots_group1.png, boxplots_group2.png, ...           (# grup bazlı boxplotlar)
- box_qe_by_source_material.png                            (qe ~ Source_Material)
- box_qe_by_pharmaceutical.png                             (qe ~ Target_Phar)
- qmax_per_pharma.png, qmax_per_feedstock.png              (max qe bar grafikler)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === AYARLAR ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET = 0

# Hariç tutulacak kolonlar (EDA dışı/kimliksel alanlar)
exclude_cols = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name",
    "Activation_Method", "%sum", "Target_Phar", "Reference"
]

# === ÇIKTI KLASÖRÜ ===
BASE_DIR = Path(__file__).parent if "__file__" in globals() else Path.cwd()
FIG_DIR  = BASE_DIR / "box_plot_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# === VERİYİ YÜKLE ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# Kullanılabilir kolonlar
cols = [c for c in df.columns if c not in exclude_cols]

# === ÖZEL FEATURE GRUPLARI (sende tanımlı şekilde) ===
group1 = [c for c in ["Agent/Sample(g/g)", "Activation_Heating_Rate (K/min)"] if c in df.columns]
group2 = [c for c in ["Soaking_Temp(K)", "Soaking_Time(min)", "Activation_Temp(K)", "Activation_Time(min)"] if c in df.columns]
group3 = [c for c in ["Total_Pore_Volume(cm3/g)","Micropore_Volume(cm3/g)","Average_Pore_Diameter(nm)","pHpzc"] if c in df.columns]

# group4: "C_percent" ile "VM(Volatile_Matter)" aralığı—kolonlar yoksa boş bırak
try:
    i1 = cols.index("C_percent")
    i2 = cols.index("VM(Volatile_Matter)")
    group4 = cols[i1:i2] if i1 < i2 else []
except ValueError:
    group4 = []

group5 = [c for c in ["Yield(%)", "VM(Volatile_Matter)", "Ash", "FC(Fixed_Carbon)"] if c in df.columns]
group6 = [c for c in ["Solution_pH", "Dosage(g/L)"] if c in df.columns]
group7 = [c for c in ["Temperature(K)", "Agitation_speed(rpm)"] if c in df.columns]
group8 = [c for c in ["BET_Surface_Area(m2/g)", "Initial_Concentration(mg/L)", "Contact_Time(min)", "qe(mg/g)"] if c in df.columns]

groups = {
    "Group 1 — Agent/Sample(g/g) & Activation_Heating_Rate (K/min)": group1,
    "Group 2 — Soaking & Activation": group2,
    "Group 3 — Remaining Features": group3,
    "Group 4 — Other Numerical Features": group4,
    "Group 5 — Yield, VM, Ash, FC": group5,
    "Group 6 — Solution_pH & Dosage(g/L)": group6,
    "Group 7 — Temperature, Agitation_speed, Contact_Time": group7,
    "Group 8 — BET_Surface_Area, Initial_Concentration, qe": group8
}

# === STİL ===
sns.set_style("whitegrid")
palette = sns.color_palette("Set2")

# === GRUP BAZLI BOXPLOT'LAR (yatay, tek figürde çok sütun) ===
for idx, (gname, gcols) in enumerate(groups.items(), start=1):
    num_cols = [c for c in gcols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        continue
    plt.figure(figsize=(12, max(6, 0.6 * len(num_cols))))
    sns.boxplot(data=df[num_cols], orient="h", palette=palette)
    plt.title(f"Boxplots — {gname}", fontsize=14, weight="bold")
    plt.xlabel("Value")
    plt.tight_layout()
    out_path = FIG_DIR / f"boxplots_group{idx}.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_path.resolve())

# === qe dağılımı: feedstock'a göre (Source_Material) ===
if "Source_Material" in df.columns and "qe(mg/g)" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="Source_Material", y="qe(mg/g)", data=df, palette="Set2")
    plt.xticks(rotation=45, ha="right")
    plt.title("qe Distribution by Source Material", fontsize=14, weight="bold")
    plt.xlabel("Source Material")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out_path = FIG_DIR / "box_qe_by_source_material.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_path.resolve())

# === qe dağılımı: pharmasötiklere göre (Target_Phar) ===
if "Target_Phar" in df.columns and "qe(mg/g)" in df.columns:
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="Target_Phar", y="qe(mg/g)", data=df, palette="Set3")
    plt.xticks(rotation=90)
    plt.title("qe Distribution by Pharmaceutical", fontsize=14, weight="bold")
    plt.xlabel("Pharmaceutical")
    plt.ylabel("qe (mg/g)")
    plt.tight_layout()
    out_path = FIG_DIR / "box_qe_by_pharmaceutical.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_path.resolve())

# === qmax: ilaç ve feedstock bazında (max qe) ===
if "Target_Phar" in df.columns and "qe(mg/g)" in df.columns:
    qmax_pharma = (
        df.groupby("Target_Phar")["qe(mg/g)"].max()
          .reset_index().rename(columns={"qe(mg/g)": "qmax"})
          .sort_values("qmax", ascending=True)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(y="Target_Phar", x="qmax", data=qmax_pharma, palette="tab20", dodge=False)
    plt.title("Qmax (max qe) per Pharmaceutical", fontsize=14, weight="bold")
    plt.xlabel("qmax (mg/g)")
    plt.ylabel("Pharmaceutical")
    for i, (val, name) in enumerate(zip(qmax_pharma["qmax"], qmax_pharma["Target_Phar"])):
        plt.text(val, i, f"{val:.1f}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    out_path = FIG_DIR / "qmax_per_pharma.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_path.resolve())

if "Source_Material" in df.columns and "qe(mg/g)" in df.columns:
    qmax_feedstock = (
        df.groupby("Source_Material")["qe(mg/g)"].max()
          .reset_index().rename(columns={"qe(mg/g)": "qmax"})
          .sort_values("qmax", ascending=True)
    )
    plt.figure(figsize=(10, 8))
    sns.barplot(y="Source_Material", x="qmax", data=qmax_feedstock, palette="tab20", dodge=False)
    plt.title("Qmax (max qe) per Feedstock", fontsize=14, weight="bold")
    plt.xlabel("qmax (mg/g)")
    plt.ylabel("Feedstock")
    for i, (val, name) in enumerate(zip(qmax_feedstock["qmax"], qmax_feedstock["Source_Material"])):  # noqa
        plt.text(val, i, f"{val:.1f}", va='center', ha='left', fontsize=8)
    plt.tight_layout()
    out_path = FIG_DIR / "qmax_per_feedstock.png"
    plt.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close()
    print("Kaydedildi:", out_path.resolve())
