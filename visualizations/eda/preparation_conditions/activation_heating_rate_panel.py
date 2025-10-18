"""
activation_heating_rate_panel.py
--------------------------------

Bu kod, aktivasyon ısıtma hızı verilerini analiz eder ve 2×2 panel halinde görselleştirir:
1) Isıtma hızı dağılımı (bar + yüzde etiketleri)
2) Atmosfer × ısıtma hızı (100% stacked bar)
3) Sıcaklık bandı × ısıtma hızı (inert atmosfer için, heatmap)
4) Sıcaklık bandı × ısıtma hızı (oksidatif atmosfer için, heatmap)

Çıktılar:
- ./figures/activation_heating_rate_panel_2x2.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- SETTINGS --------
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0
COL_RATE = "Activation_Heating_Rate (K/min)"
COL_ATM  = "Activation_Atmosphere"
COL_TEMPK= "Activation_Temp(K)"

# Discrete order to display
HR_ORDER = [3.0, 5.0, 10.0, 15.0, 20.0]

# Temperature bands (Kelvin)
def temp_band_K(TK):
    if pd.isna(TK):
        return "unknown"
    try:
        tK = float(TK)
    except:
        return "unknown"
    if tK < 773.15:       return "<773 K"
    elif tK <= 1023.15:   return "773–1023 K"
    else:                 return ">1023 K"

# Atmosphere classifier -> inert / oxidative / unknown
def classify_atm(x: str) -> str:
    if pd.isna(x):
        return "unknown"
    s = str(x).strip().lower()
    if s in {"n2", "nitrogen"}: return "inert"
    if s in {"air", "sg"}:      return "oxidative"
    if any(k in s for k in ["argon", "ar", "n2", "nitrogen"]): return "inert"
    if any(k in s for k in ["air", "steam", "co2", "co₂", "self-generated", "self generated", "sg"]):
        return "oxidative"
    return "unknown"

# -------- LOAD --------
df = pd.read_excel(PATH, sheet_name=SHEET)

# -------- STATS (print only) --------
s = pd.to_numeric(df[COL_RATE], errors="coerce").dropna()
print(f"Total rows: {len(df):,} | Non-missing '{COL_RATE}': {len(s):,}")
q = s.quantile([0.25,0.5,0.75])
print("\n=== Descriptive stats ===")
print(f"  median: {q.loc[0.5]:.1f}")
print(f"  IQR   : {q.loc[0.25]:.1f}–{q.loc[0.75]:.1f}")
print(f"  min/max: {s.min():.1f} / {s.max():.1f}")
vc = s.value_counts().reindex(HR_ORDER, fill_value=0).astype(int)
print("\n=== Frequencies ===")
for v, c in vc.items():
    print(f"  {v:g} K min⁻¹: {c} ({c/len(s)*100:.1f}%)")

# -------- PREP DATA --------
work = df.copy()
work["Rate"] = pd.to_numeric(work[COL_RATE], errors="coerce")
work = work.dropna(subset=["Rate"])
work["Rate"] = work["Rate"].astype(float)
work["AtmClass"] = work[COL_ATM].apply(classify_atm)
work["TempBand"] = work[COL_TEMPK].apply(temp_band_K)

# Crosstabs
ct_atm = pd.crosstab(work["AtmClass"], work["Rate"]).reindex(columns=HR_ORDER, fill_value=0)
ct_atm = ct_atm.loc[[i for i in ["inert","oxidative"] if i in ct_atm.index]]
ct_atm_pct = (ct_atm.div(ct_atm.sum(axis=1), axis=0) * 100).fillna(0)

def heatmap_table(subset):
    ct = pd.crosstab(subset["TempBand"], subset["Rate"]).reindex(columns=HR_ORDER, fill_value=0)
    row_order = ["<773 K", "773–1023 K", ">1023 K"]
    ct = ct.reindex(index=row_order, fill_value=0)
    return (ct.div(ct.sum(axis=1), axis=0) * 100).fillna(0)

hm_inert = heatmap_table(work[work["AtmClass"]=="inert"])
hm_oxid  = heatmap_table(work[work["AtmClass"]=="oxidative"])

# -------- PLOT 2×2 --------
fig, axes = plt.subplots(2, 2, figsize=(9, 8), dpi=150)
fig.subplots_adjust(wspace=0.5, hspace=0.4)

# (1,1) Heating rate distribution
ax = axes[0,0]
counts = vc
total  = counts.sum()
xlabels = [str(int(v)) for v in counts.index]
bars = ax.bar(xlabels, counts.values)
ax.set_title("Heating rate distribution")
ax.set_xlabel("Activation heating rate (K min⁻¹)")
ax.set_ylabel("Count")
ax.set_ylim(top=500)
for r, val in zip(bars, counts.values):
    p = 100*val/total if total>0 else 0
    ax.annotate(f"{p:.1f}%", (r.get_x()+r.get_width()/2, r.get_height()),
                ha="center", va="bottom", fontsize=7, xytext=(0,3), textcoords="offset points")

# (2,1) Atmosphere × Rate
ax = axes[1,0]
bottom = np.zeros(len(ct_atm_pct))
x = np.arange(len(ct_atm_pct.index))
for hr in HR_ORDER:
    vals = ct_atm_pct[hr].values if hr in ct_atm_pct.columns else np.zeros_like(x, dtype=float)
    ax.bar(x, vals, bottom=bottom, label=str(int(hr)))
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(ct_atm_pct.index)
ax.set_ylabel("Row percentage (%)")
ax.set_xlabel("Atmosphere")
ax.set_title("Atmosphere × Heating rate")
ax.legend(title="K min⁻¹", ncols=5, fontsize=6)
ax.set_ylim(top=130)

# Heatmap helper
def draw_heatmap(ax, data, title):
    for hr in HR_ORDER:
        if hr not in data.columns:
            data[hr] = 0.0
    data = data[HR_ORDER]
    im = ax.imshow(data.values, aspect="auto", vmin=0, vmax=data.values.max())
    ax.set_xticks(np.arange(len(HR_ORDER)))
    ax.set_xticklabels([str(int(v)) for v in HR_ORDER])
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(list(data.index))
    ax.set_xlabel("Activation heating rate (K min⁻¹)")
    ax.set_ylabel("Temperature band (K)")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i,j]:.0f}%", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row %")

# (1,2) Heatmap inert
draw_heatmap(axes[0,1], hm_inert, "Temperature band × Rate — Inert (N₂)")

# (2,2) Heatmap oxidative
draw_heatmap(axes[1,1], hm_oxid, "Temperature band × Rate — Oxidative")

labels = ["(a)", "(c)", "(b)", "(d)"]
for ax, lab in zip(axes.flat, labels):
    ax.text(-0.2, 1.05, lab, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right")

# -------- SAVE --------
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig.savefig(OUT_DIR / "activation_heating_rate_panel_2x2.png",
            bbox_inches="tight", dpi=200)
plt.show()
