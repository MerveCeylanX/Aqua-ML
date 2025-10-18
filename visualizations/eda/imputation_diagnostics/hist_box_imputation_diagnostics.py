"""
hist_box_imputation_diagnostics.py
---------------------------------

Bu script, veri setindeki bazı sayısal kolonların (Solution pH, Contact Time, Agitation Speed)
dağılımlarını incelemek için histogram + boxplot grafiklerini üretir.  
Amaç: imputation öncesi değişkenlerin eksikliklerini, dağılım şekillerini ve medyanlarını gözlemlemek.  

Çıktılar:
- Tekil grafikler (histogram + boxplot)
- Birleşik 3×2 panel (pH, Contact Time, Agitation Speed)
- Kaydedilme yeri: ./figures/  (script'in bulunduğu klasörün altında)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"
SHEET = 0
OUT_DIR = Path(__file__).resolve().parent / "figures"   # bulunduğu klasörde figures
OUT_DIR.mkdir(exist_ok=True, parents=True)

# === LOAD ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# === HELPERS ===
def prep_numeric(series):
    return pd.to_numeric(series, errors="coerce")

def plot_hist_box(fig_title, series, xlab, fname, bins=15):
    s = series.dropna()
    print(f"{fig_title} -> total: {len(series)}, missing: {series.isna().sum()} "
          f"({series.isna().mean()*100:.1f}%), median: {s.median():.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=150, constrained_layout=True)

    # Histogram
    axes[0].hist(s, bins=bins, color="darkred", edgecolor="black")
    axes[0].axvline(s.median(), color="blue", linestyle="--", label=f"Median={s.median():.2f}")
    axes[0].set_title(f"{fig_title} distribution (histogram)")
    axes[0].set_xlabel(xlab)
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Boxplot
    axes[1].boxplot(s, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightgray", color="black"))
    axes[1].axhline(s.median(), color="blue", linestyle="--")
    axes[1].set_title(f"{fig_title} (boxplot)")
    axes[1].set_ylabel(xlab)

    fig.savefig(OUT_DIR / fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

def draw_hist_box_on_axes(ax_hist, ax_box, s, title, xlab, bins=15):
    ax_hist.hist(s, bins=bins, color="darkred", edgecolor="black")
    ax_hist.axvline(s.median(), color="blue", linestyle="--", label=f"Median={s.median():.2f}")
    ax_hist.set_title(title)
    ax_hist.set_xlabel(xlab)
    ax_hist.set_ylabel("Freq")
    ax_hist.legend()

    ax_box.boxplot(s, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="lightgray", color="black"))
    ax_box.axhline(s.median(), color="blue", linestyle="--")
    ax_box.set_title(title)
    ax_box.set_ylabel(xlab)

def add_panel_labels(axes, labels):
    for ax, lab in zip(axes, labels):
        ax.text(-0.08, 1.05, lab, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", ha="right")

# === SERIES ===
ph  = prep_numeric(df["Solution_pH"])
ct  = prep_numeric(df["Contact_Time(min)"])
ag  = prep_numeric(df["Agitation_speed(rpm)"])

# === 1) INDIVIDUAL FIGS ===
plot_hist_box("Solution pH", ph, "pH", "pH_distribution_and_boxplot.png", bins=15)
plot_hist_box("Contact time", ct, "Contact time (min)", "contact_time_distribution_and_boxplot.png", bins=20)
plot_hist_box("Agitation speed", ag, "Agitation speed (rpm)", "agitation_speed_distribution_and_boxplot.png", bins=20)

# === 2) COMBINED 3×2 FIG ===
fig, axes = plt.subplots(3, 2, figsize=(12, 12), dpi=180, constrained_layout=True)

# pH
ph_valid = ph.dropna()
draw_hist_box_on_axes(axes[0,0], axes[0,1], ph_valid, "Solution pH", "pH", bins=15)

# Contact time
ct_valid = ct.dropna()
draw_hist_box_on_axes(axes[1,0], axes[1,1], ct_valid, "Contact time", "Contact time (min)", bins=20)

# Agitation speed
ag_valid = ag.dropna()
draw_hist_box_on_axes(axes[2,0], axes[2,1], ag_valid, "Agitation speed", "Agitation speed (rpm)", bins=20)

# Panel labels (a–f)
panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
add_panel_labels([axes[0,0], axes[0,1], axes[1,0], axes[1,1], axes[2,0], axes[2,1]], panel_labels)

# Layout fine-tune
fig.set_constrained_layout_pads(w_pad=0.09, h_pad=0.09, wspace=0.1, hspace=0.10)

fig.savefig(OUT_DIR / "combined_ph_ct_agitation_3x2.png", bbox_inches="tight", dpi=400)
plt.close(fig)

print("Done. Figures saved under:", OUT_DIR.resolve())
