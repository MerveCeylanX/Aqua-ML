"""
activation_agent_atmosphere.py
------------------------------

Bu script, veri setindeki **Activation_Agent** ve **Activation_Atmosphere**
kolonlarının kullanım sıklığını hesaplar ve çubuk grafikler üretir.

Çıktılar:
- figures/activation_agent_usage.png
- figures/activation_atmosphere_usage.png
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# === SETTINGS ===
PATH  = r"D:\Aqua_ML\data\Raw_data_0.xlsx" 
SHEET = 0

# === LOAD ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# === OUTPUT DIR ===
OUT_DIR = Path(__file__).parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

# --- Activation Agent ---
agent_counts = df["Activation_Agent"].value_counts().reset_index()
agent_counts.columns = ["Activation_Agent", "Count"]

plt.figure(figsize=(10,6))
sns.barplot(y="Activation_Agent", x="Count", data=agent_counts, palette="tab10")
plt.title("Usage Frequency of Activation Agents", fontsize=14, weight="bold")
plt.xlabel("Number of Records")
plt.ylabel("Activation Agent")
for i, val in enumerate(agent_counts["Count"]):
    plt.text(val, i, str(val), va='center', ha='left')
plt.tight_layout()
plt.savefig(OUT_DIR / "activation_agent_usage.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Activation Atmosphere ---
atmo_counts = df["Activation_Atmosphere"].value_counts().reset_index()
atmo_counts.columns = ["Activation_Atmosphere", "Count"]

plt.figure(figsize=(10,6))
sns.barplot(y="Activation_Atmosphere", x="Count", data=atmo_counts, palette="tab20")
plt.title("Usage Frequency of Activation Atmospheres", fontsize=14, weight="bold")
plt.xlabel("Number of Records")
plt.ylabel("Activation Atmosphere")
for i, val in enumerate(atmo_counts["Count"]):
    plt.text(val, i, str(val), va='center', ha='left')
plt.tight_layout()
plt.savefig(OUT_DIR / "activation_atmosphere_usage.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"Done. Figures saved under: {OUT_DIR.resolve()}")
