"""
lasso_feature_selection_with_plots.py
-------------------------------------
Eksik verili ve kategorik özellikli bir veri setinde, hedef 'qe(mg/g)' için
saf Lasso (LassoCV) ile özellik seçimi yapar. Sıfır olmayan ve sıfır katsayıları
çıkarır, çeşitli görselleştirmeler üretir.

Çıktı görselleri: ./figures/
"""

import pandas as pd
import numpy as np
import sklearn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

# =================== SETTINGS ====================
PATH = r"D:\Aqua_ML\data\Raw_data_enriched.xlsx"   # gerekirse değiştir
TARGET = "qe(mg/g)"                       # hedef kolon

EXCLUDE_COLS = [
    "Ce(mg/L)", "Source_Material", "Adsorbent_Name", "Activation_Agent",
    "Activation_Method", "%sum", "Target_Phar", "Reference"
]
TOP_K = 30  # grafikleri en etkili TOP_K özellik ile sınırla

# Çıktı klasörü (script'in yanına figures/)
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =================== LOAD =======================
df = pd.read_excel(PATH)
if TARGET not in df.columns:
    raise KeyError(f"TARGET '{TARGET}' not found. Available: {list(df.columns)}")
exclude_present = [c for c in EXCLUDE_COLS if c in df.columns]

y = pd.to_numeric(df[TARGET], errors="coerce")
X = df.drop(columns=[TARGET] + exclude_present)
print(f"Excluded from features: {exclude_present}")

# ================== TYPES =======================
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# ================== PREPROC =====================
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# OneHotEncoder sürüm uyumluluğu
ohe_kwargs = {}
try:
    OneHotEncoder(sparse_output=False)
    ohe_kwargs["sparse_output"] = False
except TypeError:
    ohe_kwargs["sparse"] = False

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", **ohe_kwargs))
])

pre = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
], remainder="drop")

# =================== MODEL ======================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
lasso = LassoCV(
    cv=cv,
    n_alphas=200,
    max_iter=10000,
    # alphas=None → otomatik grid
)

model = Pipeline(steps=[
    ("pre", pre),
    ("clf", lasso)
])

# y'de NaN olanları düş
keep = ~y.isna()
X_fit = X.loc[keep].copy()
y_fit = y.loc[keep].copy()

# ==================== FIT =======================
model.fit(X_fit, y_fit)

# =================== METRICS ====================
y_pred = model.predict(X_fit)
print(f"scikit-learn version: {sklearn.__version__}")
print(f"R2 (train/apparent): {r2_score(y_fit, y_pred):.3f}")
print(f"MAE (train/apparent): {mean_absolute_error(y_fit, y_pred):.3f}")
print(f"Selected alpha (Lasso): {model.named_steps['clf'].alpha_:.6f}")

# ================ FEATURE NAMES =================
def get_feature_names(preproc: ColumnTransformer) -> list:
    """ColumnTransformer içinden, OHE sonrası tüm çıktı isimlerini güvenle üret."""
    try:
        return preproc.get_feature_names_out().tolist()
    except Exception:
        names = []
        # numeric
        names += preproc.transformers_[0][2]
        # categorical (OHE)
        ohe = preproc.transformers_[1][1].named_steps["ohe"]
        bases = preproc.transformers_[1][2]
        cats = ohe.categories_
        for base, vals in zip(bases, cats):
            names += [f"{base}__{v}" for v in vals]
        return names

feat_names = get_feature_names(model.named_steps["pre"])
coefs = model.named_steps["clf"].coef_

coef_df = (pd.DataFrame({"Feature": feat_names, "Coef": coefs})
           .assign(AbsCoef=lambda d: d["Coef"].abs())
           .sort_values("AbsCoef", ascending=False))

# Non-zero ve zero ayrımı
selected = coef_df.loc[coef_df["Coef"] != 0].copy()

print("\n=== Non-zero Lasso coefficients (sorted by |coef|) ===")
if selected.empty:
    print("(No non-zero coefficients selected — veri/ön işleme gridini gözden geçirin.)")
else:
    print(selected.head(40).to_string(index=False))

# === Zero katsayıları da yazdır ===
zero_df = (coef_df.loc[coef_df["Coef"] == 0]
                    .sort_values(["Feature"]))
print("\n=== Zero Lasso coefficients (|coef|==0) ===")
if zero_df.empty:
    print("(Sıfır katsayılı özellik yok.)")
else:
    print(zero_df.to_string(index=False))

# (Opsiyonel) Excel'e kaydetmek istersen:
# with pd.ExcelWriter(OUT_DIR / "lasso_selected_features.xlsx", engine="xlsxwriter") as wr:
#     selected.to_excel(wr, sheet_name="nonzero", index=False)
#     zero_df.to_excel(wr, sheet_name="zero", index=False)

# ================== PLOTTING ====================
sns.set_style("whitegrid")

def _shorten_names(s, maxlen=45):
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."

# En etkili TOP_K özelliği al
sel_top = selected.head(TOP_K).iloc[::-1].copy()  # grafikte en büyükler üstte çıksın diye ters çevir
sel_top["FeatShort"] = sel_top["Feature"].apply(_shorten_names)

# 1) İmza katsayı bar (pozitif/negatif renkli, yatay)
plt.figure(figsize=(10, max(6, 0.35*len(sel_top))))
colors = sel_top["Coef"].apply(lambda v: "#1f77b4" if v >= 0 else "#d62728")
plt.barh(sel_top["FeatShort"], sel_top["Coef"], color=colors, edgecolor="black")
plt.axvline(0, color="black", linewidth=0.8)
plt.title(f"Lasso Coefficients (Top {len(sel_top)})")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig(OUT_DIR / "lasso_coef_signed_barh.png", dpi=300, bbox_inches="tight")
plt.close()

# 2) Lollipop (yatay, imza katsayı)
plt.figure(figsize=(10, max(6, 0.35*len(sel_top))))
y_pos = np.arange(len(sel_top))
plt.hlines(y_pos, 0, sel_top["Coef"], colors=colors, linewidth=2)
plt.plot(sel_top["Coef"], y_pos, "o", color="black", markersize=4)
plt.axvline(0, color="black", linewidth=0.8)
plt.yticks(y_pos, sel_top["FeatShort"])
plt.title(f"Lollipop — Lasso Coefficients (Top {len(sel_top)})")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig(OUT_DIR / "lasso_coef_lollipop.png", dpi=300, bbox_inches="tight")
plt.close()

# 3) Mutlak katsayı (|coef|) bar (önem sırası)
plt.figure(figsize=(10, max(6, 0.35*len(sel_top))))
plt.barh(sel_top["FeatShort"], sel_top["AbsCoef"], edgecolor="black")
plt.title(f"Absolute Coefficients |coef| (Top {len(sel_top)})")
plt.xlabel("|Coefficient|")
plt.tight_layout()
plt.savefig(OUT_DIR / "lasso_coef_abs_barh.png", dpi=300, bbox_inches="tight")
plt.close()

# 4) Actual vs Predicted (train)
plt.figure(figsize=(6, 6))
plt.scatter(y_fit, y_pred, s=18, alpha=0.7, edgecolor="white")
mn = np.nanmin([y_fit.min(), y_pred.min()])
mx = np.nanmax([y_fit.max(), y_pred.max()])
plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
plt.title(f"Actual vs Predicted (train)\nR²={r2_score(y_fit, y_pred):.3f}, MAE={mean_absolute_error(y_fit, y_pred):.2f}")
plt.xlabel("Actual qe (mg/g)")
plt.ylabel("Predicted qe (mg/g)")
plt.tight_layout()
plt.savefig(OUT_DIR / "lasso_actual_vs_pred_train.png", dpi=300, bbox_inches="tight")
plt.close()

# 5) Residual histogram
resid = y_fit - y_pred
plt.figure(figsize=(7,4))
plt.hist(resid, bins=30, edgecolor="black")
plt.title("Residuals (train)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(OUT_DIR / "lasso_residuals_hist.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figures saved to:", OUT_DIR.resolve())

# ================= EXTRA DIAGNOSTICS (opsiyonel) =================
# BET teşhisi (senin önceki çıktılarla uyumlu kalması için)
cands = [c for c in X.columns if ("bet" in c.lower()) or ("surface" in c.lower())]
print("BET candidates in X:", cands)

for c in cands:
    s = X[c]
    print(f"\n[{c}] dtype={s.dtype}, n_missing={s.isna().sum()}, nunique={s.nunique()}")
    s_num = pd.to_numeric(s, errors="coerce")
    if s.dtype == "object":
        print(f"  after to_numeric: n_missing={s_num.isna().sum()}, std={s_num.std():.3g}")
    else:
        print(f"  std={s.std():.3g}")

feat_names = get_feature_names(model.named_steps["pre"])
bet_feats = [f for f in feat_names if ("bet" in f.lower()) or ("surface" in f.lower())]
print("\nBET-related features AFTER preprocessing:", bet_feats)

if bet_feats:
    coefs_now = model.named_steps["clf"].coef_
    coef_map = dict(zip(feat_names, coefs_now))
    print({f: coef_map.get(f) for f in bet_feats})

# === OHE SONRASI (kategorikler dahil) BET korelasyonları ===
try:
    # Preprocess edilmiş tasarım matrisi (impute + scale + OHE)
    X_ohe = model.named_steps["pre"].transform(X_fit)
    feat_names_all = get_feature_names(model.named_steps["pre"])
    X_ohe_df = pd.DataFrame(X_ohe, columns=feat_names_all, index=X_fit.index)

    bet_col_name = "BET_Surface_Area(m2/g)"  # bet kolon adın farklıysa bunu güncelle
    if bet_col_name in X_ohe_df.columns:
        corr_to_bet = X_ohe_df.corr(numeric_only=True)[bet_col_name].drop(labels=[bet_col_name])

        print("\n[OHE sonrası] BET ile en yüksek + korelasyonlar:")
        print(corr_to_bet.sort_values(ascending=False).head(10).to_string())

        print("\n[OHE sonrası] BET ile en yüksek - korelasyonlar:")
        print(corr_to_bet.sort_values(ascending=True).head(10).to_string())

        # Örnek: Activation_Atmosphere dummy'leri özel liste
        aa_prefix = "Activation_Atmosphere__"
        aa_cols = [c for c in X_ohe_df.columns if c.startswith(aa_prefix)]
        aa_cols_in = [c for c in aa_cols if c in corr_to_bet.index]
        if aa_cols_in:
            aa_corr = corr_to_bet.loc[aa_cols_in].sort_values(ascending=False)
            print("\n[OHE sonrası] BET ile Activation_Atmosphere türevlerinin korelasyonları:")
            print(aa_corr.to_string())
        else:
            print("\n(Activation_Atmosphere OHE kolonları bulunamadı veya korelasyon dizininde değil.)")
    else:
        close = [c for c in X_ohe_df.columns if "bet" in c.lower() or "surface" in c.lower()]
        print(f"\n(OHE sonrası tasarımda '{bet_col_name}' bulunamadı. Olası adaylar: {close})")
except Exception as e:
    print("\n[Uyarı] OHE sonrası korelasyon hesaplanırken hata oluştu:", repr(e))
