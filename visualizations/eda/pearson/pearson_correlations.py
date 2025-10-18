"""
pearson_correlations.py
-----------------------
RAW ve ENRICHED (imputed) veri setleri için Pearson korelasyon matrislerini hesaplar.
Çıktılar:
- Heatmap PNG'leri: figures/pearson_heatmap_<girdi_stem>.png
- Tek Excel dosyası: results/pearson_corr_combined.xlsx
  - Sayfalar: 'enriched' (RAW veri için), 'imputed' (ENRICHED veri için)

Yollar (RAW_PATH, IMPUTED_PATH), sayfa (SHEET) ve hedef kolonlar (TARGET_COLS) dosya başında tanımlıdır.
Komut satırı argümanları verilirse bu yolları geçersiz kılar.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Inline configuration (formerly in a.py)
IMPUTED_PATH = r"D:\Aqua_ML\data\Raw_data_imputed.xlsx"
RAW_PATH     = r"D:\Aqua_ML\data\Raw_data_enriched.xlsx"
SHEET        = 0  # Excel için sayfa (sheet) indexi

# Yalnızca bu listedeki değişkenleri değerlendir.
TARGET_COLS: List[str] = [
    "Agent/Sample(g/g)",
    "Soaking_Time(min)",
    "Soaking_Temp(K)",
    "Activation_Time(min)",
    "Activation_Temp(K)",
    "Activation_Heating_Rate (K/min)",
    "BET_Surface_Area(m2/g)",
    "Total_Pore_Volume(cm3/g)",
    "Micropore_Volume(cm3/g)",
    "Average_Pore_Diameter(nm)",
    "pHpzc",
    "C_molar", "H_C_molar", "O_C_molar", "N_C_molar", "S_C_molar",
    "Initial_Concentration(mg/L)",
    "Solution_pH",
    "Temperature(K)",
    "Agitation_speed(rpm)",
    "Dosage(g/L)",
    "Contact_Time(min)",
    "E", "S", "A", "B", "V",
    "qe(mg/g)",
]


def _load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=SHEET if 'SHEET' in globals() else 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(
        f"Unsupported file type: {path.suffix}. Use .xlsx, .xls, or .csv"
    )


def _numeric_corr(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found to compute correlations.")
    return numeric_df.corr(method="pearson")


def _ensure_dirs(base_dir: Path) -> Tuple[Path, Path]:
    figures_dir = base_dir / "figures"
    results_dir = base_dir / "results"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, results_dir


def _save_corr_excel(corr: pd.DataFrame, out_path: Path) -> None:
    with pd.ExcelWriter(out_path) as writer:
        corr.to_excel(writer, sheet_name="pearson_correlation")


def _plot_heatmap(
    corr: pd.DataFrame,
    title: str,
    out_path: Path,
    annot: bool = False,
    cmap: str = "coolwarm",
) -> None:
    plt.figure(figsize=(max(8, corr.shape[1] * 0.5), max(6, corr.shape[0] * 0.5)))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        vmin=-1,
        vmax=1,
        cmap=cmap,
        annot=annot,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": 0.75},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _select_target_columns(df: pd.DataFrame, target_cols: Optional[List[str]]) -> pd.DataFrame:
    if target_cols is None or len(target_cols) == 0:
        return df.select_dtypes(include=[np.number])
    present = [c for c in target_cols if c in df.columns]
    if len(present) == 0:
        raise ValueError("None of TARGET_COLS are present in the dataframe.")
    return df[present].select_dtypes(include=[np.number])


def run(raw_path: Optional[str], enriched_path: Optional[str]) -> None:
    base_dir = Path(__file__).parent
    figures_dir, results_dir = _ensure_dirs(base_dir)
    combined_xlsx_path = results_dir / "pearson_corr_combined.xlsx"

    # Hold correlations to write into a single Excel file
    corr_sheets: List[tuple[str, pd.DataFrame]] = []

    if raw_path:
        print(f"[pearson] Loading RAW data: {raw_path}")
        raw_df = _load_dataframe(raw_path)
        raw_df = _select_target_columns(raw_df, TARGET_COLS)
        raw_corr = _numeric_corr(raw_df)
        raw_stem = Path(raw_path).stem
        corr_sheets.append(("enriched", raw_corr))
        _plot_heatmap(
            raw_corr,
            title="Pearson Correlation (RAW)",
            out_path=figures_dir / f"pearson_heatmap_{raw_stem}.png",
        )
        print(f"[pearson] RAW heatmap saved to {figures_dir/(f'pearson_heatmap_{raw_stem}.png')}")

    if enriched_path:
        print(f"[pearson] Loading ENRICHED data: {enriched_path}")
        enr_df = _load_dataframe(enriched_path)
        enr_df = _select_target_columns(enr_df, TARGET_COLS)
        enr_corr = _numeric_corr(enr_df)
        enr_stem = Path(enriched_path).stem
        corr_sheets.append(("imputed", enr_corr))
        _plot_heatmap(
            enr_corr,
            title="Pearson Correlation (ENRICHED)",
            out_path=figures_dir / f"pearson_heatmap_{enr_stem}.png",
        )
        print(f"[pearson] ENRICHED heatmap saved to {figures_dir/(f'pearson_heatmap_{enr_stem}.png')}")

    if not raw_path and not enriched_path:
        raise SystemExit(
            "Provide at least one input via --raw and/or --enriched"
        )

    # Write combined Excel if any correlations computed
    if corr_sheets:
        with pd.ExcelWriter(combined_xlsx_path) as writer:
            for sheet_name, corr_df in corr_sheets:
                corr_df.to_excel(writer, sheet_name=sheet_name)
        print(f"[pearson] Combined Excel saved to {combined_xlsx_path}")


if __name__ == "__main__":
    # Use inline configuration only
    run(RAW_PATH, IMPUTED_PATH)


