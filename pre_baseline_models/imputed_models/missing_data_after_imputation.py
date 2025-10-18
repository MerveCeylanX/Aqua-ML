"""
missing_data_after_imputation.py
--------------------
İmputasyon sonrası eksik veri durumunu yatay bar grafik olarak görselleştirir.
- Yalnızca "gerçek veri" sütunlarına bakar (flag/yardımcı kolonlar otomatik hariç tutulur).
- Çıktı: ./figures/missing_data_after_imputation.png

Kullanım (opsiyonel argümanlarla):
    python missing_data_plot.py --in ./data/Raw_data_imputed.xlsx --sheet 0 --out ./figures/missing_data_after_imputation.png
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_IN_PATH = r"D:\Aqua_ML\pre_baseline_models\imputed_models\data\Raw_data_imputed.xlsx"
DEFAULT_OUT_PNG = r"D:\Aqua_ML\pre_baseline_models\imputed_models\/figures/missing_data_after_imputation.png"

EXCLUDE_SUBSTRINGS = [
    "_flag", "impute", "imputed", "impute_method",
    "__", "pharm_code_norm"
]

def is_real_data_col(col_name: str, extra_excludes=None) -> bool:
    """Flag/yardımcı kolonları isim kalıbına göre hariç tut."""
    if extra_excludes is None:
        extra_excludes = []
    lowers = col_name.lower()
    for key in EXCLUDE_SUBSTRINGS + [s.lower() for s in extra_excludes]:
        if key in lowers:
            return False
    return True

def main(in_path: Path, out_png: Path, sheet: int, extra_excludes: list):
    # Girdi dosyasını doğrula
    if not in_path.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {in_path}")

    # Çıktı klasörü
    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Veriyi yükle
    df = pd.read_excel(in_path, sheet_name=sheet)

    # Yalnızca gerçek veri kolonları
    real_cols = [c for c in df.columns if is_real_data_col(c, extra_excludes)]

    if not real_cols:
        # Görsel: hiç gerçek veri kolonu yoksa bilgilendir
        plt.figure(figsize=(6, 2.5))
        plt.text(0.5, 0.5, "Görselleştirilecek gerçek veri kolonu bulunamadı.",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Bilgi] Uygun kolon yok. Bilgilendirici görsel kaydedildi: {out_png}")
        return

    # Eksik sayım ve yüzde
    missing_count = df[real_cols].isna().sum()
    missing_pct = (missing_count / len(df)) * 100.0

    # Sadece eksik kalanları göster
    md = (
        pd.DataFrame({
            "column": real_cols,
            "missing_count": missing_count.values,
            "missing_pct": missing_pct.values
        })
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=True)
    )

    if md.empty:
        # Eksik yoksa bilgilendirici görsel üret
        plt.figure(figsize=(6, 2.5))
        plt.text(0.5, 0.5, "Eksik veri bulunmuyor (imputasyon sonrası).",
                 ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] Eksik veri yok. Bilgilendirici görsel kaydedildi: {out_png}")
        return

    # Yatay bar grafik
    plt.figure(figsize=(10, max(3, 0.35 * len(md))))
    plt.barh(md["column"], md["missing_pct"], color="#0014E7")
    plt.xlabel("Eksik (%)")
    plt.ylabel("Kolon")
    plt.title("Missing Data After Imputation")

    # Çubuk etiketleri (adet + yüzde)
    for i, (pct, cnt) in enumerate(zip(md["missing_pct"].to_list(),
                                       md["missing_count"].to_list())):
        # Etiketi çubuğun biraz sağına yaz
        plt.text(pct + 0.5, i, f"{cnt} ({pct:.1f}%)", va="center")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Eksik veri grafiği kaydedildi: {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imputasyon sonrası eksik veri grafiği üretir.")
    parser.add_argument("--in", dest="in_path", type=str, default=str(DEFAULT_IN_PATH),
                        help="Girdi Excel yolu (varsayılan: ./data/Raw_data_imputed.xlsx)")
    parser.add_argument("--sheet", dest="sheet", type=int, default=0,
                        help="Sayfa numarası (varsayılan: 0)")
    parser.add_argument("--out", dest="out_png", type=str, default=str(DEFAULT_OUT_PNG),
                        help="Çıktı PNG yolu (varsayılan: ./figures/missing_data_after_imputation.png)")
    parser.add_argument("--exclude", dest="extra_excludes", type=str, nargs="*", default=[],
                        help="İsimde geçtiğinde hariç tutulacak ek anahtarlar (opsiyonel)")

    args = parser.parse_args()
    main(Path(args.in_path), Path(args.out_png), args.sheet, args.extra_excludes)
