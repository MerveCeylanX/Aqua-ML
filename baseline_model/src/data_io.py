"""
data_io.py
--------

Veri yükleme ve kaydetme yardımcıları.

"""

import os
import pandas as pd
from src.config import IN_PATH, OUT_DATA

def load_data() -> pd.DataFrame:
    """Excel dosyasını IN_PATH'ten okur ve çalışma kopyasını döndürür."""
    df_ml = pd.read_excel(IN_PATH)
    df = df_ml.copy()  # orijinali koru
    return df

def save_enriched_excel(df: pd.DataFrame) -> None:
    """
    Zenginleştirilmiş veri setini OUT_DATA yoluna Excel olarak kaydeder.
    Klasör yoksa oluşturur; hata olursa uyarı basar.
    """
    try:
        os.makedirs(os.path.dirname(OUT_DATA), exist_ok=True)
        df.to_excel(OUT_DATA, index=False)
        print(f"[OK] Zenginleştirilmiş veri kaydedildi: {OUT_DATA}")
    except Exception as e:
        print(f"[Uyarı] Excel kaydında sorun: {e}")

