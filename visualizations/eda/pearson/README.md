Pearson Correlations
====================

Amaç
----
Ham (`RAW`) ve zenginleştirilmiş/impute edilmiş (`ENRICHED`) veri setleri için Pearson korelasyon matrislerini hesaplamak, ısı haritaları ve Excel çıktıları üretmek.

Yapılandırma
------------
Yapılandırma değişkenleri doğrudan `pearson_correlations.py` içinde tanımlıdır:
- `RAW_PATH`: Normalizasyon/temizleme sonrası veri yolu
- `IMPUTED_PATH`: Impute/zenginleştirilmiş veri yolu
- `SHEET`: Excel sayfa indeksi
- `TARGET_COLS`: Korelasyona dahil edilecek kolon listesi (boş bırakılırsa sayısal tüm kolonlar)

Çalıştırma
----------
```
python pearson_correlations.py
```

Çıktılar
--------
- Isı haritaları: `figures/pearson_heatmap_<GIRDI_DOSYA_ADI>.png`
- Excel sonuçları: TEK dosya `results/pearson_corr_combined.xlsx`
  - Sayfa adları: `enriched` (RAW için), `imputed` (ENRICHED için)

Notlar
-----
- Sadece sayısal sütunlar korelasyona dahil edilir.
- Çok boyutlu veri setlerinde ısı haritaları üst üçgen maske ile çizilir.


