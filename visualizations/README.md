Visualizations
==============

Genel Bakış
-----------
Bu depo bölümü, veri keşfi ve raporlama için üretilen görselleştirme kod dosyaları ile çıktıları içerir. Özellikle `eda/` klasörü altında dağılımlar, eksik veri, aykırı değerler, farmosötik/aktif karbon ilişkileri ve hazırlama koşullarına yönelik analizler bulunur.

Klasör Yapısı
-------------
- `eda/`: Keşifçi veri analizi (Exploratory Data Analysis) kod dosyaları ve figürler
  - `distributions/`
  - `imputation_diagnostics/`
  - `missing_data/`
  - `outliers/`
  - `pharma_relations/`
  - `preparation_conditions/`
  - `pearson/`

Çalıştırma
----------
Her alt klasördeki Python kod dosyaları doğrudan çalıştırılabilir. Örnek:

```
cd eda/histograms
python basic_histograms.py
```

Çıktılar
--------
Grafikler ilgili alt klasörlerin `figures/` dizinlerine PNG (ve gerekiyorsa diğer formatlarda) olarak kaydedilir.


