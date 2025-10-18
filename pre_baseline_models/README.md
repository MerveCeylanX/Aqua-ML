# Pre-Baseline Models

Bu klasör, makine öğrenmesi modellerinin hiperparametre optimizasyonu (HPO) ve SHAP analizi sonuçlarını içerir.

## Amaç

Bu çalışma, eksik veri ile çalışan ve çalışmayan modellerin performansını değerlendirmek ve en uygun stratejiyi seçmek için yapılmıştır. Elde edilen sonuçlara göre baseline model ve stratejisi belirlenmiştir. Yapılan araştırmanın detaylı açıklamaları proje dökümanında bulunmaktadır.

## Araştırma Kapsamı

- **İmpute edilmiş veri:** Eksik değerler doldurulmuş veri ile model performansı
- **İmpute edilmemiş veri:** Ham veri + enrich işlemi ile model performansı
- **Karşılaştırma:** Her iki yaklaşımın detaylı analizi ve karşılaştırması
- **SHAP analizi:** Model kararlarının açıklanabilirliği
- **Baseline belirleme:** En iyi stratejinin seçimi

## Yapı

```
pre_baseline_models/
├── imputed_models/                    # İmpute edilmiş veri ile modeller
│   ├── data/                         # İmpute edilmiş veri dosyaları
│   ├── figures/                      # Grafik çıktıları
│   ├── results/                      # Excel sonuçları
│   └── README.md                     # Detaylı açıklamalar
├── not_imputed_models/               # İmpute edilmemiş veri ile modeller
│   ├── data/                         # Ham ve enriched veri dosyaları
│   ├── figures/                      # Grafik çıktıları
│   ├── results/                      # Excel sonuçları
│   └── README.md                     # Detaylı açıklamalar
├── imputed_notimputed_model_comparison/  # Model karşılaştırmaları
│   ├── figures/                      # Karşılaştırma grafikleri
│   └── README.md                     # Detaylı açıklamalar
└── README.md                         # Bu dosya
```

## İçerik

### 1. İmputed Models (`imputed_models/`)
- **Veri:** İmpute edilmiş veri (`Raw_data_imputed.xlsx`)
- **Modeller:** RandomForest, XGBoost, CatBoost
- **Analiz:** HPO + SHAP importance
- **Özellik:** Kategorik veriler One-Hot Encoding ile işlenir

### 2. Not-Imputed Models (`not_imputed_models/`)
- **Veri:** Ham veri + enrich işlemi (PHARM özellikleri, elemental oranlar)
- **Modeller:** XGBoost, CatBoost
- **Analiz:** HPO + SHAP importance
- **Özellik:** XGBoost'un kendi NaN handling'i kullanılır

### 3. Model Comparison (`imputed_notimputed_model_comparison/`)
- **Karşılaştırma:** RF (imputed) vs XGBoost (not-imputed)
- **Grafikler:** Predicted vs Actual, OOF error distribution
- **Analiz:** Kapsamlı performans karşılaştırması

## Kullanım

Her alt klasörde bulunan Python kod dosyalarını çalıştırarak:
- Model eğitimi ve HPO
- SHAP analizi
- Performans metrikleri
- Görselleştirmeler

üretilebilir.

## Çıktılar

- **Grafikler:** `figures/` klasörlerinde PNG formatında
- **Sonuçlar:** `results/` klasörlerinde Excel formatında
- **Metrikler:** Console çıktısı olarak

## Notlar

- Tüm modeller HPO ile optimize edilmiş parametreler kullanır
- SHAP analizi TreeExplainer ile yapılır
- Kategorik veriler için One-Hot Encoding uygulanır
- Cross-validation ile robust değerlendirme yapılır
