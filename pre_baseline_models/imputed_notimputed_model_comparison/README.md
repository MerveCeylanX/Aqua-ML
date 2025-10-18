# İmputed vs Not-İmputed Model Comparison

Bu klasör, impute edilmiş ve impute edilmemiş veri ile eğitilmiş modellerin kapsamlı karşılaştırmasını içerir.

## Yapı

```
imputed_notimputed_model_comparison/
├── figures/                          # Karşılaştırma grafikleri
│   ├── predicted_vs_actual_comparison.png  # Predicted vs Actual
│   ├── oof_pct_error_buckets_RF_vs_XGB.png  # OOF error distribution
│   ├── shap_importance_comparison.png  # SHAP importance
│   └── comprehensive_model_comparison.png  # Kapsamlı karşılaştırma
└── README.md                         # Bu dosya
```

## Kod Dosyaları

### 1. `predicted_vs_actual_comparison.py`
- **Karşılaştırma:** RF (imputed) vs XGBoost (not-imputed)
- **Grafik:** Yan yana 2 subplot
- **İçerik:** Train/Test noktaları, metrik kutuları
- **Çıktı:** `figures/predicted_vs_actual_comparison.png`

### 2. `oof_comparison.py`
- **Analiz:** Out-of-fold yüzde hata dağılımı
- **Grafik:** Yan yana yatay bar chart
- **Aralıklar:** 0–5%, 5–10%, 10–15%, 15–20%, 20–30%, 30–50%, 50–100%, 100%+
- **Çıktı:** `figures/oof_pct_error_buckets_RF_vs_XGB.png`

### 3. `shap_comparison.py`
- **Analiz:** SHAP importance karşılaştırması
- **Grafik:** Yan yana 2 subplot (individual bar format)
- **Özellikler:** Temizlenmiş feature isimleri, ilk 30 özellik
- **Çıktı:** `figures/shap_importance_comparison.png`

### 4. `comprehensive_comparison.py`
- **Kapsamlı:** Tüm karşılaştırmalar tek grafikte
- **Grafik:** 3 subplot (2x2 grid)
  - Sol üst: RF Predicted vs Actual
  - Sağ üst: XGBoost Predicted vs Actual  
  - Alt: OOF Yüzde Hata Dağılımı (tam genişlik)
- **Çıktı:** `figures/comprehensive_model_comparison.png`

## Model Karşılaştırması

### RF (İmputed Veri)
- **Veri:** `Raw_data_imputed.xlsx`
- **Preprocessing:** Pipeline (StandardScaler + OneHotEncoder)
- **Parametreler:** HPO ile optimize edilmiş
- **Renk:** Sarı (`#FECB04`)

### XGBoost (Not-İmputed Enriched Veri)
- **Veri:** `Raw_data.xlsx` + enrich işlemi
- **Preprocessing:** Direkt model (kendi NaN handling)
- **Parametreler:** HPO ile optimize edilmiş
- **Renk:** Koyu mavi (`#0014E7`)

## Enrich İşlemi (XGBoost için)

### 1. PHARM Özellikleri
- İlaç kodlarına göre E, S, A, B, V değerleri eklenir
- 21 farklı ilaç için önceden tanımlanmış değerler

### 2. Elemental Molar Oranları
- C_molar, H_C_molar, O_C_molar, N_C_molar, S_C_molar
- Atom ağırlıkları kullanılarak hesaplanır

### 3. One-Hot Encoding
- Activation_Atmosphere kategorik verisi işlenir
- Air, N2, CO2 seviyeleri ayrı feature'lar olur

## Grafik Özellikleri

### Predicted vs Actual
- **Legend:** Train/Test (kısaltma yok)
- **Metrik kutuları:** R2, RMSE, MAE (Train/Test/CV)
- **Ortak eksen limitleri:** Her iki model için aynı

### OOF Error Distribution
- **Yüzde aralıkları:** 8 farklı hata seviyesi
- **Yan yana bar:** RF (sarı) vs XGBoost (koyu mavi)
- **Yüzde değerleri:** Her bar üzerinde gösterilir

### SHAP Importance
- **Individual bar format:** Kategorik seviyeler ayrı
- **Temizlenmiş isimler:** Prefix'ler ve birimler kaldırılır
- **İlk 30 özellik:** En önemli feature'lar

## Çalıştırma

```bash
cd imputed_notimputed_model_comparison
python predicted_vs_actual_comparison.py
python oof_comparison.py
python shap_comparison.py
python comprehensive_comparison.py
```

## Çıktılar

### Grafikler
- **Predicted vs Actual:** Model performans karşılaştırması
- **OOF Distribution:** Hata dağılımı analizi
- **SHAP Importance:** Özellik önem karşılaştırması
- **Comprehensive:** Tüm analizler tek grafikte

### Console Çıktıları
- **Model metrikleri:** R2, RMSE, MAE karşılaştırması
- **SHAP importance:** En önemli 5 özellik listesi
- **OOF metrikleri:** Cross-validation sonuçları

## Notlar

- Her iki model de aynı HPO parametrelerini kullanır
- Veri preprocessing yöntemleri farklıdır
- SHAP analizi aynı yöntemle hesaplanır
- Grafikler tutarlı renk şeması kullanır
- Cross-validation ile robust karşılaştırma yapılır
