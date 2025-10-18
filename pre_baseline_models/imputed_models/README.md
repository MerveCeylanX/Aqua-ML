# İmputed Models

Bu klasör, impute edilmiş veri ile makine öğrenmesi modellerinin hiperparametre optimizasyonu (HPO) ve SHAP analizi sonuçlarını içerir.

## Yapı

```
imputed_models/
├── data/                             # İmpute edilmiş veri dosyaları
│   └── Raw_data_imputed.xlsx        # Ana veri dosyası
├── figures/                          # Grafik çıktıları
│   ├── rf_shap_grouped_bar.png      # RF SHAP grouped bar
│   ├── rf_shap_summary_transformed.png  # RF SHAP summary
│   └── ...                          # Diğer grafikler
├── results/                          # Excel sonuçları
│   └── ml_results.xlsx              # ML metrikleri ve parametreler
└── README.md                         # Bu dosya
```

## Kod Dosyaları

### 1. `impured_randomforest_hyper_and_SHAP.py`
- **Model:** RandomForest Regressor
- **Veri:** İmpute edilmiş veri (`Raw_data_imputed.xlsx`)
- **İşlemler:**
  - Hyperparameter Optimization (RandomizedSearchCV)
  - SHAP TreeExplainer analizi
  - Kategorik veriler için One-Hot Encoding
- **Çıktılar:**
  - `figures/rf_shap_grouped_bar.png`
  - `figures/rf_shap_summary_transformed.png`
  - `results/ml_results.xlsx`

## Veri İşleme

### Preprocessing
- **Numerik özellikler:** StandardScaler ile normalize edilir
- **Kategorik özellikler:** OneHotEncoder ile işlenir
- **Eksik değerler:** SimpleImputer ile doldurulur
- **Pipeline:** ColumnTransformer ile birleştirilir

### Özellikler
- **Numerik:** Agent/Sample, Soaking parametreleri, BET özellikleri, pH, molar oranlar, vb.
- **Kategorik:** Activation_Atmosphere (Air, N2, CO2)

## Model Parametreleri

### RandomForest (HPO ile optimize edilmiş)
- `n_estimators`: 200
- `max_depth`: None
- `min_samples_split`: 2
- `min_samples_leaf`: 1
- `max_features`: None

## SHAP Analizi

### TreeExplainer
- **Background data:** 1000 örnek
- **Feature perturbation:** "interventional"
- **Grafik türleri:**
  - Grouped bar: Kategorik seviyeler ana sütunda toplanır
  - Individual bar: Kategorik seviyeler ayrı ayrı gösterilir

## Çalıştırma

```bash
cd imputed_models
python impured_randomforest_hyper_and_SHAP.py
```

## Çıktılar

### Grafikler
- **SHAP Importance:** Özellik önem sıralaması
- **SHAP Summary:** Detaylı SHAP değerleri

### Metrikler
- **Train/Test:** R2, RMSE, MAE
- **Cross-Validation:** 5-fold CV sonuçları
- **SHAP:** Özellik importance değerleri

### Excel Dosyası
- **ML_Summary:** Genel performans metrikleri
- **Best_Params_RF:** En iyi hiperparametreler
- **RF_CV_Results:** Cross-validation detayları

## Notlar

- Tüm modeller impute edilmiş veri ile eğitilir
- Kategorik veriler güvenli şekilde One-Hot Encoding ile işlenir
- SHAP analizi kategorik seviyeleri ayrı ayrı gösterir
- Cross-validation ile robust değerlendirme yapılır
