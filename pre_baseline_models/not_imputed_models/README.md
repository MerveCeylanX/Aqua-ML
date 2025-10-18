# Not-İmputed Models

Bu klasör, impute edilmemiş (ham) veri ile makine öğrenmesi modellerinin hiperparametre optimizasyonu (HPO) ve SHAP analizi sonuçlarını içerir.

## Yapı

```
not_imputed_models/
├── data/                             # Ham ve enriched veri dosyaları
│   └── Raw_data_enriched.xlsx       # Enriched veri dosyası
├── figures/                          # Grafik çıktıları
│   ├── ml_results_xgboost_best.png  # XGBoost ML sonuçları
│   ├── shap_importance_xgboost_best.png  # XGBoost SHAP importance
│   ├── shap_importance_catboost.png # CatBoost SHAP importance
│   └── ...                          # Diğer grafikler
├── results/                          # Excel sonuçları
│   └── ml_results.xlsx              # ML metrikleri ve parametreler
└── README.md                         # Bu dosya
```

## Kod Dosyaları

### 1. `not_imputed_xgboost_hyper_SHAP.py`
- **Model:** XGBoost Regressor
- **Veri:** Ham veri (`Raw_data.xlsx`) + enrich işlemi
- **İşlemler:**
  - PHARM özelliklerini ekleme (E, S, A, B, V)
  - Elemental molar oranları hesaplama
  - One-Hot Encoding (Activation_Atmosphere)
  - Hyperparameter Optimization (RandomizedSearchCV)
  - SHAP TreeExplainer analizi
- **Çıktılar:**
  - `figures/ml_results_xgboost_best.png`
  - `figures/shap_importance_xgboost_best.png`
  - `results/ml_results.xlsx`

### 2. `not_imputed_best_two_SHAP.py`
- **Modeller:** XGBoost ve CatBoost
- **Veri:** Enriched veri
- **İşlemler:**
  - En iyi parametrelerle model eğitimi
  - SHAP analizi (her iki model için)
  - Individual bar format grafikler
- **Çıktılar:**
  - `figures/shap_importance_xgboost.png`
  - `figures/shap_importance_catboost.png`

## Veri İşleme

### Enrich İşlemi
1. **PHARM Özellikleri:** İlaç kodlarına göre E, S, A, B, V değerleri eklenir
2. **Elemental Oranlar:** C_molar, H_C_molar, O_C_molar, N_C_molar, S_C_molar hesaplanır
3. **One-Hot Encoding:** Activation_Atmosphere kategorik verisi işlenir

### Preprocessing
- **XGBoost:** Kendi NaN handling'i kullanılır (pipeline yok)
- **CatBoost:** Kategorik verileri otomatik işler
- **Eksik değerler:** Model tarafından yönetilir

## Model Parametreleri

### XGBoost (HPO ile optimize edilmiş)
- `n_estimators`: 1754
- `max_depth`: 7
- `learning_rate`: 0.0979
- `subsample`: 0.7425
- `colsample_bytree`: 0.8534
- `gamma`: 1.6730
- `min_child_weight`: 4
- `reg_alpha`: 0.00198
- `reg_lambda`: 0.00308

### CatBoost (Default parametreler)
- Kategorik verileri otomatik işler
- NaN değerleri kendi yönetir

## SHAP Analizi

### TreeExplainer
- **Background data:** 1000 örnek
- **Feature perturbation:** "interventional"
- **Grafik formatı:** Individual bar (kategorik seviyeler ayrı)

### Özellik İsimleri
- Birimler kaldırılır: `(mg/g)`, `(min)`, `(K)`, vb.
- Uzun isimler kısaltılır: `Soaking_Time` → `Soak_Time`
- Prefix'ler temizlenir: `num__`, `cat__` kaldırılır

## Çalıştırma

```bash
cd not_imputed_models
python not_imputed_xgboost_hyper_SHAP.py
python not_imputed_best_two_SHAP.py
```

## Çıktılar

### Grafikler
- **ML Results:** Train/Test performans grafikleri
- **SHAP Importance:** Özellik önem sıralaması (individual bar format)
- **OOF Analysis:** Out-of-fold error dağılımı

### Metrikler
- **Train/Test:** R2, RMSE, MAE
- **Cross-Validation:** 5-fold CV sonuçları
- **SHAP:** Özellik importance değerleri

### Excel Dosyası
- **ML_Summary:** Genel performans metrikleri
- **Best_Params_XGB:** En iyi hiperparametreler
- **XGB_CV_Results:** Cross-validation detayları

## Notlar

- Ham veri enrich edilerek kullanılır
- XGBoost'un kendi NaN handling'i tercih edilir
- Kategorik veriler One-Hot Encoding ile işlenir
- SHAP analizi kategorik seviyeleri ayrı ayrı gösterir
- Cross-validation ile robust değerlendirme yapılır
