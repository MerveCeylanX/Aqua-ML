# Baseline Model

Bu klasör, projenin en iyi performans gösteren baseline modelini içerir. Pre-baseline modellerin karşılaştırılması sonucunda seçilen en optimal model ve strateji burada uygulanmıştır. Bu model, Streamlit web uygulamasında da kullanılarak kullanıcılara interaktif tahmin deneyimi sunar.

## Amaç

Baseline model, tüm makine öğrenmesi deneylerinin sonucunda en iyi performansı gösteren modeldir. Bu model:
- En iyi hiperparametrelerle eğitilmiştir
- Cross-validation ile doğrulanmıştır
- Production-ready durumda joblib formatında saklanmıştır
- SHAP analizi ile açıklanabilirliği sağlanmıştır

## Araştırma Kapsamı

- **Model Seçimi:** 6 farklı ML algoritması test edilmiştir
- **Hiperparametre Optimizasyonu:** En iyi parametreler bulunmuştur
- **Cross-Validation:** 5-fold CV ile robust değerlendirme
- **Out-of-Fold Analizi:** Model güvenilirliği test edilmiştir
- **SHAP Analizi:** Model kararlarının açıklanabilirliği
- **Model Persistence:** Production kullanımı için kaydedilmiştir

## Yapı

```
baseline_model/
├── README.md                    # Bu dosya
├── main.py                      # Ana çalıştırma scripti
├── baseline_shap_analysis.py    # SHAP analizi scripti
├── requirements.txt             # Python bağımlılıkları
├── data/                        # Veri dosyaları
│   ├── Raw_data.xlsx           # Ham veri
│   └── Raw_data_enriched.xlsx  # Zenginleştirilmiş veri
├── models/                      # Eğitilmiş modeller
│   ├── best_model__*.joblib    # En iyi model (joblib format)
│   └── best_model__*.meta.json # Model metadata'sı
├── figures/                     # Görselleştirmeler
│   ├── ml_results.png          # Model karşılaştırma grafiği
│   ├── oof_error_bins__*.png   # OOF hata dağılım grafikleri
│   └── shap_importance_*.png   # SHAP önem grafikleri
└── src/                        # Kaynak kod modülleri
    ├── config.py               # Konfigürasyon
    ├── data_io.py              # Veri giriş/çıkış
    ├── estimators.py           # Model sarmalayıcıları
    ├── evaluation.py           # Model değerlendirme
    ├── features.py             # Özellik mühendisliği
    ├── imports.py              # Import yönetimi
    ├── pipelines.py            # ML pipeline'ları
    ├── preprocessing.py        # Veri ön işleme
    └── tunning.py              # Hiperparametre optimizasyonu
```

## Nasıl Çalıştırılır

### 1. Ana Model Eğitimi ve Değerlendirmesi
```bash
python main.py
```

### 2. SHAP Analizi
```bash
python baseline_shap_analysis.py
```

## Çıktılar

### Model Dosyaları
- **`models/best_model__*.joblib`:** En iyi eğitilmiş model
- **`models/best_model__*.meta.json`:** Model metadata'sı (parametreler, performans)

### Görselleştirmeler
- **`figures/ml_results.png`:** Tüm modellerin karşılaştırma grafiği
- **`figures/oof_error_bins__*.png`:** Out-of-fold hata dağılım grafikleri
- **`figures/shap_importance_*.png`:** SHAP özellik önem grafikleri

### Excel Raporları
- **`data/ML_Results.xlsx`:** Detaylı model performans raporu
- **`data/OOF_Results.xlsx`:** Out-of-fold analiz sonuçları

## Model Özellikleri

- **Algoritma:** En iyi performans gösteren algoritma (CatBoost, XGBoost, vs.)
- **Hiperparametreler:** HPO ile optimize edilmiş parametreler
- **Cross-Validation:** 5-fold CV ile doğrulanmış
- **Feature Engineering:** Domain-specific özellik mühendisliği
- **Categorical Handling:** One-hot encoding ile kategorik veri işleme
- **Missing Data:** Robust eksik veri yönetimi

## Kullanım

Bu baseline model, production ortamında kullanılmak üzere tasarlanmıştır. Model dosyası (`best_model__*.joblib`) ve metadata'sı (`best_model__*.meta.json`) ile birlikte kullanılmalıdır.

### Streamlit Uygulaması

Bu baseline model ile kaydedilen best model, Streamlit web uygulamasında da kullanılmıştır. Uygulama, kullanıcıların interaktif olarak model tahminlerini yapabilmesini sağlar.

## Gereksinimler

Tüm Python bağımlılıkları `requirements.txt` dosyasında listelenmiştir. Kurulum için:

```bash
pip install -r requirements.txt
```
