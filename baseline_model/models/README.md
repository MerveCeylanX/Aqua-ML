# Models Directory

Bu klasör, eğitilmiş baseline modellerini ve metadata dosyalarını içerir. Her model joblib formatında saklanır ve production kullanımı için hazırdır.

## Model Dosyaları

### `best_model__*.joblib`
- **Format:** Joblib pickle dosyası
- **İçerik:** Eğitilmiş ML modeli (pipeline veya direkt model)
- **Kullanım:** Production ortamında tahmin için
- **Örnek:** `best_model__CatBoost__20251016_212905.joblib`

### `best_model__*.meta.json`
- **Format:** JSON metadata dosyası
- **İçerik:** Model bilgileri ve performans metrikleri
- **İçerik:**
  - Model adı ve tipi
  - En iyi hiperparametreler
  - Cross-validation sonuçları
  - Train/test performansı
  - Feature listesi
  - Hedef değişken bilgisi
- **Örnek:** `best_model__CatBoost__20251016_212905.meta.json`

## Model Versiyonlama

Model dosyaları timestamp ile versiyonlanır:
- **Format:** `best_model__{ModelType}__{YYYYMMDD_HHMMSS}.joblib`
- **Örnek:** `best_model__CatBoost__20251016_212905.joblib`
- **Amaç:** Model geçmişini takip etmek

## Model Türleri

### CatBoost
- **Algoritma:** CatBoostRegressor
- **Özellikler:** Kategorik veri desteği, overfitting koruması
- **Kullanım:** Kategorik özellikler içeren veri setleri

### XGBoost
- **Algoritma:** XGBRegressor
- **Özellikler:** Gradient boosting, hızlı eğitim
- **Kullanım:** Büyük veri setleri, hızlı tahmin

### LightGBM
- **Algoritma:** LGBMRegressor
- **Özellikler:** Hafif, hızlı, düşük bellek kullanımı
- **Kullanım:** Büyük veri setleri, kaynak kısıtlı ortamlar

### Random Forest
- **Algoritma:** RandomForestRegressor
- **Özellikler:** Ensemble, robust, interpretable
- **Kullanım:** Kararlı tahminler, özellik önem analizi

### Support Vector Regression
- **Algoritma:** SVR
- **Özellikler:** Non-linear patterns, kernel trick
- **Kullanım:** Karmaşık pattern'ler, küçük veri setleri

### Neural Network
- **Algoritma:** MLPRegressor
- **Özellikler:** Deep learning, non-linear mapping
- **Kullanım:** Karmaşık non-linear ilişkiler

## Model Kullanımı

### Python'da Yükleme
```python
import joblib
import json

# Model yükleme
model = joblib.load('best_model__CatBoost__20251016_212905.joblib')

# Metadata yükleme
with open('best_model__CatBoost__20251016_212905.meta.json', 'r') as f:
    metadata = json.load(f)

# Tahmin yapma
predictions = model.predict(X_test)
```

### Model Bilgileri
```python
# Model tipi
print(f"Model: {metadata['best_name']}")

# Parametreler
print(f"Parametreler: {metadata['best_params']}")

# Performans
print(f"CV R2: {metadata['cv_r2']}")
print(f"Test R2: {metadata['test_r2']}")
```

## Model Seçimi

En iyi model seçimi şu kriterlere göre yapılır:
1. **Cross-validation R2 skoru**
2. **Test set performansı**
3. **Model kararlılığı**
4. **Overfitting durumu**
5. **Computational efficiency**

## Production Kullanımı

### Gereksinimler
- Python 3.8+
- Gerekli ML kütüphaneleri (requirements.txt)
- Aynı veri preprocessing pipeline'ı

### Deployment
1. Model dosyasını production sunucusuna kopyalayın
2. Metadata dosyasını da kopyalayın
3. Gerekli Python bağımlılıklarını yükleyin
4. Model yükleme kodunu test edin
5. Tahmin API'sini oluşturun

## Model Güncelleme

Yeni model eğitildiğinde:
1. Eski model dosyaları arşivlenir
2. Yeni model dosyaları oluşturulur
3. Metadata güncellenir
4. Production ortamı güncellenir
5. Performans karşılaştırması yapılır

## Güvenlik

- Model dosyaları güvenli şekilde saklanır
- Metadata dosyaları model bütünlüğünü sağlar
- Version control ile model geçmişi takip edilir
- Backup stratejisi uygulanır
