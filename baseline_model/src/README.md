# Source Code Modules

Bu klasör, baseline model için gerekli tüm kaynak kod modüllerini içerir. Her modül belirli bir işlevselliği yerine getirir ve main.py tarafından kullanılır.

## Modüller

### `config.py`
- **Amaç:** Proje konfigürasyonu ve sabitler
- **İçerik:** Veri yolları, model parametreleri, CV ayarları
- **Kullanım:** Diğer modüller tarafından import edilir

### `data_io.py`
- **Amaç:** Veri giriş/çıkış işlemleri
- **İçerik:** Excel okuma/yazma, veri yükleme fonksiyonları
- **Kullanım:** Veri yükleme ve kaydetme işlemleri

### `estimators.py`
- **Amaç:** ML algoritmalarını sklearn uyumlu hale getiren sarmalayıcılar
- **İçerik:** CatBoost, LightGBM, XGBoost wrapper sınıfları
- **Kullanım:** Pipeline'larda model olarak kullanılır

### `evaluation.py`
- **Amaç:** Model değerlendirme ve görselleştirme
- **İçerik:** CV metrikleri, grafik oluşturma, OOF analizi
- **Kullanım:** Model performans değerlendirmesi

### `features.py`
- **Amaç:** Özellik mühendisliği ve dönüşümler
- **İçerik:** Domain-specific özellik hesaplamaları
- **Kullanım:** Veri zenginleştirme işlemleri

### `imports.py`
- **Amaç:** Import yönetimi ve bağımlılık kontrolü
- **İçerik:** Gerekli kütüphanelerin import edilmesi
- **Kullanım:** Diğer modüller tarafından kullanılır

### `pipelines.py`
- **Amaç:** ML pipeline'larının tanımlanması
- **İçerik:** Preprocessing ve model pipeline'ları
- **Kullanım:** Model eğitimi ve tahmin işlemleri

### `preprocessing.py`
- **Amaç:** Veri ön işleme işlemleri
- **İçerik:** Eksik veri doldurma, scaling, encoding
- **Kullanım:** Veri hazırlık aşamasında

### `tunning.py`
- **Amaç:** Hiperparametre optimizasyonu
- **İçerik:** RandomizedSearchCV, parametre arama
- **Kullanım:** En iyi parametrelerin bulunması

## Modül İlişkileri

```
main.py
├── config.py (konfigürasyon)
├── data_io.py (veri yükleme)
├── preprocessing.py (veri hazırlık)
├── features.py (özellik mühendisliği)
├── pipelines.py (pipeline tanımları)
├── estimators.py (model sarmalayıcıları)
├── tunning.py (HPO)
└── evaluation.py (değerlendirme)
```

## Kullanım

Bu modüller doğrudan çalıştırılmaz. `main.py` tarafından import edilerek kullanılır:

```python
from src.config import *
from src.data_io import load_data
from src.preprocessing import preprocess_data
# ... diğer importlar
```

## Geliştirme

Yeni özellik eklerken:
1. İlgili modülü düzenleyin
2. `__init__.py` dosyasını güncelleyin
3. `main.py`'de gerekli import'ları ekleyin
4. Test edin ve dokümantasyonu güncelleyin
