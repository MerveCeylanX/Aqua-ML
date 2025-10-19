# Feature Selection - Özellik Seçimi

Bu klasör, adsorpsiyon kapasitesi (`qe(mg/g)`) tahmininde en önemli özellikleri belirlemek için Lasso ve Elastic Net regresyon yöntemlerini kullanarak özellik seçimi yapar.

## 📄 Betikler

### 1. `lasso_feature_selection_with_plots.py`
Lasso regresyon ile özellik seçimi yapar ve gereksiz özellikleri tamamen elemek için katsayıları sıfıra çeker.

### 2. `elasticnet_feature_selection_with_plots.py`
Elastic Net (Lasso + Ridge) ile daha dengeli özellik seçimi yapar.

## 🚀 Kullanım

```bash
# Lasso ile özellik seçimi
python lasso_feature_selection_with_plots.py

# Elastic Net ile özellik seçimi
python elasticnet_feature_selection_with_plots.py
```

## ⚙️ Ayarlar

Betik başındaki parametreleri düzenleyin:

```python
PATH = r"D:\Aqua_ML\data\Raw_data_enriched.xlsx"   # Veri dosyası yolu
TARGET = "qe(mg/g)"                                 # Hedef değişken
TOP_K = 30                                          # Grafiklerde gösterilecek özellik sayısı
```

## 📊 Çıktılar

- **Konsol**: Seçilen ve elenmiş özelliklerin listesi, model performans metrikleri (R², MAE)
- **Görseller**: `figures/` klasörüne kaydedilir
  - Katsayı grafikleri (bar, lollipop)
  - Gerçek vs Tahmin dağılımı
  - Residual histogramı

## 📦 Gereksinimler

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

