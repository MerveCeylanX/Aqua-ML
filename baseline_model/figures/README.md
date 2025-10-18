# Figures Directory

Bu klasör, baseline model için oluşturulan tüm görselleştirmeleri içerir. Grafikler model performansı, SHAP analizi ve out-of-fold değerlendirmelerini gösterir.

## Görselleştirme Türleri

### Model Karşılaştırma Grafikleri

#### `ml_results.png`
- **Açıklama:** Tüm modellerin karşılaştırma grafiği
- **İçerik:** 6 farklı ML algoritmasının performans karşılaştırması
- **Grafik Türü:** Scatter plot (Predicted vs Actual)
- **Modeller:** CatBoost, XGBoost, LightGBM, Random Forest, SVR, Neural Network
- **Metrikler:** R2, RMSE, MAE (Train/Test)
- **Kullanım:** En iyi modelin seçilmesi için

### Out-of-Fold Analiz Grafikleri

#### `oof_error_bins__PRE__*.png`
- **Açıklama:** Pre-processing sonrası OOF hata dağılımı
- **İçerik:** Model tahminlerinin gerçek değerlerden sapma yüzdesi
- **Grafik Türü:** Horizontal bar chart
- **Hata Kategorileri:** 0-5%, 5-10%, 10-20%, 20-50%, >50%
- **Kullanım:** Model güvenilirliği değerlendirmesi

#### `oof_error_bins__POST__*.png`
- **Açıklama:** Post-processing sonrası OOF hata dağılımı
- **İçerik:** Final model tahminlerinin hata dağılımı
- **Grafik Türü:** Horizontal bar chart
- **Hata Kategorileri:** 0-5%, 5-10%, 10-20%, 20-50%, >50%
- **Kullanım:** Production model performansı

### SHAP Analiz Grafikleri

#### `shap_importance_*.png`
- **Açıklama:** SHAP özellik önem grafikleri
- **İçerik:** Her özelliğin model kararlarına katkısı
- **Grafik Türü:** Horizontal bar chart
- **Özellikler:** 30 özellik (28 sayısal + 3 kategorik OHE)
- **Kullanım:** Model interpretability, feature selection

## Grafik Özellikleri

### Teknik Detaylar
- **Çözünürlük:** 200 DPI
- **Format:** PNG
- **Boyut:** Optimize edilmiş (12x10 inch)
- **Renk Paleti:** Tutarlı ve profesyonel
- **Font:** Okunabilir ve net

### Görsel Standartlar
- **Başlık:** Açıklayıcı ve bilgilendirici
- **Eksen Etiketleri:** Birimler ve açıklamalar
- **Legend:** Gerekli durumlarda
- **Grid:** Alpha=0.3 ile hafif
- **Layout:** Tight layout ile optimize

## Kullanım Senaryoları

### Araştırma ve Analiz
- Model performans karşılaştırması
- Özellik önem analizi
- Hata dağılımı incelemesi
- Model güvenilirliği değerlendirmesi

### Raporlama
- Akademik makaleler
- Teknik raporlar
- Sunumlar
- Dokümantasyon

### Model Geliştirme
- Feature selection
- Hyperparameter tuning
- Model validation
- Performance monitoring

## Grafik Güncelleme

### Otomatik Güncelleme
- Model yeniden eğitildiğinde grafikler otomatik güncellenir
- Yeni model performansı grafiklere yansır
- SHAP analizi yeni model için çalıştırılır

### Manuel Güncelleme
- `main.py` çalıştırılarak tüm grafikler yenilenir
- `baseline_shap_analysis.py` ile SHAP grafikleri güncellenir
- Eski grafikler üzerine yazılır

## Dosya Organizasyonu

### Naming Convention
- **Model karşılaştırma:** `ml_results.png`
- **OOF analizi:** `oof_error_bins__{PHASE}__{MODEL}.png`
- **SHAP analizi:** `shap_importance_{MODEL}_baseline_final.png`

### Version Control
- Grafikler timestamp ile versiyonlanır
- Eski versiyonlar arşivlenir
- En güncel grafikler aktif kullanımda

## Kalite Kontrol

### Görsel Kalite
- Yüksek çözünürlük
- Net ve okunabilir
- Profesyonel görünüm
- Tutarlı stil

### İçerik Kalitesi
- Doğru veri gösterimi
- Anlamlı başlıklar
- Uygun eksen etiketleri
- Açıklayıcı legend'lar

## Export ve Paylaşım

### Desteklenen Formatlar
- **PNG:** Web ve dokümantasyon için
- **PDF:** Yüksek kalite yazdırma için
- **SVG:** Vektör grafikler için

### Paylaşım
- Grafikler doğrudan raporlarda kullanılabilir
- Web sitelerinde gösterilebilir
- Sunumlarda kullanılabilir
- Akademik yayınlarda referans alınabilir
