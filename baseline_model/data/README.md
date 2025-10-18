# Data Directory

Bu klasör, baseline model için kullanılan veri dosyalarını ve üretilen Excel raporlarını içerir.

## Veri Dosyaları

### `Raw_data.xlsx`
- **Açıklama:** Ham veri dosyası
- **İçerik:** Normalizasyon öncesi orijinal veri
- **Kullanım:** Veri işleme ve zenginleştirme için temel kaynak

### `Raw_data_enriched.xlsx`
- **Açıklama:** Zenginleştirilmiş veri dosyası
- **İçerik:** Özellik mühendisliği uygulanmış veri
- **İçerik:**
  - PHARM özellikleri (E, S, A, B, V)
  - Elemental molar ratios (C_molar, H_C_molar, O_C_molar, N_C_molar, S_C_molar)
  - One-hot encoded kategorik değişkenler
- **Kullanım:** Model eğitimi için hazır veri


## Veri İşleme Süreci

1. **Ham Veri:** `Raw_data.xlsx` yüklenir
2. **Özellik Mühendisliği:** PHARM ve elemental özellikler eklenir
3. **Kategorik Encoding:** One-hot encoding uygulanır
4. **Zenginleştirilmiş Veri:** `Raw_data_enriched.xlsx` oluşturulur
5. **Model Eğitimi:** Zenginleştirilmiş veri kullanılır

## Veri Yapısı

### Sayısal Özellikler (28 adet)
- Agent/Sample(g/g)
- Soaking_Time(min), Soaking_Temp(K)
- Activation_Time(min), Activation_Temp(K), Activation_Heating_Rate (K/min)
- BET_Surface_Area(m2/g), Total_Pore_Volume(cm3/g), Micropore_Volume(cm3/g)
- Average_Pore_Diameter(nm), pHpzc
- C_molar, H_C_molar, O_C_molar, N_C_molar, S_C_molar
- Initial_Concentration(mg/L), Solution_pH, Temperature(K)
- Agitation_speed(rpm), Dosage(g/L), Contact_Time(min)
- E, S, A, B, V (PHARM özellikleri)

### Kategorik Özellikler (3 adet, OHE sonrası)
- Activation_Atmosphere_Air
- Activation_Atmosphere_N2
- Activation_Atmosphere_CO2

### Hedef Değişken
- qe(mg/g): Adsorpsiyon kapasitesi

## Kullanım

Bu veri dosyaları:
- Model eğitimi için kullanılır
- SHAP analizi için referans alınır
- Production ortamında tahmin için kullanılır

## Notlar

- Tüm veri dosyaları Excel formatındadır
- Veri bütünlüğü korunmuştur
- Eksik değerler uygun şekilde işlenmiştir
- Kategorik değişkenler sayısal forma dönüştürülmüştür
