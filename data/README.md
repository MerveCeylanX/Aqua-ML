Data
====

Genel Bakış
-----------
Bu klasör, analiz ve modelleme süreçlerinde kullanılan veri dosyalarını içerir.

İçerik
------
- `Raw_data_0.xlsx`: Normalizasyon öncesi ham veri (özellikle soaking time/temperature alanları işlenmeden önceki sürüm)
- `Raw_data.xlsx`: Normalizasyon/temizleme sonrası referans ham veri sürümü
- `Raw_data_imputed.xlsx`: Eksik değer doldurulmuş/impute edilmiş veri
- `Raw_data_enriched.xlsx`: Zenginleştirilmiş/özellik eklenmiş veri

Ön İşleme Notları
------------------
- Proje dokümanında belirtildiği üzere, 0 değerli "soaking time(min)" ve "soaking temperature(K)" alanları normalizasyon öncesinde özel olarak ele alınır.
- Bu alanlara ilişkin uygulanan kurallar ve dönüşümler proje dokümanında detaylandırılmıştır; burada yalnızca sürecin varlığı not düşülmüştür.

Beklenen Yapı
-------------
- Ham veriler: `.xlsx`/`.csv` gibi orijinal kaynaklardan gelen dosyalar

İşlenmiş Verilerin Kaydı ve İsimlendirme
----------------------------------------
- Veri işlendiğinde, çalışılan dosyanın klasör yapısında ilgili `data/` klasörünün içine kaydedilir.
- İsimlendirme örnekleri: `Raw_data_imputed.xlsx` (eksik değer doldurulmuş), `Raw_data_enriched.xlsx` (zenginleştirilmiş/özellik eklenmiş).
 - İsteğe bağlı: İşlenmiş dosyalar `data/processed/` alt klasöründe versiyonlanabilir.


