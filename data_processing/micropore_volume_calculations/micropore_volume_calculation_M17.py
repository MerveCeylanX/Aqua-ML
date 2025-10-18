"""
micropore_volume_calculation_M17.py
--------------------------------

Dosya ismindeki M17, verilerin alındığı makalenin referans numarasıdır. 

Amaç
-----
Veri alınan makalede sayısal tablo verilmediğinde, **arttırımlı gözenek boyutu dağılımı (PSD) grafiğinden**
sayısal değerleri okuyup (örn. WebPlotDigitizer yardımıyla) elde edilen **(R_Å ; dV/dR)** çiftlerini
kullanarak mikropor/mezopor/makropor hacim katkılarını **sayısal integrasyon** ile hesaplar.

Grafikten Alınan Veri (girdi)
--------------------------------------
- **X (R_Å)** : Gözenek *yarıçapı* (Ångström).
- **Y (dV/dR)** : Artırımlı hacim yoğunluğu (**cm³ g⁻¹ Å⁻¹**). Yani R ekseninde birim
  yarıçap başına hacim katkısı.


Hesaplama Özeti
---------------
1) Veriler parse edilir, sayısal dizilere çevrilir ve **R artan** olacak şekilde sıralanır.
2) (İsteğe bağlı) Negatif küçük salınımlar ölçüm hatası kabul edilip **0'a kırpılabilir**.
3) **Sınır noktaları** (10 Å ve 250 Å) ızgaraya *lineer enterpolasyon* ile dahil edilir ki
   bölgesel integrasyon doğru ayrılsın.
4) Trapez yöntemiyle integrasyon:
   - **Mikro**: R < 10 Å
   - **Mezo** : 10 Å ≤ R ≤ 250 Å
   - **Makro**: R > 250 Å
5) Toplam hacim (**V_total**) ve hacim-ağırlıklı **ortalama gözenek çapı (nm)** hesaplanır:
   D_nm = 2 * R_Å / 10

Çıktılar
--------
- V_micro, V_meso, V_macro, V_total : cm³/g
- d_avg_nm : hacim-ağırlıklı ortalama **çap** (nm)

"""


import numpy as np

# ---- Grafikten alınan veriler (Pore radius Å ; dV/dR cm^3 Å^-1 g^-1) ----
raw = """10.504845055573892; 0.0060000831055844585
11.529066771775444; 0.0043180616928370115
12.614577669292013; 0.0032201656887909567
13.81343508622798; 0.0024493970096236264
15.016053624471983; 0.0020213082718549885
16.498619260075365; 0.0020995106268298074
19.37049181884882; 0.003759010426189182
21.859328364410864; 0.0005196734662864392
23.7804920325489; 0.0005590298966403973
26.301981949467056; 0.000294777882517196
30.746003214998893; 0.0001944931865292968
34.4775484327474; 0.00017972413694850552
38.647691683030544; 0.00012610916274685344
46.05564749935296; 0.00007319464989991725"""

# ---- Basit trapez integratörü (uyumluluk için) ----
def _trapz_like(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    if y.size < 2: return 0.0
    return float(np.sum(0.5*(y[1:]+y[:-1])*(x[1:]-x[:-1])))

def integrate_psd_radius_angstrom(raw_text, clamp_zero=True):
    # Parse
    R_A, y = [], []
    for line in raw_text.splitlines():
        if ";" not in line: 
            continue
        a, b = [p.strip() for p in line.split(";")]
        R_A.append(float(a.replace(",", ".")))
        y.append(float(b.replace(",", ".")))
    R_A = np.array(R_A, float)
    y    = np.array(y,   float)

    # Sırala
    o = np.argsort(R_A); R_A, y = R_A[o], y[o]
    if clamp_zero: y = np.maximum(y, 0.0)   # negatif lobları kırp

    # Sınır noktalarını ekle (mikro: R < 10 Å; meso: 10–250 Å; makro: >250 Å)
    def insert_boundary(X, Y, b):
        if not (X.min() < b < X.max()): 
            return X, Y
        if np.any(np.isclose(X, b)): 
            return X, Y
        j = np.searchsorted(X, b); i = j-1
        t = (b - X[i])/(X[i+1]-X[i])
        yb = Y[i] + t*(Y[i+1]-Y[i])   # dV/dR lineer interpolasyon
        return np.insert(X, j, b), np.insert(Y, j, yb)

    R2, y2 = R_A.copy(), y.copy()
    for b in (10.0, 250.0):
        R2, y2 = insert_boundary(R2, y2, b)

    # Maskeler (sınırları iki tarafa da dahil et – trapez için güvenli)
    micro = R2 <= 10.0
    meso  = (R2 >= 10.0) & (R2 <= 250.0)
    macro = R2 >= 250.0

    # Aynı ızgarada integrasyon
    V_micro = _trapz_like(y2[micro], R2[micro]) if np.any(micro) else 0.0
    V_meso  = _trapz_like(y2[meso ], R2[meso ]) if np.any(meso ) else 0.0
    V_macro = _trapz_like(y2[macro], R2[macro]) if np.any(macro) else 0.0
    V_total = _trapz_like(y2, R2)

    # Hacim-ağırlıklı ortalama gözenek çapı (nm) – yarıçaptan (Å) çap (nm)
    D_nm2 = (2.0*R2) / 10.0
    num = _trapz_like(D_nm2 * y2, R2) if V_total > 0 else 0.0
    d_avg_nm = num / V_total if V_total > 0 else np.nan

    return V_micro, V_meso, V_macro, V_total, d_avg_nm

# ---- Çalıştır ve hesaplama sonuçlarını yazdır ----
V_micro, V_meso, V_macro, V_total, d_avg_nm = integrate_psd_radius_angstrom(raw, clamp_zero=True)
print(f"V_micro  = {V_micro:.6f} cm^3/g")
print(f"V_meso   = {V_meso:.6f} cm^3/g")
print(f"V_macro  = {V_macro:.6f} cm^3/g")
print(f"V_total  = {V_total:.6f} cm^3/g")
print(f"d_avg_nm = {d_avg_nm:.6f} nm")
print(f"Delta    = {V_total - (V_micro+V_meso+V_macro):.6e}")
