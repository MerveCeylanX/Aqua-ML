"""
micropore_volume_calculation_NX75.py
--------------------------------

Dosya ismindeki NX75, verilerin alındığı makalenin referans numarasıdır. 

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

def integrate_psd_angstrom(raw_text, clamp_zero=True, y_scale=1.0):
    """
    raw_text: 'D_A; y' satırları. D_A [Å], y [cm^3 Å^-1 g^-1] = dV/dD (doğrusal çap)
    clamp_zero: Negatif dV/dD loblarını 0'a kırpar (opsiyonel)
    y_scale: Grafikte global x10^-k çarpanı varsa burada uygula (örn. 1e-3)
    Döndürür: toplam ve alt-bölge hacimleri (cm^3/g) + hacim-ağırlıklı ortalama çap (nm)
    """
    # --- Parse ---
    D_A, y = [], []
    for line in raw_text.splitlines():
        if ";" not in line:
            continue
        a, b = [p.strip() for p in line.split(";")]
        D_A.append(float(a.replace(",", ".")))
        y.append(float(b.replace(",", ".")))
    D_A = np.array(D_A, dtype=float)
    y    = np.array(y, dtype=float) * float(y_scale)

    # --- Sırala ---
    order = np.argsort(D_A)
    D_A, y = D_A[order], y[order]

    # --- Negatifleri kırp (opsiyonel) ---
    if clamp_zero:
        y = np.maximum(y, 0.0)

    # --- Bölge sınırlarında ara nokta ekle (20 Å=2 nm, 500 Å=50 nm) ---
    def insert_boundary(D, Y, bA):
        if not (D.min() < bA < D.max()):
            return D, Y
        if np.any(np.isclose(D, bA)):
            return D, Y
        j = np.searchsorted(D, bA)
        i = j - 1
        t  = (bA - D[i]) / (D[i+1] - D[i])
        yb = Y[i] + t * (Y[i+1] - Y[i])
        return np.insert(D, j, bA), np.insert(Y, j, yb)

    D2, y2 = D_A.copy(), y.copy()
    for bA in (20.0, 500.0):
        D2, y2 = insert_boundary(D2, y2, bA)

    # --- Maskeler (Å) --          # > 50 nm
    micro = D2 <= 20.0
    meso  = (D2 >= 20.0) & (D2 <= 500.0)
    macro = D2 >= 500.0

    # --- Aynı ızgarada entegrasyon (numpy.trapezoid) ---
    V_micro = np.trapezoid(y2[micro], D2[micro]) if np.any(micro) else 0.0
    V_meso  = np.trapezoid(y2[meso ], D2[meso ]) if np.any(meso ) else 0.0
    V_macro = np.trapezoid(y2[macro], D2[macro]) if np.any(macro) else 0.0
    V_total = np.trapezoid(y2, D2)  # <<< D2,y2 üzerinde GLOBAL

    # --- Hacim-ağırlıklı ortalama gözenek çapı (nm) ---
    D_nm2 = D2 / 10.0
    num = np.trapezoid(D_nm2 * y2, D2) if V_total > 0 else 0.0
    d_avg_nm = num / V_total if V_total > 0 else np.nan

    return {
        "V_total_cm3_g": float(V_total),
        "V_micro_cm3_g": float(V_micro),
        "V_meso_cm3_g": float(V_meso),
        "V_macro_cm3_g": float(V_macro),
        "V_sum_cm3_g": float(V_micro + V_meso + V_macro),
        "avg_pore_diameter_nm": float(d_avg_nm),
    }

def print_psd_results(label, res):
    print(f"[{label}]")
    print(f"V_total  = {res['V_total_cm3_g']:.6f}  cm^3/g")
    print(f"V_micro  = {res['V_micro_cm3_g']:.6f}  cm^3/g")
    print(f"V_meso   = {res['V_meso_cm3_g']:.6f}  cm^3/g")
    print(f"V_macro  = {res['V_macro_cm3_g']:.6f}  cm^3/g")
    print(f"V_sum    = {res['V_sum_cm3_g']:.6f}  cm^3/g")
    print(f"d_avg_nm = {res['avg_pore_diameter_nm']:.6f}  nm")
    print(f"Delta (V_total - V_sum) = {res['V_total_cm3_g'] - res['V_sum_cm3_g']:.6e}\n")

# --- Grafikten alınan veriler ---
raw = """11.15485564304462; 0.07789207941047169
11.679790026246721; 0.07112660668660198
12.204724409448822; 0.07404604275211867
12.729658792650923; 0.004793007839194216
13.38582677165354; 0.0047938783771431515
14.041994750656173; 0.011560918069320927
14.82939632545932; 0.033187170011708726
15.223097112860899; 0.04539333080877327
16.272965879265094; 0.035046464963023896
16.797900262467195; 0.025494922587412885
17.585301837270343; 0.038895635557993045
18.37270341207349; 0.05853183774913707
19.291338582677163; 0.028284300283360084
20.209973753280842; 0.015416530645112142
20.997375328083997; -0.00010481276905065229
22.0472440944882; 0.013163578433292841
23.097112860892388; 0.028422019386880118
24.14698162729659; 0.020463213242623263
25.26246719160104; 0.025970497468910905
26.37795275590552; 0.02643632232538097
27.42782152230972; 0.020202225965535392
29.002624671916017; 0.019540965339531563
30.183727034120743; 0.014899082888270798
31.62729658792651; 0.012778278337098395
33.07086614173229; 0.014637573288413558
34.51443569553806; 0.01172074883674365
36.22047244094488; 0.011059662318329597
37.926509186351716; 0.01039857579991553
39.76377952755907; 0.009074313472010015
41.6010498687664; 0.003239271707951921
43.569553805774284; 0.0015171735373874229
45.40682414698163; 0.0012542710768119053
47.50656167979003; 0.0008590468479997004
49.73753280839896; 0.0007293366936097917
51.968503937007874; 0.00073229652263615
54.46194225721786; 0.000602934583425821
56.955380577427825; 0.00047357264421547807
59.58005249343834; 0.00034438481259492504
62.204724409448836; 0.00034786696439062514
65.09186351706037; 0.00021902734794965184
68.24146981627297; 0.00022320593010449752
71.25984251968507; 0.00022721040466955333
74.6719160104987; 0.00023173720200396486
78.08398950131235; 0.00023626399933837638
81.62729658792654; 0.00024096490426257777
85.43307086614175; 0.000246014024366345
89.37007874015748; 0.00011856726864364897
93.56955380577429; 0.00012413871151677192
97.90026246719162; 0.00012988426197967085
102.36220472440945; 0.00013580392003235964"""


# --- Çalıştır ve hesaplama sonuçlarını yazdır ---
res_signed  = integrate_psd_angstrom(raw, clamp_zero=False)   # negatifleri koru
res_clamped = integrate_psd_angstrom(raw, clamp_zero=True)    # negatifleri 0’a kırp

print_psd_results("SIGNED",  res_signed)
print_psd_results("CLAMPED", res_clamped)

# (opsiyonel) makaledeki toplam hacme hizalama
target_total = 0.514
scale = target_total / res_clamped["V_total_cm3_g"]
res_scaled = integrate_psd_angstrom(raw, clamp_zero=False, y_scale=scale)
print_psd_results("SCALED_to_0.514", res_scaled)
