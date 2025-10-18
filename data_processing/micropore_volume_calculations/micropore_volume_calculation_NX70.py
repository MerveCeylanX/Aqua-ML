"""
micropore_volume_calculation_NX70.py
--------------------------------

Dosya ismindeki NX70, verilerin alındığı makalenin referans numarasıdır. 

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

# --- Grafikten alınan veri ---
raw = """11,280052727729757; 0,09695172760506457
11,7975326430962; 0,09088031953320917
12,462567072482809; 0,06861987542480064
13,054566470784994; 0,004875533660440537
13,412022476580301; 0,007910544952170945
14,132278514835608; 0,017015974681189205
14,355836963642812; 0,0439971736036751
15,614058352812638; 0,05866692265214147
15,94182532158175; 0,04483856091299726
16,974706919722824; 0,03151530865712525
17,710401257231755; 0,04938969236219834
18,451142731035137; 0,07013084948252007
19,084409890803723; 0,02982659623107571
20,651991045786435; 0,02021269226125559
20,983023808628428; 0,008239301555507603
22,083002630448686; 0,033027272349412246
23,135775883397304; 0,031002480024226245
24,525222816222882; 0,02020833786915842
25,06259438639687; 0,025435389727989033
26,293204971132358; 0,024422201857742007
27,859004783893493; 0,013796495506069414
28,739284731719973; 0,01379550587150187
30,134075691210267; 0,006036770861991475
31,362311152983654; 0,003674513149274486
33,12465239085816; 0,004684336261991884
33,827391897268015; 0,0038403759027941103
35,40981757076382; 0,0026581584484113158
36,81648414506459; 0,001644772651250792
38,40128494152249; 0,0018116250393379746
39,810029748415104; 0,001978675354338627
40,68941902513079; 0,0014717845288448606
42,09608559943156; 0,000458398731684323
44,38451657341011; 0,00028719195149996823
45,61661160999691; 0,00011717273279666929
46,84900353695396; 0,00011578724440211574
48,25715456310604; -0,00005442990121468094
50,017714458758974; -0,000056409170349755455
51,778274354411906; -0,000058388439484843846
53,53883425006482; -0,00006036770861991836
56,35573008310952; -0,00006353453923604868
59,17262591615422; -0,000066701369852179
62,1655777387642; -0,00007006612738180706
65,68640063969978; -0,00024265839596071537
67,62301652491801; -0,0002448355920093015
71,32048919615944; -0,00008035832688421674
75,01796186740086; 0,00008411893824084027
78,01091369001085; 0,00008075418071121221
81,88473924118782; 0,00041366724923153075
85,93313633007877; -0,00009678626070537544
89,45455301175491; 0,00006788893133319318
93,5032469910161; -0,0002739308482949676
97,37677565182281; -0,00010965151008339447
101,95423138052044; -0,0001147976098345993
106,53198399958833; 0,000048690020722927385
111,63731080661157; -0,00012568359007752994
116,91899049357036; -0,00013162139748276735
122,02461419096386; -0,00013736127797450703
127,83475873698879; 0,000024740864188466105
133,99671837177405; 0,00001781342221569837
139,98232512662375; -0,00015754982315230315"""

# --- Çalıştır ve hesaplama sonuçlarını yazdır ---
res_signed  = integrate_psd_angstrom(raw, clamp_zero=False)   # negatifleri koru
res_clamped = integrate_psd_angstrom(raw, clamp_zero=True)    # negatifleri 0’a kırp

print_psd_results("SIGNED",  res_signed)
print_psd_results("CLAMPED", res_clamped)

# (opsiyonel) makaledeki toplam hacme hizalama
target_total = 0.520
scale = target_total / res_clamped["V_total_cm3_g"]
res_scaled = integrate_psd_angstrom(raw, clamp_zero=False, y_scale=scale)
print_psd_results("SCALED_to_0.520", res_scaled)
