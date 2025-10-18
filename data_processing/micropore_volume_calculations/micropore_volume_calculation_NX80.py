"""
micropore_volume_calculation_NX80.py
--------------------------------

Dosya ismindeki NX80, verilerin alındığı makalenin referans numarasıdır. 

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

raw = """5.618466898954702; 0.04746049376319572
5.879790940766545; 0.053222899369195484
6.010452961672472; 0.055732354960299456
6.141114982578399; 0.007963000142481497
6.533101045296169; 0.004524824164864036
7.055749128919862; 0.007871196715153558
7.186411149825782; 0.02246243669287462
7.447735191637628; 0.030455325570249853
7.970383275261325; 0.023950397005297712
8.231707317073171; 0.019025070269290047
8.754355400696864; 0.026553598953408537
9.015679442508707; 0.03519704545160161
9.538327526132408; 0.005179105734233122
10.0609756097561; 0.00016116601686462506
11.498257839721253; 0.00007001023276297091
11.890243902439027; 0.006111388158491243
12.54355400696864; 0.01484825719207801
13.066202090592332; 0.018287566545341492
13.719512195121952; 0.015964956025024922
14.503484320557492; 0.015315369869046531
15.156794425087108; 0.01336450656062587
15.810104529616728; 0.014201747341424549
16.46341463414634; 0.01578248254601504
17.247386759581886; 0.020523230962527353
18.16202090592335; 0.028795739802857392
18.94599303135888; 0.025265112754685688
19.72996515679442; 0.0154147831042835
20.644599303135884; 0.006865730606323575
21.559233449477347; 0.006123369558177785
22.47386759581881; 0.0067750605546416515
23.64982578397212; 0.007798822584614576
24.695121951219505; 0.0061272554175355876
25.871080139372822; 0.005756965402898842
27.177700348432047; 0.006873826146652329
28.484320557491287; 0.009012991723119562
29.529616724738673; 0.006040309314404871
31.097560975609753; 0.006228125850031727
32.404181184668985; 0.004556882504565873
33.972125435540065; 0.0050235094491146715
35.40940766550522; 0.006326405709622687
37.10801393728222; 0.0061426369441601875
38.93728222996515; 0.005866093286530302
40.76655052264808; 0.004753118402134618
42.59581881533101; 0.0035472067147649666
44.55574912891986; 0.003363761770915616
46.64634146341463; 0.0025299211170550368
48.73693379790941; 0.0013243332512985206
51.08885017421603; 0.001234310842842895
53.31010452961673; 0.0013300001295286495
55.792682926829265; 0.0011472028289056108
58.40592334494773; 0.0009645674390891396
61.019163763066196; 0.0009678056552206438
63.76306620209059; 0.0008782689791847348
66.76829268292683; 0.0009749297307099364
69.77351916376307; 0.0006998432703392288"""

# --- Çalıştır ve hesaplama sonuçlarını yazdır ---
res_signed  = integrate_psd_angstrom(raw, clamp_zero=False)   # negatifleri koru
res_clamped = integrate_psd_angstrom(raw, clamp_zero=True)    # negatifleri 0’a kırp

print_psd_results("SIGNED",  res_signed)
print_psd_results("CLAMPED", res_clamped)

# (opsiyonel) makaledeki toplam hacme hizalama
target_total = 0.528
scale = target_total / res_clamped["V_total_cm3_g"]
res_scaled = integrate_psd_angstrom(raw, clamp_zero=False, y_scale=scale)
print_psd_results("SCALED_to_0.528", res_scaled)
