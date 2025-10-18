"""
micropore_volume_calculation_NO37.py
--------------------------------

Dosya ismindeki NO37, verilerin alındığı makalenin referans numarasıdır. 

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

raw = """0.8228595176320859; -0.0002942577034854288
0.7504783563588919; 0.005305873898625968
0.6794824804689181; 0.010033275709309404
0.7403195968819531; 0.011705892369097741
0.6675921142629537; 0.017524206419066136
1.0767130641069667; 0.039778008017339155
1.2230915529337736; 0.027559560056450376
0.9746637075431668; 0.024069102652532905
1.9980432843962017; 0.019339969235120446
2.2002949503461746; 0.011921419686636903
2.582287394768814; 0.011266179700374315
2.393426820856625; 0.010248341265054153
2.711349816305386; 0.009956854132335158
2.586443250918471; 0.008647990326090406
2.149616593410082; 0.003848784556376786
3.1601822804683426; 0.007192401709673044
2.716660076941059; 0.006611389931861271
3.2894755829021154; 0.005737621176395889
3.1629528512347793; 0.005446942126817109
3.3546994363620097; 0.004646593496662334
3.611554434500535; 0.0028279446693929775
3.930862715332511; 0.0016637277452460214
4.436491880207447; 0.0031173538740371515
4.751644304889771; 0.004571326324174091
5.322381882775998; 0.00500665225585066
5.638226950149926; 0.006024259810273616
6.27268765566423; 0.0063140153362635995
7.035056378229088; 0.006021720120404382
8.295666076958375; 0.011837609920952154
9.878470067734687; 0.014671095731878012
12.847713846216008; 0.024047515288644405
13.515883162721975; 0.003100845889887127
14.02162776804551; 0.004481744536059269
14.657127437597227; 0.004116952718478263
15.357850960608838; 0.0026611332211636946
16.118603516893273; 0.0033870227619704485
16.755719352725407; 0.002004046187723478
17.58342736919876; 0.0005479958095117171
18.40697952952246; 0.0017101348055838517
19.294601138820028; 0.0025085209481124013
20.056623540039077; 0.0024344081801101827
20.94135913812162; 0.005050981388113676
22.081910770305264; 0.006503453112418796
22.84612654004774; 0.005047518174655624
23.987832576717402; 0.005772715072770766
24.998398263775655; 0.009116332226067031
26.204750951662206; 0.0091141388575436
27.28966028761989; 0.005621257204205497
28.496359296852233; 0.005400881387825074
29.76227925621719; 0.007871306987898952
31.28713214179551; 0.007213989073561536
32.56194101570279; 0.004084398511972613
34.02191636916702; 0.004299925829511768
35.66971333250602; 0.00618783892594206
37.194219896738524; 0.005748703459461636
39.03918914628903; 0.0034180762426442787
40.49985714244487; 0.003197238664469443
42.52906934796349; 0.0047935491877321434
44.37092170540176; 0.004426564001627706
46.59580547130007; 0.002749791485689705
48.437196066943926; 0.002673716230061275"""

def summarize_incremental_psd(raw_text, clamp_zero=True, y_scale=1.0):
    """
    raw_text: 'width_nm; dV_cm3_g' satırları (incremental volume)
    clamp_zero: Negatif dV değerlerini 0'a kırpar.
    y_scale: dV değerlerine global ölçek (örn. single-point toplamına eşitlemek için).
    """
    w, dv = [], []
    for line in raw_text.splitlines():
        if ";" not in line:
            continue
        a, b = [p.strip() for p in line.split(";")]
        w.append(float(a.replace(",", ".")))
        dv.append(float(b.replace(",", ".")))
    w = np.array(w, float)
    dv = np.array(dv, float) * float(y_scale)

    # İsteğe bağlı negatif kırpma
    if clamp_zero:
        dv = np.maximum(dv, 0.0)

    # Güvenlik: genişliğe göre sırala
    order = np.argsort(w)
    w, dv = w[order], dv[order]

    # Bölgeler (pore width, nm): mikro<2, meso 2–50, makro>50
    micro = dv[w < 2.0].sum()
    meso  = dv[(w >= 2.0) & (w <= 50.0)].sum()
    macro = dv[w > 50.0].sum()
    total = dv.sum()

    print(f"V_total  = {total:.6f}  cm^3/g")
    print(f"V_micro  = {micro:.6f}  cm^3/g")
    print(f"V_meso   = {meso:.6f}  cm^3/g")
    print(f"V_macro  = {macro:.6f}  cm^3/g")
    print(f"Delta    = {total - (micro+meso+macro):.6e}")  # kontrol

    return {"V_total": total, "V_micro": micro, "V_meso": meso, "V_macro": macro}

# Çalıştır
summarize_incremental_psd(raw, clamp_zero=True)
