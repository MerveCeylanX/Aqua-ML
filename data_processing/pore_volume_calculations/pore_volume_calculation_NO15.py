"""
pore_volume_calculation_NO15.py
--------------------------------

Dosya ismindeki NO15, verilerin alındığı makalenin referans numarasıdır. 

Amaç
-----
Bu kod, grafikten alınan (Pore diameter nm ; dV/dlog10D) verilerini
kullanarak mikro (<2 nm), mezo (2–50 nm) ve makro (>50 nm) gözenek
hacimlerini trapez yöntemiyle hesaplar.

"""

import numpy as np

# --- 1. VERİYİ YÜKLE ---
raw_data = """2.9314407807652185; 1.1904
3.15268521421618; 1.2672000000000003
3.302163605852876; 1.5244800000000003
3.4600843524318896; 1.60128
3.678855938523772; 2.15424
3.919749071175006; 1.73184
4.158493379387698; 0.8908800000000001
4.451781459736643; 0.6912
4.665925120532526; 0.6451200000000004
4.993169656535836; 0.6144000000000003
5.371406449542846; 0.5721600000000002
5.718169935407819; 0.5491199999999998
6.2811559627341635; 0.48384000000000027
6.617018469923614; 0.48384000000000027
7.421229733601797; 0.43776000000000037
8.024754377325051; 0.4147200000000004
8.814763163326894; 0.35328000000000026
9.989031751142205; 0.33408000000000015
11.381696621622812; 0.1996800000000003
13.167950451462914; 0.23424000000000023
15.39678085798729; 0.18816000000000033
18.671431346584466; 0.14207999999999998
24.10288339354869; 0.09984000000000037
33.81869296363272; 0.05760000000000032
60.29638569927142; 0.034559999999999924
232.4016000248342; 0.015360000000000262"""

# Parse et
D, y = [], []
for line in raw_data.splitlines():
    parts = line.strip().split(";")
    if len(parts) == 2:
        D.append(float(parts[0].replace(",", ".")))
        y.append(float(parts[1].replace(",", ".")))

def insert_boundary_sorted(D, y, boundary):
    """
    Boundary noktalarını (2 nm, 50 nm) log10(D) ekseninde lineer interpolasyonla ekler.
    """
    D = np.asarray(D)
    y = np.asarray(y)
    if not (D.min() < boundary < D.max()):
        return D, y
    if np.any(np.isclose(D, boundary)):
        return D, y

    j = np.searchsorted(D, boundary)
    i = j - 1
    x0, x1 = np.log10(D[i]), np.log10(D[i+1])
    y0, y1 = y[i], y[i+1]
    lb = np.log10(boundary)
    t = (lb - x0) / (x1 - x0)
    yb = y0 + t * (y1 - y0)

    D_new = np.insert(D, j, boundary)
    y_new = np.insert(y, j, yb)
    return D_new, y_new

# --- entegrasyon öncesi: sınır noktalarını ekle ---
D2, y2 = D.copy(), y.copy()
for b in [2.0, 50.0]:
    D2, y2 = insert_boundary_sorted(D2, y2, b)

logD2 = np.log10(D2)

# --- bölge maskeleri ---
mask_micro = D2 <= 2
mask_meso  = (D2 >= 2) & (D2 <= 50)
mask_macro = D2 > 50

# --- entegrasyon ---
V_total  = np.trapezoid(y,  np.log10(D))          # global
V_micro  = np.trapezoid(y2[mask_micro], logD2[mask_micro])
V_meso   = np.trapezoid(y2[mask_meso ], logD2[mask_meso ])
V_macro  = np.trapezoid(y2[mask_macro], logD2[mask_macro])
V_sum    = V_micro + V_meso + V_macro

print("Total (global):", V_total)
print("Micro:", V_micro)
print("Meso:", V_meso)
print("Macro:", V_macro)
print("Micro + Meso + Macro:", V_sum)
