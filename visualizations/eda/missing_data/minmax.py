import os
import pandas as pd

# === AYARLAR ===
PATH  = r"D:\Aqua_ML\data\Raw_data.xlsx"   # Veri dosyasının yolu
SHEET = 0                                    # Kullanılacak sayfa (ilk sheet)

# === VERİYİ YÜKLE ===
df = pd.read_excel(PATH, sheet_name=SHEET)

# === SADECE SAYISAL SÜTUNLAR ===
num_df = df.select_dtypes(include=['number'])

# === MIN, MAX, MEAN, SOFTMIN (P5), SOFTMAX (P95) HESAPLAMA ===
stats = pd.DataFrame({
    "Column": num_df.columns,
    "Min": num_df.min().values,
    "Max": num_df.max().values,
    "Mean": num_df.mean().values,
    "SoftMin (P5)": num_df.quantile(0.05).values,
    "SoftMax (P95)": num_df.quantile(0.95).values
})

# === ÇIKTI DİZİNİNİ OLUŞTUR ===
out_dir = r"D:\Aqua_ML\results"
os.makedirs(out_dir, exist_ok=True)

# === EXCEL'E KAYDETME ===
out_path = os.path.join(out_dir, "numeric_stats.xlsx")
stats.to_excel(out_path, index=False)

print(f"Hesaplanan istatistikler kaydedildi: {out_path}")
print(stats.head())
