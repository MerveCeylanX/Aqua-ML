"""
config.py
----------

Proje ayarlarını ve opsiyonel kütüphane kontrollerini içerir. 
Tüm sabitler ve model seçenekleri burada toplanmıştır.

"""

# -------------------- KULLANICI AYARLARI --------------------
IN_PATH   = r"D:\Aqua_ML\data\Raw_data.xlsx"     # ham veri
OUT_DATA  = r"D:\Aqua_ML\baseline_model\data\Raw_data_enriched.xlsx"
OUT_DIR   = r"D:\Aqua_ML\baseline_model\data"                   # çıktı klasörü

RANDOM_STATE = 42  # rastgelelik sabiti (reprodüksiyon için)
TEST_SIZE    = 0.2 # test verisi oranı

# -------------------- PAKET KONTROL (opsiyonel modeller) --------------------
have = {}

try:
    from catboost import CatBoostRegressor
    have["catboost"] = True
except Exception:
    have["catboost"] = False

try:
    from lightgbm import LGBMRegressor
    have["lightgbm"] = True
except Exception:
    have["lightgbm"] = False

try:
    from xgboost import XGBRegressor
    have["xgboost"] = True
except Exception:
    have["xgboost"] = False

try:
    from interpret.glassbox import ExplainableBoostingRegressor
    have["ebm"] = True
except Exception:
    have["ebm"] = False

# -------------------- CPU / loky fix --------------------
import os
N_JOBS = max(1, (os.cpu_count() or 1) - 1)
os.environ["LOKY_MAX_CPU_COUNT"] = str(N_JOBS)
