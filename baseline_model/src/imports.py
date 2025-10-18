"""
imports.py
-----------
Proje genelinde kullanılan kütüphaneler.
Analiz ve modelleme için temel paketleri burada toplanmıştır.

"""

# Sistem ve uyarı yönetimi
import os
import warnings
warnings.filterwarnings("ignore")

# Bilimsel hesaplama
import numpy as np
import pandas as pd

# Görselleştirme
import matplotlib.pyplot as plt

# Sklearn temel modüller
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

# Çoklu işlem desteği
from joblib import parallel_backend

