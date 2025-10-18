"""
main.py
-------

Tüm akışı uçtan uca çalıştırır:
1) Veriyi yükle → özellik zenginleştir (E,S,A,B,V + elemental oranlar)
2) Zenginleştirilmiş veriyi Excel'e kaydet
3) ML hazırlık: numerik/kategorik listeleri, tip dönüşümleri, train/test ayrımı
4) Preprocess'ler: 'passthrough' ve OHE
5) Model havuzu (Cat/LGBM/XGB + HGBR/EBM) kur → 5-fold CV + Train/Test metrikleri
6) Grafik ve özetleri kaydet + (HPO sonrası OOF ve model kaydı)
"""

import sys, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Proje modülleri
from src.data_io import load_data, save_enriched_excel
from src.features import add_pharm_features, clean_pharm_features, add_elemental_ratios
from src.config import RANDOM_STATE, TEST_SIZE, N_JOBS, OUT_DATA
from src.pipelines import build_model_pool
from src.evaluation import evaluate_and_plot, export_oof_with_pharma
from src.preprocessing import prepare_ml_data
from src.tunning import run_hpo_top2   # <-- HPO fonksiyonunu ayrı dosyadan çağırıyoruz

def main():
    # A) Veri: Yükle & zenginleştir
    df = load_data()
    df = add_pharm_features(df)
    df = clean_pharm_features(df)
    df = add_elemental_ratios(df)
    save_enriched_excel(df)

    # B + C) ML hazırlık + preprocess
    prep = prepare_ml_data(df, target_col="qe(mg/g)")

    target_col = "qe(mg/g)"
    phar_col = "Target_Phar"

    # Hedefi sayıya çevir, NaN hedefleri at
    y = pd.to_numeric(df[target_col], errors="coerce")
    X = df.copy()
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask]

    # Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # HPO + OOF için tüm veri
    X_all = pd.concat([X_train, X_test], axis=0)
    y_all = pd.concat([y_train, y_test], axis=0)
    df_meta_all = df.loc[X_all.index, [phar_col]] if phar_col in df.columns else None

    # Preprocess bilgileri
    pre_ohe   = prep["preprocessor"]
    num_feats = prep["num_feats"]
    cat_feats = prep["cat_feats"]

    # OOF analizinde farmasötik adı (train tarafı için)
    df_meta = df.loc[X_train.index, [phar_col]] if phar_col in df.columns else None

    # Native modeller için passthrough (gerekirse)
    pre_passthrough = FunctionTransformer(validate=False)

    # D) Model havuzu & değerlendirme
    models = build_model_pool(
        pre_ohe=pre_ohe,
        num_feats=num_feats,
        cat_feats=cat_feats,
    )

    ret = evaluate_and_plot(
        models=models,
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        df_meta=df_meta
    )

    # === E) TOP-2 HPO (tüm veri) + KAZANANI TÜM VERİYLE FIT ET + OOF + KAYDET ===
    from joblib import dump
    from datetime import datetime
    import json
    import numpy as np

    artifacts_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(artifacts_dir, exist_ok=True)

    res_sorted = ret.get("results_sorted", None)
    if res_sorted is None or len(res_sorted) == 0:
        raise RuntimeError("Model sonuçları boş; HPO/kaydetme yapılamıyor.")

    # Top-2 üzerinde HPO (ayrı modülden)
    best_name, best_pipe, best_score, hp_results, best_best_params = run_hpo_top2(
        models=models,
        res_sorted=res_sorted,
        X_all=X_all,
        y_all=y_all,
        random_state=RANDOM_STATE,
        n_iter=30
    )
    print(f"[HPO] Kazanan: {best_name} (CV R2={best_score:.4f})")

    # HPO kazananını TÜM veriyle yeniden fit et (nihai model)
    best_pipe.fit(X_all, y_all)

    # HPO SONRASI OOF (tüm veri) → Excel + PNG
    export_oof_with_pharma(
        best_pipe,
        X_train=X_all,
        y_train=y_all,
        out_data_path=OUT_DATA,
        cv=5,
        n_jobs=N_JOBS,
        df_meta=df_meta_all,
        phase="post",              
        tag=str(best_name)   
    )

    # Modeli kaydet
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_fname = f"best_model__{best_name or 'model'}__{ts}.joblib"
    model_path  = os.path.join(artifacts_dir, model_fname)
    dump(best_pipe, model_path)
    print(f"[OK] En iyi model kaydedildi: {model_path}")

    # Meta yaz
    meta = {
        "best_name": best_name,
        "features": num_feats + cat_feats,
        "target": target_col,
        "saved_at": ts,
        "hpo_top2": res_sorted.head(2)["model"].tolist(),
        "hpo_metric": "r2",
        "hpo_cv": 5,
        "oof_from": "post-HPO on X_all",
        "best_params": best_best_params,
        "hp_results": hp_results
    }
    with open(model_path.replace(".joblib", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
