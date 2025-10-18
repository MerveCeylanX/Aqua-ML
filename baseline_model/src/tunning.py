# src/hpo.py
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold

def get_param_distributions(model_name: str):
    # ... (senin mevcut içeriğin aynen kalsın)
    # -- burada değişiklik yok --
    if model_name == "HistGBR":
        return {
            "reg__learning_rate": np.logspace(-3, -1, 15),
            "reg__max_iter": [300, 500, 800, 1000],
            "reg__max_depth": [None, 3, 5, 7, 9],
            "reg__l2_regularization": np.logspace(-8, -2, 8),
            "reg__min_samples_leaf": [5, 10, 15, 20, 30],
            "reg__max_leaf_nodes": [None, 31, 63, 127, 255],
        }
    elif model_name.startswith("LightGBM"):  # "LightGBM-GBDT" / "LightGBM-DART"
        return {
            "reg__num_leaves": [31, 63, 127, 255],
            "reg__learning_rate": np.logspace(-3, -1, 15),
            "reg__n_estimators": [600, 900, 1200, 1500],
            "reg__boosting_type": ["dart" if "DART" in model_name else "gbdt"],
        }
    elif model_name == "XGBoost-GBTree":
        return {
            "reg__n_estimators": [400, 700, 1000, 1500],
            "reg__max_depth": [3, 5, 7, 9],
            "reg__learning_rate": np.logspace(-3, -1, 15),
            "reg__subsample": np.linspace(0.6, 1.0, 5),
            "reg__colsample_bytree": np.linspace(0.6, 1.0, 5),
            "reg__reg_lambda": np.logspace(-6, -1, 10),
            "reg__booster": ["gbtree"],
        }
    elif model_name == "CatBoost":
        return {
            "reg__depth": [4, 6, 8, 10],
            "reg__learning_rate": np.logspace(-3, -1, 15),
            "reg__n_estimators": [600, 900, 1200, 1500],
        }
    elif model_name == "EBM":
        return {
            "reg__learning_rate": np.logspace(-3, -1, 12),
            "reg__max_leaves": [2, 3, 5, 7],
            "reg__interactions": [0, 2, 4],
            "reg__max_bins": [128, 256, 512],
        }
    else:
        return {}

def run_hpo_top2(models,
                 res_sorted,
                 X_all, y_all,
                 random_state=42,
                 n_iter=30,
                 out_data_path=None,      # <-- verildiyse HP_Tuning sayfasını buraya yazar
                 return_details=True):    # <-- True ise hp_results + best_params da döner
    """
    İlk CV sonuçlarından top-2 modeli seçip HPO yapar.
    Döndürür:
      - default: (best_name, best_pipe, best_score, hp_results, best_best_params)
      - return_details=False ise yalnız (best_name, best_pipe, best_score)
    Eğer out_data_path verilirse, HP_Tuning sayfasını Excel'e yazar.
    """
    import json
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    name2pipe = {name: pipe for (name, pipe) in models if pipe is not None}

    if res_sorted is None or len(res_sorted) == 0:
        raise RuntimeError("run_hpo_top2: results_sorted boş.")

    top2 = res_sorted.head(2)["model"].tolist()
    print("[HPO] Adaylar (top-2):", top2)

    best_name, best_pipe, best_score = None, None, -np.inf
    best_best_params = None
    hp_results = []  # her aday için: {"model", "cv_r2", "best_params"}

    for cand in top2:
        base_pipe = name2pipe.get(cand)
        if base_pipe is None:
            print(f"[HPO] {cand} bulunamadı, atlandı.")
            hp_results.append({"model": cand, "cv_r2": None, "best_params": None})
            continue

        param_dist = get_param_distributions(cand)
        if not param_dist:
            print(f"[HPO] {cand} için param dağılımı yok, atlandı.")
            hp_results.append({"model": cand, "cv_r2": None, "best_params": None})
            continue

        print(f"[HPO] Başlıyor → {cand}")
        rsearch = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            verbose=1,
            refit=True,
            random_state=random_state,
        )
        rsearch.fit(X_all, y_all)
        print(f"[HPO] {cand} en iyi (CV R2) = {rsearch.best_score_:.4f}")
        print(f"[HPO] {cand} en iyi paramlar: {rsearch.best_params_}")

        # kaydet
        hp_results.append({
            "model": cand,
            "cv_r2": float(rsearch.best_score_) if rsearch.best_score_ is not None else None,
            "best_params": rsearch.best_params_
        })

        if rsearch.best_score_ > best_score:
            best_score = rsearch.best_score_
            best_name = cand
            best_pipe = rsearch.best_estimator_
            best_best_params = rsearch.best_params_

    if best_pipe is None:
        raise RuntimeError("run_hpo_top2: HPO sonucunda uygun bir model çıkmadı.")

    # İsteğe bağlı: HP_Tuning sayfasını hemen burada yaz
    if out_data_path is not None:
        try:
            rows = []
            for r in hp_results:
                rows.append({
                    "model": r.get("model"),
                    "cv_r2": r.get("cv_r2"),
                    "best_params": json.dumps(r.get("best_params"), ensure_ascii=False, default=str)
                                   if r.get("best_params") is not None else None
                })
            if rows:
                df_hp = pd.DataFrame(rows)
                with pd.ExcelWriter(out_data_path, mode="a", if_sheet_exists="replace") as xw:
                    df_hp.to_excel(xw, sheet_name="HP_Tuning", index=False)
                print(f"[OK] HP sonuçları 'HP_Tuning' sayfasına yazıldı: {out_data_path}")
        except Exception as e:
            print(f"[Uyarı] HP sonuçlarını Excel'e yazarken sorun: {e}")

    # Geri dönüş: geriye dönük uyum + detaylar
    if return_details:
        return best_name, best_pipe, best_score, hp_results, best_best_params
    return best_name, best_pipe, best_score, hp_results, best_best_params
