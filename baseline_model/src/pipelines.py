"""
pipelines.py
------------

Tek parça (end-to-end) pipeline'lar:
- DomainFE: (farmasötik + elemental oranlar + tip güvenliği + kolon seçimi)
- OHE gerektirenler (HistGBR/EBM): DomainFE → pre_ohe → reg
- Native kategorik (Cat/LGBM/XGB): DomainFE → reg

"""

from typing import List, Optional, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

from src.preprocessing import DomainFE
from src.estimators import CatBoostSk, LGBMSk, XGBSk
from src.config import have, RANDOM_STATE


def build_model_pool(
    pre_ohe,                 # OneHotEncoder içeren ColumnTransformer (prepare_ml_data'dan)
    num_feats: List[str],    # DomainFE için sayısal kolonlar
    cat_feats: List[str],    # DomainFE için kategorik kolonlar (örn. ["Target_Phar"])
) -> List[Tuple[str, Optional[Any]]]:
    """
    Çıktı: [(görünen_ad, pipeline_veya_None), ...]
    Pipeline'lar ham DataFrame alır; FE/OHE/model adımlarını kendi içinde uygular.
    """
    models: List[Tuple[str, Optional[Any]]] = []

    # --- CatBoost (native kategorik; DomainFE -> reg) ---
    if have.get("catboost", False):
        models.append((
            "CatBoost",
            Pipeline([
                ("fe", DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
                ("reg", CatBoostSk(cat_features=cat_feats))
            ])
        ))
    else:
        models.append(("CatBoost (skipped)", None))

    # --- LightGBM GBDT (native kategorik; DomainFE -> reg) ---
    if have.get("lightgbm", False):
        models.append((
            "LightGBM-GBDT",
            Pipeline([
                ("fe", DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
                ("reg", LGBMSk(boosting_type="gbdt", categorical_feature=cat_feats))
            ])
        ))
    else:
        models.append(("LightGBM-GBDT (skipped)", None))

    # --- XGBoost GBTree (native kategorik; DomainFE -> reg) ---
    if have.get("xgboost", False):
        models.append((
            "XGBoost-GBTree",
            Pipeline([
                ("fe", DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
                ("reg", XGBSk())
            ])
        ))
    else:
        models.append(("XGBoost-GBTree (skipped)", None))

    # --- HistGradientBoostingRegressor (OHE gerekli; DomainFE -> pre_ohe -> reg) ---
    models.append((
        "HistGBR",
        Pipeline([
            ("fe",  DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
            ("pre", pre_ohe),
            ("reg", HistGradientBoostingRegressor(
                max_depth=None,
                learning_rate=0.08,
                max_iter=500,
                random_state=RANDOM_STATE,
            )),
        ])
    ))

    # --- EBM (OHE gerekli; DomainFE -> pre_ohe -> reg) ---
    if have.get("ebm", False):
        from interpret.glassbox import ExplainableBoostingRegressor  # lazy import
        models.append((
            "EBM",
            Pipeline([
                ("fe",  DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
                ("pre", pre_ohe),
                ("reg", ExplainableBoostingRegressor(
                    interactions=0,
                    max_leaves=3,
                    learning_rate=0.05,
                    max_bins=256,
                    random_state=RANDOM_STATE,
                )),
            ])
        ))
    else:
        models.append(("EBM (skipped)", None))

    # --- LightGBM DART (native kategorik; DomainFE -> reg) ---
    if have.get("lightgbm", False):
        models.append((
            "LightGBM-DART",
            Pipeline([
                ("fe", DomainFE(num_feats=num_feats, cat_feats=cat_feats)),
                ("reg", LGBMSk(boosting_type="dart", categorical_feature=cat_feats))
            ])
        ))
    else:
        models.append(("LightGBM-DART (skipped)", None))

    return models
