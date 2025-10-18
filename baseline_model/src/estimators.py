"""
estimators.py
-------------

CatBoost, LightGBM ve XGBoost algoritmalarını sklearn uyumlu hale getiren
sarmalayıcı (wrapper) sınıfları içerir. Böylece modeller, Pipeline içinde ve
çapraz doğrulama süreçlerinde sorunsuz şekilde kullanılabilir.

"""

from typing import Optional, Sequence
from sklearn.base import BaseEstimator, RegressorMixin
from src.config import have  


# ----------------------- CatBoost -----------------------
class CatBoostSk(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        depth: int = 8,
        learning_rate: float = 0.05,
        n_estimators: int = 1200,
        random_state: int = 42,
        verbose: bool = False,
        allow_writing_files: bool = False,
        cat_features=None, 
    ):
        self.depth = depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.cat_features = cat_features

        # İç model oluşturmayı fit'e taşıyoruz; clone/set_params ile uyum için iyi.
        self.model_ = None

    def fit(self, X, y):
        if not have.get("catboost", False):
            raise RuntimeError("CatBoost yüklü değil.")
        from catboost import CatBoostRegressor

        # Parametrelerle her fit'te taze model kur
        self.model_ = CatBoostRegressor(
            depth=self.depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            loss_function="RMSE",
            random_seed=self.random_state,         
            verbose=self.verbose,
            allow_writing_files=self.allow_writing_files,
        )
        cat_feats = None
        if self.cat_features is not None:
            cf = list(self.cat_features)
            if len(cf) and isinstance(cf[0], str):
                # X bir pandas DataFrame olmalı
                cf = [X.columns.get_loc(c) for c in cf]
            cat_feats = cf
            
        self.model_.fit(X, y, cat_features=cat_feats)
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("Model henüz fit edilmedi veya CatBoost yüklü değil.")
        return self.model_.predict(X)


# ----------------------- LightGBM -----------------------
class LGBMSk(BaseEstimator, RegressorMixin):
    """
    LightGBM sarmalayıcı.

    - OHE yoksa: 'categorical_feature' (isim veya indeks) verilebilir.
    - OHE varsa: 'categorical_feature=None' bırakın.
    """
    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        n_estimators: int = 1500,
        random_state: int = 42,
        categorical_feature: Optional[Sequence] = None,  # isim veya indeks
    ):
        # __init__ içinde paramı ASLA değiştirme!
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.categorical_feature = categorical_feature

        # clone uyumu için iç modeli burada kurma
        self.model_ = None

    def fit(self, X, y):
        if not have.get("lightgbm", False):
            raise RuntimeError("LightGBM yüklü değil.")
        from lightgbm import LGBMRegressor

        # İç modeli her fit'te paramlarla kur
        self.model_ = LGBMRegressor(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=1,  # dış paralellik ile çakışmayı önle
        )

        # Param dönüşümü (listeye çevirme vb.) __init__’te değil, burada yapılır
        cat_feats = list(self.categorical_feature) if self.categorical_feature is not None else None

        # Not: Pipeline sonrası X OHE ile sayısal matris ise cat_feats=None ver!
        self.model_.fit(X, y, categorical_feature=cat_feats)
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ----------------------- XGBoost -----------------------
class XGBSk(BaseEstimator, RegressorMixin):
    """
    XGBoost sarmalayıcı.

    Kullanım:
    - OHE yoksa: pandas 'category' dtype ve 'enable_categorical=True' ile native kategorik desteklenir.
    - OHE varsa: zaten sayısal matris gelir; parametre kalabilir.
    """
    def __init__(
        self,
        n_estimators: int = 1500,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        booster: str = "gbtree",
        random_state: int = 42,
        enable_categorical: bool = True,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.booster = booster
        self.random_state = random_state
        self.enable_categorical = enable_categorical

        if not have.get("xgboost", False):
            self.model_ = None
        else:
            from xgboost import XGBRegressor  # geç import
            self.model_ = XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_lambda=self.reg_lambda,
                booster=self.booster,
                tree_method="hist",
                enable_categorical=self.enable_categorical,
                random_state=self.random_state,
                n_jobs=1,
            )

    def fit(self, X, y):
        if self.model_ is None:
            raise RuntimeError("XGBoost yüklü değil.")
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)
