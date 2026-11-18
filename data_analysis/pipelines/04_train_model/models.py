"""Definisi tiga model yang dibandingkan beserta ruang hyperparameter-nya.

Setiap model dibungkus dalam sklearn Pipeline:

    Pipeline([("prep", <preprocessor>), ("model", <estimator>)])

Dengan bentuk ini, imputasi MICE dan one-hot encoding ikut di-fit ulang pada
setiap fold cross-validation, sehingga tidak ada informasi dari data uji yang
bocor ke tahap preprocessing.
"""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "elastic_net_lr": "Elastic Net LR",
}
TREE_MODELS = {"xgboost", "random_forest"}


def _sklearn_version() -> tuple[int, ...]:
    """Versi scikit-learn sebagai tuple angka, mis. (1, 9, 0)."""
    parts = []
    for chunk in sklearn.__version__.split(".")[:3]:
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def build_model(name: str, preprocessor, cfg, balanced: bool = False,
                regression: bool = False, pos_weight: float = 1.0) -> Pipeline:
    """Rakit pipeline lengkap untuk satu model.

    balanced   : class weighting (hanya analisis sensitivitas, proposal §3.10.2)
    regression : luaran sekunder kontinu log(klaim/tagihan)
    pos_weight : rasio negatif/positif untuk XGBoost bila ``balanced``
    """
    seed = cfg.seed
    if regression:
        estimator = {
            "xgboost": lambda: XGBRegressor(objective="reg:squarederror", tree_method="hist",
                                            random_state=seed, n_jobs=1),
            "random_forest": lambda: RandomForestRegressor(random_state=seed, n_jobs=1),
            "elastic_net_lr": lambda: ElasticNet(max_iter=10000, random_state=seed),
        }[name]()
    elif name == "xgboost":
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=pos_weight if balanced else 1.0,
        )
    elif name == "random_forest":
        estimator = RandomForestClassifier(
            random_state=seed,
            n_jobs=1,
            class_weight="balanced" if balanced else None,
        )
    elif name == "elastic_net_lr":
        # Pembanding yang dioptimalkan setara: elastic net + saga + standardisasi
        kwargs = dict(
            solver="saga",
            max_iter=5000,
            random_state=seed,
            class_weight="balanced" if balanced else None,
        )
        # scikit-learn >= 1.8 menentukan elastic net lewat l1_ratio saja;
        # versi lama masih memerlukan penalty="elasticnet".
        if _sklearn_version() < (1, 8):
            kwargs["penalty"] = "elasticnet"
        estimator = LogisticRegression(**kwargs)
    else:
        raise ValueError(f"Model tidak dikenal: {name}")

    return Pipeline([("prep", preprocessor), ("model", estimator)])


def enabled_models(cfg) -> list[str]:
    """Nama model yang diaktifkan di config, urut sesuai penulisan proposal."""
    models = cfg["training"]["models"]
    order = ["xgboost", "random_forest", "elastic_net_lr"]
    return [m for m in order if models.get(m, {}).get("enabled", False)]


def search_space(name: str, cfg, regression: bool = False) -> dict:
    """Ruang pencarian Optuna (config) untuk inner cross-validation."""
    m = cfg["training"]["models"][name]
    return m.get("space_regression", m["space"]) if regression else m["space"]


def needs_scaling(name: str) -> bool:
    """Hanya regresi logistik yang memerlukan standardisasi variabel numerik."""
    return name == "elastic_net_lr"
