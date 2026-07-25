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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

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


def needs_class_weight(y, threshold: float = 0.35) -> bool:
    """Pembobotan kelas hanya bila proporsi kelas minoritas di bawah ambang.

    Pada data yang cukup seimbang, pembobotan menggeser probabilitas prediksi dan
    memperburuk kalibrasi - padahal kalibrasi termasuk kriteria penilaian utama.
    """
    p = float(np.mean(y))
    return min(p, 1 - p) < threshold


def build_model(name: str, preprocessor, y_train, cfg) -> Pipeline:
    """Rakit pipeline lengkap untuk satu model."""
    tr = cfg["training"]
    seed = cfg.seed
    balanced = needs_class_weight(y_train, float(tr.get("class_weight_threshold", 0.35)))

    if name == "xgboost":
        pos = float(np.sum(y_train))
        neg = float(len(y_train) - pos)
        estimator = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            scale_pos_weight=(neg / pos) if (balanced and pos) else 1.0,
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


def param_grid(name: str, cfg) -> dict:
    """Grid hyperparameter untuk inner cross-validation."""
    grid = cfg["training"]["models"][name]["grid"]
    # YAML menuliskan None sebagai null -> sudah otomatis menjadi None di Python
    return {k: list(v) for k, v in grid.items()}


def needs_scaling(name: str) -> bool:
    """Hanya regresi logistik yang memerlukan standardisasi variabel numerik."""
    return name == "elastic_net_lr"
