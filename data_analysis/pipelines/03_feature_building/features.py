"""Definisi fitur model dan pabrik preprocessor scikit-learn.

Dua hal berbeda yang sengaja dipisah:

1. **Fitur** (fungsi ``build_feature_matrix``) - transformasi deterministik yang
   tidak belajar apa pun dari data, mis. menghitung skor komponen mSOFA atau
   log durasi pra-ICU. Aman dilakukan sekali di luar cross-validation.

2. **Preprocessor** (fungsi ``make_preprocessor``) - langkah yang MEMPELAJARI
   parameter dari data: imputasi MICE, one-hot encoding, standardisasi. Objek
   ini tidak pernah di-fit di sini; ia diserahkan ke tahap 04 untuk di-fit
   HANYA pada data latih tiap fold, sesuai rencana analisis proposal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (wajib)
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Kandidat prediktor sesuai proposal (14 variabel, 4 domain) -------------
NUMERIC_BASE = [
    "age",                    # demografi
    "pre_icu_los_hours",      # administratif (ditransformasi log1p)
    "map_lowest",             # disfungsi organ
    "sf_ratio_lowest",
    "gcs_lowest",
    "creatinine_highest",
]
ORDINAL_COLS = ["jkn_class"]  # 1 < 2 < 3, dipertahankan sebagai angka berurutan
BINARY_COLS = ["mechanical_ventilation", "vasopressor_inotrope", "transfusion_prc"]
CATEGORICAL_COLS = ["sex", "icu_admission_type", "surgery_24h", "diagnosis_category"]


def build_feature_matrix(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Bangun matriks fitur X, tabel target y, dan spesifikasi kolom.

    Nilai hilang DIBIARKAN sebagai NaN - imputasi adalah tugas preprocessor
    yang di-fit di dalam fold.
    """
    derived = cfg["feature_building"]["derived"]
    out = cfg["preprocessing"]["outcome"]
    work = df.copy()

    numeric = list(NUMERIC_BASE)

    # Durasi pra-ICU sangat miring ke kanan -> log1p
    if derived.get("log_pre_icu", True):
        work["pre_icu_los_log1p"] = np.log1p(work["pre_icu_los_hours"])
        numeric = [c if c != "pre_icu_los_hours" else "pre_icu_los_log1p" for c in numeric]

    # Skor komponen mSOFA (tanpa komponen hepatik, sesuai proposal)
    components = msofa_components(work)
    if derived.get("msofa_components", True):
        work = pd.concat([work, components], axis=1)
        numeric.append("msofa_total")

    # Jumlah organ support 24 jam pertama (0-3): ringkasan intensitas terapi
    if derived.get("organ_support_count", True):
        work["organ_support_count"] = (
            work["mechanical_ventilation"] + work["vasopressor_inotrope"] + work["transfusion_prc"]
        )
        numeric.append("organ_support_count")

    binary = list(BINARY_COLS)
    if derived.get("gcs_unassessable_flag", True) and "gcs_unassessable" in work:
        binary = binary + ["gcs_unassessable"]

    feature_cols = numeric + ORDINAL_COLS + binary + CATEGORICAL_COLS
    X = work[feature_cols].copy()

    y = work[[
        "episode_id", "icu_admission_datetime", "periode",
        out["binary_col"], out["continuous_col"], out["ratio_col"],
    ]].copy()

    spec = {
        "numeric": numeric,
        "ordinal": ORDINAL_COLS,
        "binary": binary,
        "categorical": CATEGORICAL_COLS,
        "target_binary": out["binary_col"],
        "target_continuous": out["continuous_col"],
        "n_features_pre_encoding": len(feature_cols),
    }
    return X, y, spec


def msofa_components(df: pd.DataFrame) -> pd.DataFrame:
    """Skor komponen mSOFA 24 jam pertama (respirasi, kardiovaskular, SSP, ginjal).

    Komponen hepatik tidak dipakai karena bilirubin tidak termasuk variabel
    penelitian. Komponen kardiovaskular disederhanakan: tanpa data dosis
    vasopresor, pemakaian vasopresor apa pun disetarakan skor 3.
    """
    sf = df["sf_ratio_lowest"]
    resp = pd.Series(np.select(
        [sf > 512, sf > 357, sf > 214, sf > 89],
        [0, 1, 2, 3], default=4,
    ), index=df.index, dtype="float")
    resp[sf.isna()] = np.nan

    mapv, vaso = df["map_lowest"], df["vasopressor_inotrope"]
    cardio = pd.Series(np.select(
        [vaso == 1, mapv < 70],
        [3, 1], default=0,
    ), index=df.index, dtype="float")
    cardio[mapv.isna() & (vaso != 1)] = np.nan

    gcs = df["gcs_lowest"]
    cns = pd.Series(np.select(
        [gcs >= 15, gcs >= 13, gcs >= 10, gcs >= 6],
        [0, 1, 2, 3], default=4,
    ), index=df.index, dtype="float")
    cns[gcs.isna()] = np.nan

    cr = df["creatinine_highest"]
    renal = pd.Series(np.select(
        [cr < 1.2, cr < 2.0, cr < 3.5, cr < 5.0],
        [0, 1, 2, 3], default=4,
    ), index=df.index, dtype="float")
    renal[cr.isna()] = np.nan

    comp = pd.DataFrame({
        "msofa_resp": resp,
        "msofa_cardio": cardio,
        "msofa_cns": cns,
        "msofa_renal": renal,
    })
    # Total dihitung dari komponen yang tersedia (skala dijaga tetap 0-16)
    comp["msofa_total"] = comp.mean(axis=1, skipna=True) * 4
    return comp


def make_preprocessor(spec: dict, scale: bool = False, seed: int = 42) -> ColumnTransformer:
    """Rangkai langkah preprocessing yang harus di-fit di dalam fold.

    - numerik : MICE (IterativeImputer) [+ standardisasi bila ``scale=True``]
    - ordinal : median (kelas rawat jarang hilang; urutan 1<2<3 dipertahankan)
    - biner   : modus
    - kategorik: modus + one-hot encoding

    ``scale=True`` dipakai untuk elastic net logistic regression; model berbasis
    pohon tidak memerlukannya.
    """
    numeric_steps = [
        ("impute", IterativeImputer(max_iter=10, random_state=seed, sample_posterior=False)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    ordinal_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        ordinal_steps.append(("scale", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), spec["numeric"]),
            ("ord", Pipeline(ordinal_steps), spec["ordinal"]),
            ("bin", SimpleImputer(strategy="most_frequent"), spec["binary"]),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore",
                                         sparse_output=False)),
            ]), spec["categorical"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encoded_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Nama kolom setelah one-hot encoding (dipakai untuk SHAP & importance)."""
    return [str(name) for name in preprocessor.get_feature_names_out()]


def compute_vif(X: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor antarvariabel numerik (deteksi multikolinearitas).

    Dihitung pada data yang sudah dibuang barisnya bila ada nilai hilang (VIF
    bersifat deskriptif, tidak memengaruhi pemodelan).
    """
    d = X[numeric_cols].dropna()
    vifs = []
    for col in numeric_cols:
        others = [c for c in numeric_cols if c != col]
        if not others:
            continue
        A = np.column_stack([np.ones(len(d)), d[others].to_numpy()])
        beta, *_ = np.linalg.lstsq(A, d[col].to_numpy(), rcond=None)
        pred = A @ beta
        ss_res = float(((d[col].to_numpy() - pred) ** 2).sum())
        ss_tot = float(((d[col].to_numpy() - d[col].mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
        vifs.append({
            "variabel": col,
            "r2": round(r2, 4),
            "vif": round(1 / (1 - r2), 2) if r2 < 0.999 else np.inf,
        })
    return pd.DataFrame(vifs).sort_values("vif", ascending=False)
