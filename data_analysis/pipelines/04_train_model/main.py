"""TAHAP 04 - Pelatihan, evaluasi, dan interpretasi model.

Menjalankan seluruh rencana analisis pemodelan proposal dalam satu tahap:

    1. tiga model dibandingkan: XGBoost, Random Forest, Elastic Net LR
    2. nested cross-validation 5x3 untuk estimasi performa yang jujur
    3. validasi temporal (latih Mei-Des 2025, uji Jan-Mei 2026)
    4. metrik diskriminasi + kalibrasi (Tabel 4.2)
    5. interpretabilitas SHAP + permutation importance (Tabel 4.3)

Imputasi MICE, one-hot encoding, dan standardisasi berada DI DALAM pipeline,
sehingga ikut di-fit ulang pada setiap fold - tidak ada kebocoran data.

Jalankan mandiri:
    python pipelines/04_train_model/main.py
    python pipelines/04_train_model/main.py --skip-shap     # lebih cepat
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(STAGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
import numpy as np
import pandas as pd

import evaluate
import interpret
import models as model_defs
from common import viz
from common.config import load_config
from common.io_utils import load_json, save_json, save_table
from common.logging_utils import banner, get_logger
from common.runner import load_stage_module

STAGE = "04_train_model"
log = get_logger(STAGE)


def run(cfg=None, skip_shap: bool = False, **overrides) -> dict:
    """Latih, evaluasi, dan interpretasikan seluruh model."""
    cfg = cfg or load_config(**overrides)
    banner(log, "tahap 04 - pelatihan & evaluasi model")
    viz.set_theme()
    t0 = time.time()

    features = load_stage_module("03_feature_building", "features")
    fb, tr = cfg["feature_building"], cfg["training"]
    proc_dir = cfg.path("data_processed")

    X = pd.read_csv(proc_dir / fb["output_features"])
    y_table = pd.read_csv(proc_dir / fb["output_target"])
    spec = load_json(proc_dir / "feature_spec.json")

    y = y_table[spec["target_binary"]].to_numpy(dtype=int)
    train_mask = y_table["periode"].str.startswith("latih").to_numpy()
    log.info("Data: %s episode, %s variabel, prevalensi %.1f%%", len(X), X.shape[1], 100 * y.mean())
    log.info("Split temporal: %s latih / %s uji", train_mask.sum(), (~train_mask).sum())

    # -- 1. Latih & validasi tiap model -------------------------------------
    nested, temporal = {}, {}
    for name in model_defs.enabled_models(cfg):
        log.info("[%s] nested cross-validation %sx%s ...",
                 name, tr["outer_folds"], tr["inner_folds"])
        preprocessor = features.make_preprocessor(
            spec, scale=model_defs.needs_scaling(name), seed=cfg.seed)
        pipeline = model_defs.build_model(name, preprocessor, y[train_mask], cfg)
        grid = model_defs.param_grid(name, cfg)

        nested[name] = evaluate.nested_cv(name, pipeline, grid, X, y, cfg)
        auc = nested[name]["fold_metrics"]["auc_roc"]
        log.info("[%s] AUC nested CV = %.3f ± %.3f", name, auc.mean(), auc.std())

        temporal[name] = evaluate.temporal_validation(
            name, pipeline, grid, X, y, train_mask, cfg)
        log.info("[%s] AUC validasi temporal = %.3f | Brier = %.3f",
                 name, temporal[name]["metrics"]["auc_roc"], temporal[name]["metrics"]["brier"])

    # -- 2. Tabel hasil ------------------------------------------------------
    tab_dir, fig_dir = cfg.table_dir(STAGE), cfg.figure_dir(STAGE)
    comparison = evaluate.comparison_table(nested, temporal, float(tr["auc_target"]))
    fold_metrics = pd.concat([n["fold_metrics"] for n in nested.values()], ignore_index=True)
    best_params = {name: temporal[name]["best_params"] for name in temporal}

    save_table(comparison, tab_dir / "tabel_4_2_perbandingan_model.csv")
    save_table(fold_metrics, tab_dir / "01_metrik_per_fold.csv")
    save_json(best_params, tab_dir / "02_hyperparameter_terpilih.json")

    # Prediksi tersimpan agar analisis lanjutan tidak perlu melatih ulang
    predictions = y_table.loc[~train_mask, ["episode_id", "icu_admission_datetime"]].copy()
    predictions["y_true"] = y[~train_mask]
    for name in temporal:
        predictions[f"prob_{name}"] = temporal[name]["prob_test"]
    save_table(predictions, tab_dir / "03_prediksi_data_uji.csv")

    # -- 3. Simpan model ----------------------------------------------------
    model_dir = cfg.path("models")
    for name, res in temporal.items():
        joblib.dump(res["estimator"], model_dir / f"{name}_temporal.joblib")

    # -- 4. Gambar evaluasi --------------------------------------------------
    figures = [
        evaluate.plot_roc(temporal, fig_dir),
        evaluate.plot_pr(temporal, fig_dir),
        evaluate.plot_calibration(temporal, fig_dir),
        evaluate.plot_fold_auc(nested, float(tr["auc_target"]), fig_dir),
        evaluate.plot_confusion(temporal, fig_dir),
        evaluate.plot_decision_curve(temporal, fig_dir),
    ]

    # -- 5. Interpretabilitas -----------------------------------------------
    best_name = max(nested, key=lambda n: nested[n]["fold_metrics"]["auc_roc"].mean())
    shap_name = best_name if best_name in model_defs.TREE_MODELS else next(
        (n for n in nested if n in model_defs.TREE_MODELS), None)
    shap_table = pd.DataFrame()
    perm = pd.DataFrame()

    if not skip_shap and shap_name:
        ip = cfg["interpretability"]
        log.info("Analisis SHAP pada model %s ...", shap_name)
        pipeline = temporal[shap_name]["estimator"]
        X_test = X[~train_mask]
        explanation, Xt = interpret.compute_shap(pipeline, X_test)
        shap_table = interpret.shap_importance_table(explanation, Xt)
        save_table(shap_table, tab_dir / "tabel_4_3_shap_importance.csv")

        figures += [
            interpret.plot_shap_beeswarm(explanation, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_bar(shap_table, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_dependence(explanation, Xt, shap_table, fig_dir,
                                           int(ip["n_dependence_plots"])),
        ]
        figures += interpret.plot_shap_waterfall(
            explanation, temporal[shap_name]["prob_test"], fig_dir, int(ip["waterfall_cases"]))
        figures.append(interpret.plot_shap_interaction(
            pipeline, Xt, shap_table, fig_dir, int(ip["n_interaction_features"]),
            int(ip["interaction_sample"]), cfg.seed))

        log.info("Permutation importance ...")
        perm = interpret.permutation_table(
            {n: temporal[n]["estimator"] for n in temporal}, X_test, y[~train_mask], cfg)
        save_table(perm, tab_dir / "04_permutation_importance.csv")
        figures.append(interpret.plot_permutation_importance(perm, fig_dir))
    elif skip_shap:
        log.info("SHAP dilewati (--skip-shap)")

    # -- 6. Laporan ringkas --------------------------------------------------
    elapsed = time.time() - t0
    best_auc = nested[best_name]["fold_metrics"]["auc_roc"].mean()
    report = {
        "model_terbaik": model_defs.MODEL_LABELS[best_name],
        "auc_nested_cv": round(float(best_auc), 4),
        "auc_temporal": round(float(temporal[best_name]["metrics"]["auc_roc"]), 4),
        "target_auc": float(tr["auc_target"]),
        "target_tercapai": bool(best_auc >= float(tr["auc_target"])),
        "n_latih": int(train_mask.sum()),
        "n_uji": int((~train_mask).sum()),
        "hyperparameter_terpilih": best_params.get(best_name, {}),
        "durasi_detik": round(elapsed, 1),
    }
    save_json(report, cfg.path("reports") / "ringkasan_model.json")

    m = temporal[best_name]["metrics"]
    summary = {
        "Model terbaik (AUC nested CV)": model_defs.MODEL_LABELS[best_name],
        "AUC nested cross-validation": f"{best_auc:.3f}",
        "AUC validasi temporal": f"{m['auc_roc']:.3f}",
        "Target AUC ≥ 0,75": "tercapai" if best_auc >= float(tr["auc_target"]) else "belum tercapai",
        "Sensitivitas / spesifisitas": f"{m['sensitivitas']:.2f} / {m['spesifisitas']:.2f}",
        "Brier score": f"{m['brier']:.3f}",
        "Calibration intercept / slope": f"{m['calibration_intercept']:.2f} / {m['calibration_slope']:.2f}",
        "Model untuk analisis SHAP": (
            model_defs.MODEL_LABELS.get(shap_name, "-") +
            ("" if shap_name == best_name else " (TreeSHAP hanya berlaku untuk model pohon)")
        ),
        "Prediktor teratas (SHAP)": ", ".join(shap_table["variabel"].head(3)) if not shap_table.empty else "-",
        "Jumlah gambar dihasilkan": len(figures),
        "Waktu komputasi": f"{elapsed:.0f} detik",
    }

    tables = {
        "Tabel 4.2 Perbandingan performa model": comparison,
        "Metrik per outer fold": fold_metrics,
    }
    if not shap_table.empty:
        tables["Tabel 4.3 Peringkat kontribusi SHAP"] = shap_table
    if not perm.empty:
        tables["Permutation importance"] = perm

    return {
        "stage": STAGE,
        "summary": summary,
        "tables": tables,
        "figures": figures,
        "paths": {"models_dir": model_dir, "tables_dir": tab_dir, "figures_dir": fig_dir},
        "best_model": best_name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Melatih dan mengevaluasi model prediksi.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-shap", action="store_true",
                        help="Lewati analisis SHAP & permutation importance (lebih cepat)")
    args = parser.parse_args()

    result = run(load_config(args.config), skip_shap=args.skip_shap)
    for key, value in result["summary"].items():
        log.info("%-34s : %s", key, value)


if __name__ == "__main__":
    main()
