"""TAHAP 04 - Pelatihan, evaluasi, dan interpretasi model.

Menjalankan seluruh rencana analisis pemodelan proposal dalam satu tahap:

    1. tiga model dibandingkan: XGBoost, Random Forest, Elastic Net LR
    2. nested CV 5 fold x 10 ulangan, inner 3 fold + Optuna (estimasi UTAMA)
    3. analisis sensitivitas: validasi temporal (latih Mei-Des 2025, uji Jan-Mei
       2026), class weighting + rekalibrasi, analisis kasus lengkap
    4. luaran sekunder log(klaim/tagihan) dengan skema validasi yang sama
    5. model final (seluruh data) -> SHAP + permutation importance
    IK bootstrap, koreksi Holm, dan evaluasi subkelompok: tahap 05_bab4.

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


SCENARIOS = {  # nama -> keterangan (Bab 4)
    "utama": "Nested CV berulang, tanpa class weighting",
    "class_weight": "Sensitivitas: class weighting + rekalibrasi pada data latih",
    "kasus_lengkap": "Sensitivitas: analisis kasus lengkap (tanpa imputasi)",
    "log_rasio": "Luaran sekunder: regresi log(klaim/tagihan)",
}


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
    y_log = y_table[spec["target_continuous"]].to_numpy(dtype=float)
    train_mask = y_table["periode"].str.startswith("latih").to_numpy()
    groups = y_table["patient_code"].astype(str).to_numpy()   # CV per pasien
    complete = X.notna().all(axis=1).to_numpy()
    log.info("Data: %s episode, %s variabel, prevalensi %.1f%%", len(X), X.shape[1], 100 * y.mean())
    log.info("Nested CV %s fold x %s ulangan, inner %s, Optuna %s percobaan, MICE m=%s",
             tr["outer_folds"], tr["outer_repeats"], tr["inner_folds"], tr["n_trials"],
             tr["n_imputations"])

    def pipe(name, balanced=False, regression=False):
        prep = features.make_preprocessor(spec, scale=model_defs.needs_scaling(name), seed=cfg.seed)
        return model_defs.build_model(name, prep, cfg, balanced=balanced, regression=regression,
                                      pos_weight=float((y == 0).sum() / y.sum()))

    # -- 1. Latih & validasi tiap model -------------------------------------
    cv = {s: {} for s in SCENARIOS}
    temporal, final = {}, {}
    for name in model_defs.enabled_models(cfg):
        space = model_defs.search_space(name, cfg)
        t1 = time.time()
        cv["utama"][name] = evaluate.repeated_nested_cv(name, pipe(name), space, X, y, y, groups, cfg)
        auc = cv["utama"][name]["fold_metrics"]["auc_roc"]
        log.info("[%s] AUC nested CV = %.3f ± %.3f (%.0f s)", name, auc.mean(), auc.std(),
                 time.time() - t1)

        temporal[name] = evaluate.holdout(name, pipe(name), space, X, y, y, groups, train_mask, cfg)
        log.info("[%s] AUC validasi temporal = %.3f", name, temporal[name]["metrics"]["auc_roc"])
        final[name] = evaluate.final_model(pipe(name), space, X, y, y, groups, cfg)

        if tr.get("class_weight_sensitivity"):
            cv["class_weight"][name] = evaluate.repeated_nested_cv(
                name, pipe(name, balanced=True), space, X, y, y, groups, cfg, recalibrate=True)
        if tr.get("complete_case_sensitivity"):
            res = evaluate.repeated_nested_cv(
                name, pipe(name), space, X[complete].reset_index(drop=True), y[complete],
                y[complete], groups[complete], cfg)
            res["oof"]["row"] = np.flatnonzero(complete)[res["oof"]["row"]]   # indeks asli
            cv["kasus_lengkap"][name] = res
        if tr.get("secondary_outcome"):
            cv["log_rasio"][name] = evaluate.repeated_nested_cv(
                name, pipe(name, regression=True), model_defs.search_space(name, cfg, True),
                X, y_log, y, groups, cfg, regression=True)
        log.info("[%s] selesai (%.0f s)", name, time.time() - t1)

    # -- 2. Tabel hasil ------------------------------------------------------
    tab_dir, fig_dir = cfg.table_dir(STAGE), cfg.figure_dir(STAGE)
    oof = pd.concat([res["oof"].assign(skenario=s) for s, d in cv.items() for res in d.values()],
                    ignore_index=True)
    oof["episode_id"] = y_table["episode_id"].to_numpy()[oof["row"]]
    folds = pd.concat([res["fold_metrics"].assign(skenario=s) for s, d in cv.items()
                       for res in d.values()], ignore_index=True)
    save_table(oof, tab_dir / "05_prediksi_oof.csv")
    save_table(folds, tab_dir / "01_metrik_per_fold.csv")
    mean_folds = (folds.drop(columns=["ulangan", "fold"]).groupby(["skenario", "model"])
                  .agg(["mean", "std"]))
    mean_folds.columns = [f"{a}_{b}" for a, b in mean_folds.columns]
    save_table(mean_folds.reset_index(), tab_dir / "tabel_4_2_perbandingan_model.csv")

    best_params = {"final": {n: final[n]["params"] for n in final},
                   "temporal": {n: temporal[n]["best_params"] for n in temporal},
                   "ambang_final": {n: final[n]["threshold"] for n in final}}
    save_json(best_params, tab_dir / "02_hyperparameter_terpilih.json")

    predictions = y_table.loc[~train_mask, ["episode_id", "icu_admission_datetime"]].copy()
    predictions["y_true"] = y[~train_mask]
    for name in temporal:
        predictions[f"prob_{name}"] = temporal[name]["prob_test"]
        predictions[f"kelas_{name}"] = (temporal[name]["prob_test"]
                                        >= temporal[name]["metrics"]["ambang"]).astype(int)
    save_table(predictions, tab_dir / "03_prediksi_data_uji.csv")
    save_table(pd.DataFrame([{"model": n, **temporal[n]["metrics"]} for n in temporal]),
               tab_dir / "06_metrik_temporal.csv")

    # -- 3. Simpan model ----------------------------------------------------
    model_dir = cfg.path("models")
    for name in final:
        joblib.dump(final[name]["models"], model_dir / f"{name}_final_mi.joblib")

    # -- 4. Gambar evaluasi --------------------------------------------------
    n = len(y)
    nested_view = {name: {"y_test": y, "prob_test": evaluate.mean_oof(res, n),
                          "metrics": res["fold_metrics"].mean(numeric_only=True).to_dict()}
                   for name, res in cv["utama"].items()}
    where, src = "nested CV (prediksi out-of-fold)", "Rerata probabilitas out-of-fold dari 10 ulangan nested CV"
    figures = [
        evaluate.plot_roc(nested_view, fig_dir, where, "01_kurva_roc.png", src),
        evaluate.plot_pr(nested_view, fig_dir, where, "02_kurva_pr.png"),
        evaluate.plot_calibration(nested_view, fig_dir, where=where, fname="03_kalibrasi.png"),
        evaluate.plot_fold_auc({k: v for k, v in cv["utama"].items()}, float(tr["auc_target"]), fig_dir),
        evaluate.plot_decision_curve(nested_view, fig_dir, where, "06_decision_curve.png"),
        evaluate.plot_roc(temporal, fig_dir, fname="13_kurva_roc_temporal.png"),
        evaluate.plot_calibration(temporal, fig_dir, fname="14_kalibrasi_temporal.png"),
        evaluate.plot_confusion(temporal, fig_dir),
        evaluate.plot_decision_curve(temporal, fig_dir, fname="15_decision_curve_temporal.png"),
    ]

    # -- 5. Interpretabilitas (model final, seluruh data) -------------------
    best_name = max(cv["utama"], key=lambda k: cv["utama"][k]["fold_metrics"]["auc_roc"].mean())
    shap_name = best_name if best_name in model_defs.TREE_MODELS else next(
        (k for k in cv["utama"] if k in model_defs.TREE_MODELS), None)
    shap_table = pd.DataFrame()
    perm = pd.DataFrame()

    if not skip_shap and shap_name:
        ip = cfg["interpretability"]
        log.info("Analisis SHAP pada model final %s (m=%s imputasi) ...", shap_name, tr["n_imputations"])
        pipelines = final[shap_name]["models"]
        explanation, Xt = interpret.compute_shap_mi(pipelines, X)
        shap_table = interpret.shap_importance_table(explanation, Xt)
        save_table(shap_table, tab_dir / "tabel_4_3_shap_importance.csv")
        probs = np.mean([p.predict_proba(X)[:, 1] for p in pipelines], axis=0)

        figures += [
            interpret.plot_shap_beeswarm(explanation, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_bar(shap_table, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_dependence(explanation, Xt, shap_table, fig_dir,
                                           int(ip["n_dependence_plots"])),
        ]
        figures += interpret.plot_shap_waterfall(explanation, probs, fig_dir, int(ip["waterfall_cases"]))
        figures.append(interpret.plot_shap_interaction(
            pipelines[0], Xt, shap_table, fig_dir, int(ip["n_interaction_features"]),
            int(ip["interaction_sample"]), cfg.seed))

        log.info("Permutation importance (model temporal, data uji) ...")
        perm = interpret.permutation_table(
            {k: temporal[k]["estimators"][0] for k in temporal}, X[~train_mask], y[~train_mask], cfg)
        save_table(perm, tab_dir / "04_permutation_importance.csv")
        figures.append(interpret.plot_permutation_importance(perm, fig_dir))
    elif skip_shap:
        log.info("SHAP dilewati (--skip-shap)")

    # -- 6. Laporan ringkas --------------------------------------------------
    elapsed = time.time() - t0
    best = cv["utama"][best_name]["fold_metrics"]
    best_auc = best["auc_roc"].mean()
    report = {
        "model_terbaik": model_defs.MODEL_LABELS[best_name],
        "model_terbaik_key": best_name,
        "model_shap_key": shap_name,
        "auc_nested_cv": round(float(best_auc), 4),
        "auc_temporal": round(float(temporal[best_name]["metrics"]["auc_roc"]), 4),
        "target_auc": float(tr["auc_target"]),
        "target_tercapai": bool(best_auc >= float(tr["auc_target"])),
        "n_latih": int(train_mask.sum()),
        "n_uji": int((~train_mask).sum()),
        "n_kasus_lengkap": int(complete.sum()),
        "n_fitur_encoding": int(len(Xt.columns)) if not shap_table.empty else None,
        "skenario": SCENARIOS,
        "hyperparameter_terpilih": final[best_name]["params"],
        "durasi_detik": round(elapsed, 1),
    }
    save_json(report, cfg.path("reports") / "ringkasan_model.json")

    summary = {
        "Model terbaik (AUC nested CV)": model_defs.MODEL_LABELS[best_name],
        "AUC nested CV (rerata 50 fold)": f"{best_auc:.3f}",
        "AUC validasi temporal": f"{temporal[best_name]['metrics']['auc_roc']:.3f}",
        "Target AUC ≥ 0,75": "tercapai" if best_auc >= float(tr["auc_target"]) else "belum tercapai",
        "Sensitivitas / spesifisitas (ambang data latih)":
            f"{best['sensitivitas'].mean():.2f} / {best['spesifisitas'].mean():.2f}",
        "Brier score": f"{best['brier'].mean():.3f}",
        "Calibration intercept / slope":
            f"{best['calibration_intercept'].mean():.2f} / {best['calibration_slope'].mean():.2f}",
        "Model untuk analisis SHAP": model_defs.MODEL_LABELS.get(shap_name, "-"),
        "Prediktor teratas (SHAP)": ", ".join(shap_table["variabel"].head(3)) if not shap_table.empty else "-",
        "Jumlah gambar dihasilkan": len(figures),
        "Waktu komputasi": f"{elapsed:.0f} detik",
    }

    tables = {"Perbandingan performa model (rerata fold)": mean_folds.reset_index()}
    if not shap_table.empty:
        tables["Peringkat kontribusi SHAP"] = shap_table

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
    parser.add_argument("--quick", action="store_true",
                        help="Uji cepat: 1 ulangan, 3 percobaan Optuna, m=2 (bukan hasil penelitian)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.quick:
        cfg["training"].update(outer_repeats=1, n_trials=3, n_imputations=2)
    result = run(cfg, skip_shap=args.skip_shap)
    for key, value in result["summary"].items():
        log.info("%-34s : %s", key, value)


if __name__ == "__main__":
    main()
