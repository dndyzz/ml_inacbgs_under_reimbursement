"""Validasi, metrik performa, dan gambar evaluasi model.

Skema validasi sesuai proposal §3.10.4:

1. **Nested cross-validation berulang** (5 outer x 10 ulangan, 3 inner, Optuna) -
   estimasi UTAMA. Hyperparameter dipilih di inner fold, dinilai di outer fold;
   fold dikelompokkan per pasien.
2. **Validasi temporal** - latih Mei-Des 2025, uji Jan-Mei 2026; analisis
   sensitivitas, tidak dipakai memilih model.

Di setiap fold: MICE >=5 set (prediksi dirata-rata), ambang klasifikasi (Youden)
dan rekalibrasi (bila class weighting) ditentukan dari prediksi out-of-fold
DATA LATIH, lalu diterapkan apa adanya pada data uji.

Metrik: diskriminasi (AUC-ROC, PR-AUC, sensitivitas, spesifisitas, F1) dan
kalibrasi (Brier score, calibration intercept & slope).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import optuna
from joblib import Parallel, delayed
from sklearn.base import clone
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from common import viz
from models import MODEL_LABELS

STAGE = "04_train_model"
EPS = 1e-9
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------

def youden_threshold(y_true, y_prob) -> float:
    """Ambang klasifikasi optimal menurut indeks Youden (sens + spes - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return float(thresholds[int(np.argmax(tpr - fpr))])


def calibration_intercept_slope(y_true, y_prob) -> tuple[float, float]:
    """Calibration intercept & slope dari regresi logistik pada logit prediksi.

    Slope 1 dan intercept 0 berarti kalibrasi sempurna. Slope < 1 menandakan
    prediksi terlalu ekstrem (overfitting).
    """
    p = np.clip(np.asarray(y_prob, dtype=float), EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int)

    slope_model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    slope_model.fit(logit, y)
    slope = float(slope_model.coef_[0][0])

    # Intercept dihitung dengan offset (slope dikunci = 1), cara baku TRIPOD
    def neg_loglik(a: float) -> float:
        z = a + logit.ravel()
        return float(np.sum(np.log1p(np.exp(z)) - y * z))

    intercept = float(minimize_scalar(neg_loglik, bounds=(-10, 10), method="bounded").x)
    return intercept, slope


def classification_metrics(y_true, y_prob, threshold: float | None = None) -> dict:
    """Semua metrik performa untuk satu himpunan prediksi."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    thr = youden_threshold(y_true, y_prob) if threshold is None else threshold
    y_pred = (y_prob >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    intercept, slope = calibration_intercept_slope(y_true, y_prob)

    return {
        "auc_roc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "sensitivitas": tp / (tp + fn) if (tp + fn) else np.nan,
        "spesifisitas": tn / (tn + fp) if (tn + fp) else np.nan,
        "ppv": tp / (tp + fp) if (tp + fp) else np.nan,
        "npv": tn / (tn + fn) if (tn + fn) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "akurasi": (tp + tn) / len(y_true),
        "fp": int(fp),
        "fn": int(fn),
        "brier": brier_score_loss(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ambang": thr,
    }


def regression_metrics(y_true, y_pred, y_binary) -> dict:
    """Luaran sekunder log(klaim/tagihan): R², RMSE, MAE, slope kalibrasi, AUC turunan.

    AUC turunan = kemampuan prediksi log-rasio membedakan rasio <1 (log <0);
    -y_pred dipakai karena log-rasio rendah = under-reimbursement.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_true - y_pred
    slope = np.polyfit(y_pred, y_true, 1)[0] if np.std(y_pred) > 0 else np.nan
    return {
        "r2": 1 - np.sum(resid ** 2) / np.sum((y_true - y_true.mean()) ** 2),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "calibration_slope": float(slope),
        "auc_turunan": roc_auc_score(y_binary, -y_pred),
    }


# ---------------------------------------------------------------------------
# Skema validasi
# ---------------------------------------------------------------------------

def suggest(trial, space: dict) -> dict:
    """Terjemahkan ruang pencarian config -> saran parameter Optuna."""
    out = {}
    for key, spec in space.items():
        kind = spec[0]
        if kind == "int":
            out[key] = trial.suggest_int(key, spec[1], spec[2])
        elif kind in ("float", "log"):
            out[key] = trial.suggest_float(key, spec[1], spec[2], log=kind == "log")
        elif kind == "cat":
            out[key] = trial.suggest_categorical(key, spec[1])
        else:
            raise ValueError(f"Jenis ruang pencarian tidak dikenal: {spec}")
    return out


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def run_fold(pipeline, space: dict, X: pd.DataFrame, y: np.ndarray, strata: np.ndarray,
             groups: np.ndarray, train_idx, test_idx, tr: dict, seed: int,
             regression: bool = False, recalibrate: bool = False, keep_models: bool = False) -> dict:
    """Satu fold luar: tuning Optuna -> MI m set -> OOF latih -> ambang/rekalibrasi -> uji."""
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xtr, ytr = X.iloc[train_idx], y[train_idx]
    inner = StratifiedGroupKFold(n_splits=int(tr["inner_folds"]), shuffle=True, random_state=seed)
    splits = list(inner.split(Xtr, strata[train_idx], groups[train_idx]))
    scoring = tr["scoring_regression"] if regression else tr["scoring"]

    def objective(trial):
        model = clone(pipeline).set_params(**suggest(trial, space))
        return cross_val_score(model, Xtr, ytr, cv=splits, scoring=scoring, n_jobs=1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(tr["n_trials"]))
    params = study.best_params

    def predict(model, Xq):
        return model.predict(Xq) if regression else model.predict_proba(Xq)[:, 1]

    out = {"params": params}

    # MICE m set: sample_posterior + seed berbeda per set; prediksi dirata-rata
    m = int(tr["n_imputations"])
    models, preds = [], []
    for k in range(m):
        model = clone(pipeline).set_params(**params, prep__impute__sample_posterior=m > 1,
                                           prep__impute__random_state=seed + k)
        model.fit(Xtr, ytr)
        if len(test_idx):
            preds.append(predict(model, X.iloc[test_idx]))
        if keep_models:
            models.append(model)
    out["pred"] = np.mean(preds, axis=0) if preds else np.array([])
    if keep_models:
        out["models"] = models
    if regression:
        return out

    # OOF data latih (inner split yang sama) -> ambang & rekalibrasi tanpa menyentuh data uji
    oof = np.zeros(len(train_idx))
    for a_idx, b_idx in splits:
        model = clone(pipeline).set_params(**params).fit(Xtr.iloc[a_idx], ytr[a_idx])
        oof[b_idx] = predict(model, Xtr.iloc[b_idx])
    if recalibrate:
        # rekalibrasi logistik (Platt) pada logit prediksi OOF data latih
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof), ytr)
        oof = cal.predict_proba(_logit(oof))[:, 1]
        if len(test_idx):
            out["pred"] = cal.predict_proba(_logit(out["pred"]))[:, 1]
        out["calibrator"] = cal
    out["threshold"] = youden_threshold(ytr, oof)
    return out


def repeated_nested_cv(name: str, pipeline, space: dict, X: pd.DataFrame, y: np.ndarray,
                       strata: np.ndarray, groups: np.ndarray, cfg, regression: bool = False,
                       recalibrate: bool = False) -> dict:
    """Nested CV 5 fold x 10 ulangan (fold per pasien); fold luar dijalankan paralel.

    Mengembalikan prediksi out-of-fold format panjang (satu baris per episode per
    ulangan) dan metrik per fold luar.
    """
    tr = cfg["training"]
    jobs = []
    for r in range(int(tr["outer_repeats"])):
        outer = StratifiedGroupKFold(n_splits=int(tr["outer_folds"]), shuffle=True,
                                     random_state=cfg.seed + r)
        for k, (a_idx, b_idx) in enumerate(outer.split(X, strata, groups), start=1):
            jobs.append((r + 1, k, a_idx, b_idx, cfg.seed + 1000 * (r + 1) + k))

    results = Parallel(n_jobs=int(tr["n_jobs"]))(
        delayed(run_fold)(pipeline, space, X, y, strata, groups, a_idx, b_idx, tr, s,
                          regression, recalibrate)
        for _, _, a_idx, b_idx, s in jobs)

    oof_rows, fold_rows = [], []
    for (r, k, _, b_idx, _), res in zip(jobs, results):
        frame = pd.DataFrame({"model": name, "ulangan": r, "fold": k, "row": b_idx,
                              "pred": res["pred"]})
        if regression:
            metrics = regression_metrics(y[b_idx], res["pred"], strata[b_idx])
        else:
            frame["ambang"] = res["threshold"]
            frame["kelas_prediksi"] = (res["pred"] >= res["threshold"]).astype(int)
            metrics = classification_metrics(y[b_idx], res["pred"], res["threshold"])
        oof_rows.append(frame)
        fold_rows.append({"model": name, "ulangan": r, "fold": k, **metrics})
    return {
        "oof": pd.concat(oof_rows, ignore_index=True),
        "fold_metrics": pd.DataFrame(fold_rows),
        "params": [res["params"] for res in results],
    }


def holdout(name: str, pipeline, space: dict, X: pd.DataFrame, y: np.ndarray, strata: np.ndarray,
            groups: np.ndarray, train_mask: np.ndarray, cfg) -> dict:
    """Validasi temporal: tuning + MI pada periode latih, uji pada periode berikutnya."""
    tr_idx, te_idx = np.flatnonzero(train_mask), np.flatnonzero(~train_mask)
    res = run_fold(pipeline, space, X, y, strata, groups, tr_idx, te_idx, cfg["training"],
                   cfg.seed, keep_models=True)
    return {
        "model": name,
        "estimators": res["models"],
        "best_params": res["params"],
        "y_test": y[te_idx],
        "prob_test": res["pred"],
        "metrics": classification_metrics(y[te_idx], res["pred"], res["threshold"]),
        "n_train": len(tr_idx),
        "n_test": len(te_idx),
    }


def final_model(pipeline, space: dict, X: pd.DataFrame, y: np.ndarray, strata: np.ndarray,
                groups: np.ndarray, cfg) -> dict:
    """Model final: prosedur tuning yang sama pada seluruh data pengembangan (§3.10.4)."""
    res = run_fold(pipeline, space, X, y, strata, groups, np.arange(len(X)),
                   np.array([], dtype=int), cfg["training"], cfg.seed, keep_models=True)
    return {"models": res["models"], "params": res["params"], "threshold": res["threshold"]}


def mean_oof(cv: dict, n: int) -> np.ndarray:
    """Prediksi out-of-fold per episode, dirata-rata antar-ulangan (untuk gambar)."""
    return cv["oof"].groupby("row")["pred"].mean().reindex(range(n)).to_numpy()


# ---------------------------------------------------------------------------
# Gambar
# ---------------------------------------------------------------------------

def plot_roc(temporal: dict, out_dir: Path, where: str = "data uji temporal",
             fname: str = "01_kurva_roc.png", source: str | None = None) -> Path:
    """Kurva ROC ketiga model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in temporal.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["prob_test"])
        ax.plot(fpr, tpr, color=viz.MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} (AUC {res['metrics']['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color=viz.INK_MUTED,
            label="Tebakan acak")
    ax.set_xlabel("1 − spesifisitas")
    ax.set_ylabel("Sensitivitas")
    ax.set_title(f"Kurva ROC pada {where}")
    ax.grid(axis="both")
    ax.legend(loc="lower right")
    viz.annotate_source(ax, source or "Data uji = admisi ICU Jan-Mei 2026 (tidak dipakai saat pelatihan)")
    return viz.save_fig(fig, out_dir / fname)


def plot_pr(temporal: dict, out_dir: Path, where: str = "data uji temporal",
            fname: str = "02_kurva_pr.png") -> Path:
    """Kurva precision-recall (lebih informatif saat kelas tidak seimbang)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    prevalence = None
    for name, res in temporal.items():
        precision, recall, _ = precision_recall_curve(res["y_test"], res["prob_test"])
        ax.plot(recall, precision, color=viz.MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} (PR-AUC {res['metrics']['pr_auc']:.3f})")
        prevalence = float(np.mean(res["y_test"]))
    if prevalence is not None:
        ax.axhline(prevalence, linestyle="--", linewidth=1.3, color=viz.INK_MUTED,
                   label=f"Prevalensi ({prevalence:.2f})")
    ax.set_xlabel("Recall (sensitivitas)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Kurva precision-recall pada {where}")
    ax.grid(axis="both")
    ax.legend(loc="lower left")
    return viz.save_fig(fig, out_dir / fname)


def plot_calibration(temporal: dict, out_dir: Path, n_bins: int = 10, where: str = "data uji temporal",
                     fname: str = "03_kalibrasi.png") -> Path:
    """Kurva kalibrasi: probabilitas prediksi vs proporsi kejadian teramati."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color=viz.INK_MUTED,
            label="Kalibrasi sempurna")

    for name, res in temporal.items():
        p = np.asarray(res["prob_test"], dtype=float)
        y = np.asarray(res["y_test"], dtype=int)
        bins = np.quantile(p, np.linspace(0, 1, n_bins + 1))
        bins[0], bins[-1] = bins[0] - 1e-6, bins[-1] + 1e-6
        idx = np.digitize(p, bins[1:-1])
        xs, ys = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() >= 5:
                xs.append(p[mask].mean())
                ys.append(y[mask].mean())
        m = res["metrics"]
        ax.plot(xs, ys, marker="o", color=viz.MODEL_COLORS[name],
                label=(f"{MODEL_LABELS[name]} (Brier {m['brier']:.3f}, "
                       f"slope {m['calibration_slope']:.2f})"))

    ax.set_xlabel("Probabilitas prediksi")
    ax.set_ylabel("Proporsi kejadian teramati")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"Kalibrasi model pada {where}")
    ax.grid(axis="both")
    ax.legend(loc="upper left", fontsize=8)
    viz.annotate_source(ax, f"Titik = rerata per desil probabilitas prediksi ({n_bins} kelompok)")
    return viz.save_fig(fig, out_dir / fname)


def plot_fold_auc(nested: dict, auc_target: float, out_dir: Path) -> Path:
    """Sebaran AUC antar outer fold - menunjukkan kestabilan performa."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(nested)
    for i, name in enumerate(names):
        aucs = nested[name]["fold_metrics"]["auc_roc"].to_numpy()
        jitter = np.random.default_rng(0).normal(0, 0.045, len(aucs))
        ax.scatter(np.full(len(aucs), i) + jitter, aucs, s=46,
                   color=viz.MODEL_COLORS[name], alpha=0.85, zorder=3)
        ax.hlines(aucs.mean(), i - 0.24, i + 0.24, color=viz.INK_PRIMARY, linewidth=2, zorder=4)
        ax.annotate(f"{aucs.mean():.3f}", xy=(i + 0.28, aucs.mean()), fontsize=9,
                    color=viz.INK_SECONDARY, va="center")

    ax.axhline(auc_target, linestyle="--", linewidth=1.4, color=viz.STATUS["warning"])
    ax.annotate(f"target AUC {auc_target:g}", xy=(len(names) - 0.55, auc_target),
                xytext=(0, 6), textcoords="offset points", fontsize=8.5,
                color=viz.INK_SECONDARY)
    ax.set_xticks(range(len(names)), [MODEL_LABELS[n] for n in names])
    ax.set_ylabel("AUC-ROC per outer fold")
    ax.set_title("Kestabilan AUC pada nested cross-validation")
    viz.annotate_source(ax, "Garis hitam = rerata antar fold; titik = satu outer fold")
    return viz.save_fig(fig, out_dir / "04_auc_per_fold.png")


def plot_confusion(temporal: dict, out_dir: Path) -> Path:
    """Matriks konfusi tiap model pada ambang Youden dari data latih."""
    names = list(temporal)
    fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, name in zip(axes, names):
        res = temporal[name]
        thr = res["metrics"]["ambang"]
        pred = (np.asarray(res["prob_test"]) >= thr).astype(int)
        cm = confusion_matrix(res["y_test"], pred, labels=[0, 1])
        ax.imshow(cm, cmap=viz.sequential_cmap())
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm.sum()
                ax.annotate(f"{cm[i, j]:,}\n({share:.1%})", (j, i), ha="center", va="center",
                            fontsize=11, color="white" if share > 0.28 else viz.INK_PRIMARY)
        ax.set_xticks([0, 1], ["Prediksi\ntertutup", "Prediksi\nunder-reimb."], fontsize=8.5)
        ax.set_yticks([0, 1], ["Aktual\ntertutup", "Aktual\nunder-reimb."], fontsize=8.5)
        ax.grid(False)
        ax.set_title(f"{MODEL_LABELS[name]} (ambang {thr:.2f})", fontsize=10)

    fig.suptitle("Matriks konfusi pada data uji temporal", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save_fig(fig, out_dir / "05_matriks_konfusi.png")


def plot_decision_curve(temporal: dict, out_dir: Path, where: str = "data uji temporal",
                        fname: str = "06_decision_curve.png") -> Path:
    """Decision curve analysis: manfaat bersih pada berbagai ambang keputusan."""
    thresholds = np.linspace(0.05, 0.95, 91)
    fig, ax = plt.subplots(figsize=(8, 5.4))

    y = np.asarray(next(iter(temporal.values()))["y_test"], dtype=int)
    n = len(y)
    prevalence = y.mean()

    for name, res in temporal.items():
        p = np.asarray(res["prob_test"], dtype=float)
        nb = []
        for t in thresholds:
            pred = p >= t
            tp = float(np.sum(pred & (y == 1)))
            fp = float(np.sum(pred & (y == 0)))
            nb.append(tp / n - (fp / n) * (t / (1 - t)))
        ax.plot(thresholds, nb, color=viz.MODEL_COLORS[name], label=MODEL_LABELS[name])

    nb_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, nb_all, linestyle="--", linewidth=1.4, color=viz.INK_MUTED,
            label="Tandai semua episode")
    ax.axhline(0, linewidth=1.4, color=viz.BASELINE)
    ax.annotate("tidak menandai siapa pun", xy=(thresholds[-1], 0), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=8.5, color=viz.INK_MUTED)

    ax.set_xlabel("Ambang probabilitas keputusan")
    ax.set_ylabel("Manfaat bersih (net benefit)")
    ax.set_ylim(min(-0.05, prevalence - 0.65), prevalence + 0.05)
    ax.set_title(f"Decision curve analysis pada {where}")
    ax.legend(loc="upper right", fontsize=8.5)
    viz.annotate_source(ax, "Semakin tinggi kurva pada ambang yang relevan, semakin berguna model untuk pengambilan keputusan")
    return viz.save_fig(fig, out_dir / fname)
