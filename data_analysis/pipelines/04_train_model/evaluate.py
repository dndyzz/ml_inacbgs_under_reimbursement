"""Validasi, metrik performa, dan gambar evaluasi model.

Dua skema validasi dijalankan, sesuai rencana analisis proposal:

1. **Nested cross-validation** (5 outer x 3 inner) - estimasi performa yang tidak
   optimistis: hyperparameter dipilih di inner fold, dinilai di outer fold.
2. **Validasi temporal (internal-eksternal)** - dilatih pada admisi Mei-Des 2025,
   diuji pada Jan-Mei 2026, untuk melihat kestabilan terhadap pergeseran waktu.

Metrik: diskriminasi (AUC-ROC, PR-AUC, sensitivitas, spesifisitas, F1) dan
kalibrasi (Brier score, calibration intercept & slope).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from common import viz
from models import MODEL_LABELS

STAGE = "04_train_model"
EPS = 1e-9


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

    slope_model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
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
        "brier": brier_score_loss(y_true, y_prob),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "ambang": thr,
    }


# ---------------------------------------------------------------------------
# Skema validasi
# ---------------------------------------------------------------------------

def nested_cv(name: str, pipeline, grid: dict, X: pd.DataFrame, y: np.ndarray, cfg) -> dict:
    """Nested cross-validation; kembalikan metrik per fold dan prediksi out-of-fold."""
    tr = cfg["training"]
    outer = StratifiedKFold(n_splits=int(tr["outer_folds"]), shuffle=True, random_state=cfg.seed)
    inner = StratifiedKFold(n_splits=int(tr["inner_folds"]), shuffle=True, random_state=cfg.seed)

    oof = np.full(len(y), np.nan)
    fold_rows, best_params = [], []

    for fold, (train_idx, test_idx) in enumerate(outer.split(X, y), start=1):
        search = GridSearchCV(
            pipeline, grid, scoring=tr["scoring"], cv=inner,
            n_jobs=int(tr["n_jobs"]), refit=True,
        )
        search.fit(X.iloc[train_idx], y[train_idx])
        prob = search.predict_proba(X.iloc[test_idx])[:, 1]
        oof[test_idx] = prob

        metrics = classification_metrics(y[test_idx], prob)
        fold_rows.append({"model": name, "fold": fold, **metrics})
        best_params.append(search.best_params_)

    return {
        "fold_metrics": pd.DataFrame(fold_rows),
        "oof_pred": oof,
        "best_params": best_params,
        "oof_metrics": classification_metrics(y, oof),
    }


def temporal_validation(name: str, pipeline, grid: dict, X: pd.DataFrame, y: np.ndarray,
                        train_mask: np.ndarray, cfg) -> dict:
    """Latih pada periode awal, uji pada periode berikutnya (validasi temporal)."""
    tr = cfg["training"]
    inner = StratifiedKFold(n_splits=int(tr["inner_folds"]), shuffle=True, random_state=cfg.seed)
    search = GridSearchCV(pipeline, grid, scoring=tr["scoring"], cv=inner,
                          n_jobs=int(tr["n_jobs"]), refit=True)
    search.fit(X[train_mask], y[train_mask])

    prob = search.predict_proba(X[~train_mask])[:, 1]
    return {
        "model": name,
        "estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "y_test": y[~train_mask],
        "prob_test": prob,
        "metrics": classification_metrics(y[~train_mask], prob),
        "n_train": int(train_mask.sum()),
        "n_test": int((~train_mask).sum()),
    }


def comparison_table(nested: dict, temporal: dict, auc_target: float) -> pd.DataFrame:
    """Tabel 4.2 - perbandingan performa antarmodel."""
    rows = []
    for name in nested:
        folds = nested[name]["fold_metrics"]
        auc_mean, auc_sd = folds["auc_roc"].mean(), folds["auc_roc"].std()
        temp = temporal[name]["metrics"]
        rows.append({
            "model": MODEL_LABELS.get(name, name),
            "auc_nested_cv": f"{auc_mean:.3f} ± {auc_sd:.3f}",
            "auc_validasi_temporal": f"{temp['auc_roc']:.3f}",
            "pr_auc": f"{temp['pr_auc']:.3f}",
            "sensitivitas": f"{temp['sensitivitas']:.3f}",
            "spesifisitas": f"{temp['spesifisitas']:.3f}",
            "f1": f"{temp['f1']:.3f}",
            "brier": f"{temp['brier']:.3f}",
            "calibration_intercept": f"{temp['calibration_intercept']:.3f}",
            "calibration_slope": f"{temp['calibration_slope']:.3f}",
            "memenuhi_target_auc": "ya" if auc_mean >= auc_target else "tidak",
            "_auc_sort": auc_mean,
        })
    table = pd.DataFrame(rows).sort_values("_auc_sort", ascending=False).drop(columns="_auc_sort")
    return table.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gambar
# ---------------------------------------------------------------------------

def plot_roc(temporal: dict, out_dir: Path) -> Path:
    """Kurva ROC ketiga model pada data uji temporal."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in temporal.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["prob_test"])
        ax.plot(fpr, tpr, color=viz.MODEL_COLORS[name],
                label=f"{MODEL_LABELS[name]} (AUC {res['metrics']['auc_roc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color=viz.INK_MUTED,
            label="Tebakan acak")
    ax.set_xlabel("1 − spesifisitas")
    ax.set_ylabel("Sensitivitas")
    ax.set_title("Kurva ROC pada data uji temporal")
    ax.grid(axis="both")
    ax.legend(loc="lower right")
    viz.annotate_source(ax, "Data uji = admisi ICU Jan-Mei 2026 (tidak dipakai saat pelatihan)")
    return viz.save_fig(fig, out_dir / "01_kurva_roc.png")


def plot_pr(temporal: dict, out_dir: Path) -> Path:
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
    ax.set_title("Kurva precision-recall pada data uji temporal")
    ax.grid(axis="both")
    ax.legend(loc="lower left")
    return viz.save_fig(fig, out_dir / "02_kurva_pr.png")


def plot_calibration(temporal: dict, out_dir: Path, n_bins: int = 10) -> Path:
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
    ax.set_title("Kalibrasi model pada data uji temporal")
    ax.grid(axis="both")
    ax.legend(loc="upper left", fontsize=8)
    viz.annotate_source(ax, f"Titik = rerata per desil probabilitas prediksi ({n_bins} kelompok)")
    return viz.save_fig(fig, out_dir / "03_kalibrasi.png")


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
    """Matriks konfusi tiap model pada ambang Youden."""
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


def plot_decision_curve(temporal: dict, out_dir: Path) -> Path:
    """Decision curve analysis: manfaat bersih pada berbagai ambang keputusan."""
    thresholds = np.linspace(0.05, 0.85, 60)
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
    ax.set_title("Decision curve analysis")
    ax.legend(loc="upper right", fontsize=8.5)
    viz.annotate_source(ax, "Semakin tinggi kurva pada ambang yang relevan, semakin berguna model untuk pengambilan keputusan")
    return viz.save_fig(fig, out_dir / "06_decision_curve.png")
