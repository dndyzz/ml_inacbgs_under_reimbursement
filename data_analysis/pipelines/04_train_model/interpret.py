"""Interpretabilitas model: SHAP (TreeSHAP) dan permutation importance.

SHAP menjawab "variabel apa yang mendorong prediksi, ke arah mana, dan pada
pasien yang mana". Permutation importance dipakai sebagai pembanding metode:
bila peringkat keduanya sejalan, temuan lebih meyakinkan.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from common import viz
from models import MODEL_LABELS

STAGE = "04_train_model"


# ---------------------------------------------------------------------------
# Perhitungan
# ---------------------------------------------------------------------------

def transform_features(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Terapkan preprocessor terlatih -> matriks numerik dengan nama kolom."""
    prep = pipeline.named_steps["prep"]
    values = prep.transform(X)
    names = [str(n) for n in prep.get_feature_names_out()]
    return pd.DataFrame(values, columns=names, index=X.index)


def compute_shap(pipeline, X: pd.DataFrame):
    """Hitung nilai SHAP (TreeSHAP) untuk model berbasis pohon."""
    model = pipeline.named_steps["model"]
    Xt = transform_features(pipeline, X)
    explainer = shap.TreeExplainer(model)
    explanation = explainer(Xt)

    # Random Forest mengembalikan nilai untuk kedua kelas -> ambil kelas positif
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    return explanation, Xt


def shap_importance_table(explanation, Xt: pd.DataFrame) -> pd.DataFrame:
    """Tabel 4.3 - peringkat kontribusi prediktor berdasarkan analisis SHAP."""
    values = np.asarray(explanation.values)
    mean_abs = np.abs(values).mean(axis=0)

    rows = []
    for i, col in enumerate(Xt.columns):
        x = Xt[col].to_numpy(dtype=float)
        s = values[:, i]
        # Arah pengaruh: korelasi nilai fitur dengan kontribusi SHAP-nya
        corr = np.corrcoef(x, s)[0, 1] if np.std(x) > 0 and np.std(s) > 0 else np.nan
        arah = "-" if not np.isfinite(corr) else (
            "nilai tinggi menaikkan risiko" if corr > 0.05
            else "nilai tinggi menurunkan risiko" if corr < -0.05
            else "tidak monoton"
        )
        rows.append({
            "variabel": col,
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "korelasi_fitur_vs_shap": round(float(corr), 3) if np.isfinite(corr) else np.nan,
            "arah_pengaruh": arah,
        })

    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    table.insert(0, "peringkat", np.arange(1, len(table) + 1))
    total = table["mean_abs_shap"].sum()
    table["kontribusi_relatif"] = (table["mean_abs_shap"] / total).round(4) if total else np.nan
    return table


def permutation_table(pipelines: dict, X: pd.DataFrame, y: np.ndarray, cfg) -> pd.DataFrame:
    """Permutation importance (penurunan AUC saat satu variabel diacak)."""
    rows = []
    for name, pipeline in pipelines.items():
        result = permutation_importance(
            pipeline, X, y, scoring="roc_auc", n_repeats=10,
            random_state=cfg.seed, n_jobs=1,
        )
        for i, col in enumerate(X.columns):
            rows.append({
                "model": MODEL_LABELS.get(name, name),
                "variabel": col,
                "penurunan_auc": round(float(result.importances_mean[i]), 5),
                "sd": round(float(result.importances_std[i]), 5),
            })
    return pd.DataFrame(rows).sort_values(["model", "penurunan_auc"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Gambar
# ---------------------------------------------------------------------------

def plot_shap_beeswarm(explanation, out_dir: Path, max_display: int = 15) -> Path:
    """Beeswarm: sebaran kontribusi tiap variabel pada seluruh pasien uji."""
    viz.set_theme()
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                        color=viz.diverging_cmap())
    fig = plt.gcf()
    fig.set_size_inches(9, 0.42 * max_display + 2.2)
    fig.suptitle("SHAP beeswarm — kontribusi tiap variabel per pasien",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return viz.save_fig(fig, out_dir / "07_shap_beeswarm.png")


def plot_shap_bar(table: pd.DataFrame, out_dir: Path, max_display: int = 15) -> Path:
    """Peringkat global: rerata |SHAP| tiap variabel."""
    d = table.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(d) + 1.8))
    bars = ax.barh(d["variabel"], d["mean_abs_shap"], color=viz.SERIES[0], height=0.62)
    for bar, value, share in zip(bars, d["mean_abs_shap"], d["kontribusi_relatif"]):
        ax.annotate(f"{value:.3f}  ({share:.0%})",
                    xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=viz.INK_SECONDARY)
    ax.set_xlabel("Rerata |nilai SHAP| (log-odds)")
    ax.set_xlim(0, d["mean_abs_shap"].max() * 1.32)
    ax.grid(axis="x")
    ax.set_title("Kepentingan variabel global (SHAP)")
    viz.annotate_source(ax, "Persentase = kontribusi relatif terhadap total |SHAP| seluruh variabel")
    return viz.save_fig(fig, out_dir / "08_shap_importance.png")


def plot_shap_dependence(explanation, Xt: pd.DataFrame, table: pd.DataFrame,
                         out_dir: Path, n: int = 6) -> Path:
    """Dependence plot: bentuk hubungan nilai variabel dengan kontribusinya."""
    top = table["variabel"].head(n).tolist()
    values = np.asarray(explanation.values)
    ncol = 3
    nrow = int(np.ceil(len(top) / ncol))

    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, top):
        i = Xt.columns.get_loc(col)
        x = Xt[col].to_numpy(dtype=float)
        s = values[:, i]
        sc = ax.scatter(x, s, c=x, cmap=viz.sequential_cmap(), s=16, alpha=0.85,
                        edgecolor="none")
        ax.axhline(0, linewidth=1.1, color=viz.BASELINE)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Nilai SHAP", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[len(top):]:
        ax.set_visible(False)

    fig.suptitle("Dependence plot — bentuk pengaruh variabel teratas",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return viz.save_fig(fig, out_dir / "09_shap_dependence.png")


def plot_shap_waterfall(explanation, probs: np.ndarray, out_dir: Path,
                        n_cases: int = 2) -> list[Path]:
    """Waterfall: penjelasan individual pada pasien berisiko tertinggi & terendah."""
    order = np.argsort(probs)
    picks = [("risiko_tertinggi", int(order[-1])), ("risiko_terendah", int(order[0]))][:n_cases]

    paths = []
    for label, idx in picks:
        viz.set_theme()
        shap.plots.waterfall(explanation[idx], max_display=12, show=False)
        fig = plt.gcf()
        fig.set_size_inches(9.5, 6.2)
        fig.suptitle(
            f"Penjelasan individual — pasien {label.replace('_', ' ')} "
            f"(probabilitas prediksi {probs[idx]:.2f})",
            x=0.02, ha="left", fontsize=11.5, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        paths.append(viz.save_fig(fig, out_dir / f"10_shap_waterfall_{label}.png"))
    return paths


def plot_shap_interaction(pipeline, Xt: pd.DataFrame, table: pd.DataFrame,
                          out_dir: Path, n_features: int = 8, sample: int = 300,
                          seed: int = 42) -> Path:
    """Peta panas interaksi SHAP antarvariabel (eksploratif, sesuai tujuan khusus 3)."""
    model = pipeline.named_steps["model"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xt), size=min(sample, len(Xt)), replace=False)
    subset = Xt.iloc[idx]

    explainer = shap.TreeExplainer(model)
    inter = explainer.shap_interaction_values(subset)
    if isinstance(inter, list):
        inter = inter[1]
    inter = np.asarray(inter)
    if inter.ndim == 4:  # (n, f, f, kelas)
        inter = inter[..., 1]

    top = table["variabel"].head(n_features).tolist()
    pos = [Xt.columns.get_loc(c) for c in top]
    matrix = np.abs(inter).mean(axis=0)[np.ix_(pos, pos)]
    np.fill_diagonal(matrix, 0)  # fokus pada interaksi, bukan efek utama

    fig, ax = plt.subplots(figsize=(1.4 + 0.72 * len(top), 1.1 + 0.66 * len(top)))
    im = ax.imshow(matrix, cmap=viz.sequential_cmap())
    ax.set_xticks(range(len(top)), top, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)), top, fontsize=8)
    vmax = matrix.max() if matrix.size else 1
    for i in range(len(top)):
        for j in range(len(top)):
            if i == j:
                continue
            ax.annotate(f"{matrix[i, j]:.3f}", (j, i), ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[i, j] > 0.55 * vmax else viz.INK_PRIMARY)
    ax.grid(False)
    ax.set_title("Interaksi SHAP antarvariabel (rerata |nilai interaksi|)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="kekuatan interaksi")
    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "11_shap_interaksi.png")


def plot_permutation_importance(perm: pd.DataFrame, out_dir: Path, top_n: int = 12) -> Path:
    """Perbandingan permutation importance antarmodel (pembanding metode SHAP)."""
    models = perm["model"].unique().tolist()
    order = (perm.groupby("variabel")["penurunan_auc"].mean()
             .sort_values(ascending=False).head(top_n).index.tolist())[::-1]

    y = np.arange(len(order))
    height = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(order) + 2.0))
    for k, model in enumerate(models):
        d = perm[perm["model"] == model].set_index("variabel").reindex(order)
        color = next((c for n, c in viz.MODEL_COLORS.items() if MODEL_LABELS[n] == model),
                     viz.SERIES[k % len(viz.SERIES)])
        ax.barh(y + (k - (len(models) - 1) / 2) * height, d["penurunan_auc"],
                height=height * 0.92, color=color, label=model)

    ax.set_yticks(y, order, fontsize=8.5)
    ax.axvline(0, linewidth=1.2, color=viz.BASELINE)
    ax.set_xlabel("Penurunan AUC saat variabel diacak")
    ax.grid(axis="x")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Permutation importance per model")
    viz.annotate_source(ax, "Dihitung pada data uji temporal, 10 pengulangan acak per variabel")
    return viz.save_fig(fig, out_dir / "12_permutation_importance.png")
