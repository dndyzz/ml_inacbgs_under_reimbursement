"""Gambar untuk tahap feature building."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import viz

STAGE = "03_feature_building"


def plot_derived_features(X: pd.DataFrame, out_dir: Path) -> Path:
    """Sebaran fitur turunan: skor mSOFA, jumlah organ support, log pra-ICU, flag GCS."""
    panels = [c for c in
              ["msofa_total", "organ_support_count", "pre_icu_los_log1p", "gcs_unassessable"]
              if c in X]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 4.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, panels):
        s = pd.to_numeric(X[col], errors="coerce").dropna()
        if s.nunique() <= 6:  # variabel cacah/biner -> batang
            counts = s.value_counts().sort_index()
            bars = ax.bar(counts.index.astype(str), counts.values,
                          color=viz.SERIES[0], width=0.6)
            for bar, value in zip(bars, counts.values):
                ax.annotate(f"{value:,}", xy=(bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8.5, color=viz.INK_SECONDARY)
            ax.set_ylim(0, counts.max() * 1.18)
        else:
            ax.hist(s, bins=28, color=viz.SERIES[0], alpha=0.9)
            ax.axvline(s.median(), color=viz.INK_MUTED, linestyle="--", linewidth=1.4)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)

    fig.suptitle("Sebaran fitur turunan", x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return viz.save_fig(fig, out_dir / "01_fitur_turunan.png")


def plot_vif(vif: pd.DataFrame, threshold: float, out_dir: Path) -> Path:
    """VIF per variabel numerik dengan garis ambang."""
    d = vif.sort_values("vif")
    colors = [viz.STATUS["critical"] if v > threshold else viz.SERIES[0] for v in d["vif"]]

    fig, ax = plt.subplots(figsize=(8.5, 0.46 * len(d) + 2.0))
    bars = ax.barh(d["variabel"], d["vif"].replace(np.inf, d["vif"].max() * 1.2), color=colors, height=0.6)
    for bar, value in zip(bars, d["vif"]):
        ax.annotate(f"{value:.1f}", xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.INK_SECONDARY)
    ax.axvline(threshold, color=viz.INK_MUTED, linestyle="--", linewidth=1.4)
    ax.annotate(f"ambang VIF = {threshold:g}", xy=(threshold, len(d) - 0.4),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=viz.INK_MUTED)
    ax.set_xlabel("Variance Inflation Factor")
    ax.grid(axis="x")
    ax.set_title("Multikolinearitas antarvariabel numerik")
    viz.annotate_source(ax, "Merah = di atas ambang; ditoleransi karena model pohon & elastic net menangani kolinearitas")
    return viz.save_fig(fig, out_dir / "02_vif.png")


def plot_feature_correlation(X: pd.DataFrame, numeric_cols: list[str], out_dir: Path) -> Path:
    """Korelasi Spearman antarfitur numerik final."""
    cols = [c for c in numeric_cols if c in X]
    corr = X[cols].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(1.2 + 0.66 * len(cols), 1.0 + 0.58 * len(cols)))
    im = ax.imshow(corr.values, cmap=viz.diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.values[i, j]
            ax.annotate(f"{value:.2f}", (j, i), ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 else viz.INK_PRIMARY)
    ax.grid(False)
    ax.set_title("Korelasi antarfitur numerik final")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "03_korelasi_fitur.png")


def plot_feature_map(spec: dict, n_encoded: int, out_dir: Path) -> Path:
    """Berapa kolom yang masuk model dari tiap jenis variabel, sebelum vs sesudah one-hot."""
    kinds = ["numeric", "ordinal", "binary", "categorical"]
    labels = ["Numerik", "Ordinal", "Biner", "Kategorik (one-hot)"]
    before = [len(spec[k]) for k in kinds]
    after = before[:3] + [n_encoded - sum(before[:3])]

    y = np.arange(len(kinds))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(y - 0.2, before, height=0.36, color=viz.SERIES[0], label="Sebelum encoding")
    ax.barh(y + 0.2, after, height=0.36, color=viz.SERIES[1], label="Setelah encoding")
    for i, (b, a) in enumerate(zip(before, after)):
        ax.annotate(f"{b}", xy=(b, i - 0.2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)
        ax.annotate(f"{a}", xy=(a, i + 0.2), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Jumlah kolom")
    ax.set_xlim(0, max(max(before), max(after)) * 1.25)
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    ax.set_title(f"Matriks fitur: {spec['n_features_pre_encoding']} variabel → {n_encoded} kolom model")
    return viz.save_fig(fig, out_dir / "04_peta_fitur.png")
