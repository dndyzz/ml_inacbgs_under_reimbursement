"""Gambar-gambar untuk tahap EDA.

Setiap fungsi menyimpan satu PNG dan mengembalikan path-nya, sehingga notebook
cukup menampilkan file yang sama dengan hasil menjalankan lewat terminal.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import viz

STAGE = "01_eda"


def plot_missingness(df: pd.DataFrame, out_dir: Path) -> Path:
    """Batang persentase nilai hilang per kolom (hanya kolom yang bermasalah)."""
    miss = (df.isna().mean() * 100).sort_values(ascending=True)
    miss = miss[miss > 0]
    if miss.empty:
        miss = pd.Series({"(tidak ada nilai hilang)": 0.0})

    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(miss) + 1.2)))
    bars = ax.barh(miss.index, miss.values, color=viz.SERIES[0], height=0.62)
    for bar, value in zip(bars, miss.values):
        ax.annotate(f"{value:.1f}%", xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xlabel("Persentase nilai hilang (%)")
    ax.set_title("Nilai hilang per variabel")
    ax.set_xlim(0, max(miss.max() * 1.25, 1))
    ax.grid(axis="x")
    viz.annotate_source(ax, f"n = {len(df):,} baris data mentah")
    return viz.save_fig(fig, out_dir / "01_missingness.png")


def plot_numeric_distributions(df: pd.DataFrame, cols: list[str], out_dir: Path) -> Path:
    """Grid histogram variabel numerik utama."""
    cols = [c for c in cols if c in df]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.9 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(s, bins=30, color=viz.SERIES[0], alpha=0.9)
        ax.axvline(s.median(), color=viz.INK_MUTED, linestyle="--", linewidth=1.4)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Sebaran variabel numerik (garis putus-putus = median)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return viz.save_fig(fig, out_dir / "02_distribusi_numerik.png")


def plot_categorical_distributions(df: pd.DataFrame, cols: list[str], out_dir: Path) -> Path:
    """Grid batang variabel kategorik."""
    cols = [c for c in cols if c in df]
    ncol = 2
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        counts = df[col].value_counts(dropna=False).sort_values()
        labels = ["(hilang)" if pd.isna(i) else str(i) for i in counts.index]
        ax.barh(labels, counts.values, color=viz.SERIES[0], height=0.6)
        for y, value in enumerate(counts.values):
            ax.annotate(f"{value:,}", xy=(value, y), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=8, color=viz.INK_SECONDARY)
        ax.set_title(col, fontsize=10)
        ax.set_xlim(0, counts.max() * 1.18)
        ax.grid(axis="x")
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Sebaran variabel kategorik", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return viz.save_fig(fig, out_dir / "03_distribusi_kategorik.png")


def plot_cost_vs_claim(df: pd.DataFrame, out_dir: Path) -> Path:
    """Dua panel: sebaran biaya vs klaim, dan sebaran rasio klaim/biaya."""
    d = df.dropna(subset=["total_hospital_billing", "inacbg_claim"]).copy()
    d["ratio"] = d["inacbg_claim"] / d["total_hospital_billing"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    billing = d["total_hospital_billing"] / 1e6
    claim = d["inacbg_claim"] / 1e6
    under = d["ratio"] < 1
    ax.scatter(billing[~under], claim[~under], s=12, alpha=0.6,
               color=viz.SERIES[0], edgecolor="none", label="Klaim menutup biaya")
    ax.scatter(billing[under], claim[under], s=12, alpha=0.6,
               color=viz.STATUS["critical"], edgecolor="none", label="Under-reimbursement")
    lim = float(np.nanpercentile(billing, 99))
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.5, color=viz.INK_MUTED,
            label="Garis impas")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Tagihan rumah sakit (Rp juta)")
    ax.set_ylabel("Klaim INA-CBGs (Rp juta)")
    ax.set_title("Klaim vs tagihan per episode")
    ax.grid(axis="both")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    ax.hist(d["ratio"].clip(upper=3), bins=45, color=viz.SERIES[0], alpha=0.9)
    ax.axvline(1.0, color=viz.STATUS["critical"], linewidth=2)
    ax.annotate("titik impas\n(rasio = 1)", xy=(1.0, ax.get_ylim()[1] * 0.88),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color=viz.STATUS["critical"], va="top")
    ax.set_xlabel("Rasio klaim / tagihan")
    ax.set_ylabel("Jumlah episode")
    ax.set_title("Sebaran rasio klaim terhadap tagihan")
    viz.annotate_source(ax, f"n = {len(d):,} episode • area kiri garis merah = under-reimbursement")

    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "04_biaya_vs_klaim.png")


def plot_prevalence_by_group(prev_tables: dict[str, pd.DataFrame], out_dir: Path) -> Path:
    """Plot titik + IK 95% prevalensi under-reimbursement per subkelompok."""
    frames = [t for t in prev_tables.values() if not t.empty]
    data = pd.concat(frames, ignore_index=True)
    data["label"] = data["variabel"] + ": " + data["kategori"].astype(str)
    data = data.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(4.0, 0.34 * len(data) + 1.5)))
    y = np.arange(len(data))
    ax.hlines(y, data["ik95_bawah"], data["ik95_atas"],
              color=viz.BASELINE, linewidth=2.5)
    ax.scatter(data["prevalensi"], y, s=42, color=viz.SERIES[0], zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(data["label"], fontsize=8)
    ax.set_xlabel("Prevalensi under-reimbursement (IK 95%)")
    ax.set_title("Prevalensi under-reimbursement menurut subkelompok")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.grid(axis="x")
    viz.annotate_source(ax, "Titik = proporsi episode dengan rasio klaim/tagihan < 1")
    return viz.save_fig(fig, out_dir / "05_prevalensi_subkelompok.png")


def plot_numeric_by_outcome(df: pd.DataFrame, cols: list[str], outcome: str, out_dir: Path) -> Path:
    """Boxplot variabel numerik menurut status under-reimbursement."""
    cols = [c for c in cols if c in df]
    ncol = 3
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        groups = [
            pd.to_numeric(df.loc[df[outcome] == g, col], errors="coerce").dropna()
            for g in (0, 1)
        ]
        bp = ax.boxplot(groups, patch_artist=True, widths=0.55,
                        medianprops=dict(color=viz.INK_PRIMARY, linewidth=1.6),
                        flierprops=dict(markersize=2.5, markerfacecolor=viz.INK_MUTED,
                                        markeredgecolor="none", alpha=0.5))
        for patch, color in zip(bp["boxes"], [viz.SERIES[0], viz.STATUS["critical"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor(viz.SURFACE)
            patch.set_linewidth(2)
        ax.set_xticklabels(["Tertutup", "Under-reimb."], fontsize=8)
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle("Variabel numerik menurut status under-reimbursement",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return viz.save_fig(fig, out_dir / "06_numerik_vs_outcome.png")


def plot_correlation(df: pd.DataFrame, cols: list[str], out_dir: Path) -> Path:
    """Peta panas korelasi Spearman antarvariabel numerik (deteksi kolinearitas)."""
    cols = [c for c in cols if c in df]
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(cols), 0.9 + 0.55 * len(cols)))
    im = ax.imshow(corr.values, cmap=viz.diverging_cmap(), vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            value = corr.values[i, j]
            ax.annotate(f"{value:.2f}", (j, i), ha="center", va="center", fontsize=7,
                        color="white" if abs(value) > 0.55 else viz.INK_PRIMARY)
    ax.grid(False)
    ax.set_title("Korelasi Spearman antarvariabel numerik")
    fig.colorbar(im, ax=ax, shrink=0.8, label="koefisien korelasi")
    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "07_korelasi.png")


def plot_temporal(df: pd.DataFrame, outcome: str, out_dir: Path) -> Path:
    """Volume episode dan prevalensi outcome per bulan (dasar pembagian temporal)."""
    d = df.dropna(subset=["icu_admission_datetime"]).copy()
    d["bulan"] = pd.to_datetime(d["icu_admission_datetime"]).dt.to_period("M").dt.to_timestamp()
    monthly = d.groupby("bulan").agg(
        episode=("episode_id", "count"), prevalensi=(outcome, "mean")
    ).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)

    ax = axes[0]
    ax.bar(monthly["bulan"], monthly["episode"], width=20,
           color=viz.SERIES[0], label="Jumlah episode")
    ax.set_ylabel("Episode per bulan")
    ax.set_title("Volume admisi ICU per bulan")

    ax = axes[1]
    ax.plot(monthly["bulan"], monthly["prevalensi"], marker="o",
            color=viz.SERIES[1], label="Prevalensi under-reimbursement")
    ax.axhline(monthly["prevalensi"].mean(), linestyle="--", linewidth=1.3,
               color=viz.INK_MUTED, label="Rerata periode")
    ax.set_ylabel("Prevalensi")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Prevalensi under-reimbursement per bulan")
    ax.legend(loc="lower right", fontsize=8)
    viz.annotate_source(ax, "Stabilitas antarbulan menjadi dasar validasi internal-eksternal (split temporal)")

    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "08_tren_bulanan.png")
