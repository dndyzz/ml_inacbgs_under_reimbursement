"""Gambar untuk tahap preprocessing: alur seleksi dan bentuk outcome akhir."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import viz

STAGE = "02_preprocessing"


def plot_selection_flow(flow: pd.DataFrame, out_dir: Path) -> Path:
    """Diagram alur seleksi subjek (Gambar 3.1) bergaya CONSORT.

    Kolom kiri = jumlah episode yang bertahan, kolom kanan = alasan dikeluarkan.
    """
    steps = flow.to_dict("records")
    exclusions = [s for s in steps if s["n_keluar"] > 0]
    # Satu kotak utama untuk kondisi awal + satu setelah tiap langkah eksklusi
    n_main = len(exclusions) + 1
    counts = [steps[0]["n_sisa"]] + [s["n_sisa"] for s in exclusions]

    row_h, box_h = 1.0, 0.52
    fig, ax = plt.subplots(figsize=(10.5, 1.18 * n_main + 0.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.25, n_main * row_h)
    ax.axis("off")

    def box(x, y, w, h, text, color, text_color=viz.INK_PRIMARY, fontsize=9.5, weight="normal"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.08",
            facecolor=color, edgecolor="none",
        ))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=text_color, linespacing=1.4, fontweight=weight)

    for i in range(n_main):
        y = (n_main - 1 - i) * row_h
        is_last = i == n_main - 1
        if i == 0:
            label, face, ink, weight = (
                f"{steps[0]['tahap']}\nn = {counts[i]:,} episode", "#cde2fb", viz.INK_PRIMARY, "normal")
        elif is_last:
            label, face, ink, weight = (
                f"Kohort analisis\nn = {counts[i]:,} episode", viz.SERIES[0], "white", "bold")
        else:
            label, face, ink, weight = (
                f"Tersisa\nn = {counts[i]:,} episode", "#eef4fb", viz.INK_PRIMARY, "normal")
        box(0.3, y, 4.4, box_h, label, face, text_color=ink, weight=weight)

        if is_last:
            continue

        # panah turun ke kotak berikutnya
        y_next = (n_main - 2 - i) * row_h + box_h
        ax.add_patch(FancyArrowPatch(
            (2.5, y), (2.5, y_next), arrowstyle="-|>", mutation_scale=12,
            color=viz.BASELINE, linewidth=1.5,
        ))
        # cabang ke kanan menuju kotak eksklusi
        y_mid = (y + y_next) / 2
        ax.add_patch(FancyArrowPatch(
            (2.5, y_mid), (5.1, y_mid), arrowstyle="-|>", mutation_scale=11,
            color=viz.BASELINE, linewidth=1.2,
        ))
        step = exclusions[i]
        label = textwrap.fill(step["tahap"].replace("Dikeluarkan: ", ""), 48)
        box(5.2, y_mid - 0.24, 4.6, 0.48,
            f"{label}\nn = {step['n_keluar']:,}", "#f7e8e2",
            text_color=viz.INK_SECONDARY, fontsize=8.2)

    ax.set_title("Alur seleksi subjek (Gambar 3.1)", loc="left", pad=12)
    return viz.save_fig(fig, out_dir / "01_alur_seleksi.png")


def plot_cohort_shrinkage(flow: pd.DataFrame, out_dir: Path) -> Path:
    """Batang jumlah episode tersisa setelah tiap langkah seleksi."""
    d = flow[flow["tahap"] != "Kohort analisis akhir"].copy()
    labels = [textwrap.fill(t.replace("Dikeluarkan: ", "− "), 34) for t in d["tahap"]]

    fig, ax = plt.subplots(figsize=(9, 0.72 * len(d) + 2.2))
    colors = [viz.SERIES[0]] + [viz.SERIES[1]] * (len(d) - 1)
    bars = ax.barh(range(len(d)), d["n_sisa"], color=colors, height=0.6)
    ax.set_yticks(range(len(d)), labels, fontsize=8.5)
    ax.invert_yaxis()
    for i, (bar, n_sisa, n_keluar) in enumerate(zip(bars, d["n_sisa"], d["n_keluar"])):
        note = f"{n_sisa:,}" + (f"  (−{n_keluar:,})" if n_keluar else "")
        ax.annotate(note, xy=(n_sisa, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xlabel("Jumlah episode tersisa")
    ax.set_xlim(0, d["n_sisa"].max() * 1.2)
    ax.grid(axis="x")
    ax.set_title("Dampak tiap langkah pembersihan terhadap ukuran kohort")
    return viz.save_fig(fig, out_dir / "02_dampak_pembersihan.png")


def plot_outcome_overview(df: pd.DataFrame, cfg, out_dir: Path) -> Path:
    """Tiga panel: proporsi outcome, sebaran log-rasio, komposisi split temporal."""
    out = cfg["preprocessing"]["outcome"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))

    # (a) proporsi outcome
    ax = axes[0]
    counts = df[out["binary_col"]].value_counts().sort_index()
    labels = ["Klaim menutup biaya", "Under-reimbursement"]
    colors = [viz.SERIES[0], viz.STATUS["critical"]]
    bars = ax.bar(labels, counts.values, color=colors, width=0.55)
    for bar, value in zip(bars, counts.values):
        ax.annotate(f"{value:,}\n({value/len(df):.1%})",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=9.5, color=viz.INK_SECONDARY)
    ax.set_ylabel("Jumlah episode")
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_title("Outcome primer (biner)")
    ax.tick_params(axis="x", labelsize=9)

    # (b) outcome sekunder
    ax = axes[1]
    ax.hist(df[out["continuous_col"]], bins=40, color=viz.SERIES[0], alpha=0.9)
    ax.axvline(0, color=viz.STATUS["critical"], linewidth=2)
    ax.annotate("titik impas\nlog rasio = 0", xy=(0, ax.get_ylim()[1] * 0.9),
                xytext=(8, 0), textcoords="offset points", fontsize=9,
                color=viz.STATUS["critical"], va="top")
    ax.set_xlabel("log(klaim / tagihan)")
    ax.set_ylabel("Jumlah episode")
    ax.set_title("Outcome sekunder (magnitudo)")

    # (c) komposisi periode
    ax = axes[2]
    comp = pd.crosstab(df["periode"], df[out["binary_col"]], normalize="index")
    idx = np.arange(len(comp))
    bottom = np.zeros(len(comp))
    for col, color, label in zip([0, 1], colors, labels):
        if col not in comp:
            continue
        ax.bar(idx, comp[col], bottom=bottom, color=color, width=0.5,
               label=label, edgecolor=viz.SURFACE, linewidth=2)
        bottom += comp[col].to_numpy()
    ax.set_xticks(idx, comp.index, fontsize=9)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Komposisi outcome per periode")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncols=2)

    fig.tight_layout()
    return viz.save_fig(fig, out_dir / "03_outcome_kohort.png")


def plot_missingness_after(report: pd.DataFrame, out_dir: Path) -> Path:
    """Nilai hilang yang tersisa pada kohort analisis (bahan strategi imputasi)."""
    d = report[report["n_hilang"] > 0].sort_values("persen_hilang")
    if d.empty:
        d = pd.DataFrame({"variabel": ["(tidak ada)"], "persen_hilang": [0.0]})

    fig, ax = plt.subplots(figsize=(8, max(2.8, 0.45 * len(d) + 1.4)))
    bars = ax.barh(d["variabel"], d["persen_hilang"], color=viz.SERIES[0], height=0.6)
    for bar, value in zip(bars, d["persen_hilang"]):
        ax.annotate(f"{value:.1f}%", xy=(value, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=viz.INK_SECONDARY)
    ax.set_xlabel("Persentase nilai hilang pada kohort analisis (%)")
    ax.set_xlim(0, max(d["persen_hilang"].max() * 1.3, 1))
    ax.grid(axis="x")
    ax.set_title("Sisa nilai hilang setelah pembersihan")
    viz.annotate_source(ax, "Nilai hilang TIDAK diimputasi di sini; imputasi MICE dilakukan di dalam fold (tahap 04)")
    return viz.save_fig(fig, out_dir / "04_sisa_nilai_hilang.png")
