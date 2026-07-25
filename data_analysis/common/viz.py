"""Tema visual tunggal untuk seluruh figure penelitian.

Semua gambar memakai palet, ketebalan garis, dan tipografi yang sama supaya
lampiran tesis terlihat sebagai satu sistem. Figure selalu disimpan sebagai PNG
300 dpi lalu ditutup; notebook menampilkan file PNG tersebut, sehingga hasil
menjalankan lewat ``python main.py`` dan lewat notebook identik.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Palet kategorik (urutan slot tetap, tidak pernah diputar) -------------
SERIES = [
    "#2a78d6",  # 1 biru
    "#eb6834",  # 2 oranye
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 kuning
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 hijau
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 merah
]

# Warna tetap per model: identitas mengikuti entitas, bukan peringkat performa
MODEL_COLORS = {
    "xgboost": SERIES[0],
    "random_forest": SERIES[1],
    "elastic_net_lr": SERIES[2],
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "elastic_net_lr": "Elastic Net LR",
}

# --- Warna status (tidak pernah dipakai sebagai warna seri) ---------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- Chrome & tinta -------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Ramp sekuensial satu hue (biru, terang -> gelap) untuk peta panas
SEQUENTIAL_HEX = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
    "#2a78d6", "#256abf", "#184f95", "#0d366b",
]
# Ramp divergen biru <-> merah dengan titik tengah abu netral
DIVERGING_HEX = ["#184f95", "#3987e5", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#a52625"]


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("viz_sequential", SEQUENTIAL_HEX)


def diverging_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("viz_diverging", DIVERGING_HEX)


def set_theme() -> None:
    """Terapkan rcParams global. Aman dipanggil berkali-kali."""
    # Di luar notebook, pakai backend non-GUI agar tidak membuka jendela.
    if not _in_notebook():
        matplotlib.use("Agg", force=False)

    plt.rcParams.update({
        "figure.figsize": (8.0, 5.0),
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.alpha": 1.0,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "text.color": INK_PRIMARY,
        "axes.prop_cycle": plt.cycler(color=SERIES),
    })


def save_fig(fig, path: Path, close: bool = True) -> Path:
    """Simpan figure ke PNG 300 dpi dan kembalikan path-nya."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    if close:
        plt.close(fig)
    return path


def annotate_source(ax, text: str) -> None:
    """Catatan kecil di bawah gambar (mis. jumlah observasi / sumber data)."""
    ax.annotate(
        text,
        xy=(0, -0.16), xycoords="axes fraction",
        fontsize=8, color=INK_MUTED, ha="left", va="top",
    )


def rupiah_juta(x, _pos=None) -> str:
    """Formatter sumbu untuk rupiah dalam satuan juta."""
    return f"{x / 1e6:,.0f}"


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False
