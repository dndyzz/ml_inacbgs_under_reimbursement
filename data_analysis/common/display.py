"""Penampil hasil pipeline di dalam notebook.

Setiap ``run()`` mengembalikan dict dengan kunci:
    stage    : nama tahap
    summary  : dict berisi angka-angka ringkas
    tables   : dict nama -> DataFrame (pratinjau data)
    figures  : list Path gambar PNG yang sudah tersimpan
    paths    : dict artefak file yang dihasilkan

Fungsi di sini merapikan dict tersebut menjadi tampilan notebook. Di luar
notebook, fungsi-fungsi ini tetap aman dipanggil (jatuh ke ``print``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _display(obj: Any) -> None:
    try:
        from IPython.display import display  # type: ignore

        display(obj)
    except Exception:
        print(obj)


def _markdown(text: str) -> None:
    try:
        from IPython.display import Markdown, display  # type: ignore

        display(Markdown(text))
    except Exception:
        print(text)


def show_summary(result: dict) -> None:
    """Tampilkan angka-angka ringkas sebuah tahap sebagai tabel kecil."""
    summary = result.get("summary", {})
    if not summary:
        return
    df = pd.DataFrame(
        {"Keterangan": list(summary.keys()), "Nilai": list(summary.values())}
    )
    _markdown(f"**Ringkasan tahap `{result.get('stage', '?')}`**")
    _display(df.style.hide(axis="index") if hasattr(df, "style") else df)


def show_tables(result: dict, max_rows: int = 8, only: list[str] | None = None) -> None:
    """Pratinjau tabel hasil tahap (default 8 baris pertama)."""
    tables = result.get("tables", {})
    for name, table in tables.items():
        if only and name not in only:
            continue
        df = pd.read_csv(table) if isinstance(table, (str, Path)) else table
        _markdown(f"**{name}** — {df.shape[0]:,} baris x {df.shape[1]} kolom")
        _display(df.head(max_rows))


def show_figures(result: dict, width: int = 820, only: list[str] | None = None) -> None:
    """Tampilkan gambar PNG yang dihasilkan tahap ini, inline di notebook."""
    figures = result.get("figures", [])
    try:
        from IPython.display import Image, display  # type: ignore
    except Exception:
        print("Gambar tersimpan di:", *[str(f) for f in figures], sep="\n  ")
        return

    for fig_path in figures:
        fig_path = Path(fig_path)
        if only and fig_path.stem not in only:
            continue
        _markdown(f"*{fig_path.stem.replace('_', ' ')}*")
        display(Image(filename=str(fig_path), width=width))


def show_result(
    result: dict,
    max_rows: int = 8,
    width: int = 820,
    tables: bool = True,
    figures: bool = True,
) -> None:
    """Tampilkan ringkasan + pratinjau tabel + semua gambar sekaligus."""
    show_summary(result)
    if tables:
        show_tables(result, max_rows=max_rows)
    if figures:
        show_figures(result, width=width)


def show_file(path: str | Path, width: int = 820) -> None:
    """Tampilkan satu file gambar berdasarkan path."""
    try:
        from IPython.display import Image, display  # type: ignore

        display(Image(filename=str(path), width=width))
    except Exception:
        print("Gambar:", path)
