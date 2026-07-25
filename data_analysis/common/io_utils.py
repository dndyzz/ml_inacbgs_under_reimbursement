"""Penyimpanan artefak (tabel, JSON, model) dengan konvensi seragam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Simpan DataFrame ke CSV UTF-8-BOM agar rapi dibuka di Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding="utf-8-sig")
    return path


def load_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)


def save_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=_fallback)
    return path


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _fallback(obj: Any) -> Any:
    """Serialisasi objek non-JSON (numpy, Path, Timestamp)."""
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
