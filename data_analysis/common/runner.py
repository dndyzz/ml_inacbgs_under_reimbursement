"""Jembatan notebook <-> pipeline.

Setiap ``pipelines/<tahap>/main.py`` berisi fungsi ``run(cfg, **overrides)``.
Modul ini memuat file tersebut berdasarkan nama tahap sehingga notebook cukup
menulis ``run_pipeline("01_eda")`` tanpa mengurus sys.path atau nama paket
(folder berawalan angka tidak bisa di-import biasa).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .config import PROJECT_ROOT

PIPELINE_DIR = PROJECT_ROOT / "pipelines"


def list_pipelines() -> list[str]:
    """Nama semua tahap pipeline, terurut."""
    return sorted(
        p.name for p in PIPELINE_DIR.iterdir()
        if p.is_dir() and (p / "main.py").exists()
    )


def load_pipeline(stage: str) -> ModuleType:
    """Muat main.py sebuah tahap sebagai modul bernama ``pipeline_<tahap>``."""
    stage_dir = PIPELINE_DIR / stage
    main_py = stage_dir / "main.py"
    if not main_py.exists():
        raise FileNotFoundError(
            f"Tahap '{stage}' tidak ditemukan. Pilihan: {list_pipelines()}"
        )

    # Modul pendamping (generator.py, plots.py, ...) diimpor relatif ke folder
    # tahapnya, jadi foldernya harus ada di sys.path.
    for extra in (str(PROJECT_ROOT), str(stage_dir)):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    module_name = f"pipeline_{stage}"
    spec = importlib.util.spec_from_file_location(module_name, main_py)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def load_stage_module(stage: str, module: str) -> ModuleType:
    """Muat modul pendamping sebuah tahap, mis. ``features.py`` milik tahap 03.

    Dipakai ketika satu tahap perlu memakai ulang definisi tahap sebelumnya
    (tahap 04 memakai ``make_preprocessor`` dari tahap 03) tanpa menduplikasi kode.
    """
    path = PIPELINE_DIR / stage / f"{module}.py"
    if not path.exists():
        raise FileNotFoundError(f"Modul {module}.py tidak ada pada tahap '{stage}'")

    module_name = f"pipeline_{stage}_{module}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def run_pipeline(stage: str, cfg: Any = None, **overrides: Any) -> dict:
    """Jalankan satu tahap dan kembalikan dict hasilnya (tabel, figure, ringkasan)."""
    module = load_pipeline(stage)
    return module.run(cfg=cfg, **overrides)


def run_all(stages: list[str] | None = None, cfg: Any = None) -> dict[str, dict]:
    """Jalankan seluruh tahap berurutan; kembalikan hasil per tahap."""
    results: dict[str, dict] = {}
    for stage in stages or list_pipelines():
        results[stage] = run_pipeline(stage, cfg=cfg)
    return results
