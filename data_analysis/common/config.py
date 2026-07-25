"""Pemuat konfigurasi dan resolusi path proyek.

Semua pipeline memanggil ``load_config()`` sehingga hanya ada satu sumber
kebenaran (config/config.yaml) untuk seed, path, dan hyperparameter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# data_analysis/ -> akar proyek analisis
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """Pembungkus tipis di atas dict YAML.

    Akses bisa lewat atribut (``cfg.training``) maupun kunci
    (``cfg["training"]["outer_folds"]``). Path relatif di YAML otomatis
    diresolusi terhadap akar proyek dan foldernya dibuat bila belum ada.
    """

    def __init__(self, data: dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self.root = root
        self._paths = {
            key: (root / value) for key, value in data.get("paths", {}).items()
        }

    # -- akses dasar ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as exc:  # pragma: no cover - hanya untuk pesan error
            raise AttributeError(f"Kunci konfigurasi '{key}' tidak ada") from exc

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return self._data

    # -- path ----------------------------------------------------------
    def path(self, key: str, *parts: str, create: bool = True) -> Path:
        """Kembalikan path absolut untuk salah satu entri ``paths`` di YAML."""
        base = self._paths[key]
        if create:
            base.mkdir(parents=True, exist_ok=True)
        return base.joinpath(*parts) if parts else base

    def figure_dir(self, stage: str) -> Path:
        """Subfolder gambar per tahap, mis. outputs/figures/01_eda."""
        d = self.path("figures") / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def table_dir(self, stage: str) -> Path:
        d = self.path("tables") / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def seed(self) -> int:
        return int(self._data["project"]["seed"])


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Baca config/config.yaml (atau path lain) dan terapkan override dangkal.

    Contoh::

        cfg = load_config(data_generation={"n_episodes": 500})
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(section), dict):
            data[section] = {**data[section], **value}
        else:
            data[section] = value

    return Config(data, root=cfg_path.resolve().parents[1])
