"""TAHAP 03 - Feature building.

Mengubah kohort analisis menjadi matriks fitur yang siap dimodelkan:

    - transformasi deterministik (log durasi pra-ICU, skor komponen mSOFA,
      jumlah organ support, penanda GCS tidak dapat dinilai)
    - penetapan peran tiap kolom: numerik / ordinal / biner / kategorik
    - pemeriksaan multikolinearitas (VIF) sebagai informasi, bukan penyaring

Imputasi, one-hot encoding, dan standardisasi TIDAK dijalankan di sini. Ketiganya
dirakit sebagai objek preprocessor (``features.make_preprocessor``) yang baru
di-fit di dalam fold pada tahap 04, agar tidak terjadi kebocoran data.

Jalankan mandiri:
    python pipelines/03_feature_building/main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(STAGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd

import feature_plots as plots
import features
from common import viz
from common.config import load_config
from common.io_utils import save_json, save_table
from common.logging_utils import banner, get_logger

STAGE = "03_feature_building"
log = get_logger(STAGE)

DOMAIN_LABEL = {
    "age": "Demografi/administratif",
    "sex": "Demografi/administratif",
    "jkn_class": "Demografi/administratif",
    "icu_admission_type": "Demografi/administratif",
    "pre_icu_los_log1p": "Demografi/administratif",
    "mechanical_ventilation": "Intervensi/organ support",
    "vasopressor_inotrope": "Intervensi/organ support",
    "transfusion_prc": "Intervensi/organ support",
    "surgery_24h": "Intervensi/organ support",
    "organ_support_count": "Intervensi/organ support (turunan)",
    "map_lowest": "Disfungsi organ (mSOFA)",
    "sf_ratio_lowest": "Disfungsi organ (mSOFA)",
    "gcs_lowest": "Disfungsi organ (mSOFA)",
    "creatinine_highest": "Disfungsi organ (mSOFA)",
    "msofa_total": "Disfungsi organ (turunan)",
    "gcs_unassessable": "Disfungsi organ (penanda sedasi)",
    "diagnosis_category": "Kasus",
}


def run(cfg=None, **overrides) -> dict:
    """Bangun matriks fitur dan simpan ke data/processed/."""
    cfg = cfg or load_config(**overrides)
    banner(log, "tahap 03 - feature building")
    viz.set_theme()

    fb = cfg["feature_building"]
    cohort_path = cfg.path("data_interim") / fb["input_file"]
    if not cohort_path.exists():
        raise FileNotFoundError(f"{cohort_path} tidak ada. Jalankan tahap 02 lebih dulu.")

    cohort = pd.read_csv(cohort_path, parse_dates=["icu_admission_datetime"])
    log.info("Kohort dibaca: %s episode", len(cohort))

    X, y, spec = features.build_feature_matrix(cohort, cfg)
    log.info("Matriks fitur: %s baris x %s variabel (pra-encoding)", *X.shape)

    # Nama kolom setelah encoding - hanya untuk metadata/pelaporan.
    # Preprocessor di-fit pada periode latih saja agar disiplin split tetap terjaga.
    preprocessor = features.make_preprocessor(spec, scale=False, seed=cfg.seed)
    train_mask = y["periode"].str.startswith("latih")
    preprocessor.fit(X[train_mask])
    encoded_names = features.encoded_feature_names(preprocessor)
    log.info("Jumlah kolom setelah one-hot encoding: %s", len(encoded_names))

    vif = features.compute_vif(X, spec["numeric"])

    # -- simpan -------------------------------------------------------------
    proc_dir = cfg.path("data_processed")
    x_path = proc_dir / fb["output_features"]
    y_path = proc_dir / fb["output_target"]
    spec_path = proc_dir / "feature_spec.json"
    save_table(X, x_path)
    save_table(y, y_path)
    save_json({**spec, "encoded_feature_names": encoded_names,
               "n_features_post_encoding": len(encoded_names)}, spec_path)

    spec_table = _spec_table(spec)
    tab_dir = cfg.table_dir(STAGE)
    save_table(spec_table, tab_dir / "01_spesifikasi_fitur.csv")
    save_table(vif, tab_dir / "02_vif.csv")
    save_table(pd.DataFrame({"kolom_model": encoded_names}), tab_dir / "03_kolom_setelah_encoding.csv")
    log.info("Matriks fitur disimpan: %s", x_path)

    # -- gambar -------------------------------------------------------------
    fig_dir = cfg.figure_dir(STAGE)
    figures = [
        plots.plot_derived_features(X, fig_dir),
        plots.plot_vif(vif, float(fb["vif_threshold"]), fig_dir),
        plots.plot_feature_correlation(X, spec["numeric"], fig_dir),
        plots.plot_feature_map(spec, len(encoded_names), fig_dir),
    ]

    n_high_vif = int((vif["vif"] > float(fb["vif_threshold"])).sum())
    summary = {
        "Episode": f"{len(X):,}",
        "Variabel pra-encoding": spec["n_features_pre_encoding"],
        "Kolom model pasca one-hot": len(encoded_names),
        "Fitur numerik": len(spec["numeric"]),
        "Fitur biner": len(spec["binary"]),
        "Fitur kategorik": len(spec["categorical"]),
        "Variabel dengan VIF di atas ambang": n_high_vif,
        "Sel kosong pada matriks fitur": f"{int(X.isna().sum().sum()):,} (diimputasi di tahap 04)",
        "Prevalensi outcome": f"{y[spec['target_binary']].mean():.1%}",
    }

    return {
        "stage": STAGE,
        "summary": summary,
        "tables": {
            "Spesifikasi fitur": spec_table,
            "VIF": vif,
            "Pratinjau matriks fitur": X.head(20),
            "Pratinjau target": y.head(20),
        },
        "figures": figures,
        "paths": {"X": x_path, "y": y_path, "spec": spec_path},
    }


def _spec_table(spec: dict) -> pd.DataFrame:
    """Tabel peran tiap variabel: jenis, domain penelitian, dan perlakuan."""
    treatment = {
        "numeric": "MICE di dalam fold (+ standardisasi khusus elastic net)",
        "ordinal": "Imputasi median, urutan 1<2<3 dipertahankan",
        "binary": "Imputasi modus",
        "categorical": "Imputasi modus + one-hot encoding (drop kategori pertama)",
    }
    rows = []
    for kind in ["numeric", "ordinal", "binary", "categorical"]:
        for col in spec[kind]:
            rows.append({
                "variabel": col,
                "jenis": kind,
                "domain": DOMAIN_LABEL.get(col, "-"),
                "perlakuan_preprocessing": treatment[kind],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Membangun matriks fitur model.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Nama file kohort di data/interim/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.to_dict()["feature_building"]["input_file"] = args.input

    result = run(cfg)
    for key, value in result["summary"].items():
        log.info("%-38s : %s", key, value)


if __name__ == "__main__":
    main()
