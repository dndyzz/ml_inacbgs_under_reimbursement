"""TAHAP 02 - Preprocessing.

Mengubah tabel mentah menjadi kohort analisis yang siap dimodelkan:

    1. turunkan variabel waktu (durasi pra-ICU, lama rawat ICU) dan rasio SpO2/FiO2
    2. periksa rentang nilai wajar -> nilai mustahil dijadikan hilang
    3. terapkan kriteria inklusi & eksklusi (menghasilkan Gambar 3.1)
    4. tetapkan outcome pada akhir episode (biner + magnitudo log-rasio)
    5. beri label periode untuk validasi temporal

Yang SENGAJA tidak dilakukan di sini: imputasi, encoding, dan standardisasi.
Ketiganya harus terjadi di dalam fold validasi (tahap 04) agar tidak bocor.

Jalankan mandiri:
    python pipelines/02_preprocessing/main.py
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

import cleaning
import prep_plots as plots
from common import viz
from common.config import load_config
from common.io_utils import save_table
from common.logging_utils import banner, get_logger

STAGE = "02_preprocessing"
log = get_logger(STAGE)

# Kolom yang dipantau nilai hilangnya (semua kandidat prediktor)
PREDICTOR_COLS = [
    "age", "sex", "jkn_class", "icu_admission_type", "pre_icu_los_hours",
    "mechanical_ventilation", "vasopressor_inotrope", "transfusion_prc", "surgery_24h",
    "map_lowest", "sf_ratio_lowest", "gcs_lowest", "creatinine_highest",
    "diagnosis_category",
]

DATE_COLS = [
    "hospital_admission_datetime", "icu_admission_datetime",
    "icu_discharge_datetime", "hospital_discharge_datetime",
]


def run(cfg=None, **overrides) -> dict:
    """Bersihkan data mentah dan hasilkan kohort analisis."""
    cfg = cfg or load_config(**overrides)
    banner(log, "tahap 02 - preprocessing")
    viz.set_theme()

    pre = cfg["preprocessing"]
    raw_path = cfg.path("data_raw") / pre["input_file"]
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} tidak ditemukan. Jalankan tahap 00 lebih dulu.")

    df = pd.read_csv(raw_path, parse_dates=DATE_COLS)
    log.info("Data mentah: %s baris x %s kolom", *df.shape)

    # 1-2. turunan + pemeriksaan rentang
    df = cleaning.build_derived_columns(df)
    df, range_report = cleaning.apply_range_checks(df, pre["plausible_ranges"])
    n_out_of_range = int(range_report["n_di_luar_batas"].sum())
    log.info("Nilai di luar batas wajar diubah menjadi hilang: %s", n_out_of_range)

    # 3. seleksi subjek
    cohort, flow = cleaning.apply_selection(df, cfg)
    log.info("Kohort analisis: %s episode (dari %s baris mentah)", len(cohort), len(df))

    # 4-5. outcome + label periode
    cohort = cleaning.define_outcome(cohort, cfg)
    cohort = cleaning.add_temporal_split(cohort, cfg["training"]["temporal_split_date"])

    missing_report = cleaning.missingness_report(cohort, PREDICTOR_COLS)

    # -- simpan ------------------------------------------------------------
    out_path = cfg.path("data_interim") / pre["output_file"]
    save_table(cohort, out_path)
    tab_dir = cfg.table_dir(STAGE)
    save_table(flow, tab_dir / "01_alur_seleksi.csv")
    save_table(range_report, tab_dir / "02_pemeriksaan_rentang.csv")
    save_table(missing_report, tab_dir / "03_nilai_hilang_kohort.csv")
    log.info("Kohort disimpan: %s", out_path)

    # -- gambar ------------------------------------------------------------
    fig_dir = cfg.figure_dir(STAGE)
    figures = [
        plots.plot_selection_flow(flow, fig_dir),
        plots.plot_cohort_shrinkage(flow, fig_dir),
        plots.plot_outcome_overview(cohort, cfg, fig_dir),
        plots.plot_missingness_after(missing_report, fig_dir),
    ]

    # -- ringkasan ---------------------------------------------------------
    out = pre["outcome"]
    n_event = int(cohort[out["binary_col"]].sum())
    periode = cohort["periode"].value_counts()
    summary = {
        "Baris mentah": f"{len(df):,}",
        "Kohort analisis akhir": f"{len(cohort):,} episode",
        "Total dikeluarkan": f"{len(df) - len(cohort):,} episode",
        "Nilai di luar batas -> hilang": n_out_of_range,
        "Kejadian under-reimbursement": f"{n_event:,} ({n_event/len(cohort):.1%})",
        "Median rasio klaim/tagihan": f"{cohort[out['ratio_col']].median():.2f}",
        "Median selisih (klaim − tagihan)": f"Rp {cohort['selisih_rupiah'].median()/1e6:,.1f} juta",
        "Periode latih": f"{periode.get('latih (Mei-Des 2025)', 0):,} episode",
        "Periode uji": f"{periode.get('uji (Jan-Mei 2026)', 0):,} episode",
        "Variabel prediktor dengan nilai hilang": int((missing_report["n_hilang"] > 0).sum()),
    }

    return {
        "stage": STAGE,
        "summary": summary,
        "tables": {
            "Alur seleksi subjek": flow,
            "Pemeriksaan rentang nilai": range_report,
            "Nilai hilang pada kohort": missing_report,
            "Pratinjau kohort analisis": cohort.head(20),
        },
        "figures": figures,
        "paths": {"cohort": out_path, "figures_dir": fig_dir, "tables_dir": tab_dir},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocessing kohort penelitian.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Nama file CSV mentah di data/raw/")
    parser.add_argument("--output", default=None, help="Nama file kohort di data/interim/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.to_dict()["preprocessing"]["input_file"] = args.input
    if args.output:
        cfg.to_dict()["preprocessing"]["output_file"] = args.output

    result = run(cfg)
    for key, value in result["summary"].items():
        log.info("%-42s : %s", key, value)


if __name__ == "__main__":
    main()
