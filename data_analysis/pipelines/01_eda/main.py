"""TAHAP 01 - Exploratory Data Analysis (EDA).

Melihat data APA ADANYA sebelum disentuh: sebaran tiap variabel, pola nilai
hilang, hubungan biaya vs klaim, prevalensi under-reimbursement beserta IK 95%,
kolinearitas antarvariabel, dan kestabilan antarbulan.

Tahap ini sengaja diletakkan SEBELUM preprocessing. Keputusan pembersihan
(rentang wajar, kriteria eksklusi, strategi imputasi) harus lahir dari apa yang
terlihat di sini, bukan sebaliknya.

Catatan: rasio klaim/tagihan dihitung di sini hanya untuk keperluan eksplorasi.
Definisi outcome resmi (termasuk penyaringan kriteria inklusi/eksklusi)
ditetapkan pada tahap 02.

Jalankan mandiri:
    python pipelines/01_eda/main.py
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

import eda_plots as plots
import profiling
from common import viz
from common.config import load_config
from common.io_utils import save_table
from common.logging_utils import banner, get_logger

STAGE = "01_eda"
log = get_logger(STAGE)

# Variabel yang diprofilkan (nama kolom pada data mentah)
NUMERIC_COLS = [
    "age", "pre_icu_los_hours", "map_lowest", "sf_ratio_lowest", "gcs_lowest",
    "creatinine_highest", "icu_los_days", "total_hospital_billing", "inacbg_claim",
]
CATEGORICAL_COLS = [
    "sex", "jkn_class", "icu_admission_type", "diagnosis_category", "surgery_24h",
    "mechanical_ventilation", "vasopressor_inotrope", "transfusion_prc",
    "oxygen_device", "gcs_note", "discharge_status",
]
PREVALENCE_GROUPS = [
    "jkn_class", "icu_admission_type", "diagnosis_category", "surgery_24h",
    "mechanical_ventilation", "vasopressor_inotrope", "transfusion_prc",
]
CORRELATION_COLS = [
    "age", "pre_icu_los_hours", "map_lowest", "sf_ratio_lowest", "gcs_lowest",
    "creatinine_highest", "total_hospital_billing", "inacbg_claim",
]


def run(cfg=None, **overrides) -> dict:
    """Profilkan data mentah dan hasilkan tabel + gambar eksplorasi."""
    cfg = cfg or load_config(**overrides)
    banner(log, "tahap 01 - exploratory data analysis")
    viz.set_theme()

    raw_path = cfg.path("data_raw") / cfg["eda"]["input_file"]
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} belum ada. Jalankan tahap 00 dulu "
            f"(python pipelines/00_data_generation/main.py) atau letakkan data nyata di data/raw/."
        )
    df = pd.read_csv(raw_path, parse_dates=[
        "hospital_admission_datetime", "icu_admission_datetime",
        "icu_discharge_datetime", "hospital_discharge_datetime",
    ])
    log.info("Data mentah dibaca: %s baris x %s kolom", *df.shape)

    df = _add_exploration_columns(df)
    fig_dir, tab_dir = cfg.figure_dir(STAGE), cfg.table_dir(STAGE)

    # -- Tabel --------------------------------------------------------------
    overview = profiling.dataset_overview(df)
    num_desc = profiling.describe_numeric(df, NUMERIC_COLS)
    cat_desc = profiling.describe_categorical(df, CATEGORICAL_COLS)

    analysable = df.dropna(subset=["under_reimbursement_eksploratif"])
    prev_tables = {
        col: profiling.prevalence_by_group(analysable, "under_reimbursement_eksploratif", col)
        for col in PREVALENCE_GROUPS if col in df
    }
    prevalence = pd.concat(prev_tables.values(), ignore_index=True)
    baseline = profiling.baseline_table(
        analysable, "under_reimbursement_eksploratif",
        numeric_cols=[c for c in NUMERIC_COLS if c not in ("total_hospital_billing", "inacbg_claim")],
        categorical_cols=CATEGORICAL_COLS[:8],
    )

    save_table(overview, tab_dir / "01_ringkasan_kolom.csv")
    save_table(num_desc, tab_dir / "02_deskriptif_numerik.csv")
    save_table(cat_desc, tab_dir / "03_deskriptif_kategorik.csv")
    save_table(prevalence, tab_dir / "04_prevalensi_subkelompok.csv")
    save_table(baseline, tab_dir / "tabel_4_1_karakteristik_subjek.csv")

    # -- Gambar -------------------------------------------------------------
    figures = [
        plots.plot_missingness(df, fig_dir),
        plots.plot_numeric_distributions(df, NUMERIC_COLS, fig_dir),
        plots.plot_categorical_distributions(df, CATEGORICAL_COLS, fig_dir),
        plots.plot_cost_vs_claim(df, fig_dir),
        plots.plot_prevalence_by_group(prev_tables, fig_dir),
        plots.plot_numeric_by_outcome(
            analysable,
            [c for c in NUMERIC_COLS if c not in ("total_hospital_billing", "inacbg_claim")],
            "under_reimbursement_eksploratif", fig_dir,
        ),
        plots.plot_correlation(df, CORRELATION_COLS, fig_dir),
        plots.plot_temporal(analysable, "under_reimbursement_eksploratif", fig_dir),
    ]
    log.info("%s gambar tersimpan di %s", len(figures), fig_dir)

    # -- Ringkasan ----------------------------------------------------------
    k = int(analysable["under_reimbursement_eksploratif"].sum())
    n = len(analysable)
    lo, hi = profiling.wilson_ci(k, n)
    summary = {
        "Baris data mentah": f"{len(df):,}",
        "Baris duplikat (episode_id ganda)": f"{int(df['episode_id'].duplicated().sum()):,}",
        "Kolom dengan nilai hilang": int((df.isna().sum() > 0).sum()),
        "Episode dengan data finansial lengkap": f"{n:,}",
        "Prevalensi under-reimbursement (eksploratif)": f"{k/n:.1%} (IK95% {lo:.1%}-{hi:.1%})",
        "Median rasio klaim/tagihan": f"{analysable['claim_to_billing_ratio'].median():.2f}",
        "Median lama rawat ICU": f"{df['icu_los_days'].median():.1f} hari",
        "Episode ICU < 24 jam (calon eksklusi)": f"{int((df['icu_los_days'] < 1).sum()):,}",
    }

    return {
        "stage": STAGE,
        "summary": summary,
        "tables": {
            "Ringkasan kolom": overview,
            "Deskriptif numerik": num_desc,
            "Prevalensi per subkelompok": prevalence,
            "Tabel 4.1 Karakteristik subjek": baseline,
        },
        "figures": figures,
        "paths": {"figures_dir": fig_dir, "tables_dir": tab_dir},
    }


def _add_exploration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Kolom turunan yang diperlukan untuk eksplorasi (bukan hasil preprocessing)."""
    df = df.copy()
    df["pre_icu_los_hours"] = (
        df["icu_admission_datetime"] - df["hospital_admission_datetime"]
    ).dt.total_seconds() / 3600
    df["sf_ratio_lowest"] = df["spo2_lowest"] / df["fio2_lowest"]
    df["claim_to_billing_ratio"] = df["inacbg_claim"] / df["total_hospital_billing"]
    df["under_reimbursement_eksploratif"] = (df["claim_to_billing_ratio"] < 1).astype(float)
    df.loc[df["claim_to_billing_ratio"].isna(), "under_reimbursement_eksploratif"] = pd.NA
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA data mentah penelitian.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--input", default=None, help="Nama file CSV di data/raw/")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.input:
        cfg.to_dict()["eda"]["input_file"] = args.input

    result = run(cfg)
    for key, value in result["summary"].items():
        log.info("%-45s : %s", key, value)


if __name__ == "__main__":
    main()
