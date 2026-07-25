"""TAHAP 00 - Pembangkitan data dummy.

Membuat dataset sintetis berisi episode rawat ICU dewasa lengkap dengan total
tagihan rumah sakit dan nilai klaim INA-CBGs, sesuai variabel pada Tabel 3.1
proposal. Tahap ini HANYA dipakai selama data RSCM belum tersedia; begitu data
nyata masuk, letakkan file CSV-nya di ``data/raw/`` dan mulai dari tahap 01.

Jalankan mandiri:
    python pipelines/00_data_generation/main.py
    python pipelines/00_data_generation/main.py --n-episodes 2000 --seed 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Bootstrap agar file ini bisa dijalankan langsung dari mana saja
ROOT = Path(__file__).resolve().parents[2]
STAGE_DIR = Path(__file__).resolve().parent
for _p in (str(ROOT), str(STAGE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import numpy as np

from common import viz
from common.config import load_config
from common.io_utils import save_table
from common.logging_utils import banner, get_logger
from generator import build_data_dictionary, generate_raw_dataset

STAGE = "00_data_generation"
log = get_logger(STAGE)


def run(cfg=None, **overrides) -> dict:
    """Bangkitkan data dummy dan simpan ke data/raw/."""
    cfg = cfg or load_config(**overrides)
    banner(log, "tahap 00 - pembangkitan data dummy")

    gen_cfg = cfg["data_generation"]
    log.info("Membangkitkan %s episode (seed=%s)", gen_cfg["n_episodes"], cfg.seed)

    df = generate_raw_dataset(cfg)
    dictionary = build_data_dictionary()

    raw_path = cfg.path("data_raw") / gen_cfg["output_file"]
    dict_path = cfg.path("data_raw") / gen_cfg["dictionary_file"]
    save_table(df, raw_path)
    save_table(dictionary, dict_path)
    log.info("Data mentah disimpan: %s (%s baris x %s kolom)", raw_path.name, *df.shape)

    figures = [_plot_sanity_check(df, cfg)]

    complete = df.dropna(subset=["total_hospital_billing", "inacbg_claim"])
    ratio = complete["inacbg_claim"] / complete["total_hospital_billing"]

    summary = {
        "Jumlah baris dibangkitkan": f"{len(df):,}",
        "Jumlah kolom": df.shape[1],
        "Periode admisi ICU": f"{gen_cfg['study_start']} s.d. {gen_cfg['study_end']}",
        "Median tagihan RS": f"Rp {complete['total_hospital_billing'].median()/1e6:,.1f} juta",
        "Median klaim INA-CBGs": f"Rp {complete['inacbg_claim'].median()/1e6:,.1f} juta",
        "Prevalensi under-reimbursement (mentah)": f"{(ratio < 1).mean():.1%}",
        "Baris dengan minimal 1 nilai hilang": f"{df.isna().any(axis=1).sum():,}",
    }

    return {
        "stage": STAGE,
        "summary": summary,
        "tables": {
            "Pratinjau data mentah": df.head(20),
            "Kamus data (Tabel 3.1)": dictionary,
        },
        "figures": figures,
        "paths": {"raw_data": raw_path, "data_dictionary": dict_path},
    }


def _plot_sanity_check(df, cfg) -> Path:
    """Sebar tagihan vs klaim dengan garis impas - pemeriksaan cepat generator."""
    viz.set_theme()
    d = df.dropna(subset=["total_hospital_billing", "inacbg_claim"])
    billing = d["total_hospital_billing"] / 1e6
    claim = d["inacbg_claim"] / 1e6
    under = claim < billing

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(billing[~under], claim[~under], s=14, alpha=0.65,
               color=viz.SERIES[0], edgecolor="none", label="Klaim menutup biaya")
    ax.scatter(billing[under], claim[under], s=14, alpha=0.65,
               color=viz.STATUS["critical"], edgecolor="none", label="Under-reimbursement")

    lim = float(np.nanpercentile(billing, 99.5))
    ax.plot([0, lim], [0, lim], color=viz.INK_MUTED, linewidth=1.5, linestyle="--",
            label="Garis impas (klaim = biaya)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Total tagihan rumah sakit (Rp juta)")
    ax.set_ylabel("Klaim INA-CBGs (Rp juta)")
    ax.set_title("Klaim vs tagihan per episode — data dummy")
    ax.grid(axis="both")
    ax.legend(loc="upper left")
    viz.annotate_source(ax, f"n = {len(d):,} episode dengan data finansial lengkap • data sintetis")

    return viz.save_fig(fig, cfg.figure_dir(STAGE) / "00_klaim_vs_tagihan.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangkitkan data dummy penelitian.")
    parser.add_argument("--config", default=None, help="Path config YAML alternatif")
    parser.add_argument("--n-episodes", type=int, default=None, help="Jumlah episode")
    parser.add_argument("--seed", type=int, default=None, help="Seed acak")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.n_episodes:
        cfg.to_dict()["data_generation"]["n_episodes"] = args.n_episodes
    if args.seed:
        cfg.to_dict()["project"]["seed"] = args.seed

    result = run(cfg)
    for key, value in result["summary"].items():
        log.info("%-42s : %s", key, value)


if __name__ == "__main__":
    main()
