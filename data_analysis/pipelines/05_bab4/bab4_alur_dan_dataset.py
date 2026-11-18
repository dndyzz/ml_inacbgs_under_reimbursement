"""Gambar 4.1 alur seleksi (ringkas) + dataset siap machine learning (variabel of interest).

1. figures/05_bab4/03_alur_seleksi.png: eksklusi luaran (17) dan pemeriksaan kualitas data (216) sebagai
   dua kotak, sesuai keputusan peneliti/pembimbing (rincian tetap di tables/02_preprocessing/01_alur_seleksi.csv).
2. <akar proyek>/Dataset_ML_Under-reimbursement_ICU_RSCM.xlsx (+ .csv): 801 episode, ID studi
   (tanpa no. RM / ID episode RS), 14 prediktor dalam satuan asli + penanda GCS-sedasi, luaran.
   Nilai hilang dibiarkan kosong: imputasi, encoding, dan standardisasi dilakukan di dalam fold.
   Peta ID studi -> ID asli: tables/05_bab4/peta_id_studi.csv (tidak untuk dibagikan).
Jalankan:  python pipelines/05_bab4/bab4_alur_dan_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from common import viz
from common.config import load_config

PROJECT = ROOT.parents[1]
OUT_XLSX = PROJECT / "Dataset_ML_Under-reimbursement_ICU_RSCM.xlsx"

# kolom dataset: (nama kolom, sumber di kohort, nama Tabel 3.1, domain, peran, satuan/kode)
COLUMNS = [
    ("id_episode", None, "ID episode studi", "Identitas", "ID", "EP0001 dst."),
    ("id_pasien", None, "ID pasien studi", "Identitas", "ID (pengelompokan fold)",
     "PS0001 dst.; episode dari pasien yang sama berbagi ID"),
    ("bulan_masuk_icu", None, "Bulan masuk unit perawatan intensif", "Waktu", "Pembagian temporal", "YYYY-MM"),
    ("periode_validasi_temporal", "periode", "Periode validasi temporal", "Waktu", "Pembagian temporal",
     "latih = Mei–Des 2025; uji = Jan–Mei 2026"),
    ("usia", "age", "Usia", "Demografi dan administratif", "Prediktor (IV)", "tahun"),
    ("jenis_kelamin", "sex", "Jenis kelamin", "Demografi dan administratif", "Prediktor (IV)",
     "Laki-laki / Perempuan"),
    ("kelas_rawat_jkn", "jkn_class", "Kelas rawat JKN", "Demografi dan administratif", "Prediktor (IV)",
     "1 / 2 / 3 (ordinal)"),
    ("tipe_admisi_icu", "icu_admission_type", "Tipe admisi unit perawatan intensif",
     "Demografi dan administratif", "Prediktor (IV)", "Elektif / Emergensi"),
    ("durasi_rawat_pra_icu_jam", "pre_icu_los_hours", "Durasi rawat pra-unit perawatan intensif",
     "Demografi dan administratif", "Prediktor (IV)", "jam (model memakai log(1 + jam))"),
    ("ventilasi_mekanik", "mechanical_ventilation", "Ventilasi mekanik", "Intervensi/organ support",
     "Prediktor (IV)", "1 = ya, 0 = tidak"),
    ("vasopresor_inotropik", "vasopressor_inotrope", "Vasopresor/inotropik", "Intervensi/organ support",
     "Prediktor (IV)", "1 = ya, 0 = tidak"),
    ("transfusi_darah", "transfusion_prc", "Transfusi darah", "Intervensi/organ support", "Prediktor (IV)",
     "1 = ya, 0 = tidak"),
    ("pembedahan_24jam", "surgery_24h", "Pembedahan 24 jam", "Intervensi/organ support", "Prediktor (IV)",
     "Tidak / Elektif / Emergensi"),
    ("map_terendah", "map_lowest", "MAP terendah", "Disfungsi organ", "Prediktor (IV)", "mmHg"),
    ("rasio_spo2_fio2_terendah", "sf_ratio_lowest", "Rasio SpO₂/FiO₂ terendah", "Disfungsi organ",
     "Prediktor (IV)", "rasio"),
    ("gcs_terendah", "gcs_lowest", "GCS terendah", "Disfungsi organ", "Prediktor (IV)", "3–15"),
    ("gcs_tidak_dapat_dinilai_sedasi", "gcs_unassessable", "Penanda GCS tidak dapat dinilai karena sedasi",
     "Disfungsi organ", "Prediktor (missing indicator)", "1 = ya, 0 = tidak"),
    ("kreatinin_tertinggi", "creatinine_highest", "Kreatinin tertinggi", "Disfungsi organ", "Prediktor (IV)",
     "mg/dL"),
    ("kategori_diagnosis_utama", "diagnosis_category", "Kategori diagnosis utama", "Kasus", "Prediktor (IV)",
     "9 kategori sistem organ"),
    ("total_tagihan_rs", "total_hospital_billing", "Total tagihan episode", "Luaran", "Pembentuk luaran",
     "rupiah"),
    ("klaim_inacbg_final", "inacbg_claim", "Klaim INA-CBGs episode (final, disetujui BPJS Kesehatan)",
     "Luaran", "Pembentuk luaran", "rupiah"),
    ("rasio_klaim_tagihan", "claim_to_billing_ratio", "Rasio klaim terhadap biaya riil", "Luaran",
     "Luaran (deskriptif)", "klaim / tagihan"),
    ("under_reimbursement", "under_reimbursement", "Under-reimbursement (variabel dependen)", "Luaran",
     "Luaran primer (DV)", "1 = rasio < 1, 0 = rasio ≥ 1"),
    ("log_rasio_klaim_tagihan", "log_claim_to_billing_ratio", "Logaritma rasio", "Luaran",
     "Luaran sekunder (DV)", "ln(klaim / tagihan)"),
]


def flow_figure(flow: pd.DataFrame, n_under: int, path: Path) -> Path:
    viz.set_theme()
    n0 = int(flow.n_sisa.iloc[0])
    is_outcome = flow.tahap.str.contains("outcome tidak dapat ditentukan")
    assert is_outcome.sum() == 1
    ex_outcome = int(flow.n_keluar[is_outcome].iloc[0])
    n1 = n0 - ex_outcome
    ex_quality = int(flow.n_keluar[~is_outcome].sum())
    n = n1 - ex_quality
    assert n == int(flow.n_sisa.iloc[-1])
    fmt = lambda k: f"{k:,}".replace(",", ".")

    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis("off")

    def box(x, y, w, h, txt, fc, bold=False, color=viz.INK_PRIMARY):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="none"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=10.5, color=color,
                fontweight="bold" if bold else "normal")

    arrow = dict(arrowstyle="-|>", color=viz.BASELINE, lw=1.6)
    main_c, excl_c = "#cde2fb", "#f6e7e1"
    box(3, 9.1, 4.6, 1.0, f"Episode dalam data mart ICU Dewasa JKN\n(Mei 2025–Mei 2026)\nn = {fmt(n0)}", main_c)
    box(3, 6.6, 4.6, 0.8, f"Tersisa\nn = {fmt(n1)}", "#edf4fd")
    box(3, 4.1, 4.6, 0.8, f"Kohort analisis\nn = {fmt(n)} episode", viz.SERIES[0], bold=True, color="white")
    box(7.6, 7.85, 4.2, 0.95, f"Luaran tidak dapat ditentukan\n(klaim atau tagihan kosong/nol)\nn = {ex_outcome}",
        excl_c)
    box(7.6, 5.35, 4.2, 0.8, f"Tidak lolos pemeriksaan kualitas data\nn = {ex_quality}", excl_c)
    pct = lambda k: f"{100 * k / n:.1f}".replace(".", ",")
    box(1.6, 1.7, 2.8, 1.0, f"Under-reimbursement\n(rasio < 1)\nn = {n_under} ({pct(n_under)}%)", "#f6c9c8")
    box(4.6, 1.7, 2.8, 1.0, f"Tidak under-reimbursement\n(rasio ≥ 1)\nn = {n - n_under} ({pct(n - n_under)}%)",
        "#cde2fb")
    for y0, y1 in ((8.55, 7.05), (6.15, 4.55)):
        ax.annotate("", (3, y1), (3, y0), arrowprops=arrow)
    for y in (7.85, 5.35):
        ax.annotate("", (5.45, y), (3, y), arrowprops=arrow)
    for x in (1.6, 4.6):
        ax.annotate("", (x, 2.25), (3, 3.65), arrowprops=arrow)
    fig.tight_layout()
    return viz.save_fig(fig, path)


def build_dataset(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = d.sort_values(["icu_admission_datetime", "episode_id"]).reset_index(drop=True)
    pid = {p: f"PS{i + 1:04d}" for i, p in enumerate(pd.unique(d.patient_code))}
    out = pd.DataFrame({"id_episode": [f"EP{i + 1:04d}" for i in range(len(d))],
                        "id_pasien": d.patient_code.map(pid),
                        "bulan_masuk_icu": pd.to_datetime(d.icu_admission_datetime).dt.strftime("%Y-%m")})
    for col, src, *_ in COLUMNS:
        if src:
            out[col] = d[src]
    out["periode_validasi_temporal"] = out.periode_validasi_temporal.str.split(" ").str[0]
    out["pembedahan_24jam"] = out.pembedahan_24jam.str.capitalize()
    dx = out.kategori_diagnosis_utama.str.replace("Sistem ", "")
    out["kategori_diagnosis_utama"] = dx.str[0].str.upper() + dx.str[1:]
    out["kelas_rawat_jkn"] = out.kelas_rawat_jkn.astype(int)
    out["durasi_rawat_pra_icu_jam"] = out.durasi_rawat_pra_icu_jam.round(1)
    out["rasio_spo2_fio2_terendah"] = out.rasio_spo2_fio2_terendah.round(1)
    out = out[[c[0] for c in COLUMNS]]
    key = pd.DataFrame({"id_episode": out.id_episode, "id_pasien": out.id_pasien,
                        "episode_id_asli": d.episode_id, "patient_code_asli": d.patient_code})
    kamus = pd.DataFrame([{"No": i + 1, "Kolom": c, "Nama variabel (Tabel 3.1)": nm, "Domain": dom,
                           "Peran": role, "Satuan / kode": unit,
                           "Data hilang, n": int(out[c].isna().sum())}
                          for i, (c, _, nm, dom, role, unit) in enumerate(COLUMNS)])
    return out, kamus, key


def main() -> None:
    cfg = load_config()
    assert cfg["project"]["data_mode"] == "REAL"
    d = pd.read_csv(cfg.path("data_interim") / cfg["preprocessing"]["output_file"])
    flow = pd.read_csv(cfg.table_dir("02_preprocessing") / "01_alur_seleksi.csv")
    assert int(flow.n_sisa.iloc[-1]) == len(d)
    fig = flow_figure(flow, int(d.under_reimbursement.sum()), cfg.figure_dir("05_bab4") / "03_alur_seleksi.png")

    data, kamus, key = build_dataset(d)
    # pemeriksaan: dataset = data yang masuk model (urutan berbeda, isi sama)
    X = pd.read_csv(cfg.path("data_processed") / cfg["feature_building"]["output_features"])
    assert len(data) == len(X) == 801 and data.id_pasien.nunique() == d.patient_code.nunique()
    assert np.allclose(np.sort(data.durasi_rawat_pra_icu_jam), np.sort(np.expm1(X.pre_icu_los_log1p)), atol=0.051)
    assert data.under_reimbursement.sum() == d.under_reimbursement.sum()
    assert ((data.rasio_klaim_tagihan < 1) == (data.under_reimbursement == 1)).all()
    assert not data.columns.str.contains("asli|no_rm|nama").any()

    catatan = pd.DataFrame({"Catatan": [
        "Satu baris = satu episode rawat inap; kohort analisis akhir (n = 801; 773 pasien).",
        "Kolom id_pasien dipakai untuk mengelompokkan fold (episode pasien yang sama berada di fold yang sama).",
        "Nilai hilang sengaja dibiarkan kosong. MICE, one-hot encoding, dan standardisasi dilakukan di dalam "
        "setiap fold validasi, bukan pada dataset ini.",
        "Prediktor model: 14 variabel bebas + penanda GCS tidak dapat dinilai karena sedasi.",
        "total_tagihan_rs, klaim_inacbg_final, dan rasio_klaim_tagihan hanya untuk membentuk/mendeskripsikan "
        "luaran dan tidak boleh dipakai sebagai prediktor (kebocoran luaran).",
        "Klaim INA-CBGs final = nilai klaim yang telah disetujui BPJS Kesehatan.",
        "Tidak memuat nomor rekam medis, nama, atau ID episode rumah sakit.",
    ]})
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        data.to_excel(xw, sheet_name="Dataset", index=False)
        kamus.to_excel(xw, sheet_name="Kamus_Variabel", index=False)
        catatan.to_excel(xw, sheet_name="Catatan", index=False)
        for name, df in (("Dataset", data), ("Kamus_Variabel", kamus), ("Catatan", catatan)):
            ws = xw.sheets[name]
            ws.freeze_panes(1, 2 if name == "Dataset" else 0)
            for j, c in enumerate(df.columns):
                width = max(len(str(c)), *(len(str(v)) for v in df[c].head(200))) + 2
                ws.set_column(j, j, min(width, 70))
        xw.sheets["Dataset"].autofilter(0, 0, len(data), len(data.columns) - 1)
    data.to_csv(OUT_XLSX.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    key.to_csv(cfg.table_dir("05_bab4") / "peta_id_studi.csv", index=False, encoding="utf-8-sig")
    print(fig, OUT_XLSX, OUT_XLSX.with_suffix(".csv"), sep="\n")
    print(kamus.to_string(index=False))


if __name__ == "__main__":
    main()
