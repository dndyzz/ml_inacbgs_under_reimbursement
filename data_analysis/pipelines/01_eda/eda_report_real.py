"""Laporan EDA untuk Data Mart RSCM (data nyata dari tim IT).

1. Petakan `Data Mart ICU Dewasa JKN-rev.xlsx` ke skema kamus data pipeline
   APA ADANYA (hanya ganti nama kolom & kode nilai, tanpa pembersihan) ->
   data/raw/icu_inacbg_REAL_datamart.csv  (pola *_REAL* di-gitignore).
2. Jalankan tahap 01_eda yang sudah ada (config.yaml mode REAL -> outputs_real/,
   terpisah dari output data dummy yang ikut di-commit).
3. Tambah cek kualitas data khusus data mart + saran cleaning, lalu tulis
   satu laporan HTML mandiri: outputs_real/reports/laporan_eda_datamart.html

Jalankan:  python pipelines/01_eda/eda_report_real.py [--xlsx PATH]
"""

from __future__ import annotations

import argparse
import base64
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from common.config import load_config
from common.runner import load_pipeline

DEFAULT_XLSX = ROOT.parent / "data_raw" / "Data Mart ICU Dewasa JKN-rev.xlsx"
CSV_NAME = "icu_inacbg_REAL_datamart.csv"

YA = {"Ya": 1, "Tidak": 0}


def to_schema(dm: pd.DataFrame) -> pd.DataFrame:
    """Ganti nama kolom & kode nilai ke kamus data pipeline. Nilai TIDAK dibersihkan."""
    return pd.DataFrame({
        # no_rm sengaja tidak dibawa
        "episode_id": dm.kode_subjek.astype(str) + "-" + dm.id_episode_rs.astype(str),
        "patient_code": dm.kode_subjek.astype(str),
        "hospital_admission_datetime": dm.tgl_jam_masuk_rs,
        "icu_admission_datetime": dm.tgl_jam_masuk_icu,
        "icu_discharge_datetime": dm.tgl_jam_keluar_icu,
        "hospital_discharge_datetime": dm.tgl_jam_keluar_rs,
        "age": dm.usia_tahun,
        "sex": dm.jenis_kelamin.map({"m": "Laki-laki", "f": "Perempuan"}),
        "jkn_class": dm.kelas_rawat_jkn,
        "icu_admission_type": dm.tipe_admisi_icu.replace({"Emergency": "Emergensi"}),
        "icu_admission_source": dm.sumber_admisi_icu,
        "diagnosis_icd10": dm.dx_utama_icd10,
        "diagnosis_category": dm.kategori_dx,
        "mechanical_ventilation": dm.ventilasi_mekanik_24j.map(YA),
        "vasopressor_inotrope": dm.vasopresor_inotropik_24j.map(YA),
        "transfusion_prc": dm.transfusi_prc_24j.map(YA),
        "surgery_24h": dm.pembedahan_24j.map({"Tidak": "tidak", "Ya-Elektif": "elektif",
                                              "Ya-Emergensi": "emergensi"}),
        "map_lowest": dm.map_terendah_24j,
        "map_source": dm.map_sumber,
        "spo2_lowest": dm.spo2_pada_sf_min,
        "fio2_lowest": dm.fio2_final,
        "oxygen_device": dm.alat_oksigen_sf_min.replace({"Ventilator Mekanik": "Ventilator"}),
        "consciousness_lowest": dm.kesadaran_terendah_24j,
        "gcs_lowest": dm.gcs_terendah_24j,
        "gcs_e": dm.gcs_terendah_24j_e,
        "gcs_m": dm.gcs_terendah_24j_m,
        "gcs_v": dm.gcs_terendah_24j_v,
        # kode kamus data: hanya "tidak dapat dinilai" yang jadi flag gcs_unassessable
        "gcs_note": np.where(dm.gcs_sumber == "Tidak dapat dinilai (sedasi)",
                             "tersedasi_seluruh_24_jam", "tersedia"),
        "gcs_source": dm.gcs_sumber,
        "creatinine_highest": dm.kreatinin_tertinggi_24j,
        "icu_los_days": dm.lama_rawat_icu_jam / 24,
        "post_icu_los_days": (dm.tgl_jam_keluar_rs - dm.tgl_jam_keluar_icu).dt.total_seconds() / 86400,
        "discharge_status": dm.meninggal_episode,
        "icu_discharge_status": dm.status_keluar_icu,
        "inacbg_code": dm.kode_inacbg_final,
        "inacbg_severity_level": dm.severity_level_final,
        "total_hospital_billing": dm.total_tagihan_episode,
        "inacbg_claim": dm.klaim_inacbg_final,
        "icu_billing": dm.biaya_icu,
        "under_reimbursement_it": dm.under_reimbursement,   # versi IT, untuk dibandingkan
    })


def quality_checks(d: pd.DataFrame) -> pd.DataFrame:
    """Setiap cek = (temuan, n baris, saran). n dihitung, bukan ditulis tangan."""
    r = d.inacbg_claim / d.total_hospital_billing
    gcs_sum = d.gcs_e + d.gcs_m + d.gcs_v
    rows = [
        ("Outcome", "Klaim atau tagihan kosong",
         d.inacbg_claim.isna() | d.total_hospital_billing.isna(),
         "Eksklusi (kriteria: data finansial tidak lengkap)."),
        ("Outcome", "Klaim = 0 (tagihan > 0) — kemungkinan pending/ditolak/dispute",
         (d.inacbg_claim == 0) & (d.total_hospital_billing > 0),
         "Konfirmasi ke IT/casemix. Bila belum final → eksklusi; jangan dihitung under-reimbursement."),
        ("Outcome", "Tagihan = 0",
         d.total_hospital_billing == 0, "Eksklusi (rasio tidak terdefinisi)."),
        ("Outcome", "Kolom under_reimbursement IT = 1 padahal rasio tidak bisa dihitung",
         (d.under_reimbursement_it == 1) & r.isna(),
         "Jangan pakai kolom IT; hitung ulang label dari klaim/tagihan."),
        ("Outcome", "Rasio klaim/tagihan > 2 (over-reimbursement besar)",
         r > 2, "Pertahankan (plausibel pada CBG ventilator J-1-20/J-1-02), tapi verifikasi sampel."),
        ("Waktu", "Keluar ICU setelah keluar RS",
         d.icu_discharge_datetime > d.hospital_discharge_datetime,
         "Tanya IT; tidak memengaruhi prediktor (pasca-24 jam)."),
        ("Waktu", "Tanggal keluar RS kosong", d.hospital_discharge_datetime.isna(),
         "Cek apakah episode belum selesai → tagihan belum final."),
        ("Administratif", "Kelas rawat JKN = 0 (di luar kode 1/2/3)", d.jkn_class == 0,
         "Tanya arti kode 0. Bila tidak diketahui → missing lalu imputasi."),
        ("Intervensi", "Ventilasi 24 jam = Tidak, tapi alat O₂ saat S/F terendah = Ventilator",
         (d.mechanical_ventilation == 0) & (d.oxygen_device == "Ventilator"),
         "Tanya IT definisi & jendela waktu kedua kolom; tentukan satu sumber kebenaran."),
        ("Intervensi", "Admisi elektif dengan pembedahan 'elektif' (seluruhnya)",
         (d.icu_admission_type == "Elektif") & (d.surgery_24h == "elektif"),
         "Curiga surgery_24h diturunkan dari tipe admisi, bukan jadwal OK 24 jam. Minta data kamar operasi."),
        ("Disfungsi organ", "FiO₂ > 1", d.fio2_lowest > 1, "Salah input → missing."),
        ("Disfungsi organ", "SpO₂ < 50%", d.spo2_lowest < 50, "Salah input → missing."),
        ("Disfungsi organ", "MAP < 30 mmHg", d.map_lowest < 30, "Verifikasi; bila artefak → missing."),
        ("Disfungsi organ", "Kreatinin > 20 mg/dL", d.creatinine_highest > 20,
         "Plausibel (ESRD/AKI berat) — pertahankan, verifikasi."),
        ("Disfungsi organ", "GCS kosong padahal 'Kesadaran Menurun'",
         d.gcs_lowest.isna() & (d.consciousness_lowest == "Kesadaran Menurun"),
         "Minta IT ambil komponen E/M/V dari lembar observasi; sisanya imputasi (MICE, di dalam fold)."),
        ("Disfungsi organ", "GCS kosong karena sedasi ('Tidak dapat dinilai')",
         d.gcs_lowest.isna() & (d.gcs_source == "Tidak dapat dinilai (sedasi)"),
         "Pertahankan sebagai missing informatif + flag gcs_unassessable."),
        ("Disfungsi organ", "Total GCS ≠ E+M+V", gcs_sum.notna() & (gcs_sum != d.gcs_lowest),
         "Hitung ulang dari komponen."),
        ("Disfungsi organ", "GCS = 15 padahal dicatat 'Sedasi, sadar penuh'",
         (d.gcs_lowest == 15) & (d.gcs_source == "Sedasi, sadar penuh"),
         "Tanya IT: diukur atau diisi otomatis dari teks 'Sadar Penuh'?"),
        ("Kasus", "Diagnosis utama ICD-10 kosong", d.diagnosis_icd10.isna(),
         "Kategori tetap terisi — tanya IT asal kategorinya."),
        ("Kasus", "Neoplasma jinak (D10–D36) dikategorikan hemato-onkologi",
         d.diagnosis_icd10.fillna("").str.match(r"D(1\d|2\d|3[0-6])")
         & (d.diagnosis_category == "Sistem hemato-onkologi"),
         "Kategorikan ulang dari ICD-10 per sistem organ sesuai definisi operasional."),
        ("Biaya", "Biaya ICU > total tagihan episode", d.icu_billing > d.total_hospital_billing,
         "Tanya IT; tidak dipakai sebagai prediktor."),
    ]
    out = pd.DataFrame([(dom, t, int(m.fillna(False).sum()), s) for dom, t, m, s in rows],
                       columns=["domain", "temuan", "n", "saran"])
    out["%"] = (100 * out.n / len(d)).round(1)
    return out[["domain", "temuan", "n", "%", "saran"]]


def class_distribution(d: pd.DataFrame) -> pd.DataFrame:
    from profiling import wilson_ci   # pipelines/01_eda sudah ada di sys.path via load_pipeline
    r = d.inacbg_claim / d.total_hospital_billing
    scen = {
        "Semua episode dengan rasio terhitung (apa adanya)": r.notna() & np.isfinite(r),
        "Setelah eksklusi klaim = 0 / tagihan = 0": (d.inacbg_claim > 0) & (d.total_hospital_billing > 0),
    }
    rows = []
    for name, m in scen.items():
        u = int((r[m] < 1).sum()); n = int(m.sum()); lo, hi = wilson_ci(u, n)
        rows.append({"skenario": name, "n": n, "under-reimbursement (<1)": u,
                     "adekuat (≥1)": n - u, "prevalensi": f"{u/n:.1%}",
                     "IK95%": f"{lo:.1%}–{hi:.1%}", "rasio adekuat : under": f"1 : {u/(n-u):.1f}"})
    return pd.DataFrame(rows)


def missing_table(d: pd.DataFrame) -> pd.DataFrame:
    ok = (d.inacbg_claim > 0) & (d.total_hospital_billing > 0)
    d = d[ok].copy()
    d["sf"] = d.spo2_lowest / d.fio2_lowest
    y = d.inacbg_claim / d.total_hospital_billing < 1
    preds = ["age", "sex", "jkn_class", "icu_admission_type", "hospital_admission_datetime",
             "mechanical_ventilation", "vasopressor_inotrope", "transfusion_prc", "surgery_24h",
             "map_lowest", "sf", "gcs_lowest", "creatinine_highest", "diagnosis_category"]
    rows = []
    for c in preds:
        miss = d[c].isna() | ((c == "jkn_class") & (d[c] == 0))
        rows.append({"prediktor": c, "hilang": int(miss.sum()), "% hilang": round(100 * miss.mean(), 1),
                     "prevalensi under bila hilang": f"{y[miss].mean():.1%}" if miss.any() else "-",
                     "prevalensi under bila ada": f"{y[~miss].mean():.1%}"})
    return pd.DataFrame(rows)


def multi_episode(d: pd.DataFrame) -> dict:
    vc = d.patient_code.value_counts()
    return {"Episode": len(d), "Pasien unik": int(vc.size),
            "Pasien dengan >1 episode": int((vc > 1).sum()),
            "Episode milik pasien berulang": int(vc[vc > 1].sum())}


# ----------------------------------------------------------------- HTML
CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--rule:#e1e0d9;--acc:#2a78d6;--warn:#fff4d6}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--ink:#ededea;--ink2:#a9a8a2;--rule:#33332f;--acc:#6da7ec;--warn:#3a3118}}
body{background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:26px;margin:0 0 4px}h2{margin-top:40px;border-bottom:2px solid var(--rule);padding-bottom:4px}
.sub{color:var(--ink2)}table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}
th,td{border-bottom:1px solid var(--rule);padding:5px 8px;text-align:left;vertical-align:top}
th{background:color-mix(in srgb,var(--acc) 10%,transparent)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto}img{max-width:100%;background:#fff;border:1px solid var(--rule);margin:8px 0}
.box{background:var(--warn);padding:10px 14px;border-radius:6px;margin:12px 0}
ol li{margin-bottom:8px}
"""


def tbl(df: pd.DataFrame) -> str:
    def td(v):
        num = isinstance(v, (int, float, np.integer, np.floating))
        return f'<td class="n">{v:,}</td>' if num and not isinstance(v, bool) else f"<td>{html.escape(str(v))}</td>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(td(v) for v in row) + "</tr>" for row in df.itertuples(index=False))
    return f'<div class="wrap"><table><tr>{head}</tr>{body}</table></div>'


def img(p: Path, caption: str) -> str:
    b = base64.b64encode(Path(p).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{b}"><figcaption class="sub">{html.escape(caption)}</figcaption></figure>'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default=str(DEFAULT_XLSX))
    a = ap.parse_args()

    dm = pd.read_excel(a.xlsx)
    d = to_schema(dm)
    cfg = load_config()   # config.yaml mode REAL: output ke outputs_real/ (di-gitignore)
    assert cfg["eda"]["input_file"] == CSV_NAME, "config.yaml belum mode REAL"
    d.to_csv(cfg.path("data_raw") / CSV_NAME, index=False)

    eda = load_pipeline("01_eda").run(cfg)       # kode EDA yang sudah ada, tanpa diubah
    figs = {Path(f).stem: f for f in eda["figures"]}

    qc, cls, miss, me = quality_checks(d), class_distribution(d), missing_table(d), multi_episode(d)
    t41 = eda["tables"]["Tabel 4.1 Karakteristik subjek"]
    num = eda["tables"]["Deskriptif numerik"]
    prev = eda["tables"]["Prevalensi per subkelompok"]
    rdir = cfg.path("reports")
    for name, t in {"qc_datamart": qc, "distribusi_kelas": cls, "missing_per_prediktor": miss}.items():
        t.to_csv(cfg.table_dir("01_eda") / f"{name}.csv", index=False)

    s = eda["summary"]
    period = f"{d.icu_admission_datetime.min():%d %b %Y} – {d.icu_admission_datetime.max():%d %b %Y}"
    parts = [
        f"<h1>Laporan EDA — Data Mart ICU Dewasa JKN</h1>",
        f'<p class="sub">Sumber: {html.escape(Path(a.xlsx).name)} · {len(dm):,} baris × {dm.shape[1]} kolom · '
        f"masuk ICU {period}. Data dibaca apa adanya; belum ada pembersihan.</p>",
        "<h2>1. Ringkasan</h2>", tbl(pd.DataFrame({"Butir": list(s) + list(me),
                                                    "Nilai": list(s.values()) + [f"{v:,}" for v in me.values()]})),
        "<h2>2. Distribusi kelas outcome</h2>",
        "<p>Outcome = rasio klaim INA-CBGs final / total tagihan episode; &lt; 1 = under-reimbursement. "
        "Label dihitung ulang dari klaim dan tagihan, bukan dari kolom <code>under_reimbursement</code> IT.</p>",
        tbl(cls),
        '<div class="box">Kelas minoritas adalah episode <b>adekuat</b> (±155). Ini yang membatasi jumlah '
        "parameter model (EPV), bukan total 1.000-an episode. AUC dan kalibrasi harus dilaporkan; "
        "akurasi tidak informatif karena model yang selalu menebak “under” sudah ~85% benar.</div>",
        img(figs.pop("04_biaya_vs_klaim"), "Tagihan vs klaim dan sebaran rasio klaim/tagihan"),
        "<h2>3. Karakteristik subjek menurut status outcome (draf Tabel 4.1)</h2>",
        "<p>Numerik: median (IQR), uji Mann-Whitney. Kategorik: n (%), uji chi-square. Deskriptif, bukan inferensi.</p>",
        tbl(t41),
        "<h2>4. Statistik deskriptif variabel numerik</h2>", tbl(num),
        "<h2>5. Prevalensi under-reimbursement per subkelompok</h2>", tbl(prev),
        "<h2>6. Nilai hilang pada 14 prediktor</h2>",
        "<p>Di antara episode dengan klaim &gt; 0 dan tagihan &gt; 0. Kelas JKN = 0 dihitung sebagai hilang.</p>",
        tbl(miss),
        "<h2>7. Cek kualitas data</h2>", tbl(qc),
        "<h2>8. Gambar</h2>",
        *[img(p, k.replace("_", " ")) for k, p in figs.items()],
        "<h2>9. Saran cleaning (urut prioritas)</h2>", CLEANING,
    ]
    out = rdir / "laporan_eda_datamart.html"
    out.write_text(f'<!doctype html><html lang="id"><head><meta charset="utf-8">'
                   f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                   f"<title>Laporan EDA Data Mart</title><style>{CSS}</style></head><body>"
                   + "\n".join(parts) + "</body></html>", encoding="utf-8")
    print(out)


CLEANING = """
<ol>
<li><b>Hitung ulang label outcome</b> dari <code>klaim / tagihan</code>; buang kolom <code>under_reimbursement</code> IT
(memberi label 1 pada baris tanpa rasio).</li>
<li><b>Eksklusi episode tanpa outcome valid</b>: klaim/tagihan kosong, tagihan = 0, dan klaim = 0 yang belum final
(setelah konfirmasi IT). Catat jumlahnya untuk diagram alur.</li>
<li><b>Nilai mustahil → missing</b>: FiO₂ &gt; 1, SpO₂ &lt; 50%, MAP &lt; 30. Hitung ulang S/F setelahnya.
Batas ini sudah ada di <code>preprocessing.plausible_ranges</code>; sesuaikan <code>sf_ratio_lowest</code> dan kreatinin (data nyata maks 22,2).</li>
<li><b>GCS</b>: hitung ulang total dari E+M+V; pisahkan “tidak dapat dinilai (sedasi)” sebagai flag; sisanya imputasi
MICE <i>di dalam fold</i>. Jangan complete-case — kelas adekuat turun dari 155 ke 66 dan missing-nya tidak acak.</li>
<li><b>Kelas JKN = 0</b>: tunggu arti kode dari IT; sementara perlakukan sebagai missing.</li>
<li><b>pembedahan_24j</b>: jangan dipakai sebelum IT memastikan sumbernya (jadwal kamar operasi 24 jam pertama).
Bila memang turunan tipe admisi, variabel ini redundan dan harus dikeluarkan atau diambil ulang.</li>
<li><b>ventilasi_mekanik_24j vs alat O₂</b>: tetapkan satu definisi; bila alat O₂ = ventilator pada jam ke-0–24,
ventilasi = Ya.</li>
<li><b>kategori_dx</b>: kategorikan ulang dari <code>dx_utama_icd10</code> per sistem organ (definisi operasional),
bukan per bab neoplasma; gabungkan kategori dengan n &lt; 20 (obstetri) ke “lain-lain”.</li>
<li><b>Pasien berulang</b>: pertahankan semua episode, tapi CV harus <code>GroupKFold</code> per <code>patient_code</code>
supaya episode pasien yang sama tidak ada di latih dan uji sekaligus.</li>
<li><b>Variabel pasca-24 jam</b> (lama rawat, status keluar, severity, kode CBG, biaya ICU) hanya untuk deskripsi —
tidak boleh masuk X.</li>
<li><b>Privasi</b>: <code>no_rm</code> sudah tidak dibawa ke CSV analisis; <code>kode_subjek</code> tetap per pasien.</li>
</ol>
"""

if __name__ == "__main__":
    main()
