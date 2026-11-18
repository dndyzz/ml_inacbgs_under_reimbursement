"""Pembersihan data, penerapan kriteria seleksi, dan penetapan outcome.

Urutan kerja mengikuti Bab 3 proposal:
    turunkan variabel waktu  ->  periksa rentang wajar  ->  terapkan kriteria
    inklusi/eksklusi  ->  tetapkan outcome pada akhir episode

Prinsip yang dijaga: TIDAK ADA imputasi di tahap ini. Nilai hilang dibiarkan
sebagai NaN dan baru diimputasi di dalam fold validasi (tahap 04), sesuai
rencana analisis untuk mencegah kebocoran data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung variabel yang di proposal didefinisikan sebagai selisih/rasio."""
    df = df.copy()

    # Durasi rawat pra-ICU: selisih timestamp masuk RS dan masuk ICU
    df["pre_icu_los_hours"] = (
        df["icu_admission_datetime"] - df["hospital_admission_datetime"]
    ).dt.total_seconds() / 3600

    # Lama rawat ICU dari timestamp (dipakai untuk kriteria eksklusi <24 jam)
    df["icu_los_hours"] = (
        df["icu_discharge_datetime"] - df["icu_admission_datetime"]
    ).dt.total_seconds() / 3600

    # Data mart memetakan seluruh bab neoplasma ke hemato-onkologi; neoplasma
    # jinak dikembalikan ke sistem organ lokasinya (definisi operasional)
    if "diagnosis_icd10" in df:
        df["diagnosis_category_it"] = df["diagnosis_category"]
        df["diagnosis_category"] = recategorize_diagnosis(df["diagnosis_icd10"],
                                                          df["diagnosis_category"])

    # GCS total kosong karena V tak dapat dinilai (terintubasi, TIDAK tersedasi):
    # V diturunkan dari E dan M (Meredith dkk., J Trauma 1998;44:839-44).
    # Pasien tersedasi tetap hilang -> ditangani flag gcs_unassessable + MICE.
    if {"gcs_e", "gcs_m", "gcs_v"} <= set(df):
        v_est = (-0.3756 + 0.5713 * df["gcs_m"] + 0.4233 * df["gcs_e"]).clip(1, 5)
        fill = (df["gcs_lowest"].isna() & df["gcs_e"].notna() & df["gcs_m"].notna()
                & df["gcs_v"].isna() & (df["gcs_note"] == "tersedia"))
        df.loc[fill, "gcs_lowest"] = (df["gcs_e"] + df["gcs_m"] + v_est).round()[fill]
        df["gcs_v_derived"] = fill.astype(int)

    # Rasio SpO2/FiO2 terendah
    df["sf_ratio_lowest"] = df["spo2_lowest"] / df["fio2_lowest"]

    # Penanda GCS tidak dapat dinilai karena sedasi (dipertahankan sebagai fitur)
    df["gcs_unassessable"] = (df["gcs_note"] != "tersedia").astype(int)

    return df


# Neoplasma jinak D10-D36 -> sistem organ lokasi, memakai 10 kategori formulir
# penelitian (Lampiran 3). Tidak ada kategori muskuloskeletal/kulit -> Lain-lain.
# Neoplasma ganas (C), in situ (D00-D09), perilaku tak pasti (D37-D48) dan
# hematologi (D50-D89) tetap hemato-onkologi.
_BENIGN_SITE = {
    **dict.fromkeys(["D12", "D13"], "Sistem gastrointestinal/hati"),
    **dict.fromkeys(["D14", "D15"], "Sistem respirasi"),
    **dict.fromkeys(["D25", "D26", "D27", "D28"], "Sistem obstetri/ginekologi"),
    **dict.fromkeys(["D29", "D30", "D34", "D35"], "Sistem renal/metabolik"),
    **dict.fromkeys(["D32", "D33"], "Sistem neurologi"),
}
_BENIGN_SUBSITE = {  # pengecualian tingkat 4 karakter
    "D15.1": "Sistem kardiovaskular",                     # jantung
    "D35.2": "Sistem neurologi", "D35.3": "Sistem neurologi", "D35.4": "Sistem neurologi",
    "D36.1": "Sistem neurologi",                          # saraf tepi
}


def recategorize_diagnosis(icd: pd.Series, category: pd.Series) -> pd.Series:
    """Koreksi kategori IT; kode lain/kosong memakai kategori IT apa adanya.

    - neoplasma jinak D10-D36 -> sistem organ lokasi
    - T80-T88 (komplikasi tindakan medis/bedah) bukan trauma -> Lain-lain
    - obstetri/ginekologi (n < 20, OR tak stabil) digabung ke Lain-lain
    """
    code = icd.fillna("").astype(str).str.strip().str.upper()
    benign = code.str.match(r"D(1\d|2\d|3[0-6])")
    new = code.str[:5].map(_BENIGN_SUBSITE).fillna(code.str[:3].map(_BENIGN_SITE)).fillna("Lain-lain")
    out = category.where(~benign, new)
    out = out.mask(code.str.match(r"T8[0-8]"), "Lain-lain")
    return out.replace({"Sistem obstetri/ginekologi": "Lain-lain"})


def apply_range_checks(df: pd.DataFrame, ranges: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nilai di luar batas fisiologis wajar dianggap salah input -> jadi NaN.

    Sengaja diubah menjadi hilang (bukan dibuang barisnya) agar episode tetap
    masuk analisis dan nilainya diimputasi bersama nilai hilang lain.
    """
    df = df.copy()
    rows = []
    for col, (low, high) in ranges.items():
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        invalid = ((s < low) | (s > high)) & s.notna()
        df.loc[invalid, col] = np.nan
        rows.append({
            "variabel": col,
            "batas_bawah": low,
            "batas_atas": high,
            "n_di_luar_batas": int(invalid.sum()),
            "persen": round(100 * invalid.mean(), 2),
        })
    return df, pd.DataFrame(rows)


def apply_selection(df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Terapkan kriteria inklusi/eksklusi; kembalikan kohort + tabel alur seleksi.

    Tabel alur inilah isi Gambar 3.1 (Alur Seleksi Subjek).
    """
    pre = cfg["preprocessing"]
    flow = [{"tahap": "Episode terekstraksi dari SIM RS", "n_keluar": 0, "n_sisa": len(df)}]

    def drop(mask: pd.Series, label: str, frame: pd.DataFrame) -> pd.DataFrame:
        kept = frame.loc[~mask]
        flow.append({"tahap": label, "n_keluar": int(mask.sum()), "n_sisa": len(kept)})
        return kept

    df = drop(df["episode_id"].duplicated(keep="first"),
              "Dikeluarkan: duplikat episode", df)
    df = drop(df["age"].isna() | (df["age"] < pre["min_age"]),
              f"Dikeluarkan: usia < {pre['min_age']} tahun atau tidak diketahui", df)
    df = drop(df["icu_los_hours"] < pre["min_icu_hours"],
              f"Dikeluarkan: lama rawat ICU < {pre['min_icu_hours']} jam", df)
    df = drop(df["pre_icu_los_hours"].isna(),
              "Dikeluarkan: durasi pra-ICU tidak dapat dihitung (rujukan tanpa waktu masuk)", df)
    # klaim = 0 / tagihan = 0 diperlakukan sama dengan kosong: klaim belum final
    # (pending/ditolak) bukan under-reimbursement, dan rasio /0 tak terdefinisi
    df = drop(~(df["total_hospital_billing"] > 0) | ~(df["inacbg_claim"] > 0),
              "Dikeluarkan: data tagihan/klaim tidak lengkap atau nol (outcome tidak dapat ditentukan)", df)

    # Episode dengan data tidak konsisten dikeluarkan (bukan dikoreksi). Data mart
    # RSCM belum terkonfirmasi IT; sampel tetap > target 600 (lihat Bab 4).
    if "hospital_discharge_datetime" in df:
        df = drop(df["hospital_discharge_datetime"].isna(),
                  "Dikeluarkan: tanggal keluar RS kosong (tagihan mungkin belum final)", df)
        df = drop(df["icu_discharge_datetime"] > df["hospital_discharge_datetime"],
                  "Dikeluarkan: keluar ICU tercatat setelah keluar RS", df)
    if "icu_billing" in df:
        df = drop(df["icu_billing"] > df["total_hospital_billing"],
                  "Dikeluarkan: biaya ICU melebihi total tagihan episode", df)
    df = drop(df["jkn_class"].isna(),
              "Dikeluarkan: kelas rawat JKN tidak valid (kode di luar 1-3)", df)
    df = drop((df["mechanical_ventilation"] == 0) & (df["oxygen_device"] == "Ventilator"),
              "Dikeluarkan: status ventilasi bertentangan dengan alat O2 (ventilator)", df)

    flow[-1]["tahap"] = flow[-1]["tahap"]
    flow.append({"tahap": "Kohort analisis akhir", "n_keluar": 0, "n_sisa": len(df)})
    return df.reset_index(drop=True), pd.DataFrame(flow)


def define_outcome(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Tetapkan outcome pada level episode (dihitung di akhir episode rawat).

    Primer  : biner under-reimbursement (rasio klaim/tagihan < 1)
    Sekunder: log rasio klaim/tagihan (simetris terhadap titik impas)
    """
    out = cfg["preprocessing"]["outcome"]
    df = df.copy()
    df[out["ratio_col"]] = df["inacbg_claim"] / df["total_hospital_billing"]
    df[out["binary_col"]] = (df[out["ratio_col"]] < 1).astype(int)
    df[out["continuous_col"]] = np.log(df[out["ratio_col"]])
    df["selisih_rupiah"] = df["inacbg_claim"] - df["total_hospital_billing"]
    return df


def add_temporal_split(df: pd.DataFrame, split_date: str) -> pd.DataFrame:
    """Beri label periode latih/uji untuk validasi internal-eksternal temporal."""
    df = df.copy()
    cutoff = pd.Timestamp(split_date)
    df["periode"] = np.where(
        df["icu_admission_datetime"] < cutoff, "latih (Mei-Des 2025)", "uji (Jan-Mei 2026)"
    )
    return df


def missingness_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ringkasan nilai hilang pada kohort akhir (menjadi dasar strategi MICE)."""
    rows = []
    for col in cols:
        if col not in df:
            continue
        rows.append({
            "variabel": col,
            "n_hilang": int(df[col].isna().sum()),
            "persen_hilang": round(100 * df[col].isna().mean(), 2),
            "strategi": "MICE di dalam fold" if df[col].isna().any() else "-",
        })
    return pd.DataFrame(rows).sort_values("persen_hilang", ascending=False)
