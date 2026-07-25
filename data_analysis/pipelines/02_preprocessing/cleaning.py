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

    # Rasio SpO2/FiO2 terendah
    df["sf_ratio_lowest"] = df["spo2_lowest"] / df["fio2_lowest"]

    # Penanda GCS tidak dapat dinilai karena sedasi (dipertahankan sebagai fitur)
    df["gcs_unassessable"] = (df["gcs_note"] != "tersedia").astype(int)

    return df


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
    df = drop(df["total_hospital_billing"].isna() | df["inacbg_claim"].isna(),
              "Dikeluarkan: data tagihan/klaim tidak lengkap (outcome tidak dapat ditentukan)", df)

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
