"""Pembangkit data dummy episode rawat ICU + klaim INA-CBGs.

Data dibangkitkan dengan struktur kausal yang meniru mekanisme nyata, bukan
angka acak lepas, supaya pipeline hilir benar-benar teruji:

    keparahan laten (z)
        -> tanda vital & disfungsi organ 24 jam (MAP, SpO2/FiO2, GCS, kreatinin)
        -> organ support 24 jam (ventilator, vasopresor, transfusi, bedah)
        -> lama rawat ICU & bangsal
        -> total tagihan rumah sakit (biaya riil, mengikuti konsumsi sumber daya)

    kelompok diagnosis + severity level INA-CBGs
        -> nilai klaim (paket, TIDAK mengikuti intensitas aktual)

    rasio klaim/tagihan < 1  ->  under-reimbursement

Karena klaim ditentukan paket sedangkan tagihan mengikuti konsumsi sumber daya,
selisihnya bisa diprediksi dari variabel 24 jam pertama - persis hipotesis
penelitian. Semua angka biaya memakai satuan rupiah dan berada pada orde yang
dilaporkan literatur Indonesia (biaya ICU ~Rp 6,6 juta/hari).

CATATAN: seluruh isi file ini adalah DATA SINTETIS. Tidak ada data pasien nyata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameter domain
# ---------------------------------------------------------------------------

# 7 kelompok diagnosis menurut sistem organ -> 6 variabel indikator setelah
# one-hot, sehingga total parameter model ~20 (sesuai perhitungan besar sampel).
DIAGNOSIS = {
    #                      proporsi  efek keparahan  tarif dasar INA-CBGs (Rp)
    "Sepsis/Infeksi":      (0.20,    0.55,           30_000_000),
    "Kardiovaskular":      (0.17,    0.30,           34_000_000),
    "Respirasi":           (0.16,    0.40,           28_000_000),
    "Neurologi":           (0.16,    0.25,           31_000_000),
    "Gastrointestinal":    (0.13,    0.15,           29_000_000),
    "Trauma/Bedah":        (0.10,    0.20,           35_000_000),
    "Lainnya":             (0.08,    0.00,           26_000_000),
}

SEX = {"Laki-laki": 0.55, "Perempuan": 0.45}
JKN_CLASS = {1: 0.15, 2: 0.30, 3: 0.55}

# Biaya satuan (rupiah)
ICU_COST_PER_DAY = 6_660_000       # Nur et al. 2024 (RSHS Bandung)
WARD_COST_PER_DAY = 1_250_000
SURGERY_COST = {"tidak": 0, "elektif": 18_000_000, "emergensi": 26_000_000}
TRANSFUSION_COST = 4_200_000
DIAGNOSTIC_BASE_COST = 5_500_000
DIAGNOSTIC_PER_ICU_DAY = 850_000

# Pengali tarif & biaya menurut kelas rawat JKN
CLASS_COST_MULT = {1: 1.15, 2: 1.00, 3: 0.92}
CLASS_CLAIM_MULT = {1: 1.10, 2: 1.00, 3: 0.93}
SEVERITY_CLAIM_MULT = {1: 1.00, 2: 1.35, 3: 1.80}

OXYGEN_DEVICE_NONVENT = ["Udara ruangan", "Nasal kanul", "Simple mask", "NRM"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_raw_dataset(cfg) -> pd.DataFrame:
    """Bangkitkan tabel mentah seperti hasil ekstraksi SIM RS + rekam medis."""
    gen = cfg["data_generation"]
    rng = np.random.default_rng(cfg.seed)
    n = int(gen["n_episodes"])

    # -- 1. Konteks admisi --------------------------------------------------
    diag_names = list(DIAGNOSIS)
    diag_probs = np.array([DIAGNOSIS[d][0] for d in diag_names])
    diag_probs = diag_probs / diag_probs.sum()
    diagnosis = rng.choice(diag_names, size=n, p=diag_probs)
    diag_severity = np.array([DIAGNOSIS[d][1] for d in diagnosis])

    admission_type = rng.choice(["Emergensi", "Elektif"], size=n, p=[0.68, 0.32])
    is_emergency = (admission_type == "Emergensi").astype(float)

    sex = rng.choice(list(SEX), size=n, p=list(SEX.values()))
    jkn_class = rng.choice(list(JKN_CLASS), size=n, p=list(JKN_CLASS.values()))

    age = np.clip(rng.normal(53, 16, n), 18, 92).round(0)

    # -- 2. Keparahan laten (tidak ikut disimpan; hanya penggerak) ----------
    z = (
        diag_severity
        + 0.45 * is_emergency
        + 0.012 * (age - 53)
        + rng.normal(0, 0.85, n)
    )

    # -- 3. Organ support 24 jam pertama ------------------------------------
    is_resp = (diagnosis == "Respirasi").astype(float)
    is_neuro = (diagnosis == "Neurologi").astype(float)
    is_sepsis = (diagnosis == "Sepsis/Infeksi").astype(float)
    is_cardio = (diagnosis == "Kardiovaskular").astype(float)
    is_trauma = (diagnosis == "Trauma/Bedah").astype(float)
    is_gi = (diagnosis == "Gastrointestinal").astype(float)

    p_vent = _sigmoid(-0.85 + 1.15 * z + 1.25 * is_resp + 0.85 * is_neuro + 0.35 * is_emergency)
    mech_vent = rng.binomial(1, p_vent)

    p_vaso = _sigmoid(-1.30 + 1.30 * z + 1.05 * is_sepsis + 0.95 * is_cardio + 0.45 * mech_vent)
    vasopressor = rng.binomial(1, p_vaso)

    # Pembedahan 24 jam: elektif hampir selalu pascaoperasi terjadwal
    p_elective_surg = _sigmoid(1.30 - 0.25 * z) * (1 - is_emergency)
    p_emergency_surg = _sigmoid(-1.55 + 0.55 * z + 1.30 * is_trauma + 0.75 * is_gi) * is_emergency
    u = rng.random(n)
    surgery = np.where(
        u < p_elective_surg, "elektif",
        np.where(u < p_elective_surg + p_emergency_surg, "emergensi", "tidak"),
    )
    surg_any = (surgery != "tidak").astype(float)

    p_transf = _sigmoid(-2.15 + 0.75 * z + 1.15 * is_trauma + 0.85 * is_gi + 0.70 * surg_any)
    transfusion = rng.binomial(1, p_transf)

    # -- 4. Disfungsi organ 24 jam (komponen mSOFA) -------------------------
    map_lowest = np.clip(
        82 - 12.5 * z - 6.5 * vasopressor + rng.normal(0, 7.5, n), 30, 125
    ).round(0)

    sf_ratio = np.clip(
        395 - 82 * z - 55 * mech_vent + rng.normal(0, 42, n), 60, 476
    )
    # SpO2 dan FiO2 dicatat terpisah di rekam medis; rasio dihitung saat preprocessing
    spo2 = np.clip(np.where(sf_ratio < 150, rng.normal(91, 3.5, n), rng.normal(96, 2.2, n)), 70, 100).round(0)
    fio2 = np.clip(spo2 / sf_ratio, 0.21, 1.00).round(2)
    oxygen_device = np.where(
        mech_vent == 1,
        "Ventilator",
        rng.choice(OXYGEN_DEVICE_NONVENT, size=n, p=[0.25, 0.40, 0.20, 0.15]),
    )

    gcs = np.clip(
        np.round(15 - 2.6 * np.maximum(z, -0.5) - 3.2 * is_neuro + rng.normal(0, 1.4, n)), 3, 15
    )

    creatinine = np.clip(
        np.exp(-0.05 + 0.38 * z + rng.normal(0, 0.42, n)), 0.2, 15
    ).round(2)

    # -- 5. Durasi rawat ----------------------------------------------------
    pre_icu_hours = np.where(
        is_emergency == 1,
        np.exp(rng.normal(2.6, 1.15, n)),      # median ~13 jam, ekor panjang
        np.exp(rng.normal(2.1, 0.70, n)),      # elektif: masuk cepat pascaoperasi
    )
    pre_icu_hours = np.clip(pre_icu_hours, 0.5, 1400).round(1)

    icu_days = np.clip(
        np.exp(0.55 + 0.42 * z + 0.55 * mech_vent + 0.25 * vasopressor + rng.normal(0, 0.52, n)),
        0.3, 45,
    )

    mortality_p = _sigmoid(-2.4 + 0.95 * z + 0.5 * mech_vent)
    died = rng.binomial(1, mortality_p)
    post_icu_days = np.where(
        died == 1, 0.0,
        np.clip(np.exp(1.15 + 0.20 * z + 0.35 * surg_any + rng.normal(0, 0.62, n)), 0, 60),
    )

    # -- 6. Tagihan rumah sakit (biaya riil, mengikuti konsumsi sumber daya) -
    cost_mult = np.array([CLASS_COST_MULT[c] for c in jkn_class])
    icu_intensity = 1 + 0.28 * mech_vent + 0.16 * vasopressor
    icu_cost = icu_days * ICU_COST_PER_DAY * cost_mult * icu_intensity
    pre_icu_cost = (pre_icu_hours / 24) * WARD_COST_PER_DAY * cost_mult * 1.15
    post_icu_cost = post_icu_days * WARD_COST_PER_DAY * cost_mult
    surgery_cost = np.array([SURGERY_COST[s] for s in surgery]) * cost_mult
    transfusion_cost = transfusion * TRANSFUSION_COST
    diagnostic_cost = (DIAGNOSTIC_BASE_COST + DIAGNOSTIC_PER_ICU_DAY * icu_days) * cost_mult

    total_billing = (
        icu_cost + pre_icu_cost + post_icu_cost + surgery_cost
        + transfusion_cost + diagnostic_cost
    ) * np.exp(rng.normal(0, 0.11, n))

    # -- 7. Klaim INA-CBGs (paket; tidak melihat lama rawat aktual) ---------
    support_count = mech_vent + vasopressor + transfusion
    severity_level = np.where(
        (support_count >= 2) | (creatinine > 3.0), 3,
        np.where((support_count == 1) | (surg_any == 1), 2, 1),
    )
    base_claim = np.array([DIAGNOSIS[d][2] for d in diagnosis])
    claim_mult = np.array([CLASS_CLAIM_MULT[c] for c in jkn_class])
    sev_mult = np.array([SEVERITY_CLAIM_MULT[s] for s in severity_level])

    inacbg_claim = (
        base_claim * sev_mult * claim_mult * np.exp(rng.normal(0, 0.09, n))
    )

    # Kalibrasi skala tarif agar prevalensi under-reimbursement mendekati target.
    target = gen.get("target_prevalence")
    if target:
        # skala s sedemikian rupa sehingga P(claim * s < billing) = target
        scale = float(np.quantile(total_billing / inacbg_claim, 1 - target))
    else:
        scale = float(gen.get("claim_scale", 1.0))
    inacbg_claim = inacbg_claim * scale

    # -- 8. Timestamp -------------------------------------------------------
    start = pd.Timestamp(gen["study_start"])
    end = pd.Timestamp(gen["study_end"])
    span_minutes = int((end - start).total_seconds() // 60)
    icu_admit = start + pd.to_timedelta(rng.integers(0, span_minutes, n), unit="m")
    hosp_admit = icu_admit - pd.to_timedelta(pre_icu_hours, unit="h")
    icu_discharge = icu_admit + pd.to_timedelta(icu_days * 24, unit="h")
    hosp_discharge = icu_discharge + pd.to_timedelta(post_icu_days * 24, unit="h")

    df = pd.DataFrame({
        "episode_id": [f"EP{idx:06d}" for idx in range(1, n + 1)],
        "patient_code": [f"RM-{v:06d}" for v in rng.integers(100000, 999999, n)],
        "hospital_admission_datetime": hosp_admit.round("min"),
        "icu_admission_datetime": icu_admit.round("min"),
        "icu_discharge_datetime": icu_discharge.round("min"),
        "hospital_discharge_datetime": hosp_discharge.round("min"),
        "age": age.astype(int),
        "sex": sex,
        "jkn_class": jkn_class,
        "icu_admission_type": admission_type,
        "referral_flag": rng.binomial(1, 0.22, n),
        "diagnosis_category": diagnosis,
        "mechanical_ventilation": mech_vent,
        "vasopressor_inotrope": vasopressor,
        "transfusion_prc": transfusion,
        "surgery_24h": surgery,
        "map_lowest": map_lowest,
        "spo2_lowest": spo2,
        "fio2_lowest": fio2,
        "gcs_lowest": gcs,
        "gcs_note": "tersedia",
        "oxygen_device": oxygen_device,
        "creatinine_highest": creatinine,
        "icu_los_days": icu_days.round(2),
        "post_icu_los_days": post_icu_days.round(2),
        "discharge_status": np.where(died == 1, "Meninggal", "Hidup"),
        "inacbg_severity_level": severity_level,
        "total_hospital_billing": total_billing.round(0),
        "inacbg_claim": inacbg_claim.round(0),
    })

    df = _inject_missingness(df, cfg, rng)
    df = _inject_dirty_rows(df, cfg, rng)
    return df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)


def _inject_missingness(df: pd.DataFrame, cfg, rng) -> pd.DataFrame:
    """Sisipkan nilai hilang dengan mekanisme MAR (bergantung variabel lain).

    Contoh paling penting: GCS hilang justru pada pasien tersedasi (terventilasi),
    persis masalah yang dibahas di definisi operasional proposal.
    """
    frac = cfg["data_generation"]["missing_fraction"]
    n = len(df)

    # GCS: peluang hilang jauh lebih besar pada pasien terventilasi
    base = frac["gcs_lowest"] * 0.4
    p_gcs = np.clip(base + 0.32 * df["mechanical_ventilation"].to_numpy(), 0, 0.9)
    sedated = rng.random(n) < p_gcs
    df.loc[sedated, "gcs_lowest"] = np.nan
    df.loc[sedated, "gcs_note"] = "tersedasi_seluruh_24_jam"

    # SpO2/FiO2: lebih sering tidak terdokumentasi pada pasien tanpa ventilator
    p_sf = frac["sf_ratio_lowest"] * (1.6 - 0.9 * df["mechanical_ventilation"].to_numpy())
    miss_sf = rng.random(n) < p_sf
    df.loc[miss_sf, ["spo2_lowest", "fio2_lowest"]] = np.nan

    for col, key in [("creatinine_highest", "creatinine_highest"), ("map_lowest", "map_lowest")]:
        mask = rng.random(n) < frac[key]
        df.loc[mask, col] = np.nan

    return df


def _inject_dirty_rows(df: pd.DataFrame, cfg, rng) -> pd.DataFrame:
    """Sisipkan baris yang harus tersaring pipeline preprocessing.

    Tanpa ini, pipeline 02 tidak pernah teruji: kriteria eksklusi, duplikat, dan
    pemeriksaan rentang nilai tidak akan pernah memicu apa pun.
    """
    frac = cfg["data_generation"]["dirty_fraction"]
    n = len(df)
    idx = df.index.to_numpy()

    # (a) ICU < 24 jam -> kriteria eksklusi 1
    k = int(frac["icu_los_lt_24h"] * n)
    pick = rng.choice(idx, size=k, replace=False)
    short = rng.uniform(2, 23.5, k) / 24
    df.loc[pick, "icu_los_days"] = short.round(2)
    df.loc[pick, "icu_discharge_datetime"] = (
        df.loc[pick, "icu_admission_datetime"] + pd.to_timedelta(short * 24, unit="h")
    )

    # (b) Pasien rujukan tanpa waktu masuk RS asal -> kriteria eksklusi 2
    k = int(frac["missing_pre_icu_time"] * n)
    pick = rng.choice(idx, size=k, replace=False)
    df.loc[pick, "hospital_admission_datetime"] = pd.NaT
    df.loc[pick, "referral_flag"] = 1

    # (c) Data finansial tidak lengkap -> kriteria eksklusi 3
    k = int(frac["missing_financial"] * n)
    pick = rng.choice(idx, size=k, replace=False)
    half = len(pick) // 2
    df.loc[pick[:half], "total_hospital_billing"] = np.nan
    df.loc[pick[half:], "inacbg_claim"] = np.nan

    # (d) Nilai di luar batas wajar (salah input) -> tertangkap QC rentang
    k = max(3, int(0.005 * n))
    pick = rng.choice(idx, size=k, replace=False)
    df.loc[pick[: k // 3], "age"] = 150
    df.loc[pick[k // 3 : 2 * k // 3], "map_lowest"] = 5
    df.loc[pick[2 * k // 3 :], "creatinine_highest"] = 99.9

    # (e) Baris duplikat (ekstraksi ganda dari SIM RS)
    k = int(frac["duplicate_rows"] * n)
    dup = df.loc[rng.choice(idx, size=k, replace=False)].copy()
    df = pd.concat([df, dup], ignore_index=True)

    return df


def build_data_dictionary() -> pd.DataFrame:
    """Kamus data mengikuti Tabel 3.1 Batasan Operasional Variabel Penelitian."""
    rows = [
        ("episode_id", "Identitas episode rawat inap (unit analisis)", "ID", "-", "SIM RS", "Identitas"),
        ("patient_code", "Kode pasien teranonimkan", "ID", "-", "SIM RS", "Identitas"),
        ("hospital_admission_datetime", "Waktu masuk rumah sakit", "Datetime", "-", "RME", "Waktu"),
        ("icu_admission_datetime", "Waktu masuk ICU (t0 jendela prediktor 24 jam)", "Datetime", "-", "RME", "Waktu"),
        ("icu_discharge_datetime", "Waktu keluar ICU", "Datetime", "-", "RME", "Waktu"),
        ("hospital_discharge_datetime", "Waktu keluar rumah sakit", "Datetime", "-", "RME", "Waktu"),
        ("age", "Usia saat masuk ICU", "Rasio", "tahun", "RME", "Demografi/administratif"),
        ("sex", "Jenis kelamin biologis", "Nominal", "-", "RME", "Demografi/administratif"),
        ("jkn_class", "Kelas rawat sesuai kepesertaan JKN", "Ordinal", "1/2/3", "SIM RS", "Demografi/administratif"),
        ("icu_admission_type", "Jalur masuk ICU (elektif/emergensi)", "Nominal", "-", "RME", "Demografi/administratif"),
        ("referral_flag", "Status pasien rujukan", "Nominal", "0/1", "RME", "Demografi/administratif"),
        ("diagnosis_category", "Kelompok diagnosis utama menurut sistem organ", "Nominal", "-", "RME", "Kasus"),
        ("mechanical_ventilation", "Ventilasi mekanik invasif dalam 24 jam pertama", "Nominal", "0/1", "RME/penagihan", "Intervensi"),
        ("vasopressor_inotrope", "Vasopresor dan/atau inotropik IV dalam 24 jam pertama", "Nominal", "0/1", "RME/penagihan", "Intervensi"),
        ("transfusion_prc", "Transfusi PRC dalam 24 jam pertama", "Nominal", "0/1", "Bank darah", "Intervensi"),
        ("surgery_24h", "Pembedahan dalam 24 jam pertama (tidak/elektif/emergensi)", "Nominal", "-", "RME kamar operasi", "Intervensi"),
        ("map_lowest", "MAP terendah dalam 24 jam pertama", "Rasio", "mmHg", "Lembar observasi", "Disfungsi organ"),
        ("spo2_lowest", "SpO2 terendah dalam 24 jam pertama", "Rasio", "%", "Lembar observasi", "Disfungsi organ"),
        ("fio2_lowest", "FiO2 pada saat SpO2 terendah", "Rasio", "fraksi", "Lembar observasi", "Disfungsi organ"),
        ("gcs_lowest", "GCS terendah bebas sedasi dalam 24 jam pertama", "Ordinal", "3-15", "Lembar observasi", "Disfungsi organ"),
        ("gcs_note", "Keterangan ketersediaan GCS (tersedia/tersedasi)", "Nominal", "-", "Lembar observasi", "Disfungsi organ"),
        ("oxygen_device", "Jenis alat oksigen (variabel pendamping estimasi FiO2)", "Nominal", "-", "Lembar observasi", "Disfungsi organ"),
        ("creatinine_highest", "Kreatinin serum tertinggi dalam 24 jam pertama", "Rasio", "mg/dL", "Laboratorium", "Disfungsi organ"),
        ("icu_los_days", "Lama rawat ICU", "Rasio", "hari", "RME", "Deskriptif (bukan prediktor)"),
        ("post_icu_los_days", "Lama rawat pasca-ICU", "Rasio", "hari", "RME", "Deskriptif (bukan prediktor)"),
        ("discharge_status", "Status keluar (hidup/meninggal)", "Nominal", "-", "RME", "Deskriptif (bukan prediktor)"),
        ("inacbg_severity_level", "Severity level kode INA-CBGs", "Ordinal", "1/2/3", "Data klaim", "Deskriptif (bukan prediktor)"),
        ("total_hospital_billing", "Total tagihan rumah sakit satu episode", "Rasio", "Rp", "Sistem keuangan", "Output"),
        ("inacbg_claim", "Nilai klaim INA-CBGs final satu episode", "Rasio", "Rp", "Data klaim BPJS", "Output"),
    ]
    return pd.DataFrame(
        rows,
        columns=["variabel", "definisi_operasional", "skala_ukur", "satuan", "cara_ukur", "domain"],
    )
