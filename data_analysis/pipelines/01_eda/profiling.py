"""Tabel-tabel deskriptif untuk tahap EDA.

Semua fungsi murni (DataFrame masuk -> DataFrame keluar) supaya mudah diuji dan
dipakai ulang saat data RSCM yang sebenarnya tersedia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Satu baris per kolom: tipe, jumlah hilang, jumlah nilai unik, contoh isi."""
    rows = []
    for col in df.columns:
        s = df[col]
        example = s.dropna().iloc[0] if s.notna().any() else "-"
        rows.append({
            "kolom": col,
            "tipe": str(s.dtype),
            "n_terisi": int(s.notna().sum()),
            "n_hilang": int(s.isna().sum()),
            "persen_hilang": round(100 * s.isna().mean(), 2),
            "n_unik": int(s.nunique(dropna=True)),
            "contoh_nilai": example,
        })
    return pd.DataFrame(rows)


def describe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Ringkasan numerik lengkap (rerata+SB dan median+IQR, keduanya dilaporkan)."""
    rows = []
    for col in cols:
        if col not in df:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        # Uji normalitas hanya sebagai penunjuk penyajian di tabel karakteristik
        normal_p = stats.shapiro(s.sample(min(len(s), 500), random_state=0))[1] if len(s) >= 3 else np.nan
        rows.append({
            "variabel": col,
            "n": len(s),
            "rerata": round(s.mean(), 2),
            "simpang_baku": round(s.std(), 2),
            "median": round(s.median(), 2),
            "q1": round(s.quantile(0.25), 2),
            "q3": round(s.quantile(0.75), 2),
            "min": round(s.min(), 2),
            "maks": round(s.max(), 2),
            "p_shapiro": round(normal_p, 4) if pd.notna(normal_p) else np.nan,
            "distribusi": "normal" if pd.notna(normal_p) and normal_p > 0.05 else "tidak normal",
        })
    return pd.DataFrame(rows)


def describe_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Frekuensi dan persentase tiap kategori."""
    rows = []
    for col in cols:
        if col not in df:
            continue
        counts = df[col].value_counts(dropna=False)
        total = len(df)
        for value, n in counts.items():
            rows.append({
                "variabel": col,
                "kategori": "(hilang)" if pd.isna(value) else value,
                "n": int(n),
                "persen": round(100 * n / total, 1),
            })
    return pd.DataFrame(rows)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Interval kepercayaan 95% proporsi (metode Wilson, stabil untuk p ekstrem)."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def prevalence_by_group(df: pd.DataFrame, outcome: str, group: str) -> pd.DataFrame:
    """Prevalensi outcome per kategori sebuah variabel, dengan IK 95%."""
    rows = []
    for value, sub in df.groupby(group, dropna=False):
        k = int(sub[outcome].sum())
        n = int(sub[outcome].notna().sum())
        lo, hi = wilson_ci(k, n)
        rows.append({
            "variabel": group,
            "kategori": "(hilang)" if pd.isna(value) else value,
            "n": n,
            "kejadian": k,
            "prevalensi": round(k / n, 4) if n else np.nan,
            "ik95_bawah": round(lo, 4),
            "ik95_atas": round(hi, 4),
        })
    return pd.DataFrame(rows).sort_values("prevalensi", ascending=False)


def baseline_table(
    df: pd.DataFrame,
    outcome: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Tabel 4.1 - karakteristik subjek keseluruhan dan menurut status outcome.

    Numerik disajikan median (IQR) dan diuji Mann-Whitney; kategorik disajikan
    n (%) dan diuji chi-square. Uji bersifat deskriptif, bukan inferensi utama.
    """
    grp0 = df[df[outcome] == 0]
    grp1 = df[df[outcome] == 1]
    rows = []

    for col in numeric_cols:
        if col not in df:
            continue
        all_s = pd.to_numeric(df[col], errors="coerce")
        s0 = pd.to_numeric(grp0[col], errors="coerce").dropna()
        s1 = pd.to_numeric(grp1[col], errors="coerce").dropna()
        p = stats.mannwhitneyu(s0, s1, alternative="two-sided")[1] if len(s0) > 1 and len(s1) > 1 else np.nan
        rows.append({
            "variabel": col,
            "kategori": "median (IQR)",
            "keseluruhan": _fmt_iqr(all_s),
            "tidak_under_reimbursement": _fmt_iqr(s0),
            "under_reimbursement": _fmt_iqr(s1),
            "p": _fmt_p(p),
        })

    for col in categorical_cols:
        if col not in df:
            continue
        table = pd.crosstab(df[col], df[outcome])
        p = stats.chi2_contingency(table)[1] if table.shape[0] > 1 and table.shape[1] > 1 else np.nan
        for value in df[col].dropna().unique():
            m_all, m0, m1 = df[col] == value, grp0[col] == value, grp1[col] == value
            rows.append({
                "variabel": col,
                "kategori": str(value),
                "keseluruhan": _fmt_n_pct(m_all.sum(), len(df)),
                "tidak_under_reimbursement": _fmt_n_pct(m0.sum(), len(grp0)),
                "under_reimbursement": _fmt_n_pct(m1.sum(), len(grp1)),
                "p": _fmt_p(p) if value == df[col].dropna().unique()[0] else "",
            })

    return pd.DataFrame(rows)


def _fmt_iqr(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "-"
    return f"{s.median():,.1f} ({s.quantile(0.25):,.1f}-{s.quantile(0.75):,.1f})"


def _fmt_n_pct(k: int, n: int) -> str:
    return f"{int(k):,} ({100 * k / n:.1f}%)" if n else "-"


def _fmt_p(p: float) -> str:
    if pd.isna(p):
        return "-"
    return "<0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")
