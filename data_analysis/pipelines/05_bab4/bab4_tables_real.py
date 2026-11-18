"""Tabel Bab 4 dari kohort data nyata (mode REAL).

Tabel 4.1 karakteristik subjek menurut status under-reimbursement (§3.10.1)
Tabel 4.2 analisis bivariat (OR kasar) dan multivariat (OR suaian; MICE m=20 pada
          seluruh data + aturan Rubin; seluruh 14 variabel bebas serentak) (§3.10.3)
Tabel 4.3 performa model, nested CV berulang (rerata 50 fold luar + IK 95% bootstrap
          per pasien), ΔAUC berpasangan terhadap Elastic Net + koreksi Holm (§3.10.5)
Tabel 4.4 kepentingan SHAP model final (§3.10.6)
Tabel 4.5 evaluasi subkelompok model terbaik (§3.10.2)
Tabel 4.6 analisis sensitivitas: temporal, class weighting + rekalibrasi, kasus lengkap
Tabel 4.7 luaran sekunder log(klaim/tagihan)

Keluaran: outputs_real/tables/05_bab4/*.csv + ringkasan.json (gitignored).
Jalankan setelah tahap 02-04:  python pipelines/05_bab4/bab4_tables_real.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from common import viz
from common.config import load_config

M_IMPUTATIONS = 20
REF_DX = "Sistem hemato-onkologi"   # kategori terbesar sebagai rujukan

# (label, kolom, skala OR) - skala: OR per kenaikan sebesar nilai ini
NUMERIC = [
    ("Usia (tahun)", "age", 10, "per 10 tahun"),
    ("Durasi rawat pra-ICU (jam)", "pre_icu_los_hours", 24, "per 24 jam"),
    ("MAP terendah (mmHg)", "map_lowest", 10, "per 10 mmHg"),
    ("Rasio SpO₂/FiO₂ terendah", "sf_ratio_lowest", 50, "per 50"),
    ("GCS terendah", "gcs_lowest", 1, "per 1 poin"),
    ("Kreatinin serum tertinggi (mg/dL)", "creatinine_highest", 1, "per 1 mg/dL"),
]
BINARY = [
    ("Ventilasi mekanik invasif", "mechanical_ventilation"),
    ("Vasopresor/inotropik", "vasopressor_inotrope"),
    ("Transfusi PRC", "transfusion_prc"),
]
CATEG = [  # (label, kolom, urutan level, rujukan)
    ("Jenis kelamin", "sex", ["Perempuan", "Laki-laki"], "Perempuan"),
    ("Tipe admisi ICU", "icu_admission_type", ["Elektif", "Emergensi"], "Elektif"),
    ("Pembedahan dalam 24 jam", "surgery_24h", ["tidak", "elektif", "emergensi"], "tidak"),
]
LEVEL_LABEL = {"tidak": "Tidak", "elektif": "Elektif", "emergensi": "Emergensi"}


# ---------------------------------------------------------------------- format
def f(x, d=1):
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fp(p):
    return "<0,001" if p < 0.001 else f(p, 3)


def n_pct(k, n):
    return f"{k} ({f(100 * k / n)})"


# ------------------------------------------------------------------ tabel 4.1
def hospital_days(d: pd.DataFrame) -> pd.Series:
    """Lama rawat inap episode (hari, minimal 1) untuk tagihan per hari (tujuan khusus 1)."""
    los = (pd.to_datetime(d.hospital_discharge_datetime)
           - pd.to_datetime(d.hospital_admission_datetime)).dt.total_seconds() / 86400
    return los.clip(lower=1)


def wilson(k: int, n: int, z: float = 1.96):
    """IK 95% Wilson untuk proporsi."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, c - h, c + h


def describe_numeric(s: pd.Series, y: pd.Series):
    s1, s0 = s[y == 1].dropna(), s[y == 0].dropna()
    if abs(s.skew()) < 0.5:
        fmt = lambda v: f"{f(v.mean())} ± {f(v.std())}"     # noqa: E731
        p, tag = stats.ttest_ind(s1, s0, equal_var=False).pvalue, "*"
    else:
        fmt = lambda v: f"{f(v.median())} ({f(v.quantile(.25))}–{f(v.quantile(.75))})"  # noqa: E731
        p, tag = stats.mannwhitneyu(s1, s0).pvalue, "**"
    return fmt(s.dropna()), fmt(s1), fmt(s0), fp(p), tag


def categorical_p(ct: pd.DataFrame) -> float:
    if ct.shape == (2, 2) and (stats.contingency.expected_freq(ct.values) < 5).any():
        return stats.fisher_exact(ct.values).pvalue
    return stats.chi2_contingency(ct.values, correction=False).pvalue


def table_41(d: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(jkn_label=d.jkn_class.map({1: "Kelas 1", 2: "Kelas 2", 3: "Kelas 3"}),
                 severity=d.inacbg_severity_level.where(
                     d.inacbg_severity_level.isin(["I", "II", "III"])))
    y = d.under_reimbursement
    n, n1, n0 = len(d), int(y.sum()), int((y == 0).sum())
    rows = []

    def head(label):
        rows.append([label, "", "", "", ""])

    def cat_block(label, col, levels, show=None, p_override=None):
        sub = d[d[col].notna()]
        ct = pd.crosstab(sub[col], sub.under_reimbursement).reindex(levels).fillna(0)
        p = fp(categorical_p(ct)) if p_override is None else p_override
        miss = int(d[col].isna().sum())
        rows.append([label + (f" (hilang {miss})" if miss else ""), "", "", "", p])
        for lv in (show or levels):
            k = int((sub[col] == lv).sum())
            k1, k0 = int(ct.loc[lv, 1]), int(ct.loc[lv, 0])
            rows.append([f"   {LEVEL_LABEL.get(lv, lv)}", n_pct(k, len(sub)),
                         n_pct(k1, int((sub.under_reimbursement == 1).sum())),
                         n_pct(k0, int((sub.under_reimbursement == 0).sum())), ""])

    head("Demografi dan administratif")
    tot, a, b, p, t = describe_numeric(d.age, y)
    rows.append([f"Usia (tahun){t}", tot, a, b, p])
    cat_block("Jenis kelamin, n (%)", "sex", ["Laki-laki", "Perempuan"])
    cat_block("Kelas rawat JKN, n (%)", "jkn_label", ["Kelas 1", "Kelas 2", "Kelas 3"])
    cat_block("Tipe admisi ICU, n (%)", "icu_admission_type", ["Elektif", "Emergensi"])
    tot, a, b, p, t = describe_numeric(d.pre_icu_los_hours, y)
    rows.append([f"Durasi rawat pra-ICU (jam){t}", tot, a, b, p])

    head("Intervensi dalam 24 jam pertama, n (%)")
    for label, col in BINARY:
        ct = pd.crosstab(d[col], y)
        rows.append([label, n_pct(int(d[col].sum()), n), n_pct(int(d.loc[y == 1, col].sum()), n1),
                     n_pct(int(d.loc[y == 0, col].sum()), n0), fp(categorical_p(ct))])
    cat_block("Pembedahan", "surgery_24h", ["tidak", "elektif", "emergensi"])

    head("Disfungsi organ dalam 24 jam pertama")
    for label, col, *_ in NUMERIC[2:]:
        tot, a, b, p, t = describe_numeric(d[col], y)
        miss = int(d[col].isna().sum())
        rows.append([f"{label}{t}" + (f" (hilang {miss})" if miss else ""), tot, a, b, p])
    k = int(d.gcs_unassessable.sum())
    rows.append(["GCS tidak dapat dinilai (sedasi), n (%)", n_pct(k, n),
                 n_pct(int(d.loc[y == 1, "gcs_unassessable"].sum()), n1),
                 n_pct(int(d.loc[y == 0, "gcs_unassessable"].sum()), n0),
                 fp(categorical_p(pd.crosstab(d.gcs_unassessable, y)))])

    head("Kasus")
    levels = d.diagnosis_category.value_counts().index.tolist()
    levels = [lv for lv in levels if lv != "Lain-lain"] + ["Lain-lain"]
    cat_block("Kategori diagnosis utama, n (%)", "diagnosis_category", levels)

    head("Luaran dan pembiayaan (deskriptif)")
    money = [("Total tagihan RS (Rp juta)**", d.total_hospital_billing / 1e6),
             ("Klaim INA-CBGs (Rp juta)**", d.inacbg_claim / 1e6),
             ("Tagihan per hari rawat (Rp juta)**", d.total_hospital_billing / 1e6 / hospital_days(d)),
             ("Rasio klaim/tagihan**", d.claim_to_billing_ratio),
             ("Log rasio klaim/tagihan**", d.log_claim_to_billing_ratio),
             ("Selisih klaim − tagihan (Rp juta)**", d.selisih_rupiah / 1e6)]
    for label, s in money:
        dd = 2 if "asio" in label else 1
        q = lambda v: f"{f(v.median(), dd)} ({f(v.quantile(.25), dd)}–{f(v.quantile(.75), dd)})"  # noqa: E731
        rows.append([label, q(s), q(s[y == 1]), q(s[y == 0]), "–"])
    cat_block("Severity level INA-CBGs, n (%)", "severity", ["I", "II", "III"])
    tot, a, b, p, t = describe_numeric(d.icu_los_hours / 24, y)
    rows.append([f"Lama rawat ICU (hari){t}", tot, a, b, p])
    sub = d[d.discharge_status.notna()]
    ct = pd.crosstab(sub.discharge_status, sub.under_reimbursement)
    miss = int(d.discharge_status.isna().sum())
    rows.append(["Meninggal dalam episode, n (%)" + (f" (hilang {miss})" if miss else ""),
                 n_pct(int((sub.discharge_status == 'Meninggal').sum()), len(sub)),
                 n_pct(int(ct.loc['Meninggal', 1]), int(ct[1].sum())),
                 n_pct(int(ct.loc['Meninggal', 0]), int(ct[0].sum())), fp(categorical_p(ct))])

    return pd.DataFrame(rows, columns=["Variabel", f"Total (n = {n})",
                                       f"Under-reimbursement (n = {n1})",
                                       f"Tidak under-reimbursement (n = {n0})", "p"])


# ------------------------------------------------------------------ tabel 4.2
def logit(X: np.ndarray, y: np.ndarray, iters: int = 100):
    """Regresi logistik tanpa penalti (Newton-Raphson); kembalikan beta & kovarians."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta, np.linalg.inv(H)


def design(d: pd.DataFrame) -> pd.DataFrame:
    """Matriks rancangan 14 variabel bebas; nama kolom = label baris Tabel 4.2."""
    X = pd.DataFrame(index=d.index)
    for label, col, scale, unit in NUMERIC:
        X[f"{label.split(' (')[0]} ({unit})"] = d[col] / scale
    X["Kelas rawat JKN (per kenaikan kelas)"] = d.jkn_class
    for label, col in BINARY:
        X[label] = d[col]
    for label, col, levels, ref in CATEG:
        for lv in levels:
            if lv != ref:
                X[f"{label}: {LEVEL_LABEL.get(lv, lv)} (vs {LEVEL_LABEL.get(ref, ref)})"] = (
                    d[col] == lv).astype(float)
    for lv in sorted(d.diagnosis_category.unique()):
        if lv != REF_DX:
            X[f"Diagnosis: {lv} (vs hemato-onkologi)"] = (d.diagnosis_category == lv).astype(float)
    return X


def groups_of(X: pd.DataFrame) -> list[list[str]]:
    """Kolom per variabel (dummies satu variabel dianalisis bersama pada OR kasar)."""
    out, seen = [], set()
    for c in X.columns:
        key = c.split(":")[0] if ":" in c else c
        if key not in seen:
            seen.add(key)
            out.append([k for k in X.columns if (k.split(":")[0] if ":" in k else k) == key])
    return out


def ci_text(b, se, q):
    return f"{f(np.exp(b), 2)} ({f(np.exp(b - q * se), 2)}–{f(np.exp(b + q * se), 2)})"


def hosmer_lemeshow(y, p, g=10):
    dec = pd.qcut(p, g, labels=False, duplicates="drop")
    t = pd.DataFrame({"y": y, "p": p, "g": dec}).groupby("g").agg(o=("y", "sum"), e=("p", "sum"),
                                                                 n=("y", "size"))
    chi = (((t.o - t.e) ** 2) / (t.e * (1 - t.e / t.n))).sum()
    return float(chi), float(stats.chi2.sf(chi, len(t) - 2))


def table_42(d: pd.DataFrame, seed: int):
    y = d.under_reimbursement.to_numpy(float)
    X = design(d)

    # --- bivariat: tiap variabel sendiri, data tersedia
    crude = {}
    for cols in groups_of(X):
        ok = X[cols].notna().all(axis=1).to_numpy()
        b, cov = logit(X.loc[ok, cols].to_numpy(float), y[ok])
        for i, c in enumerate(cols, start=1):
            se = np.sqrt(cov[i, i])
            crude[c] = (ci_text(b[i], se, 1.96), fp(2 * stats.norm.sf(abs(b[i] / se))))

    # --- multivariat: MICE m kali pada seluruh data (outcome ikut model imputasi)
    betas, variances, probs, vifs = [], [], [], None
    mat = np.column_stack([X.to_numpy(float), y])
    for m in range(M_IMPUTATIONS):
        imp = IterativeImputer(sample_posterior=True, max_iter=10, random_state=seed + m)
        Xi = pd.DataFrame(imp.fit_transform(mat)[:, :-1], columns=X.columns)
        jk = "Kelas rawat JKN (per kenaikan kelas)"
        Xi[jk] = Xi[jk].round().clip(1, 3)
        Xi["GCS terendah (per 1 poin)"] = Xi["GCS terendah (per 1 poin)"].round().clip(3, 15)
        b, cov = logit(Xi.to_numpy(), y)
        betas.append(b[1:])
        variances.append(np.diag(cov)[1:])
        probs.append(1 / (1 + np.exp(-(b[0] + Xi.to_numpy() @ b[1:]))))
        if vifs is None:
            Z = Xi.to_numpy()
            vifs = pd.Series(np.diag(np.linalg.inv(np.corrcoef(Z, rowvar=False))), index=X.columns)
    Q, U = np.mean(betas, axis=0), np.mean(variances, axis=0)
    B = np.var(betas, axis=0, ddof=1)
    T = U + (1 + 1 / M_IMPUTATIONS) * B
    r = (1 + 1 / M_IMPUTATIONS) * B / U
    df = (M_IMPUTATIONS - 1) * (1 + 1 / r) ** 2
    q = stats.t.ppf(0.975, df)
    p_adj = 2 * stats.t.sf(np.abs(Q / np.sqrt(T)), df)

    # --- sensitivitas: analisis kasus lengkap (tanpa imputasi)
    # Kategori biner dengan sel nol (separasi sempurna) tak terestimasi: episodenya
    # dikeluarkan dari model kasus lengkap dan barisnya ditulis "–".
    ok = X.notna().all(axis=1).to_numpy()
    sep = [c for c in X.columns if set(X.loc[ok, c].unique()) <= {0.0, 1.0}
           and (pd.crosstab(X.loc[ok, c], y[ok]).values == 0).any()]
    ok_cc = ok & ~X[sep].eq(1).any(axis=1).to_numpy()
    keep = [c for c in X.columns if c not in sep]
    bc, covc = logit(X.loc[ok_cc, keep].to_numpy(float), y[ok_cc])
    cc = {c: (bc[i + 1], np.sqrt(covc[i + 1, i + 1])) for i, c in enumerate(keep)}
    # SE(log OR) >5: kuasi-separasi/kolinearitas (tipe admisi ~ pembedahan elektif) -> tak stabil
    unstable = [c for c, (_, se_) in cc.items() if not np.isfinite(se_) or se_ > 5]
    cc = {c: v for c, v in cc.items() if c not in unstable}

    rows = []
    for i, c in enumerate(X.columns):
        b_, s_ = cc.get(c, (np.nan, np.nan))
        rows.append([c, *crude[c], ci_text(Q[i], np.sqrt(T[i]), q[i]), fp(p_adj[i]),
                     ci_text(b_, s_, 1.96) if c in cc else "–",
                     fp(2 * stats.norm.sf(abs(b_ / s_))) if c in cc else "–", f(vifs[c], 2)])
    tab = pd.DataFrame(rows, columns=["Variabel", "OR kasar (IK 95%)", "p kasar",
                                      "OR suaian (IK 95%)", "p suaian",
                                      "OR suaian kasus lengkap (IK 95%)", "p kasus lengkap", "VIF"])
    pbar = np.mean(probs, axis=0)
    hl, hl_p = hosmer_lemeshow(y, pbar)
    fit = {
        "n": int(len(y)), "m_imputasi": M_IMPUTATIONS, "auc_apparent": roc_auc_score(y, pbar),
        "hl_chi2": hl, "hl_p": hl_p, "vif_max": float(vifs.max()),
        "vif_max_var": str(vifs.idxmax()), "n_vif_gt5": int((vifs > 5).sum()),
        "fmi_max": float((r / (1 + r)).max()),
        "n_kasus_lengkap": int(ok.sum()), "n_kasus_lengkap_model": int(ok_cc.sum()),
        "event_kasus_lengkap": int(y[ok_cc].sum()), "separasi_kasus_lengkap": sep,
        "tidak_stabil_kasus_lengkap": unstable,
    }
    return tab, fit


# ------------------------------------------------------------------ bootstrap
MODELS = {"xgboost": "XGBoost", "random_forest": "Random Forest", "elastic_net_lr": "Elastic Net LR"}
REF_MODEL = "elastic_net_lr"          # pembanding formal (tujuan khusus 3)


def auc_fast(y, p):
    """AUC-ROC berbasis peringkat (setara Mann-Whitney); NaN bila satu kelas saja."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def cal_fast(y, p, iters=25):
    """Calibration intercept (slope dikunci 1) & slope via Newton-Raphson."""
    pc = np.clip(p, 1e-9, 1 - 1e-9)           # sama dengan tahap 04 (EPS)
    lp = np.log(pc / (1 - pc))
    if y.min() == y.max():
        return np.nan, np.nan
    a = 0.0
    for _ in range(iters):                     # offset: y ~ a + lp
        q = 1 / (1 + np.exp(-np.clip(a + lp, -30, 30)))
        a += (y - q).sum() / max((q * (1 - q)).sum(), 1e-12)
    X = np.column_stack([np.ones_like(lp), lp])
    b = np.array([0.0, 1.0])
    with np.errstate(all="ignore"):
        for _ in range(iters):                 # y ~ b0 + b1 lp
            q = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
            H = (X * (q * (1 - q))[:, None]).T @ X
            try:
                step = np.linalg.solve(H, X.T @ (y - q))
            except np.linalg.LinAlgError:
                break
            b += step
            if np.abs(step).max() < 1e-8:
                return a, b[1]
    # Newton tidak konvergen (prediksi ekstrem) -> lbfgs, sama seperti tahap 04
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(lp.reshape(-1, 1), y)
    return a, float(lr.coef_[0][0])


def clf_metrics(y, p, c):
    """Metrik klasifikasi cepat untuk bootstrap (c = kelas prediksi pada ambang data latih)."""
    tp, fp = int(((c == 1) & (y == 1)).sum()), int(((c == 1) & (y == 0)).sum())
    fn, tn = int(((c == 0) & (y == 1)).sum()), int(((c == 0) & (y == 0)).sum())
    a, b = cal_fast(y, p)
    return {
        "auc": auc_fast(y, p),
        "pr_auc": average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,
        "sens": tp / (tp + fn) if tp + fn else np.nan,
        "spes": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
        "intercept": a, "slope": b, "fp": fp, "fn": fn,
    }


def reg_metrics(y, yhat, yb):
    resid = y - yhat
    sst = np.sum((y - y.mean()) ** 2)
    return {
        "r2": 1 - np.sum(resid ** 2) / sst if sst else np.nan,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "slope": np.polyfit(yhat, y, 1)[0] if np.std(yhat) > 0 else np.nan,
        "auc": auc_fast(yb, -yhat),
    }


def boot_indices(rows: np.ndarray, patients: np.ndarray, n_boot: int, seed: int) -> list:
    """Bootstrap per pasien: pasien diambil dengan pengembalian, seluruh episodenya ikut."""
    codes, inv = np.unique(patients[rows], return_inverse=True)
    members = [rows[inv == k] for k in range(len(codes))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[k] for k in rng.integers(0, len(codes), len(codes))])
            for _ in range(n_boot)]


def repeat_matrix(oof: pd.DataFrame, n: int, col: str) -> np.ndarray:
    """(ulangan x episode): prediksi OOF per ulangan; NaN untuk episode di luar skenario."""
    piv = oof.pivot(index="ulangan", columns="row", values=col).reindex(columns=range(n))
    return piv.to_numpy(dtype=float)


def evaluate_repeats(fn, y, P, C, idx):
    """Metrik pada indeks idx untuk setiap ulangan, lalu dirata-rata antar-ulangan."""
    vals = [fn(y[idx], P[r, idx], None if C is None else C[r, idx]) for r in range(len(P))]
    return pd.DataFrame(vals).mean().to_dict()


def evaluate_folds(fn, y, P, C, F, idx):
    """Rerata metrik per fold luar (5 fold x 10 ulangan) pada indeks idx.

    Estimator yang sama dengan titik estimasi proposal ("rerata metrik pada
    lipatan uji lingkar luar"), sehingga IK bootstrap konsisten dengan titiknya.
    """
    vals = []
    for r in range(len(P)):
        fr = F[r, idx]
        for k in np.unique(fr[np.isfinite(fr)]):
            sel = idx[fr == k]
            vals.append(fn(y[sel], P[r, sel], C[r, sel]))
    return pd.DataFrame(vals).mean().to_dict()


def _boot_chunk(fn, y, P, C, F, boots):
    import warnings
    warnings.filterwarnings("ignore")
    return [evaluate_folds(fn, y, P, C, F, b) if F is not None else evaluate_repeats(fn, y, P, C, b)
            for b in boots]


def boot_summary(fn, y, P, C, boots, point: dict, F=None):
    """Titik + IK 95% persentil + sebaran bootstrap (untuk uji selisih berpasangan).

    F (matriks fold) diberikan -> metrik per fold dirata-rata; tanpa F -> metrik
    pada prediksi OOF gabungan per ulangan (dipakai untuk subkelompok kecil).
    """
    chunks = np.array_split(np.arange(len(boots)), 16)
    parts = Parallel(n_jobs=-1)(delayed(_boot_chunk)(fn, y, P, C, F, [boots[i] for i in c])
                                for c in chunks if len(c))
    draws = pd.DataFrame([row for part in parts for row in part])
    lo, hi = draws.quantile(0.025), draws.quantile(0.975)
    return {k: (point.get(k, np.nan), lo.get(k, np.nan), hi.get(k, np.nan)) for k in draws}, draws


def p_boot(delta: np.ndarray) -> float:
    """p dua sisi bootstrap untuk H0: selisih = 0 (batas bawah 1/B)."""
    d = delta[np.isfinite(delta)]
    return max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / len(d))


def holm(pvals: dict) -> dict:
    """Koreksi Holm untuk perbandingan berganda."""
    order = sorted(pvals, key=pvals.get)
    m, out, run = len(order), {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def fp_holm(p, floor):
    return f"≤{f(floor, 3)}" if p <= floor + 1e-12 else fp(p)


def ci3(t, d=3):
    v, lo, hi = t
    return "–" if not np.isfinite(v) else f"{f(v, d)} ({f(lo, d)}–{f(hi, d)})"


class Ctx:
    """Data bersama tahap 05: y, pasien, prediksi OOF, ukuran bootstrap."""

    def __init__(self, cfg, d):
        tab = cfg.table_dir("04_train_model")
        yt = pd.read_csv(cfg.path("data_processed") / cfg["feature_building"]["output_target"])
        self.y = yt[cfg["preprocessing"]["outcome"]["binary_col"]].to_numpy(int)
        self.ylog = yt[cfg["preprocessing"]["outcome"]["continuous_col"]].to_numpy(float)
        self.pat = yt.patient_code.astype(str).to_numpy()
        self.ep = yt.episode_id.astype(str).to_numpy()
        self.n = len(self.y)
        assert (yt.episode_id.astype(str).to_numpy() == d.episode_id.astype(str).to_numpy()).all()
        self.oof = pd.read_csv(tab / "05_prediksi_oof.csv")
        self.folds = pd.read_csv(tab / "01_metrik_per_fold.csv")
        self.temporal = pd.read_csv(tab / "03_prediksi_data_uji.csv")
        self.temporal_metrics = pd.read_csv(tab / "06_metrik_temporal.csv").set_index("model")
        self.B = int(cfg["training"]["n_boot"])
        self.seed = cfg.seed
        self.all_boots = boot_indices(np.arange(self.n), self.pat, self.B, self.seed)

    def mats(self, scen, model, reg=False):
        o = self.oof[(self.oof.skenario == scen) & (self.oof.model == model)]
        P = repeat_matrix(o, self.n, "pred")
        C = None if reg else repeat_matrix(o, self.n, "kelas_prediksi")
        F = repeat_matrix(o, self.n, "fold")
        rows = np.flatnonzero(np.isfinite(P[0]))
        return P, C, rows, F

    def fold_mean(self, scen, model):
        return self.folds[(self.folds.skenario == scen) & (self.folds.model == model)].mean(numeric_only=True)


FOLD_KEY = {"auc": "auc_roc", "pr_auc": "pr_auc", "sens": "sensitivitas", "spes": "spesifisitas",
            "ppv": "ppv", "f1": "f1", "brier": "brier", "intercept": "calibration_intercept",
            "slope": "calibration_slope", "fp": "fp", "fn": "fn", "r2": "r2", "rmse": "rmse",
            "mae": "mae"}


def scenario_stats(ctx: Ctx, scen: str, reg=False):
    """Per model: titik (rerata 50 fold luar, sesuai proposal) + IK bootstrap per pasien."""
    out, draws = {}, {}
    for k in MODELS:
        P, C, rows, F = ctx.mats(scen, k, reg)
        boots = ctx.all_boots if len(rows) == ctx.n else boot_indices(rows, ctx.pat, ctx.B, ctx.seed)
        fm = ctx.fold_mean(scen, k)
        point = {m: fm[c] for m, c in FOLD_KEY.items() if c in fm}
        if reg:   # kolom "kelas" diisi luaran biner agar AUC turunan bisa dihitung
            point["auc"], point["slope"] = fm["auc_turunan"], fm["calibration_slope"]
            yb = np.tile(ctx.y, (len(P), 1))
            out[k], draws[k] = boot_summary(reg_metrics, ctx.ylog, P, yb, boots, point, F)
            continue
        out[k], draws[k] = boot_summary(clf_metrics, ctx.y, P, C, boots, point, F)
        if scen == "utama":
            # FP/FN: jumlah episode pada seluruh kohort, rerata antar-ulangan
            tot = evaluate_repeats(clf_metrics, ctx.y, P, C, rows)
            out[k]["fp"] = (tot["fp"], np.nan, np.nan)
            out[k]["fn"] = (tot["fn"], np.nan, np.nan)
    return out, draws


def delta_tests(draws: dict, points: dict):
    """ΔAUC berpasangan (bootstrap yang sama) + koreksi Holm atas 3 perbandingan."""
    pairs = [("xgboost", "elastic_net_lr"), ("random_forest", "elastic_net_lr"),
             ("xgboost", "random_forest")]
    res, raw = {}, {}
    for a, b in pairs:
        d = draws[a]["auc"].to_numpy() - draws[b]["auc"].to_numpy()
        res[(a, b)] = (points[a]["auc"][0] - points[b]["auc"][0], *np.nanpercentile(d, [2.5, 97.5]))
        raw[(a, b)] = p_boot(d)
    adj = holm(raw)
    return {k: {"delta": res[k], "p": raw[k], "p_holm": adj[k]} for k in pairs}


def table_43(ctx: Ctx):
    stats_, draws = scenario_stats(ctx, "utama")
    tests = delta_tests(draws, stats_)
    floor = 1 / ctx.B
    rows = []
    for k, label in MODELS.items():
        s = stats_[k]
        if k == REF_MODEL:
            dtxt = ptxt = "rujukan"
        else:
            t = tests[(k, REF_MODEL)]
            dtxt = f"{f(t['delta'][0], 3)} ({f(t['delta'][1], 3)} s.d. {f(t['delta'][2], 3)})"
            ptxt = fp_holm(t["p_holm"], floor)
        rows.append({"model": label, "auc": ci3(s["auc"]), "delta": dtxt, "p": ptxt,
                     "pr_auc": ci3(s["pr_auc"]), "sens": ci3(s["sens"]), "spes": ci3(s["spes"]),
                     "f1": ci3(s["f1"]), "fp": f(s["fp"][0], 1), "fn": f(s["fn"][0], 1),
                     "brier": ci3(s["brier"]), "intercept": ci3(s["intercept"], 2),
                     "slope": ci3(s["slope"], 2), "auc_sd": f(ctx.fold_mean("utama", k)["auc_roc"], 3)})
    xr = tests[("xgboost", "random_forest")]
    info = {
        "xgb_vs_rf": {"delta": xr["delta"], "p": xr["p"], "p_holm": xr["p_holm"]},
        "tests": {f"{a}_vs_{b}": v for (a, b), v in tests.items()},
        "auc_point": {k: stats_[k]["auc"] for k in MODELS},
        "auc_sd": {k: float(ctx.folds[(ctx.folds.skenario == "utama") & (ctx.folds.model == k)]
                            .auc_roc.std()) for k in MODELS},
        "ambang_median": {k: float(ctx.oof[(ctx.oof.skenario == "utama") & (ctx.oof.model == k)]
                                   .ambang.median()) for k in MODELS},
        "best": max(MODELS, key=lambda k: stats_[k]["auc"][0]),
        "stats": {k: {m: list(v) for m, v in s.items()} for k, s in stats_.items()},
    }
    return pd.DataFrame(rows), info


# ------------------------------------------------------------- tabel 4.5 (subkelompok)
def subgroup_masks(d: pd.DataFrame):
    return [
        ("Jenis kelamin", "Laki-laki", d.sex == "Laki-laki"),
        ("Jenis kelamin", "Perempuan", d.sex == "Perempuan"),
        ("Usia", "<60 tahun", d.age < 60),
        ("Usia", "≥60 tahun", d.age >= 60),
        ("Kelas rawat JKN", "Kelas 1", d.jkn_class == 1),
        ("Kelas rawat JKN", "Kelas 2", d.jkn_class == 2),
        ("Kelas rawat JKN", "Kelas 3", d.jkn_class == 3),
        ("Tipe admisi ICU", "Elektif", d.icu_admission_type == "Elektif"),
        ("Tipe admisi ICU", "Emergensi", d.icu_admission_type == "Emergensi"),
    ]


def table_45(ctx: Ctx, d: pd.DataFrame, model: str):
    P, C, _, _ = ctx.mats("utama", model)
    rows, raw = [], []
    for i, (grp, level, mask) in enumerate(subgroup_masks(d)):
        idx = np.flatnonzero(mask.to_numpy())
        n1 = int(ctx.y[idx].sum())
        point = evaluate_repeats(clf_metrics, ctx.y, P, C, idx)
        boots = boot_indices(idx, ctx.pat, ctx.B, ctx.seed + 7 * i)
        s, _ = boot_summary(clf_metrics, ctx.y, P, C, boots, point)
        # <10 kejadian atau <10 non-kejadian: estimasi tidak stabil -> tidak dilaporkan
        small = n1 < 10 or len(idx) - n1 < 10
        show = (lambda t, dd=3: "n kecil") if small else ci3
        rows.append({"Subkelompok": grp, "Kategori": level,
                     "n (kejadian)": f"{len(idx)} ({n1})",
                     "AUC-ROC (IK 95%)": show(s["auc"]),
                     "Calibration intercept (IK 95%)": show(s["intercept"], 2),
                     "Calibration slope (IK 95%)": show(s["slope"], 2),
                     "Sensitivitas (IK 95%)": show(s["sens"]),
                     "Spesifisitas (IK 95%)": show(s["spes"]),
                     "kecil": small})
        raw.append({"grp": grp, "level": level, "n": len(idx), "n1": n1,
                    **{k: list(v) for k, v in s.items()}})
    return pd.DataFrame(rows), raw


# ------------------------------------------------------------- tabel 4.6 (sensitivitas)
def temporal_stats(ctx: Ctx):
    t = ctx.temporal
    y = t.y_true.to_numpy(int)
    pts = pd.Series(ctx.pat, index=ctx.ep).reindex(t.episode_id.astype(str)).to_numpy()
    boots = boot_indices(np.arange(len(y)), pts, ctx.B, ctx.seed + 99)
    out, draws = {}, {}
    for k in MODELS:
        p, c = t[f"prob_{k}"].to_numpy(), t[f"kelas_{k}"].to_numpy()
        point = clf_metrics(y, p, c)
        dr = pd.DataFrame([clf_metrics(y[b], p[b], c[b]) for b in boots])
        out[k] = {m: (point[m], dr[m].quantile(0.025), dr[m].quantile(0.975)) for m in dr}
        draws[k] = dr
    return out, delta_tests(draws, out), {"n": len(y), "n1": int(y.sum())}


def table_46(ctx: Ctx):
    """Analisis sensitivitas: temporal, class weighting + rekalibrasi, kasus lengkap."""
    main_, _ = scenario_stats(ctx, "utama")
    temp, temp_tests, temp_n = temporal_stats(ctx)
    cw, _ = scenario_stats(ctx, "class_weight")
    cc, _ = scenario_stats(ctx, "kasus_lengkap")
    n_cc = int(ctx.mats("kasus_lengkap", "xgboost")[2].size)
    n1_cc = int(ctx.y[ctx.mats("kasus_lengkap", "xgboost")[2]].sum())
    blocks = [("Analisis utama: nested CV berulang (n = " + f(ctx.n, 0) + ")", main_),
              (f"Validasi temporal: uji Januari–Mei 2026 (n = {temp_n['n']})", temp),
              ("Class weighting + rekalibrasi pada data latih (n = " + f(ctx.n, 0) + ")", cw),
              (f"Analisis kasus lengkap (n = {n_cc})", cc)]
    rows = []
    for title, st in blocks:
        rows.append([title, "", "", "", "", ""])
        for k, label in MODELS.items():
            s = st[k]
            rows.append([f"   {label}", ci3(s["auc"]), f(s["brier"][0], 3), f(s["intercept"][0], 2),
                         f(s["slope"][0], 2), f"{f(s['sens'][0], 3)} / {f(s['spes'][0], 3)}"])
    tab = pd.DataFrame(rows, columns=["Skenario / model", "AUC-ROC (IK 95%)", "Brier score",
                                      "Calibration intercept", "Calibration slope",
                                      "Sensitivitas / spesifisitas"])
    info = {"temporal_tests": {f"{a}_vs_{b}": v for (a, b), v in temp_tests.items()},
            "temporal_n": temp_n, "n_cc": n_cc, "n1_cc": n1_cc,
            "temporal": {k: {m: list(v) for m, v in s.items()} for k, s in temp.items()},
            "cw": {k: {m: list(v) for m, v in s.items()} for k, s in cw.items()},
            "cc": {k: {m: list(v) for m, v in s.items()} for k, s in cc.items()}}
    return tab, info


# ------------------------------------------------------------- tabel 4.7 (log-rasio)
def table_47(ctx: Ctx):
    st, _ = scenario_stats(ctx, "log_rasio", reg=True)
    rows = []
    for k, label in MODELS.items():
        s = st[k]
        rows.append({"model": label, "r2": ci3(s["r2"]), "rmse": ci3(s["rmse"]), "mae": ci3(s["mae"]),
                     "slope": ci3(s["slope"], 2), "auc": ci3(s["auc"])})
    return pd.DataFrame(rows), {k: {m: list(v) for m, v in s.items()} for k, s in st.items()}


# ------------------------------------------------------------------ tabel 4.4
def table_44(tab_dir: Path, top: int):
    s = pd.read_csv(tab_dir / "tabel_4_3_shap_importance.csv").head(top)
    return pd.DataFrame({
        "Peringkat": s.peringkat, "Variabel": s.variabel.map(viz.feature_label),
        "Rerata |SHAP|": s.mean_abs_shap.map(lambda v: f(v, 4)),
        "Kontribusi relatif (%)": (100 * s.kontribusi_relatif).map(lambda v: f(v, 1)),
        "Arah pengaruh": s.arah_pengaruh.str.replace("nilai tinggi ", "nilai tinggi → "),
    })


def main() -> None:
    cfg = load_config()
    assert cfg["project"]["data_mode"] == "REAL", "skrip ini khusus data nyata"
    seed = cfg["project"]["seed"]
    d = pd.read_csv(cfg.path("data_interim") / cfg["preprocessing"]["output_file"])
    src = cfg.table_dir("04_train_model")
    out = cfg.table_dir("05_bab4")
    ctx = Ctx(cfg, d)
    report = json.loads((cfg.path("reports") / "ringkasan_model.json").read_text(encoding="utf-8"))

    t41 = table_41(d)
    t42, fit = table_42(d, seed)
    t43, perf = table_43(ctx)
    t44 = table_44(src, cfg["interpretability"]["shap_max_display"])
    t45, sub = table_45(ctx, d, report["model_terbaik_key"])
    t46, sens = table_46(ctx)
    t47, sec = table_47(ctx)
    for name, t in [("tabel_4_1", t41), ("tabel_4_2", t42), ("tabel_4_3", t43), ("tabel_4_4", t44),
                    ("tabel_4_5", t45), ("tabel_4_6", t46), ("tabel_4_7", t47)]:
        t.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    flow = pd.read_csv(cfg.table_dir("02_preprocessing") / "01_alur_seleksi.csv")
    params = json.loads((src / "02_hyperparameter_terpilih.json").read_text())
    n_ev = int(d.under_reimbursement.sum())
    summary = {"regresi_multivariat": fit, "performa": perf, "subkelompok": sub,
               "sensitivitas": sens, "luaran_sekunder": sec, "alur": flow.to_dict("records"),
               "hyperparameter": params, "model": report, "n_kohort": len(d),
               "n_latih": int((d.periode.str.startswith("latih")).sum()),
               "n_event": n_ev, "proporsi_ik95": list(wilson(n_ev, len(d))),
               "tagihan_per_hari": {k: float(v) for k, v in
                                    (d.total_hospital_billing / 1e6 / hospital_days(d))
                                    .describe().items()},
               "cfg_training": {k: cfg["training"][k] for k in
                                ("outer_folds", "outer_repeats", "inner_folds", "n_trials",
                                 "n_imputations")},
               "n_boot": ctx.B, "kategori_dx": d.diagnosis_category.value_counts().to_dict(),
               "gcs_v_derived": int(d.get("gcs_v_derived", pd.Series(0)).sum())}
    (out / "ringkasan.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                        encoding="utf-8")
    for t in (t41, t42, t43, t44, t45, t46, t47):
        print(t.to_string(), end="\n\n")


if __name__ == "__main__":
    main()
