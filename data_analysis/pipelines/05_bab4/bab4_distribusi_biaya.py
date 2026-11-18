"""Gambar 4.2: sebaran tagihan, klaim INA-CBGs, dan under-reimbursement (tujuan khusus 1).

Proposal etik §1.5.2 butir 1: mendeskripsikan total tagihan RS, rerata tagihan per hari,
nilai klaim INA-CBGs, dan proporsi under-reimbursement. Tabel 4.1 memuat median/RIK;
skrip ini menambah sebaran dan agregat rupiahnya.

Keluaran: figures/05_bab4/01_distribusi_biaya.png + tables/05_bab4/distribusi_biaya.json
Jalankan setelah tahap 02:  python pipelines/05_bab4/bab4_distribusi_biaya.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

from common import viz
from common.config import load_config

RED, BLUE = "#e34948", "#2a78d6"
BANDS = [0, .25, .5, .75, 1, 1.25, np.inf]
BAND_LABELS = ["<25%", "25–<50%", "50–<75%", "75–<100%", "100–<125%", "≥125%"]


def koma(v, d=1):
    return f"{v:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def summarize(d: pd.DataFrame) -> dict:
    b, c, r = d.total_hospital_billing, d.inacbg_claim, d.claim_to_billing_ratio
    u = d.under_reimbursement == 1
    gap_pct = 100 * (1 - r[u])
    bands = pd.cut(r, BANDS, right=False, labels=BAND_LABELS).value_counts().reindex(BAND_LABELS)
    grp = lambda g: (d.groupby(g).agg(n=("episode_id", "size"), under=("under_reimbursement", "mean"),
                                      rasio_median=("claim_to_billing_ratio", "median"),
                                      selisih_miliar=("selisih_rupiah", "sum"))
                     .assign(selisih_miliar=lambda x: x.selisih_miliar / 1e9)
                     .to_dict("index"))
    share_icu = d.icu_billing / b
    return {
        "n": len(d), "n_under": int(u.sum()),
        "total_tagihan_miliar": b.sum() / 1e9, "total_klaim_miliar": c.sum() / 1e9,
        "cakupan_agregat": c.sum() / b.sum(), "selisih_bersih_miliar": (c - b).sum() / 1e9,
        "defisit_under_miliar": (c - b)[u].sum() / 1e9, "surplus_tidak_under_miliar": (c - b)[~u].sum() / 1e9,
        "rerata_selisih_juta": (c - b).mean() / 1e6, "sb_selisih_juta": (c - b).std() / 1e6,
        "persen_tak_tertutup_under_median_q1_q3": np.percentile(gap_pct, [50, 25, 75]).tolist(),
        "pita_cakupan": {k: int(v) for k, v in bands.items()},
        "porsi_tagihan_icu_median_q1_q3": np.percentile(share_icu, [50, 25, 75]).tolist(),
        "per_kategori_dx": grp("diagnosis_category"), "per_kelas": grp("jkn_class"),
        "per_tipe_admisi": grp("icu_admission_type"),
    }


def plot(d: pd.DataFrame, s: dict, path: Path) -> Path:
    viz.set_theme()
    u = d.under_reimbursement == 1
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    (a, b), (c, e) = ax
    pct_fmt = FuncFormatter(lambda v, _: koma(v, 2).rstrip("0").rstrip(","))

    # A. sebaran rasio klaim/tagihan (skala log)
    bins = np.geomspace(0.03, 7, 41)
    a.hist([d.claim_to_billing_ratio[u], d.claim_to_billing_ratio[~u]], bins=bins, stacked=True,
           color=[RED, BLUE], label=[f"Under-reimbursement (n = {u.sum()})",
                                     f"Tidak under-reimbursement (n = {(~u).sum()})"])
    a.axvline(1, color=viz.INK_PRIMARY, lw=1, ls="--")
    a.text(1.06, a.get_ylim()[1] * 0.55, "titik impas\n(rasio = 1)", fontsize=8.5, va="top")
    a.set_xscale("log")
    a.set_xticks([0.05, 0.1, 0.25, 0.5, 1, 2, 4])
    a.xaxis.set_major_formatter(pct_fmt)
    a.xaxis.set_minor_formatter(NullFormatter())
    a.set_xlabel("Rasio klaim INA-CBGs / total tagihan RS (skala log)")
    a.set_ylabel("Jumlah episode")
    a.set_ylim(0, a.get_ylim()[1] * 1.3)
    a.legend(loc="upper right", fontsize=8.5, frameon=True, facecolor=viz.SURFACE, edgecolor="none",
             framealpha=1)

    # B. tagihan vs klaim per episode
    for m, col in [(u, RED), (~u, BLUE)]:
        b.scatter(d.total_hospital_billing[m] / 1e6, d.inacbg_claim[m] / 1e6, s=9, alpha=.55,
                  color=col, edgecolor="none")
    lim = (3, 800)
    b.plot(lim, lim, color=viz.INK_PRIMARY, lw=1, ls="--")
    b.text(15.5, 48, "klaim = tagihan", fontsize=8.5, ha="left")
    b.set(xscale="log", yscale="log", xlim=(14, 800), ylim=lim,
          xlabel="Total tagihan RS per episode (Rp juta, skala log)",
          ylabel="Klaim INA-CBGs per episode (Rp juta, skala log)")
    for axis in (b.xaxis, b.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: koma(v, 0)))
        axis.set_minor_formatter(NullFormatter())
    b.set_xticks([20, 50, 100, 200, 500])
    b.set_yticks([5, 10, 20, 50, 100, 200, 500])
    b.text(0.97, 0.04, "Titik di bawah garis = under-reimbursement", transform=b.transAxes,
           ha="right", fontsize=8.5, color=viz.INK_SECONDARY)

    # C. pita cakupan klaim terhadap tagihan
    counts = np.array(list(s["pita_cakupan"].values()))
    colors = ["#a52625", RED, "#f0a3a2", "#f6c9c8", "#9ec5f4", BLUE]
    bars = c.bar(BAND_LABELS, counts, color=colors)
    for bar, k in zip(bars, counts):
        c.text(bar.get_x() + bar.get_width() / 2, k + 4, f"{k}\n({koma(100 * k / s['n'])}%)",
               ha="center", va="bottom", fontsize=8.5)
    c.set_ylim(0, counts.max() * 1.25)
    c.set_xlabel("Klaim INA-CBGs sebagai persentase total tagihan RS")
    c.set_ylabel("Jumlah episode")
    c.tick_params(axis="x", labelsize=8.5)

    # E. selisih bersih per kategori diagnosis
    g = pd.DataFrame(s["per_kategori_dx"]).T.sort_values("selisih_miliar", ascending=False)
    names = [i.replace("Sistem ", "") for i in g.index]
    names = [x[0].upper() + x[1:] for x in names]
    e.barh(names, g.selisih_miliar, color=[RED if v < 0 else BLUE for v in g.selisih_miliar])
    e.axvline(0, color=viz.BASELINE, lw=1)
    for y, (v, row) in enumerate(g.iterrows()):
        e.text(row.selisih_miliar - 0.4, y, f"{koma(row.selisih_miliar, 1)} ({koma(100 * row.under, 0)}%; "
               f"n = {int(row.n)})", va="center", ha="right", fontsize=8)
    e.set_xlim(g.selisih_miliar.min() * 1.75, 2)
    e.set_xlabel("Selisih bersih klaim − tagihan (Rp miliar)")
    e.xaxis.set_major_formatter(FuncFormatter(lambda v, _: koma(v, 0)))
    e.text(0.0, -0.17, "Label: selisih (proporsi under-reimbursement; n)", transform=e.transAxes,
           fontsize=8, color=viz.INK_MUTED)

    for axis, lab in zip((a, b, c, e), "ABCD"):
        axis.set_title(lab, loc="left", fontweight="bold")
    fig.tight_layout()
    return viz.save_fig(fig, path)


def main() -> None:
    cfg = load_config()
    assert cfg["project"]["data_mode"] == "REAL", "skrip ini khusus data nyata"
    d = pd.read_csv(cfg.path("data_interim") / cfg["preprocessing"]["output_file"])
    # pemeriksaan: label luaran dan selisih konsisten dengan definisi proposal
    assert ((d.claim_to_billing_ratio < 1) == (d.under_reimbursement == 1)).all()
    assert np.allclose(d.selisih_rupiah, d.inacbg_claim - d.total_hospital_billing)
    s = summarize(d)
    assert sum(s["pita_cakupan"].values()) == s["n"]
    assert abs(s["defisit_under_miliar"] + s["surplus_tidak_under_miliar"] - s["selisih_bersih_miliar"]) < 1e-9
    fig = plot(d, s, cfg.figure_dir("05_bab4") / "01_distribusi_biaya.png")
    out = cfg.table_dir("05_bab4") / "distribusi_biaya.json"
    out.write_text(json.dumps(s, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(fig, out, json.dumps({k: v for k, v in s.items() if not k.startswith("per_")}, indent=1, default=float))


if __name__ == "__main__":
    main()
