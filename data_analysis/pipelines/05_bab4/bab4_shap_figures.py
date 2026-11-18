"""Gambar SHAP ulang dari model final tersimpan (tanpa melatih ulang tahap 04).

Dipakai bila hanya tampilan yang berubah (label definisi operasional). Nilai SHAP identik
dengan tahap 04 karena model dan data sama. Menulis ulang 07-10 di figures/04_train_model,
gabungan waterfall figures/05_bab4/02_shap_waterfall.png, dan tables/05_bab4/tabel_4_4.csv.
Jalankan:  python pipelines/05_bab4/bab4_shap_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "pipelines" / "04_train_model"), str(Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import joblib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import interpret
from bab4_tables_real import table_44
from common import viz
from common.config import load_config


def main() -> None:
    cfg = load_config()
    assert cfg["project"]["data_mode"] == "REAL"
    fb, ip = cfg["feature_building"], cfg["interpretability"]
    X = pd.read_csv(cfg.path("data_processed") / fb["output_features"])
    report = json.loads((cfg.path("reports") / "ringkasan_model.json").read_text(encoding="utf-8"))
    name = report["model_terbaik_key"]
    pipelines = joblib.load(cfg.path("models") / f"{name}_final_mi.joblib")

    explanation, Xt = interpret.compute_shap_mi(pipelines, X)
    table = interpret.shap_importance_table(explanation, Xt)
    old = pd.read_csv(cfg.table_dir("04_train_model") / "tabel_4_3_shap_importance.csv")
    # pemeriksaan: SHAP yang dihitung ulang = hasil tahap 04
    assert table.variabel.tolist() == old.variabel.tolist()
    assert np.allclose(table.mean_abs_shap, old.mean_abs_shap, atol=1e-5)

    probs = np.mean([p.predict_proba(X)[:, 1] for p in pipelines], axis=0)
    fig_dir = cfg.figure_dir("04_train_model")
    figs = [interpret.plot_shap_beeswarm(explanation, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_bar(table, fig_dir, int(ip["shap_max_display"])),
            interpret.plot_shap_dependence(explanation, Xt, table, fig_dir, int(ip["n_dependence_plots"]))]
    wf = interpret.plot_shap_waterfall(explanation, probs, fig_dir, 2)

    # gabungan dua waterfall untuk Bab 4 (A = risiko tertinggi, B = risiko terendah)
    viz.set_theme()
    imgs = [mpimg.imread(p) for p in wf]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 12.6))
    for ax, img, lab in zip(axes, imgs, "AB"):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(lab, loc="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    figs.append(viz.save_fig(fig, cfg.figure_dir("05_bab4") / "02_shap_waterfall.png"))

    t44 = table_44(cfg.table_dir("04_train_model"), int(ip["shap_max_display"]))
    t44.to_csv(cfg.table_dir("05_bab4") / "tabel_4_4.csv", index=False, encoding="utf-8-sig")

    hi, lo = int(np.argmax(probs)), int(np.argmin(probs))
    shown = interpret.for_display(explanation)
    cases = {}
    for key, i in (("tertinggi", hi), ("terendah", lo)):
        contrib = pd.Series(shown.values[i], index=shown.feature_names)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
        cases[key] = {"prob": float(probs[i]), "base": float(np.mean(shown.base_values)),
                      "top": {k: [float(v), float(shown.data[i][shown.feature_names.index(k)])]
                              for k, v in top.items()}}
    (cfg.table_dir("05_bab4") / "shap_waterfall_kasus.json").write_text(
        json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    print(*figs, sep="\n")
    print(t44.Variabel.tolist())
    print(json.dumps(cases, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
