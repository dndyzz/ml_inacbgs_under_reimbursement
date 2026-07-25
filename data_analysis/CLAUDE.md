# CLAUDE.md — data_analysis

Panduan teknis pipeline. Konteks penelitiannya ada di `../CLAUDE.md`; baca itu dulu.

## Menjalankan

```bash
# sekali di awal
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name icu-inacbg --display-name "Python (ICU INA-CBGs)"
```

```bash
# seluruh pipeline lewat terminal
python pipelines/00_data_generation/main.py
python pipelines/01_eda/main.py
python pipelines/02_preprocessing/main.py
python pipelines/03_feature_building/main.py
python pipelines/04_train_model/main.py          # --skip-shap untuk versi cepat
```

Atau buka `run_all_pipelines.ipynb` → **Run All Cells**. Keduanya menjalankan kode yang sama
dan menghasilkan file yang sama; notebook hanya menambah tampilan inline.

## Arsitektur

```
data_analysis/
├── config/config.yaml      # SATU-SATUNYA tempat angka & pengaturan
├── common/                 # infrastruktur bersama (bukan logika penelitian)
│   ├── config.py           # load_config() + resolusi path
│   ├── io_utils.py         # save_table / save_json
│   ├── logging_utils.py    # get_logger, banner
│   ├── viz.py              # tema matplotlib, palet, save_fig
│   ├── runner.py           # run_pipeline(), load_stage_module()
│   └── display.py          # show_result() untuk notebook
├── pipelines/
│   ├── 00_data_generation/ generator.py
│   ├── 01_eda/             profiling.py, eda_plots.py
│   ├── 02_preprocessing/   cleaning.py, prep_plots.py
│   ├── 03_feature_building/ features.py, feature_plots.py
│   └── 04_train_model/     models.py, evaluate.py, interpret.py
├── data/{raw,interim,processed}/
├── outputs/{figures,tables,models,reports}/
└── run_all_pipelines.ipynb
```

### Aliran data antartahap

| Tahap | Baca | Tulis |
|---|---|---|
| 00 data_generation | — | `data/raw/icu_inacbg_raw.csv`, `data_dictionary.csv` |
| 01 eda | `data/raw/` | tabel + gambar EDA (tidak mengubah data) |
| 02 preprocessing | `data/raw/` | `data/interim/analytic_cohort.csv` |
| 03 feature_building | `data/interim/` | `data/processed/X_features.csv`, `y_target.csv`, `feature_spec.json` |
| 04 train_model | `data/processed/` | `outputs/models/*.joblib`, Tabel 4.2 & 4.3, 13 gambar |

## Kontrak setiap pipeline

Wajib dipatuhi saat menambah atau mengubah tahap:

```python
STAGE = "05_nama_tahap"

def run(cfg=None, **overrides) -> dict:
    cfg = cfg or load_config(**overrides)
    ...
    return {
        "stage":   STAGE,
        "summary": {"Keterangan": "nilai"},   # dict skalar, ditampilkan sebagai tabel
        "tables":  {"Nama tabel": DataFrame}, # dipratinjau di notebook
        "figures": [Path, ...],               # PNG yang sudah tersimpan
        "paths":   {"nama_artefak": Path},
    }

def main() -> None:      # antarmuka CLI: argparse --config, dst.
    ...

if __name__ == "__main__":
    main()
```

Di awal setiap `main.py` ada bootstrap `sys.path` (naik dua level ke akar proyek + folder tahap
sendiri) supaya file bisa dijalankan langsung dari mana saja.

## Aturan yang gampang dilanggar

1. **Nama modul pendamping harus unik lintas tahap.** Notebook memuat semua `main.py` dalam satu
   proses Python; dua file bernama `plots.py` di tahap berbeda akan saling menimpa di
   `sys.modules`. Karena itu namanya `eda_plots.py`, `prep_plots.py`, `feature_plots.py`.
   Jika menambah modul baru, beri awalan nama tahap.
2. **Tidak boleh ada imputasi/encoding/standardisasi di luar fold.** Semua langkah yang
   *belajar* dari data harus berada di dalam `Pipeline([("prep", ...), ("model", ...)])` supaya
   ikut di-fit ulang per fold. Tahap 03 hanya *merakit* preprocessor, tidak mem-fit-nya
   (kecuali sekali pada periode latih, semata untuk mengambil nama kolom).
3. **Jangan memakai variabel pasca-24-jam sebagai prediktor.** `icu_los_days`,
   `post_icu_los_days`, `discharge_status`, `inacbg_severity_level`, `total_hospital_billing`,
   dan `inacbg_claim` hanya boleh dipakai untuk deskripsi atau pembentukan outcome. Memasukkan
   salah satunya ke `X` = kebocoran outcome dan AUC palsu ~1,0.
4. **Figure selalu disimpan lalu ditutup** (`viz.save_fig`), tidak pernah `plt.show()`. Notebook
   menampilkan file PNG-nya, sehingga hasil terminal dan notebook identik.
5. **Angka apa pun yang bisa berubah masuk `config/config.yaml`**, bukan ditulis keras di kode.

## Menambah tahap baru

1. Buat folder `pipelines/05_nama_tahap/` berisi `main.py` (+ modul pendamping berawalan unik).
2. Ikuti kontrak `run()` di atas.
3. Tambahkan bagian konfigurasinya di `config/config.yaml`.
4. Tambahkan satu sel markdown + satu sel kode di `run_all_pipelines.ipynb`.
   `run_pipeline("05_nama_tahap")` otomatis menemukannya — tidak ada daftar tahap yang perlu diperbarui.

## Beralih ke data RSCM yang sebenarnya

1. Letakkan CSV di `data/raw/` (jangan di-commit; lihat `.gitignore`).
2. Ubah `project.data_mode: REAL` dan `eda.input_file` / `preprocessing.input_file` di config.
3. Samakan nama kolom dengan kamus data di `data/raw/data_dictionary.csv`, atau tambahkan
   pemetaan nama kolom di `pipelines/02_preprocessing/cleaning.py`.
4. **Lewati tahap 00.** Tahap 00 hanya untuk data dummy.
5. Periksa ulang `preprocessing.plausible_ranges` terhadap data nyata.

## Lingkungan yang diuji

Python 3.11.5 (Windows 11) · numpy 2.4 · pandas 3.0 · scikit-learn 1.9 · xgboost 3.2 ·
shap 0.51 · matplotlib 3.11. Kode menjaga kompatibilitas mundur pada titik yang diketahui
berubah (mis. argumen `penalty` pada `LogisticRegression` scikit-learn < 1.8).

Waktu jalan penuh dari nol: ± 2 menit untuk 1.200 episode dummy.
