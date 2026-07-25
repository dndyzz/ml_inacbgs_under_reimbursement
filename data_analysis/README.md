# Pipeline Analisis — Prediksi Under-Reimbursement INA-CBGs di ICU

Pipeline machine learning untuk penelitian *Prediksi Under-Reimbursement Klaim INA-CBGs pada
Pasien Unit Perawatan Intensif Dewasa di RSUPN Dr. Cipto Mangunkusumo Berdasarkan Data 24 Jam
Pertama Perawatan ICU*.

> ⚠️ **Data di repositori ini sepenuhnya sintetis.** Data pasien RSCM belum tersedia. Seluruh
> angka performa yang muncul adalah hasil pada data dummy dan **bukan temuan penelitian**.

## Instalasi

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m ipykernel install --user --name icu-inacbg --display-name "Python (ICU INA-CBGs)"
```

## Menjalankan

**Lewat notebook (interaktif, semua tabel & gambar tampil inline):**

```bash
.venv\Scripts\jupyter lab run_all_pipelines.ipynb
```

Lalu **Run → Run All Cells**.

**Lewat terminal (setiap tahap berdiri sendiri):**

```bash
python pipelines/00_data_generation/main.py
python pipelines/01_eda/main.py
python pipelines/02_preprocessing/main.py
python pipelines/03_feature_building/main.py
python pipelines/04_train_model/main.py
```

Keduanya menjalankan kode yang sama dan menulis file hasil yang sama.

## Tahapan

| Tahap | Isi | Keluaran utama |
|---|---|---|
| **00 data_generation** | Membangkitkan 1.200 episode dummy sesuai Tabel 3.1 proposal, termasuk data hilang dan baris yang harus tersaring | `data/raw/icu_inacbg_raw.csv` |
| **01 eda** | Melihat data apa adanya: sebaran, nilai hilang, biaya vs klaim, prevalensi per subkelompok, tren bulanan | 8 gambar, Tabel 4.1 |
| **02 preprocessing** | Variabel turunan, pemeriksaan rentang, kriteria inklusi/eksklusi, penetapan outcome, label periode | `data/interim/analytic_cohort.csv`, Gambar 3.1 |
| **03 feature_building** | Skor mSOFA, jumlah organ support, log pra-ICU, penanda GCS tersedasi, VIF, spesifikasi kolom | `data/processed/X_features.csv` |
| **04 train_model** | XGBoost / Random Forest / Elastic Net LR, nested CV 5×3, validasi temporal, kalibrasi, SHAP, permutation importance | Tabel 4.2, Tabel 4.3, 13 gambar, model `.joblib` |

## Hasil

- `outputs/figures/<tahap>/` — PNG 300 dpi, siap dilampirkan ke laporan
- `outputs/tables/<tahap>/` — CSV, termasuk Tabel 4.1, 4.2, dan 4.3
- `outputs/models/` — pipeline terlatih (preprocessing + model dalam satu objek)
- `outputs/reports/ringkasan_model.json` — ringkasan performa model terbaik

## Pengaturan

Semua angka yang bisa diubah ada di `config/config.yaml`: jumlah sampel dummy, periode data,
kriteria eksklusi, rentang nilai wajar, grid hyperparameter, jumlah fold, tanggal pemisah
validasi temporal, dan pengaturan SHAP.

## Dokumentasi lanjutan

- `CLAUDE.md` — arsitektur, kontrak antar-pipeline, aturan anti-kebocoran data, cara menambah tahap
- `../CLAUDE.md` — konteks penelitian, definisi outcome, daftar prediktor
