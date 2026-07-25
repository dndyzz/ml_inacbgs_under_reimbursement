# CLAUDE.md — Proposal dr. Ade Ariadi

Panduan konteks untuk sesi Claude Code pada repositori ini. Baca ini lebih dulu sebelum
mengubah apa pun.

## Tentang penelitian

**Judul.** Prediksi under-reimbursement klaim INA-CBGs pada pasien unit perawatan intensif
dewasa di RSUPN Dr. Cipto Mangunkusumo berdasarkan data 24 jam pertama perawatan ICU:
pendekatan machine learning.

**Peneliti.** dr. Ade Ariadi (PPDS Subspesialis Anestesiologi & Terapi Intensif, FKUI-RSCM).
Pembimbing: dr. Navy Lolong W, Sp.An-TI, Subsp.TI(K) dan dr. Adhrie Sugiarto, Sp.An-TI, Subsp.TI(K).

**Pertanyaan penelitian.** Dapatkah model machine learning berbasis variabel klinis dan
administratif 24 jam pertama ICU memprediksi under-reimbursement dengan AUC-ROC ≥ 0,75 dan
kalibrasi baik; dan mana yang terbaik di antara XGBoost, Random Forest, dan Elastic Net
Logistic Regression?

**Desain.** Kohort retrospektif, unit analisis = **episode rawat inap**, data Mei 2025–Mei 2026.
Prediktor diambil dari 24 jam pertama admisi ICU pertama; outcome ditentukan di akhir episode,
sehingga model bersifat *strictly forward-looking*. Pelaporan mengikuti **TRIPOD+AI**.

### Definisi outcome (jangan diubah tanpa persetujuan peneliti)

| Outcome | Definisi |
|---|---|
| Primer (biner) | `under_reimbursement = 1` bila **total klaim INA-CBGs episode / total tagihan RS episode < 1** |
| Sekunder (kontinu) | `log(klaim / tagihan)` — simetris terhadap titik impas |

Tagihan mencakup seluruh episode: pra-ICU + ICU + pasca-ICU.

### 14 prediktor (4 domain)

| Domain | Variabel |
|---|---|
| Demografi & administratif (5) | usia, jenis kelamin, kelas rawat JKN (1/2/3), tipe admisi ICU (elektif/emergensi), durasi rawat pra-ICU |
| Intervensi / organ support (4) | ventilasi mekanik invasif, vasopresor/inotropik, transfusi PRC, pembedahan 24 jam (tidak/elektif/emergensi) |
| Disfungsi organ, kriteria mSOFA tanpa komponen hepatik (4) | MAP terendah, rasio SpO₂/FiO₂ terendah, GCS terendah (bebas sedasi), kreatinin serum tertinggi |
| Kasus (1) | kategori diagnosis utama menurut sistem organ |

Catatan penting: GCS pasien yang tersedasi sepanjang 24 jam dinyatakan **hilang** dan diberi
penanda "tidak dapat dinilai karena sedasi" — penanda itu ikut menjadi fitur, jangan dibuang.

### Rencana analisis (ringkas)

- Preprocessing: MICE **di dalam fold**, one-hot untuk nominal, ordinal encoding untuk kelas rawat,
  standardisasi hanya untuk regresi logistik, class weighting untuk menjaga kalibrasi.
- Model: XGBoost, Random Forest, Elastic Net LR (pembanding yang dioptimalkan setara).
- Validasi: nested CV 5 outer × 3 inner + validasi temporal (latih Mei–Des 2025, uji Jan–Mei 2026).
- Metrik: AUC-ROC (target ≥ 0,75), sensitivitas, spesifisitas, F1; kalibrasi via Brier score dan
  calibration intercept & slope.
- Interpretabilitas: SHAP (TreeSHAP) global + individual, plus eksplorasi interaksi.
- Tanpa seleksi variabel stepwise; reduksi dimensi diserahkan pada regularisasi masing-masing algoritma.

## Peta folder

```
Proposal dr Ari/
├── CLAUDE.md                  # berkas ini
├── .gitignore
├── Proposal/                  # dokumen proposal (.docx, .pptx)
│   ├── proposal_gabungan_#17_UI-Format.docx   <- VERSI TERBARU, jadikan rujukan
│   ├── Editan #1/ , Editan #2/                 # revisi per bab
│   └── ...                                     # versi #3 s.d. #16 (riwayat)
└── data_analysis/             # pipeline machine learning (lihat CLAUDE.md di dalamnya)
```

## Aturan kerja

1. **Rujukan isi penelitian selalu versi proposal dengan nomor tertinggi** (saat ini `#17`).
   Bila diminta menyesuaikan kode dengan proposal, baca versi itu, bukan versi lama.
2. **Jangan mengubah file .docx/.pptx** kecuali diminta eksplisit. Dokumen ini sedang dalam
   proses bimbingan; perubahan diam-diam berbahaya.
3. **Data pasien nyata tidak boleh masuk repositori.** Isi `data_analysis/data/` saat ini
   seluruhnya sintetis. Pola nama file data nyata sudah diblokir di `.gitignore`.
4. **Angka hasil model dari data dummy bukan temuan penelitian.** Jangan menuliskannya ke
   dalam dokumen proposal atau laporan.
5. Bahasa: dokumentasi, komentar kode, dan label gambar dalam **bahasa Indonesia**; nama
   variabel dan fungsi dalam bahasa Inggris.

## Istilah

| Singkatan | Arti |
|---|---|
| INA-CBGs | Indonesia Case-Based Groups (tarif paket JKN) |
| JKN / BPJS | Jaminan Kesehatan Nasional / penyelenggaranya |
| mSOFA | modified Sequential Organ Failure Assessment |
| SF ratio | rasio SpO₂/FiO₂ |
| TRIPOD+AI | standar pelaporan model prediksi klinis |
| RSCM | RSUPN Dr. Cipto Mangunkusumo |
