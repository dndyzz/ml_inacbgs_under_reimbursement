"""Gabungkan spreadsheet.csv (coding INA-CBGs) + REGISTER PASIEN ICU menjadi satu tabel
kohort yang memakai NAMA KOLOM PERSIS seperti protokol permintaan data IT.

Nama, urutan, dan daftar kode kolom dibaca saat runtime dari
`Permintaan_Data_IT_RSCM_ICU_INACBG.xlsx` sheet '3. Template Data', jadi tidak bisa
melenceng dari protokol.

ATURAN PENGISIAN (penting, jangan diubah tanpa alasan):
  1. Kolom protokol hanya diisi bila nilainya memenuhi definisi operasional protokol.
     Bila sumber lokal tidak bisa memenuhinya, kolom DIKOSONGKAN dan alasannya ditulis
     pada kolom `catatan` -- ini persis instruksi protokol bagian E ("Data belum tersedia").
     Mengisi kolom protokol dengan proksi akan merusak arti kolom itu saat data IT tiba.
  2. Variabel yang hanya ada di sumber lokal, atau yang bersifat proksi, ditaruh pada
     kolom tambahan berawalan `x_` SETELAH kolom `catatan`. Kolom `x_` bukan bagian
     protokol; hapus saja bila data IT sudah lengkap.
  3. `no_rekam_medis` diisi HANYA untuk penautan. Buang kolom itu (dan `x_*_tautan`)
     sebelum dataset dipakai analisis / dibagikan -- lihat DEIDENT_DROP.

Prasyarat: jalankan compile_icu.py lebih dulu supaya spreadsheet.csv ada.
Keluaran: cohort_protokol.csv + ringkasan audit ke stdout.
"""
import os
import re

import numpy as np
import openpyxl
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROTO = os.path.join(os.path.dirname(BASE), "Permintaan_Data_IT_RSCM_ICU_INACBG.xlsx")
CODING = os.path.join(BASE, "spreadsheet.csv")
REGISTERS = [
    os.path.join(BASE, "data_raw", "REGISTER PASIEN ICU 2025.xlsx"),
    os.path.join(BASE, "data_raw", "REGISTER PASIEN ICU 2026.xlsx"),
]
OUT = os.path.join(BASE, "cohort_protokol.csv")

# kolom yang harus dibuang sebelum analisis / berbagi data
DEIDENT_DROP = ["no_rekam_medis", "x_nama_sumber", "x_sumber_tautan", "x_delta_tautan_hari"]

# kolom register yang dipakai: kunci internal -> regex pencocok header
REG_COLS = [
    ("nrm", r"^NRM"), ("nama", r"^NAMA$"), ("lahir", r"LAHIR"), ("jk", r"JENIS KELAMIN"),
    ("prio", r"^PRIORITAS"), ("masuk", r"^TANGGAL MASUK"), ("asal", r"^ASAL MASUK"),
    ("keluar", r"^TANGGAL KELUAR"), ("pindah", r"^PINDAH KE"), ("jaminan", r"^JAMINAN"),
    ("surg", r"^SURGICAL"), ("med", r"^MEDICAL"), ("ventin", r"PEMASANGAN"),
]

# bab ICD-10 -> kategori sistem organ protokol (daftar kode ada di Kamus Data no. 29)
ICD10_MAP = [
    (r"^[AB]", "sepsis-infeksi sistemik"),
    (r"^C", "hemato-onkologi"),
    (r"^D", "hemato-onkologi"),          # D00-D48 neoplasma, D50-D89 darah
    (r"^E", "renal-metabolik"),
    (r"^G", "neurologi"),
    (r"^I", "kardiovaskular"),
    (r"^J", "respirasi"),
    (r"^K", "gastrointestinal-hati"),
    (r"^N", "renal-metabolik"),
    (r"^O", "obstetri-ginekologi"),
    (r"^[ST]", "trauma"),
]
ASAL_GROUPS = [
    ("kamar operasi", r"\bOK\b|KANIGARA|KAMAR OPERASI"),
    ("IGD", r"IGD|EMERGEN"),
    ("HCU", r"HCU"),
    ("ICU lain", r"\bICU\b"),
    ("rawat inap", r"GED|GD A|LT ?\d|BEDAH|RAWAT|EXE|KENCANA"),
]


def nrm_key(v):
    """NRM -> deret angka tanpa nol depan; '' bila tak ada angka."""
    digits = re.sub(r"\D", "", "" if v is None else str(v))
    return digits.lstrip("0")


def name_key(v):
    """Nama -> huruf kapital dan spasi saja, untuk pencocokan cadangan."""
    return re.sub(r"[^A-Z ]", "", ("" if v is None else str(v)).upper()).strip()


def to_dt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return pd.NaT
    s = str(v).strip()
    if not s or s in ("nan", "None", "NaT", "#VALUE!", "#REF!"):
        return pd.NaT
    # dua gaya bercampur di sumber: ISO '2026-01-02 21:50:00' dan '14/01/2026 19:20'.
    # ISO dicoba lebih dulu supaya dayfirst tidak dipaksakan ke tanggal yang sudah ISO.
    for kwargs in ({"format": "ISO8601"}, {"dayfirst": True}):
        try:
            return pd.to_datetime(s, **kwargs)
        except Exception:
            pass
    return pd.NaT


def icd10_category(code):
    """Kode ICD-10 -> kategori sistem organ protokol."""
    if not code:
        return ""
    if code.startswith("R57.2"):
        return "sepsis-infeksi sistemik"      # syok septik
    if code.startswith("R57"):
        return "kardiovaskular"               # syok lain
    for pat, label in ICD10_MAP:
        if re.match(pat, code):
            return label
    return "lain-lain"


def codes_in(text, pat):
    """Ambil kode berurutan tanpa duplikat dari teks diagnosis/prosedur gabungan."""
    seen, out = set(), []
    for m in re.findall(pat, "" if text is None else str(text)):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def prioritas_label(text):
    """PRIORITAS register -> 'Prioritas I' / 'Prioritas II'.

    Sebagian bulan menulis tanpa kata 'Prioritas' ('Ventilasi Mekanik',
    'Monitoring Hemodinamik') -- keduanya dinormalkan supaya tidak jadi kategori sendiri.
    """
    up = ("" if text is None else str(text)).upper()
    if re.search(r"PRIORITAS I\b|VENTILASI MEKANIK", up):
        return "Prioritas I"
    if re.search(r"PRIORITAS II\b|MONITORING HEMODINAMIK", up):
        return "Prioritas II"
    return ""


def group_asal(text):
    up = ("" if text is None else str(text)).upper()
    if not up.strip() or up in ("NAN", "NONE"):
        return ""
    for label, pat in ASAL_GROUPS:
        if re.search(pat, up):
            return label
    return "lain"


def load_protocol_columns():
    wb = openpyxl.load_workbook(PROTO, data_only=True)
    row = next(wb["3. Template Data"].iter_rows(max_row=1))
    cols = [c.value for c in row if c.value]
    wb.close()
    return cols


def load_register():
    rows = []
    for path in REGISTERS:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if ws.title.upper() in ("MASTER", "REKAP ICU"):
                continue                      # template kosong / daftar tindakan n=73
            it = ws.iter_rows(values_only=True)
            try:
                head = [re.sub(r"\s+", " ", str(c)).strip().upper() if c is not None else ""
                        for c in next(it)]
            except StopIteration:
                continue
            idx = {}
            for i, h in enumerate(head):
                for key, pat in REG_COLS:
                    if key not in idx and re.search(pat, h):
                        idx[key] = i
            for raw in it:
                rec = {k: (raw[idx[k]] if k in idx and idx[k] < len(raw) else None)
                       for k, _ in REG_COLS}
                if not rec["nama"] and not rec["nrm"]:
                    continue                  # baris padding template
                rec["reg_sheet"] = ws.title
                rows.append(rec)
        wb.close()
    reg = pd.DataFrame(rows)
    reg["nrm_k"] = reg["nrm"].map(nrm_key)
    reg["nama_k"] = reg["nama"].map(name_key)
    reg["masuk_dt"] = reg["masuk"].map(to_dt)
    reg["keluar_dt"] = reg["keluar"].map(to_dt)
    reg["lahir_dt"] = reg["lahir"].map(to_dt)
    reg["ventin_dt"] = reg["ventin"].map(to_dt)
    return reg.reset_index(drop=True)


def link(coding, reg):
    """Tautkan tiap baris coding ke satu baris register.

    NRM lebih dulu; bila satu NRM punya beberapa admisi, pilih yang tanggal masuknya
    paling dekat. Nama dipakai hanya bila NRM gagal. Cara dan selisihnya dilaporkan
    supaya tautan yang meragukan bisa diperiksa manual, bukan dipercaya diam-diam.
    """
    by_nrm = reg.groupby("nrm_k").groups
    by_name = reg.groupby("nama_k").groups
    out = []
    for _, c in coding.iterrows():
        cands, how = None, ""
        if c["nrm_k"] and c["nrm_k"] in by_nrm:
            cands, how = list(by_nrm[c["nrm_k"]]), "NRM"
        elif c["nama_k"] and c["nama_k"] in by_name:
            cands, how = list(by_name[c["nama_k"]]), "nama"
        if not cands:
            out.append((None, 0, "tidak cocok", np.nan))
            continue
        sub = reg.loc[cands]
        if pd.notna(c["masuk_dt"]) and sub["masuk_dt"].notna().any():
            delta = (sub["masuk_dt"] - c["masuk_dt"]).abs().dt.total_seconds() / 86400
            j = delta.idxmin()
            out.append((j, len(cands), how + "+tanggal", round(float(delta.loc[j]), 2)))
        else:
            out.append((sub.index[0], len(cands), how + " (tanpa tanggal)", np.nan))
    return pd.DataFrame(out, columns=["reg_i", "n_kandidat", "cara", "delta_hari"],
                        index=coding.index)


def build():
    cols = load_protocol_columns()
    cod = pd.read_csv(CODING, dtype=str, keep_default_na=False)
    cod["ratio"] = pd.to_numeric(cod["Coverage (%)"], errors="coerce")
    cod = cod[cod["ratio"].notna()].reset_index(drop=True)   # outcome = syarat masuk kohort
    cod["nrm_k"] = cod["NRM"].map(nrm_key)
    cod["nama_k"] = cod["Nama"].map(name_key)
    cod["masuk_dt"] = cod["Tanggal & Jam Masuk"].map(to_dt)

    reg = load_register()
    mt = link(cod, reg)
    pick = lambda col: mt["reg_i"].map(lambda i: reg.loc[i, col] if pd.notna(i) else None)

    n = len(cod)
    df = pd.DataFrame({c: [""] * n for c in cols})
    notes = [[] for _ in range(n)]

    # --- A. identitas -----------------------------------------------------------
    df["kode_subjek"] = [f"ICU-{i + 1:04d}" for i in range(n)]
    df["no_rekam_medis"] = cod["NRM"].str.strip()
    df["nama_unit_icu"] = cod["source_sheet"].str.strip()
    # no_episode_rawat_inap: tidak ada nomor episode di kedua sumber

    # --- B. waktu ---------------------------------------------------------------
    masuk = pick("masuk_dt").map(to_dt).fillna(cod["masuk_dt"])
    keluar = pick("keluar_dt").map(to_dt).fillna(cod["Tanggal & Jam Keluar"].map(to_dt))
    fmt = lambda s: s.dt.strftime("%Y-%m-%d %H:%M").fillna("")
    df["tgl_masuk_icu"] = fmt(masuk)
    df["tgl_keluar_icu"] = fmt(keluar)
    # tgl_masuk_rs / tgl_keluar_rs / pasien_rujukan: tidak ada di sumber lokal

    # status_keluar_rs: hanya 'meninggal' yang bisa dipastikan berlaku di level episode.
    # 'hidup' saat keluar ICU TIDAK berarti hidup saat keluar RS -> dibiarkan kosong.
    disp = (pick("pindah").astype(str).str.upper().fillna("")
            + " " + cod["Alasan Discharge"].str.upper())
    df.loc[disp.str.contains("MENINGGAL", na=False), "status_keluar_rs"] = "meninggal"

    # --- C. demografi & administratif -------------------------------------------
    umur = ((masuk - pick("lahir_dt").map(to_dt)).dt.days / 365.25).round(0)
    umur = umur.where((umur >= 0) & (umur <= 110))
    umur_cod = pd.to_numeric(cod["Usia (tahun)"], errors="coerce")
    umur = umur.fillna(umur_cod.where((umur_cod >= 0) & (umur_cod <= 110)))
    df["usia"] = umur.map(lambda v: "" if pd.isna(v) else str(int(v)))

    jk = cod["JK"].str.strip().str.upper().where(lambda s: s.isin(["L", "P"]))
    jk_reg = pick("jk").astype(str).str.strip().str.upper().str[:1]
    df["jenis_kelamin"] = jk.fillna(jk_reg.where(jk_reg.isin(["L", "P"]))).fillna("")

    bayar = (cod["Tipe Pembayaran"].str.upper().str.strip() + " "
             + pick("jaminan").astype(str).str.upper())
    df.loc[bayar.str.contains("BPJS|JKN", na=False), "peserta_jkn"] = "1"
    df.loc[bayar.str.contains("UMUM|PERUSAHAAN", na=False)
           & ~bayar.str.contains("BPJS|JKN", na=False), "peserta_jkn"] = "0"
    # kelas_rawat_jkn: tidak ada di sumber lokal (yang ada hanya jenis penjamin)
    # tipe_admisi_icu: elektif/emergensi tidak terdokumentasi -> proksi di x_asal_masuk_unit
    # durasi_pra_icu_jam: butuh tgl_masuk_rs yang tidak ada

    # --- D. intervensi 24 jam ---------------------------------------------------
    # Timestamp ventilator kosong = tidak diventilasi (tervalidasi silang dgn PRIORITAS:
    # Prioritas I -> 1228/1250 ada timestamp, Prioritas II -> 869/949 kosong).
    # Hati-hati: 'Lama Penggunaan Ventilator' adalah formula teks atas kedua timestamp,
    # jadi pasangan kosong ikut ter-render "0 hours 00 minutes" -- jangan dibaca sebagai nol.
    v_cod = cod["Tanggal & Jam Pemasangan Awal Ventilator"].map(to_dt)
    v_reg = pick("ventin_dt").map(to_dt)
    vstart = v_cod.fillna(v_reg)
    ada_durasi = cod["Lama Penggunaan Ventilator (jam)"].str.contains(r"^\d+ hours", na=False)
    jam_ke_vent = (vstart - masuk).dt.total_seconds() / 3600

    vent = pd.Series("", index=cod.index)
    vent[vstart.isna() & ada_durasi] = "0"                       # tidak diventilasi
    vent[vstart.notna() & jam_ke_vent.between(-2, 24)] = "1"     # mulai dalam 24 jam
    df["ventilasi_mekanik_24j"] = vent
    # ventilasi ada tapi waktunya tak bisa dipastikan -> kosong, dicatat
    tak_pasti = vstart.notna() & ~jam_ke_vent.between(-2, 24)
    # vasopresor_inotropik_24j / transfusi_prc_24j: tidak ada data waktu pemberian
    # pembedahan_24j: kode 0/1/2 butuh pisah elektif-emergensi yang tidak terekam
    #                 -> biner ada di x_pasca_bedah

    # --- E. disfungsi organ: TIDAK ADA SATU PUN di sumber lokal ------------------

    # --- F. kasus ---------------------------------------------------------------
    dx_text = cod["Diagnosis (ICD-10)"] + " " + cod["Bukti Untuk Claim (Diagnosis)"]
    dx_codes = dx_text.map(lambda t: codes_in(t, r"[A-TV-Z]\d{2}(?:\.\d{1,2})?"))
    df["diagnosis_utama_icd10"] = dx_codes.map(lambda l: l[0] if l else "")
    df["diagnosis_sekunder_icd10"] = dx_codes.map(lambda l: ";".join(l[1:]))
    df["kategori_diagnosis_utama"] = df["diagnosis_utama_icd10"].map(icd10_category)
    tx_text = cod["Tindakan (ICD-9)"] + " " + cod["Bukti Untuk Claim (Tindakan)"]
    df["prosedur_icd9cm"] = tx_text.map(lambda t: ";".join(codes_in(t, r"\d{2}\.\d{1,2}")))

    # --- G. biaya & klaim -------------------------------------------------------
    rp = lambda s: pd.to_numeric(s.str.strip(), errors="coerce")
    tag, klaim = rp(cod["Tagihan RS (Rp)"]), rp(cod["Claim INA CBGs (Rp)"])
    num = lambda s: s.map(lambda v: "" if pd.isna(v) else f"{v:.0f}")
    df["total_tagihan_episode"] = num(tag)
    df["klaim_inacbg_final"] = num(klaim)
    df["kode_inacbg"] = cod["Kode Grouper INA CBGs"].str.strip()
    df["deskripsi_inacbg"] = cod["Nama Grouper INA CBGs"].str.strip()
    df["severity_level_inacbg"] = cod["Kode Grouper INA CBGs"].str.extract(
        r"-(I{1,3})$")[0].fillna("")
    df["rasio_klaim_tagihan"] = cod["ratio"].map(lambda v: f"{v:.3f}")
    df["under_reimbursement"] = (cod["ratio"] < 1).astype(int).astype(str)
    # status_klaim / tagihan_pra_icu / tagihan_icu / tagihan_pasca_icu: tidak ada

    # --- kolom tambahan non-protokol --------------------------------------------
    extra = pd.DataFrame(index=cod.index)
    extra["x_prioritas_icu"] = pick("prio").map(prioritas_label)
    extra["x_asal_masuk_unit"] = pick("asal").map(group_asal)
    surg = pd.to_numeric(pick("surg"), errors="coerce")
    med = pd.to_numeric(pick("med"), errors="coerce")
    extra["x_pasca_bedah"] = np.where(surg == 1, "1", np.where(med == 1, "0", ""))
    extra["x_cmg_inacbg"] = cod["Kode Grouper INA CBGs"].str.extract(r"^([A-Z]{1,2})")[0].fillna("")
    extra["x_ventilasi_kapanpun"] = np.where(vstart.notna(), "1",
                                             np.where(ada_durasi, "0", ""))
    extra["x_lama_ventilator_jam"] = cod["Lama Penggunaan Ventilator (jam)"].str.extract(
        r"^(\d+) hours")[0].where(vstart.notna()).fillna("")
    extra["x_nama_sumber"] = cod["Nama"].str.strip()
    extra["x_sumber_tautan"] = mt["cara"]
    extra["x_delta_tautan_hari"] = mt["delta_hari"].map(lambda v: "" if pd.isna(v) else f"{v:g}")
    extra["x_bulan_sumber"] = cod["source_month"]

    # --- catatan per baris ------------------------------------------------------
    for i in range(n):
        notes[i].append("tagihan dari lembar coding ICU; belum terverifikasi mencakup "
                        "seluruh episode (pra-ICU + ICU + pasca-ICU)")
    for i in cod.index[mt["cara"].eq("tidak cocok")]:
        notes[i].append("tidak tertaut register: variabel prioritas/asal masuk/pasca-bedah kosong")
    for i in cod.index[mt["cara"].str.startswith("nama")]:
        notes[i].append("tertaut lewat nama, bukan NRM -- perlu verifikasi manual")
    for i in cod.index[mt["n_kandidat"] > 1]:
        notes[i].append(f"{int(mt.loc[i, 'n_kandidat'])} admisi register untuk NRM ini "
                        "(admisi ulang) -- episode dipilih dari tanggal terdekat")
    for i in cod.index[mt["delta_hari"] > 30]:
        notes[i].append(f"selisih tanggal masuk coding vs register {mt.loc[i, 'delta_hari']:g} hari "
                        "-- kemungkinan episode keliru")
    for i in cod.index[tak_pasti]:
        notes[i].append("ventilasi mekanik terekam tetapi awalnya di luar 24 jam pertama "
                        "atau waktunya tak dapat dipastikan -> ventilasi_mekanik_24j dikosongkan")
    for i in cod.index[df["usia"].eq("")]:
        notes[i].append("usia tidak dapat dihitung (tanggal lahir/masuk hilang atau di luar rentang)")
    df["catatan"] = ["; ".join(x) for x in notes]

    out = pd.concat([df[cols], extra], axis=1)
    return out, cod, reg, mt, tak_pasti


def report(out, cod, reg, mt, tak_pasti):
    proto = load_protocol_columns()
    n = len(out)
    print(f"kohort: {n} baris (semua punya rasio klaim/tagihan)")
    print(f"register: {len(reg)} baris admisi ICU, {reg['nrm_k'].nunique()} NRM unik")
    print(f"tertaut register: {mt['reg_i'].notna().sum()} ({mt['reg_i'].notna().mean() * 100:.1f}%)")
    print("\ncara penautan:")
    for k, v in mt["cara"].value_counts().items():
        print(f"  {v:5d}  {k}")
    print(f"  ambigu (>1 admisi utk NRM yg sama): {(mt['n_kandidat'] > 1).sum()}")
    print(f"  selisih tanggal >30 hari          : {(mt['delta_hari'] > 30).sum()}")

    print(f"\n=== {len(proto)} kolom protokol: keterisian ===")
    for c in proto:
        k = (out[c].astype(str).str.strip() != "").sum()
        tag = "" if k else "   <- KOSONG TOTAL"
        print(f"  {k:5d} ({k / n * 100:5.1f}%)  {c}{tag}")
    kosong = [c for c in proto if (out[c].astype(str).str.strip() != "").sum() == 0]
    print(f"\nkolom protokol kosong total: {len(kosong)}/{len(proto)}")
    print("  " + ", ".join(kosong))

    print("\n=== kolom tambahan (non-protokol) ===")
    for c in out.columns:
        if c.startswith("x_"):
            k = (out[c].astype(str).str.strip() != "").sum()
            print(f"  {k:5d} ({k / n * 100:5.1f}%)  {c}")

    print("\n=== distribusi variabel yang terpakai ===")
    for c in ["jenis_kelamin", "peserta_jkn", "ventilasi_mekanik_24j", "status_keluar_rs",
              "severity_level_inacbg", "kategori_diagnosis_utama", "under_reimbursement",
              "x_prioritas_icu", "x_asal_masuk_unit", "x_pasca_bedah"]:
        vc = out[c].replace("", "(kosong)").value_counts()
        print(f"  [{c}] " + ", ".join(f"{k}={v}" for k, v in vc.head(11).items()))
    usia = pd.to_numeric(out["usia"], errors="coerce")
    print(f"  [usia] median {usia.median():.0f} (IQR {usia.quantile(.25):.0f}-"
          f"{usia.quantile(.75):.0f}), <18 th: {(usia < 18).sum()}")
    print(f"  ventilasi terekam tapi waktu tak pasti: {int(tak_pasti.sum())}")

    print("\n=== silang mutu: PRIORITAS x ventilasi_mekanik_24j ===")
    print(pd.crosstab(out["x_prioritas_icu"].replace("", "(kosong)"),
                      out["ventilasi_mekanik_24j"].replace("", "(kosong)")).to_string())

    # duplikat episode: NRM + klaim + tagihan identik
    dup = out.duplicated(subset=["no_rekam_medis", "klaim_inacbg_final",
                                 "total_tagihan_episode"], keep=False)
    print(f"\nkandidat baris ganda (NRM+klaim+tagihan identik): {dup.sum()}")

    # complete-case untuk model cadangan 7 variabel
    seven = ["usia", "jenis_kelamin", "ventilasi_mekanik_24j", "kategori_diagnosis_utama",
             "x_prioritas_icu", "x_asal_masuk_unit", "x_pasca_bedah"]
    cc = (out[seven].astype(str).apply(lambda s: s.str.strip() != "")).all(axis=1)
    y = pd.to_numeric(out["under_reimbursement"], errors="coerce")
    print(f"\ncomplete-case 7 variabel cadangan: {cc.sum()} ({cc.mean() * 100:.1f}%)"
          f" -- under_reimbursement=1: {int(y[cc].sum())}, =0: {int((1 - y[cc]).sum())}")
    print(f"buang sebelum analisis/berbagi: {', '.join(DEIDENT_DROP)}")


def demo():
    assert nrm_key("0123-45-67") == "1234567"
    assert nrm_key(" 987-65-43") == "9876543"
    assert nrm_key(None) == "" and nrm_key("-") == ""
    assert name_key(" ANIK  WIDIASTUTI ") == "ANIK  WIDIASTUTI"
    assert name_key("Nama, Ke-2 (x)") == "NAMA KE X"
    assert pd.isna(to_dt("#VALUE!")) and pd.isna(to_dt(""))
    assert to_dt("14/01/2026 19:20:00").strftime("%Y-%m-%d %H:%M") == "2026-01-14 19:20"
    assert to_dt("2026-01-02 21:50:00").hour == 21
    assert icd10_category("J90") == "respirasi"
    assert icd10_category("C25.0") == "hemato-onkologi"
    assert icd10_category("R57.2") == "sepsis-infeksi sistemik"   # syok septik, bukan kardio
    assert icd10_category("R57.1") == "kardiovaskular"
    assert icd10_category("S06.5") == "trauma"
    assert icd10_category("F32") == "lain-lain" and icd10_category("") == ""
    # kode diambil berurutan, duplikat dibuang -> primer = kode pertama
    got = codes_in("Other benign mammary dysplasiasN60.8 Primer Benign neoplasm of breastD24 "
                   "Sekunder Hypertrophy of breastN62 Sekunder", r"[A-TV-Z]\d{2}(?:\.\d{1,2})?")
    assert got == ["N60.8", "D24", "N62"], got
    assert codes_in("x85.32 y33.27 z85.32", r"\d{2}\.\d{1,2}") == ["85.32", "33.27"]
    assert group_asal("OK KANIGARA/ A - BEDAH LT 4") == "kamar operasi"   # OK menang atas ruangan
    assert group_asal("IGD ZONA MERAH") == "IGD"
    assert group_asal("") == "" and group_asal(None) == ""
    assert group_asal("GD A 401") == "rawat inap"
    assert group_asal("planet mars") == "lain"
    assert prioritas_label("PRIORITAS I VENTILASI MEKANIK") == "Prioritas I"
    assert prioritas_label("Ventilasi Mekanik") == "Prioritas I"       # varian tanpa 'Prioritas'
    assert prioritas_label("PRIORITAS II MONITORING HEMODINAMIK") == "Prioritas II"
    assert prioritas_label("monitoring hemodinamik") == "Prioritas II"
    assert prioritas_label("") == "" and prioritas_label(None) == ""
    print("demo ok")


if __name__ == "__main__":
    demo()
    out, cod, reg, mt, tak_pasti = build()
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    report(out, cod, reg, mt, tak_pasti)
    print(f"\n{len(out)} baris x {len(out.columns)} kolom -> {OUT}")
