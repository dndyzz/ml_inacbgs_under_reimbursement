"""Gabungkan semua sheet ber-keyword ICU dari data_raw/*.xlsx menjadi satu spreadsheet.csv.

Header antar-sheet sedikit berbeda (typo label, satu label kosong, satu kolom
kosong di depan pada file Juni/Juli). Kolom disejajarkan berdasarkan POSISI
setelah kolom depan tanpa nama dibuang, lalu diverifikasi lewat nama-nama jangkar
(nama, NRM, claim, tagihan). Sheet yang gagal verifikasi dilaporkan, tidak
dimasukkan diam-diam.
"""
import csv
import glob
import os
import re
from datetime import datetime

import openpyxl

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "data_raw")
OUT = os.path.join(BASE, "spreadsheet.csv")
META = ["source_file", "source_month", "source_sheet"]

CANON = [
    "No", "Final", "Nama", "JK", "Usia (tahun)", "NRM", "DPJP", "Tipe Pembayaran",
    "Tanggal & Jam Masuk", "Tanggal & Jam Keluar", "Lama Perawatan (hari)",
    "Alasan Discharge", "Tanggal & Jam Pemasangan Awal Ventilator",
    "Tanggal & Jam Pelepasan Ventilator", "Lama Penggunaan Ventilator (jam)",
    "Group Waktu Penggunaan Ventilator", "Diagnosis (ICD-10)",
    "Bukti Untuk Claim (Diagnosis)", "Tindakan (ICD-9)", "Bukti Untuk Claim (Tindakan)",
    "RESUME MEDIS Sementara", "Kode Grouper INA CBGs", "Nama Grouper INA CBGs",
    "Claim INA CBGs (Rp)", "Tagihan RS (Rp)", "Selisih (Rp)", "Coverage (%)",
    "Update Billing", "Pengajuan Obat/Alat NON-FORNAS",
]
# kolom jangkar: posisi -> nama ternormalisasi yang harus cocok bila label tidak kosong
ANCHORS = {2: "nama", 5: "nrm", 23: "claiminacbgsrp", 24: "tagihanrsrp"}
NOT_EMPTY = (2, 5, 23, 24)  # nama / NRM / claim / tagihan -> baris template dibuang


def clean(v):
    if v is None:
        return ""
    if isinstance(v, datetime):
        return v.isoformat(sep=" ")
    return str(v).replace("\n", " ").strip()


def norm(name):
    """Nama kolom -> kunci banding: buang catatan '*...' dan semua non-alfanumerik."""
    return re.sub(r"[^a-z0-9]", "", name.split("*")[0].lower())


def month_of(fname):
    m = re.search(r"IPILB (.+)\.xlsx$", fname)
    return m.group(1) if m else fname


def align(head):
    """Jumlah kolom kosong di depan yang harus dibuang, atau None bila tak cocok."""
    for off in range(len(head) - len(CANON) + 1):
        if any(head[:off]):  # kolom yang dibuang harus benar-benar tanpa nama
            break
        shifted = head[off:]
        if all(not shifted[i] or norm(shifted[i]) == a for i, a in ANCHORS.items()):
            return off
    return None


def main():
    rows, skipped = [], []
    for path in sorted(glob.glob(os.path.join(RAW, "*.xlsx"))):
        fname = os.path.basename(path)
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet in wb.sheetnames:
            if "ICU" not in sheet.upper():
                continue
            it = wb[sheet].iter_rows(values_only=True)
            try:
                head = [clean(c) for c in next(it)]
            except StopIteration:
                continue
            while head and not head[-1]:
                head.pop()
            off = align(head)
            if off is None:
                skipped.append(f"{fname}/{sheet}: header tidak dikenali -> {head[:6]}")
                continue
            for raw in it:
                rec = [clean(c) for c in raw]
                if any(rec[:off]):
                    skipped.append(f"{fname}/{sheet}: ada data di kolom depan yang dibuang")
                    break
                rec = rec[off:off + len(CANON)]
                rec += [""] * (len(CANON) - len(rec))
                if rec[0].upper() == "CONTOH":  # baris contoh pada template
                    continue
                if not any(rec[i] for i in NOT_EMPTY):  # baris template kosong
                    continue
                rows.append([fname, month_of(fname), sheet] + rec)
        wb.close()

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(META + CANON)
        w.writerows(rows)

    print(f"{len(rows)} baris -> {OUT}")
    for s in skipped:
        print("SKIP", s)


def demo():
    assert clean(None) == ""
    assert clean(datetime(2026, 1, 2, 21, 50)) == "2026-01-02 21:50:00"
    assert clean("a\nb ") == "a b"
    assert month_of("A Coding INACBGs - IPILB Januari 2026.xlsx") == "Januari 2026"
    assert norm("No.") == norm("No") == "no"
    assert norm("Tagihan RS (Rp)\n*terhitung otomatis") == "tagihanrsrp"
    assert align(CANON) == 0
    assert align([""] + CANON) == 1                      # file Juni/Juli
    assert align(["", ""] + CANON) == 2
    bad = list(CANON)
    bad[12] = ""                                          # label hilang -> tetap diterima
    assert align(bad) == 0
    bad[2] = "Nama Grouper"                               # jangkar salah -> ditolak
    assert align(bad) is None
    assert align(["Nama"] + CANON) is None                # kolom depan berisi nama
    print("demo ok")


if __name__ == "__main__":
    demo()
    main()
