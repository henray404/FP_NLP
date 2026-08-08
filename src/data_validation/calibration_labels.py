"""
Label kalibrasi manual untuk judge `judge_quality.py`, disimpan sebagai KODE.

Kenapa di `src/` dan bukan `data/`: `.gitignore:9` membuang seluruh `data/*` dan hanya
mengecualikan `data/Final/*_v3.*`. Itulah sebabnya `easy_clean_v2.jsonl` sempat hilang dan
memutus kalibrasi -- anotasi manual yang ikut hilang tidak bisa dibuat ulang tanpa menganotasi
260 baris sekali lagi. Label di sini bertahan karena ia file sumber, bukan data.

Isi: 22 indeks anotasi manual pada sampel `seed=7` dari `easy_clean_v2`, **0-indexed** terhadap
daftar keluaran `sampel_kalibrasi()` (bukan terhadap berkas aslinya). Sudah diverifikasi: 6/6
probe cocok.

Pemakaian: bandingkan vonis judge (`*_q1.progress` / `*_q2.progress` dari `judge_quality.py`
mode `--limit 260 --seed 7`) terhadap `Q1_POSITIF` / `Q2_POSITIF` untuk menghitung
presisi/recall sebelum judge dilepas ke data penuh.

Usage:
    python -m src.data_validation.calibration_labels --self-check
    python -m src.data_validation.calibration_labels --self-check \\
        --input data/Final/easy_clean_v2.jsonl   # sekalian cek sampelnya bisa dibentuk
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Parameter sampel kalibrasi. Harus identik dengan `--limit 260 --seed 7` di judge_quality.py;
# kalau salah satu berubah, seluruh indeks di bawah tidak lagi menunjuk baris yang sama.
SEED = 7
UKURAN_SAMPEL = 260

# --- Kategori anotasi manual (0-indexed terhadap sampel) ----------------------------------

C1_FABRIKASI = [53, 108, 111, 140, 203, 215, 220]   # soal rusak: instruksi/aktivitas/terpotong -> positif Q1
C2_KORUPSI = [25, 144, 222]                         # LaTeX/OCR rusak -> positif Q1
C3_MISMATCH = [164, 187, 169, 190]                  # nilai gold salah -> sinyal Q3, BUKAN positif Q2
C4_BLEED = [48, 80, 230, 254]                       # cacat di kolom `cara` (sudah dibuang) -> tak terpakai
C5_CARA_ENG = [58, 103, 124, 207]                   # cacat di `cara` -> tak terpakai

KATEGORI: dict[str, list[int]] = {
    "C1_FABRIKASI": C1_FABRIKASI,
    "C2_KORUPSI": C2_KORUPSI,
    "C3_MISMATCH": C3_MISMATCH,
    "C4_BLEED": C4_BLEED,
    "C5_CARA_ENG": C5_CARA_ENG,
}

# --- Turunan: acuan evaluasi judge -------------------------------------------------------

# Q1 menilai keterjawaban soal. Soal fabrikasi dan soal korup dua-duanya harus divonis TIDAK.
Q1_POSITIF = sorted(C1_FABRIKASI + C2_KORUPSI)      # 10 baris

# Q2 hanya menilai BENTUK/TIPE gold, bukan nilainya. Yang pasti: baris 108, gold-nya
# `\begin{array}` alias tabel.
Q2_POSITIF = [108]

# Kandidat tambahan Q2, belum pasti -- dipisah supaya tidak diam-diam ikut dihitung sebagai
# ground truth. Pakai lewat `q2_positif(sertakan_ragu=True)` kalau ingin batas atas.
Q2_POSITIF_RAGU = [111, 207]

# C3 SENGAJA tidak masuk Q2: keempat baris itu nilainya salah, tetapi bentuk gold-nya wajar
# semua (`(7, 8)`, `160\pi`, `149`). Q2 tidak menilai kebenaran nilai, jadi memasukkannya akan
# menghukum judge untuk sesuatu yang memang bukan tugasnya.


def q2_positif(sertakan_ragu: bool = False) -> list[int]:
    """Acuan positif Q2. `sertakan_ragu=True` menambahkan kandidat yang belum pasti."""
    return sorted(Q2_POSITIF + Q2_POSITIF_RAGU) if sertakan_ragu else sorted(Q2_POSITIF)


def sampel_kalibrasi(path_v2: str | Path) -> list[dict]:
    """Bentuk ulang sampel 260 baris yang dianotasi, dari berkas v2.

    Meniru persis pengambilan sampel di `judge_quality.run()` (`_idx` dulu, baru `sample`)
    supaya indeks di modul ini menunjuk baris yang sama.
    """
    path = Path(path_v2)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} tidak ada. Berkas v2 masuk .gitignore dan mungkin sudah hilang; "
            "bangun ulang dari sumbernya sebelum memakai fungsi ini."
        )
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):
        r["_idx"] = i
    if len(rows) < UKURAN_SAMPEL:
        raise ValueError(
            f"{path} hanya punya {len(rows)} baris, butuh minimal {UKURAN_SAMPEL} "
            "-- berkasnya bukan easy_clean_v2 yang dipakai saat anotasi."
        )
    return random.Random(SEED).sample(rows, UKURAN_SAMPEL)


def self_check(input_path: str | Path | None = None) -> None:
    """Self-check: tak ada indeks duplikat antar kategori, semua < UKURAN_SAMPEL."""
    semua: list[int] = []
    for nama, idxs in KATEGORI.items():
        assert idxs, f"{nama} kosong"
        duplikat_internal = {i for i in idxs if idxs.count(i) > 1}
        assert not duplikat_internal, f"{nama} punya indeks dobel: {sorted(duplikat_internal)}"
        for i in idxs:
            assert isinstance(i, int), f"{nama} memuat indeks non-int: {i!r}"
            assert 0 <= i < UKURAN_SAMPEL, f"{nama} indeks {i} di luar 0..{UKURAN_SAMPEL - 1}"
        semua.extend(idxs)

    tumpang_tindih = {i for i in semua if semua.count(i) > 1}
    assert not tumpang_tindih, f"indeks dipakai lebih dari satu kategori: {sorted(tumpang_tindih)}"
    assert len(semua) == 22, f"total anotasi harus 22, dapat {len(semua)}"

    assert Q1_POSITIF == sorted(C1_FABRIKASI + C2_KORUPSI)
    assert len(Q1_POSITIF) == 10, f"Q1_POSITIF harus 10 baris, dapat {len(Q1_POSITIF)}"
    assert set(Q2_POSITIF) <= set(semua), "Q2_POSITIF menunjuk baris yang tidak dianotasi"
    assert set(Q2_POSITIF_RAGU) <= set(semua), "Q2_POSITIF_RAGU menunjuk baris tak dianotasi"
    assert not set(Q2_POSITIF) & set(C3_MISMATCH), "C3 bukan positif Q2 -- lihat catatan di atas"
    assert q2_positif() == [108]
    assert q2_positif(sertakan_ragu=True) == [108, 111, 207]

    print(f"self-check OK: {len(semua)} indeks, {len(KATEGORI)} kategori, "
          f"Q1_POSITIF={len(Q1_POSITIF)}, Q2_POSITIF={len(Q2_POSITIF)}"
          f" (+{len(Q2_POSITIF_RAGU)} ragu)")

    if input_path is not None:
        sampel = sampel_kalibrasi(input_path)
        assert len(sampel) == UKURAN_SAMPEL
        print(f"sampel OK: {len(sampel)} baris dari {input_path}")
        for nama, idxs in KATEGORI.items():
            contoh = str(sampel[idxs[0]].get("soal", ""))[:60].replace("\n", " ")
            print(f"  {nama:14} idx {idxs[0]:3}: {contoh}...")


def main() -> None:
    ap = argparse.ArgumentParser(description="Label kalibrasi manual judge_quality")
    ap.add_argument("--self-check", action="store_true", help="validasi label lalu keluar")
    ap.add_argument("--input", default=None,
                    help="opsional: berkas v2 untuk sekalian menguji sampel_kalibrasi()")
    args = ap.parse_args()

    if args.self_check:
        self_check(args.input)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
