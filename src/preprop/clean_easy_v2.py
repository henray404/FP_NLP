"""
Bangun ulang `easy_clean_v2.jsonl` dari `easy_clean.jsonl` — deterministik, tanpa GPU.

Kenapa modul ini ada: logikanya semula hanya hidup di
`notebooks/revisi/07_bersihkan_easy_clean.ipynb`, sementara `easy_clean_v2.jsonl` sendiri
masuk `.gitignore` (`data/*` dibuang kecuali `data/Final/*_v3.*`). Akibatnya berkas v2 —
satu-satunya acuan indeks bagi label kalibrasi di `calibration_labels.py` — bisa hilang dan
kalibrasi judge ikut mati. Dengan logika di `src/`, v2 selalu bisa dibangun ulang persis sama
tanpa membuka notebook.

Aturan pembersihan (urutan cek = prioritas, supaya satu baris tidak dihitung dua kali):
  1. `jawaban_bare_letter` — jawaban cuma huruf pilihan ganda "A".."E"
  2. `mc_block_in_soal`    — blok opsi A-E masih utuh di dalam `soal`
  3. `cara_fabricated`     — `cara` mengaku kekurangan informasi lalu mengarang soal sendiri
  4. `english_leak`        — soal berbahasa Inggris nyelip
  5. `REPAIR`              — jawaban berbentuk "C. 24 cm" -> prefix hurufnya ditanggalkan
  6. `KEEP`                — lolos apa adanya

Hasil verifikasi acuan (dari notebook 07, harus tetap sama): 2.686 -> 2.336 baris, dengan
`jawaban_bare_letter` 278, `REPAIR` 107, `english_leak` 50, `cara_fabricated` 22.

Usage:
    python -m src.preprop.clean_easy_v2 --self-check
    python -m src.preprop.clean_easy_v2                       # pakai path default
    python -m src.preprop.clean_easy_v2 --input data/Final/easy_clean.jsonl \\
        --output data/Final/easy_clean_v2.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BARE_LETTER = re.compile(r"^[A-E]$")
LETTER_VALUE = re.compile(r"^([A-E])\.\s*(.+)$", re.S)

# Frasa yang muncul kalau LLM mengaku soal aslinya terpotong lalu mengarang isinya sendiri.
# Daftar ini hasil sampling manual di notebook 07; jangan diubah tanpa menganotasi ulang,
# karena indeks di `calibration_labels.py` menunjuk sampel dari keluaran aturan ini.
CONFUSION_PHRASES = [
    "kita perlu lebih banyak informasi",
    "saya perlu melihat soal aslinya",
    "berdasarkan informasi yang diberikan, saya akan membuat",
    "namun, tanpa informasi lebih lanjut",
    "tanpa informasi tambahan yang lengkap",
    "kita tidak memiliki informasi",
    "namun, kita perlu lebih jelas",
]

# Angka acuan dari notebook 07. Dipakai `--self-check` sebagai pagar regresi supaya
# perubahan diam-diam pada aturan langsung ketahuan.
ACUAN_INPUT = 2686
ACUAN_KEEP_TOTAL = 2336
ACUAN_ALASAN = {
    "jawaban_bare_letter": 278,
    "REPAIR": 107,
    "english_leak": 50,
    "cara_fabricated": 22,
}

DEFAULT_INPUT = Path("data/Final/easy_clean.jsonl")
DEFAULT_OUTPUT = Path("data/Final/easy_clean_v2.jsonl")


def cara_fabricated(cara: str) -> bool:
    """True kalau `cara` memuat pengakuan kekurangan informasi (soal kemungkinan dikarang)."""
    c = (cara or "").lower()
    return any(p in c for p in CONFUSION_PHRASES)


def classify(row: dict) -> str:
    """'KEEP' | 'REPAIR' | alasan-drop. Urutan cek menentukan prioritas alasan."""
    from src.cot_synthesis.to_chatml import is_indonesian
    from src.preprop.filter_rules import is_multiple_choice

    jawaban = (row.get("jawaban") or "").strip()
    soal = row.get("soal") or ""
    cara = row.get("cara") or ""

    if BARE_LETTER.match(jawaban):
        return "jawaban_bare_letter"
    if is_multiple_choice(soal):
        return "mc_block_in_soal"
    if cara_fabricated(cara):
        return "cara_fabricated"
    if soal.strip() and not is_indonesian(soal):
        return "english_leak"
    if LETTER_VALUE.match(jawaban):
        return "REPAIR"
    return "KEEP"


def perbaiki(row: dict) -> dict:
    """Salinan baris dengan prefix huruf pilihan ganda ditanggalkan dari `jawaban`."""
    m = LETTER_VALUE.match((row.get("jawaban") or "").strip())
    if m is None:
        raise ValueError("baris ini bukan kandidat REPAIR")
    return {**row, "jawaban": m.group(2).strip(), "_repair": "stripped_letter_prefix"}


def bersihkan(rows: list[dict]) -> tuple[list[dict], list[dict], Counter]:
    """Pisahkan baris jadi (bersih, dibuang, hitungan-alasan). Tidak menyentuh `rows`."""
    alasan: Counter = Counter()
    bersih_rows: list[dict] = []
    dibuang: list[dict] = []

    for r in rows:
        vonis = classify(r)
        alasan[vonis] += 1
        if vonis == "KEEP":
            bersih_rows.append(r)
        elif vonis == "REPAIR":
            bersih_rows.append(perbaiki(r))
        else:
            dibuang.append({**r, "_dropped_reason": vonis})

    assert len(bersih_rows) + len(dibuang) == len(rows)
    return bersih_rows, dibuang, alasan


def _tulis(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT,
        *, tulis_dibuang: bool = True) -> dict:
    """Bangun v2 dari v1. Mengembalikan statistik untuk dicetak/diperiksa notebook."""
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} tidak ada — sumber v1 wajib tersedia")

    rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
    bersih_rows, dibuang, alasan = bersihkan(rows)

    _tulis(output_path, bersih_rows)
    path_dibuang = output_path.with_name(f"{output_path.stem}_dropped.jsonl")
    if tulis_dibuang:
        _tulis(path_dibuang, dibuang)

    return {
        "input": len(rows),
        "bersih": len(bersih_rows),
        "dibuang": len(dibuang),
        "alasan": dict(alasan),
        "output_path": str(output_path),
        "dropped_path": str(path_dibuang) if tulis_dibuang else None,
    }


def verifikasi_ulang(bersih_rows: list[dict]) -> Counter:
    """Jalankan ulang detektor di berkas keluaran. Semua hitungan kritis harus 0."""
    sisa: Counter = Counter()
    for r in bersih_rows:
        jawaban = (r.get("jawaban") or "").strip()
        if BARE_LETTER.match(jawaban):
            sisa["jawaban_bare_letter"] += 1
        if LETTER_VALUE.match(jawaban):
            sisa["letter_prefix_remaining"] += 1
        if cara_fabricated(r.get("cara") or ""):
            sisa["cara_fabricated"] += 1
    return sisa


def self_check(input_path: Path | None = None) -> None:
    """Self-check CPU: aturan klasifikasi + perbaikan, tanpa menyentuh berkas apa pun."""
    assert classify({"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": "C"}) \
        == "jawaban_bare_letter"
    assert classify({"soal": "Pilih jawaban benar.\nA. 1\nB. 2\nC. 3\nD. 4", "jawaban": "12"}) \
        == "mc_block_in_soal"
    assert classify({"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": "14",
                     "cara": "Saya perlu melihat soal aslinya untuk menjawab."}) \
        == "cara_fabricated"
    assert classify({"soal": "What is the value of the sum of two and three numbers here?",
                     "jawaban": "5"}) == "english_leak"
    assert classify({"soal": "Berapa panjang sisi miring segitiga tersebut?",
                     "jawaban": "C. 24 cm"}) == "REPAIR"
    assert classify({"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": "14"}) == "KEEP"

    # Prioritas: bare-letter dicek sebelum blok pilihan ganda, jadi alasannya bukan mc_block.
    assert classify({"soal": "Pilih.\nA. 1\nB. 2\nC. 3\nD. 4", "jawaban": "B"}) \
        == "jawaban_bare_letter"

    assert perbaiki({"jawaban": "C. 24 cm"})["jawaban"] == "24 cm"
    assert perbaiki({"jawaban": "A.  Rp. 742.500,00"})["jawaban"] == "Rp. 742.500,00"

    contoh = [
        {"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": "14"},
        {"soal": "Berapa panjang sisi miring segitiga tersebut?", "jawaban": "C. 24 cm"},
        {"soal": "Berapa nilai dari 2 + 2 pada bilangan bulat?", "jawaban": "D"},
    ]
    bersih_rows, dibuang, alasan = bersihkan(contoh)
    assert len(bersih_rows) == 2 and len(dibuang) == 1
    assert alasan["KEEP"] == 1 and alasan["REPAIR"] == 1 and alasan["jawaban_bare_letter"] == 1
    assert verifikasi_ulang(bersih_rows)["letter_prefix_remaining"] == 0
    print("self-check OK: 8 kasus klasifikasi, 2 perbaikan, 1 batch mini")

    if input_path is not None:
        rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
        bersih_rows, dibuang, alasan = bersihkan(rows)
        print(f"data nyata: {len(rows)} -> {len(bersih_rows)} bersih, {len(dibuang)} dibuang")
        beda = {k: (alasan.get(k, 0), v) for k, v in ACUAN_ALASAN.items() if alasan.get(k, 0) != v}
        if len(rows) == ACUAN_INPUT:
            assert len(bersih_rows) == ACUAN_KEEP_TOTAL, \
                f"acuan notebook 07: {ACUAN_KEEP_TOTAL} bersih, dapat {len(bersih_rows)}"
            assert not beda, f"rincian alasan menyimpang dari acuan (dapat, acuan): {beda}"
            print("cocok dengan acuan notebook 07")
        else:
            print(f"CATATAN: input {len(rows)} baris, acuan {ACUAN_INPUT} — pagar regresi dilewati")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangun ulang easy_clean_v2.jsonl (deterministik)")
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--self-check", action="store_true",
                    help="validasi aturan lalu keluar; tambah --input untuk sekalian cek acuan")
    args = ap.parse_args()

    if args.self_check:
        p = Path(args.input)
        self_check(p if p.exists() else None)
        return

    stats = run(Path(args.input), Path(args.output))
    for k, v in stats.items():
        print(f"{k:14}: {v}")


if __name__ == "__main__":
    main()
