"""
SKENARIO 1 (revisi) — bake-off teacher CoT tier 14B.

Beda dari `compare_teachers.py` (versi lama, teacher 7B): modul ini menilai kandidat
dengan LIMA metrik, bukan hanya retensi/cakupan. Dua metrik baru dipakai untuk menguji
hipotesis, bukan sekadar mencari model terbaik:

    retensi%     = kandidat lolos judge / total kandidat        (makin tinggi makin bagus)
    cakupan%     = soal terjawab benar / soal dicoba            (makin tinggi makin bagus)
    format%      = kandidat memuat \\boxed{...}                  -> H1
    indonesia%   = kandidat lolos is_indonesian()               -> H2
    skor_efektif = cakupan% x indonesia% / 100                  -> kriteria pemenang

Kenapa skor efektif, bukan cakupan mentah: `to_chatml(id_only=True)` membuang CoT yang
dominan Inggris, jadi solusi benar tapi berbahasa Inggris TETAP terbuang sebelum training.
Cakupan mentah menyesatkan kalau bahasanya salah.

Hipotesis yang diuji:
    H1 — spesialis matematika (OpenMath-Nemotron) gagal di format/bahasa Indonesia.
    H2 — R1-Distill paling parah language-mixing-nya.
Kedua hipotesis dilaporkan apa adanya, termasuk kalau SALAH.

Catatan lisensi: kalau `openmath-14b` menang, data CoT turunannya membawa kewajiban
atribusi CC-BY-4.0. `attribution_note()` menghasilkan kalimatnya untuk bab Metodologi.

CATATAN ANGKA HISTORIS: retensi 37,08% / cakupan 66,61% milik teacher 7B lama diukur
pada data yang lebih kotor (pra-v4). JANGAN dimasukkan ke tabel yang sama — pakai
`HISTORIS_7B` dan laporkan terpisah.

Usage:
    python -m src.cot_synthesis.bakeoff --self-check          # tanpa GPU, tanpa file
    python -m src.cot_synthesis.bakeoff --demo                # tabel contoh
    python -m src.cot_synthesis.bakeoff \\
        --candidates data/cot/bakeoff_qwen3-14b.jsonl \\
        --correct    data/cot/bakeoff_qwen3-14b_correct.jsonl \\
        --tag qwen3-14b --out data/eval/s1_teachers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .to_chatml import is_indonesian
from .utils import extract_boxed, read_jsonl

# Tiga kandidat tier 14B. Repo ID sudah diverifikasi ada via HF API (2026-08-08).
# Parameter sama -> perbedaan hasil berasal dari filosofi training, bukan ukuran model.
KANDIDAT_14B: dict[str, dict[str, str]] = {
    "qwen3-14b": {
        "model": "Qwen/Qwen3-14B",
        "lisensi": "apache-2.0",
        "filosofi": "generalis multibahasa",
    },
    "r1-distill-14b": {
        "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "lisensi": "mit",
        "filosofi": "distilasi reasoning R1",
    },
    "openmath-14b": {
        "model": "nvidia/OpenMath-Nemotron-14B",
        "lisensi": "cc-by-4.0",
        "filosofi": "spesialis matematika",
    },
}

# Angka teacher 7B dari paper lama. Diukur pada data pra-v4 yang lebih kotor ->
# TIDAK sebanding dengan tabel di atas. Dilaporkan terpisah sebagai konteks historis.
HISTORIS_7B: dict[str, float] = {"retensi_pct": 37.08, "cakupan_pct": 66.61}

# Lisensi yang menuntut atribusi pada data turunan.
_LISENSI_ATRIBUSI = {"cc-by-4.0"}

_KOLOM_META = ("model", "lisensi", "filosofi")


def hitung_metrik(kandidat: list[dict], kept: int, problems_covered: int) -> dict:
    """Lima metrik untuk satu teacher.

    kandidat         : baris mentah `bakeoff_<tag>.jsonl` (sebelum judge).
    kept             : jumlah kandidat yang lolos judge (`run_filter()['kept']`).
    problems_covered : soal unik yang punya minimal satu solusi benar.
    """
    n = len(kandidat)
    if n == 0:
        raise ValueError("daftar kandidat kosong -- generate belum jalan?")
    dicoba = len({r.get("id", "") for r in kandidat if r.get("id")})

    retensi = 100 * kept / n
    cakupan = (100 * problems_covered / dicoba) if dicoba else 0.0
    format_pct = 100 * sum(extract_boxed(r.get("text", "")) is not None for r in kandidat) / n
    indonesia_pct = 100 * sum(is_indonesian(r.get("text", "")) for r in kandidat) / n
    return {
        "kandidat": n,
        "benar": kept,
        "soal_dicoba": dicoba,
        "soal_terjawab": problems_covered,
        "retensi_pct": round(retensi, 2),
        "cakupan_pct": round(cakupan, 2),
        "format_pct": round(format_pct, 2),
        "indonesia_pct": round(indonesia_pct, 2),
        "skor_efektif": round(cakupan * indonesia_pct / 100, 2),
    }


def _meta(tag: str, meta: dict | None = None) -> dict:
    """Metadata model/lisensi/filosofi untuk satu tag teacher."""
    info = meta or KANDIDAT_14B.get(tag, {})
    return {k: info.get(k, "") for k in _KOLOM_META}


def nilai_teacher(tag: str, candidates_path: str | Path, kept: int,
                  problems_covered: int, *, meta: dict | None = None) -> dict:
    """Baca file kandidat satu teacher lalu hitung metriknya + metadata model/lisensi."""
    rows = read_jsonl(candidates_path)
    hasil = hitung_metrik(rows, kept, problems_covered)
    hasil.update(_meta(tag, meta))
    hasil["file_kandidat"] = str(candidates_path)
    return hasil


def peringkat(hasil: dict[str, dict]) -> list[tuple[str, dict]]:
    """Urutkan teacher dari skor efektif tertinggi. Tie-break: cakupan, lalu nama."""
    return sorted(hasil.items(),
                  key=lambda kv: (-kv[1]["skor_efektif"], -kv[1]["cakupan_pct"], kv[0]))


def pemenang(hasil: dict[str, dict]) -> str:
    """Tag teacher dengan skor efektif tertinggi."""
    if not hasil:
        raise ValueError("belum ada teacher yang dinilai")
    return peringkat(hasil)[0][0]


def dua_teratas(hasil: dict[str, dict]) -> list[str]:
    """Dua teacher teratas — dipakai issue #9 (union dua teacher)."""
    return [tag for tag, _ in peringkat(hasil)[:2]]


def uji_hipotesis(hasil: dict[str, dict]) -> dict[str, dict]:
    """Verdict H1 dan H2 dari angka, bukan dari dugaan.

    H1 benar bila `openmath-14b` punya format% ATAU indonesia% terendah.
    H2 benar bila `r1-distill-14b` punya indonesia% terendah.
    Verdict None = kandidat terkait tidak ada di hasil (tidak bisa disimpulkan).
    """
    def _terendah(metrik: str) -> str | None:
        return min(hasil, key=lambda t: hasil[t][metrik]) if hasil else None

    fmt_min, id_min = _terendah("format_pct"), _terendah("indonesia_pct")
    h1 = None if "openmath-14b" not in hasil else (
        fmt_min == "openmath-14b" or id_min == "openmath-14b")
    h2 = None if "r1-distill-14b" not in hasil else (id_min == "r1-distill-14b")
    return {
        "H1": {"klaim": "spesialis matematika gagal di format/bahasa Indonesia",
               "verdict": h1, "format_terendah": fmt_min, "indonesia_terendah": id_min},
        "H2": {"klaim": "R1-Distill paling parah language mixing",
               "verdict": h2, "indonesia_terendah": id_min},
    }


def attribution_note(tag_pemenang: str, hasil: dict[str, dict] | None = None) -> str:
    """Kalimat atribusi untuk bab Metodologi bila teacher pemenang berlisensi CC-BY.
    String kosong kalau lisensinya tidak menuntut atribusi."""
    info = (hasil or {}).get(tag_pemenang) or KANDIDAT_14B.get(tag_pemenang, {})
    lisensi = (info.get("lisensi") or "").lower()
    if lisensi not in _LISENSI_ATRIBUSI:
        return ""
    return (f"Data CoT pada penelitian ini disintesis oleh {info.get('model', tag_pemenang)} "
            f"yang dirilis di bawah lisensi {lisensi.upper()}. Sesuai ketentuan lisensi "
            f"tersebut, dataset turunan mencantumkan atribusi kepada pembuat model asal.")


def render_table(hasil: dict[str, dict], tag_pemenang: str | None = None) -> str:
    """Tabel markdown 3 baris x 5 metrik. Baris pemenang ditandai."""
    menang = tag_pemenang or (pemenang(hasil) if hasil else None)
    baris = ["| teacher | retensi% | cakupan% | format% | Indonesia% | skor efektif |",
             "|---|---|---|---|---|---|"]
    for tag, m in peringkat(hasil):
        tanda = " **(PEMENANG)**" if tag == menang else ""
        baris.append(f"| {tag}{tanda} | {m['retensi_pct']:.2f} | {m['cakupan_pct']:.2f} | "
                     f"{m['format_pct']:.2f} | {m['indonesia_pct']:.2f} | {m['skor_efektif']:.2f} |")
    return "\n".join(baris)


def render_historis() -> str:
    """Tabel terpisah untuk angka teacher 7B lama. JANGAN digabung ke tabel utama."""
    return ("| teacher (historis, data pra-v4) | retensi% | cakupan% |\n"
            "|---|---|---|\n"
            f"| r1-distill-7b | {HISTORIS_7B['retensi_pct']:.2f} | "
            f"{HISTORIS_7B['cakupan_pct']:.2f} |")


def ringkas(hasil: dict[str, dict]) -> dict:
    """Bundel lengkap S1: metrik, pemenang, dua teratas, verdict hipotesis, atribusi."""
    menang = pemenang(hasil)
    return {
        "teachers": hasil,
        "pemenang": menang,
        "dua_teratas": dua_teratas(hasil),
        "hipotesis": uji_hipotesis(hasil),
        "atribusi": attribution_note(menang, hasil),
        "historis_7b": HISTORIS_7B,
    }


def simpan(ringkasan: dict, path: str | Path) -> Path:
    """Tulis ringkasan S1 ke JSON (dipakai issue #6 dan #9)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ringkasan, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ── contoh sintetis untuk --self-check / --demo (CPU, tanpa file, tanpa GPU) ────────
def _baris_sintetis(n: int, n_boxed: int, n_indo: int) -> list[dict]:
    """n kandidat palsu: n_boxed di antaranya memuat \\boxed{}, n_indo berbahasa Indonesia."""
    keluar = []
    for i in range(n):
        teks = ("Langkah pertama kita hitung nilai dari persamaan tersebut. " if i < n_indo
                else "First we compute the value of the given equation so that the answer is. ")
        if i < n_boxed:
            teks += r"\boxed{42}"
        keluar.append({"id": f"p{i % max(n // 4, 1)}", "text": teks})
    return keluar


def _contoh() -> dict[str, dict]:
    """Hasil sintetis bergaya H1+H2 benar: spesialis matematika kalah karena bahasa."""
    profil = {
        # tag             (n, boxed, indo, kept, covered)
        "qwen3-14b":      (80, 76, 70, 44, 17),
        "r1-distill-14b": (80, 70, 22, 48, 18),
        "openmath-14b":   (80, 26, 30, 52, 19),
    }
    return {
        tag: {**hitung_metrik(_baris_sintetis(n, boxed, indo), kept, covered), **_meta(tag)}
        for tag, (n, boxed, indo, kept, covered) in profil.items()
    }


def self_check() -> None:
    """Uji aritmetika metrik + pemilihan pemenang. CPU, tanpa GPU, tanpa file eksternal."""
    kandidat = [
        {"id": "a", "text": r"Jadi nilai yang dicari adalah \boxed{7}"},   # ID + boxed
        {"id": "a", "text": r"Therefore the answer is \boxed{7}"},         # EN + boxed
        {"id": "b", "text": "Maka bilangan tersebut adalah tujuh"},        # ID tanpa boxed
        {"id": "c", "text": "The value we need to find is seven"},         # EN tanpa boxed
    ]
    m = hitung_metrik(kandidat, kept=2, problems_covered=2)
    assert m["kandidat"] == 4 and m["soal_dicoba"] == 3, m
    assert m["retensi_pct"] == 50.0, m
    assert m["cakupan_pct"] == round(200 / 3, 2), m
    assert m["format_pct"] == 50.0, m
    assert m["indonesia_pct"] == 50.0, m
    assert m["skor_efektif"] == round(200 / 3 * 50 / 100, 2), m

    kosong_ditolak = False
    try:
        hitung_metrik([], 0, 0)
    except ValueError:
        kosong_ditolak = True
    assert kosong_ditolak, "kandidat kosong seharusnya ValueError"

    contoh = _contoh()
    r = ringkas(contoh)
    assert r["pemenang"] == "qwen3-14b", r["pemenang"]
    assert r["dua_teratas"][0] == "qwen3-14b" and len(r["dua_teratas"]) == 2, r["dua_teratas"]
    assert r["hipotesis"]["H1"]["verdict"] is True, r["hipotesis"]
    assert r["hipotesis"]["H2"]["verdict"] is True, r["hipotesis"]
    assert attribution_note("openmath-14b") != "", "cc-by-4.0 wajib memicu atribusi"
    assert attribution_note("qwen3-14b") == "", "apache-2.0 tidak butuh atribusi"
    assert len(render_table(contoh).splitlines()) == 5, "tabel harus 3 baris + 2 header"

    print("self-check OK: metrik, peringkat, hipotesis, atribusi, tabel")


def demo() -> None:
    """Cetak tabel contoh supaya bentuk keluaran terlihat tanpa menjalankan GPU."""
    contoh = _contoh()
    r = ringkas(contoh)
    print(render_table(contoh, r["pemenang"]))
    print("\nPEMENANG    :", r["pemenang"])
    print("DUA TERATAS :", ", ".join(r["dua_teratas"]), " (dipakai union #9)")
    print("H1", r["hipotesis"]["H1"]["verdict"], "| H2", r["hipotesis"]["H2"]["verdict"])
    print("\nangka historis (JANGAN digabung ke tabel di atas):")
    print(render_historis())
    if r["atribusi"]:
        print("\nATRIBUSI:", r["atribusi"])
    print("\n(angka di atas SINTETIS -- hanya contoh format, bukan hasil pengukuran)")


def main() -> None:
    ap = argparse.ArgumentParser(description="S1 revisi: bake-off teacher CoT tier 14B")
    ap.add_argument("--self-check", action="store_true", help="uji modul di CPU, tanpa GPU")
    ap.add_argument("--demo", action="store_true", help="cetak tabel contoh (angka sintetis)")
    ap.add_argument("--candidates", help="jsonl kandidat satu teacher")
    ap.add_argument("--correct", help="jsonl hasil judge teacher yang sama")
    ap.add_argument("--tag", help="tag teacher, mis. qwen3-14b")
    ap.add_argument("--out", default="data/eval/s1_teachers.json")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return
    if args.demo:
        demo()
        return
    if not (args.candidates and args.correct and args.tag):
        ap.error("butuh --candidates, --correct, dan --tag (atau pakai --self-check/--demo)")

    benar = read_jsonl(args.correct)
    hasil = {args.tag: nilai_teacher(args.tag, args.candidates, kept=len(benar),
                                     problems_covered=len({r.get("id") for r in benar}))}
    r = ringkas(hasil)
    print(render_table(hasil, r["pemenang"]))
    print("\nringkasan ->", simpan(r, args.out))


if __name__ == "__main__":
    main()
