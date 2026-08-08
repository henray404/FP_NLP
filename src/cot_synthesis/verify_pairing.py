"""
Verifikasi eksplisit bahwa cot.jsonl dan nocot.jsonl berdiri di atas HIMPUNAN SOAL IDENTIK.

Ini yang mengisolasi efek CoT di Skenario 3: base model, hyperparameter, dan daftar soal
dibuat sama persis; satu-satunya yang berbeda adalah bentuk target (penalaran penuh vs
hanya \\boxed{}). Kalau himpunan soalnya berbeda sedikit saja, selisih akurasi antar-arm
tidak lagi bisa diklaim sebagai efek CoT.

to_chatml.py memang menulis kedua file berpasangan per baris, tapi properti itu harus
DIUJI dari file hasilnya -- bukan diasumsikan dari kode yang menghasilkannya (file bisa
ditimpa, digabung dari beberapa shard, atau difilter ulang belakangan).

Pemeriksaan:
1. Jumlah baris sama.
2. Pasangan per indeks: soal ke-i di cot.jsonl == soal ke-i di nocot.jsonl.
3. Himpunan soal identik (menangkap kasus urutan teracak tapi isi sama).
4. Multiset soal identik (duplikat pun harus sama banyak).
5. Target nocot berisi HANYA \\boxed{...}; target cot tidak boleh kosong.

Keluar dengan status 1 kalau ada yang gagal, supaya bisa dipakai sebagai gerbang di notebook.

Pemakaian:
    python -m src.cot_synthesis.verify_pairing
    python -m src.cot_synthesis.verify_pairing data/sft/train/cot.jsonl data/sft/train/nocot.jsonl
    python -m src.cot_synthesis.verify_pairing --self-check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .prompt_wrap import unwrap, wrap
from .utils import read_jsonl

# Target arm nocot yang sah: spasi opsional, lalu \boxed{...}, tanpa teks lain.
_ONLY_BOXED_RE = re.compile(r"^\s*\\boxed\{.*\}\s*$", re.DOTALL)

MAX_CONTOH = 3  # jumlah contoh ketidakcocokan yang dicetak, biar output tetap terbaca


def _pesan(row: dict, peran: str) -> str:
    """Isi pesan pertama dengan `role` tertentu dari satu baris ChatML."""
    for m in row.get("messages", []):
        if m.get("role") == peran:
            return m.get("content", "")
    return ""


def _soal_list(rows: list[dict], mode: str) -> list[str]:
    return [unwrap(_pesan(r, "user"), mode) for r in rows]


def run(cot_path: str | Path, nocot_path: str | Path) -> dict:
    """Bandingkan kedua file. `ok=True` berarti semua pemeriksaan lolos."""
    cot_rows = read_jsonl(cot_path)
    nocot_rows = read_jsonl(nocot_path)

    cot_soal = _soal_list(cot_rows, "cot")
    nocot_soal = _soal_list(nocot_rows, "nocot")

    masalah: list[str] = []
    if len(cot_rows) != len(nocot_rows):
        masalah.append(f"jumlah baris beda: cot={len(cot_rows)} nocot={len(nocot_rows)}")

    beda_indeks = [i for i, (a, b) in enumerate(zip(cot_soal, nocot_soal)) if a != b]
    if beda_indeks:
        contoh = ", ".join(str(i) for i in beda_indeks[:MAX_CONTOH])
        masalah.append(f"{len(beda_indeks)} baris tidak berpasangan pada indeks: {contoh} ...")

    set_cot, set_nocot = set(cot_soal), set(nocot_soal)
    hanya_cot = set_cot - set_nocot
    hanya_nocot = set_nocot - set_cot
    if hanya_cot or hanya_nocot:
        masalah.append(
            f"himpunan soal beda: hanya-cot={len(hanya_cot)} hanya-nocot={len(hanya_nocot)}")
    if Counter(cot_soal) != Counter(nocot_soal):
        masalah.append("multiset soal beda (jumlah duplikat per soal tidak sama)")

    nocot_bukan_boxed = sum(
        1 for r in nocot_rows if not _ONLY_BOXED_RE.match(_pesan(r, "assistant")))
    if nocot_bukan_boxed:
        masalah.append(f"{nocot_bukan_boxed} target nocot bukan hanya \\boxed{{...}}")
    cot_kosong = sum(1 for r in cot_rows if not _pesan(r, "assistant").strip())
    if cot_kosong:
        masalah.append(f"{cot_kosong} target cot kosong")

    return {
        "ok": not masalah,
        "baris_cot": len(cot_rows),
        "baris_nocot": len(nocot_rows),
        "soal_unik_cot": len(set_cot),
        "soal_unik_nocot": len(set_nocot),
        "soal_duplikat_cot": len(cot_soal) - len(set_cot),
        "pasangan_indeks_tidak_cocok": len(beda_indeks),
        "hanya_di_cot": len(hanya_cot),
        "hanya_di_nocot": len(hanya_nocot),
        "nocot_bukan_boxed": nocot_bukan_boxed,
        "cot_kosong": cot_kosong,
        "masalah": masalah,
    }


# -------------------------------
# Self-check (CPU, tanpa GPU, tanpa data nyata)
# -------------------------------

def _tulis_chatml(rows: list[tuple[str, str]], mode: str, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for soal, target in rows:
            f.write(json.dumps({"messages": [
                {"role": "user", "content": wrap(soal, mode)},
                {"role": "assistant", "content": target},
            ]}, ensure_ascii=False) + "\n")


def self_check() -> bool:
    """Uji kasus lolos dan tiga kasus gagal pada file sintetis di direktori sementara."""
    lolos = True
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        cot = [("Berapa 2+5?", "Jumlahkan: 2+5=7. Jadi \\boxed{7}"),
               ("Nilai x jika x-3=4?", "x = 4+3 = 7 sehingga \\boxed{7}")]
        nocot = [("Berapa 2+5?", "\\boxed{7}"), ("Nilai x jika x-3=4?", "\\boxed{7}")]

        _tulis_chatml(cot, "cot", d / "cot.jsonl")
        _tulis_chatml(nocot, "nocot", d / "nocot.jsonl")
        baik = run(d / "cot.jsonl", d / "nocot.jsonl")
        print(f"[1] pasangan identik  -> ok={baik['ok']} baris={baik['baris_cot']}")
        if not baik["ok"]:
            lolos = False

        rusak = [("Berapa 2+6?", "\\boxed{8}"), ("Nilai x jika x-3=4?", "\\boxed{7}")]
        _tulis_chatml(rusak, "nocot", d / "nocot_rusak.jsonl")
        buruk = run(d / "cot.jsonl", d / "nocot_rusak.jsonl")
        print(f"[2] satu soal diganti -> ok={buruk['ok']} masalah={buruk['masalah']}")
        if buruk["ok"]:
            lolos = False

        bocor = [("Berapa 2+5?", "2+5=7 jadi \\boxed{7}"), ("Nilai x jika x-3=4?", "\\boxed{7}")]
        _tulis_chatml(bocor, "nocot", d / "nocot_bocor.jsonl")
        b = run(d / "cot.jsonl", d / "nocot_bocor.jsonl")
        print(f"[3] langkah bocor     -> ok={b['ok']} nocot_bukan_boxed={b['nocot_bukan_boxed']}")
        if b["ok"] or b["nocot_bukan_boxed"] != 1:
            lolos = False

        s = "Berapa nilai $x$ jika $2x=10$?"
        if unwrap(wrap(s, "cot")) != s or unwrap(wrap(s, "nocot")) != s:
            print("[4] unwrap GAGAL mengembalikan soal asli")
            lolos = False
        else:
            print("[4] unwrap bolak-balik -> ok")

    print("SELF-CHECK:", "LOLOS" if lolos else "GAGAL")
    return lolos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Verifikasi cot.jsonl dan nocot.jsonl memakai himpunan soal identik")
    ap.add_argument("cot", nargs="?", default="data/sft/train/cot.jsonl")
    ap.add_argument("nocot", nargs="?", default="data/sft/train/nocot.jsonl")
    ap.add_argument("--self-check", action="store_true",
                    help="uji modul ini pada data sintetis (CPU, tanpa GPU)")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if self_check() else 1)

    hasil = run(args.cot, args.nocot)
    for k, v in hasil.items():
        if k != "masalah":
            print(f"{k:30} {v}")
    if hasil["ok"]:
        print("\nOK: himpunan soal cot.jsonl dan nocot.jsonl IDENTIK dan berpasangan per baris.")
        sys.exit(0)
    print("\nGAGAL:")
    for m in hasil["masalah"]:
        print(f"  - {m}")
    sys.exit(1)


if __name__ == "__main__":
    main()
