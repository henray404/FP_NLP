"""
Ukur distribusi panjang TOKEN dataset SFT dengan tokenizer Qwen -> tetapkan `max_seq_length`.

Kenapa harus diukur ulang tiap ganti teacher: angka lama (p50 683, p99 2001, maks 3233)
diukur dari keluaran `DeepSeek-R1-Distill-Qwen-7B` pada data lama. Teacher 14B -- apalagi
Qwen3-14B yang punya thinking mode -- bisa jauh lebih panjang. `max_seq_length` terlalu kecil
memotong ekor penalaran (contoh terpotong justru mengajari model berhenti sebelum \\boxed{});
terlalu besar memboroskan VRAM dan waktu tanpa manfaat.

Yang dihitung, untuk tiap file ChatML:
- `asisten` : panjang target saja (yang benar-benar dilatih saat train_on_responses_only=true)
- `total`   : prompt + target (yang harus muat di dalam max_seq_length)

Rekomendasi `max_seq_length` = anak tangga 2^k terkecil yang menampung p99 `total`. p99, bukan
maks: satu-dua outlier ekstrem tidak sepadan dengan menggandakan biaya seluruh training --
outlier itu terpotong, dan jumlahnya dilaporkan sebagai `terpotong_pada_rekomendasi`.

Tokenizer: default `Qwen/Qwen2.5-3B` (student terbesar, lihat configs/cot_3b.yaml). Kalau
`transformers` tidak terpasang atau tokenizer gagal diunduh, modul jatuh ke estimator kasar
char/CHARS_PER_TOKEN dan MENANDAI hasilnya `estimasi=True` -- angka estimasi tidak boleh
dipakai untuk menetapkan max_seq_length final, hanya untuk menguji jalur kodenya.

Pemakaian:
    python -m src.cot_synthesis.token_stats
    python -m src.cot_synthesis.token_stats data/sft/train/cot.jsonl --tokenizer Qwen/Qwen2.5-3B
    python -m src.cot_synthesis.token_stats --out data/sft/train/token_stats.json
    python -m src.cot_synthesis.token_stats --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .utils import read_jsonl

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B"
# Estimator cadangan: ~3,5 karakter per token untuk teks Indonesia + LaTeX pada BPE Qwen.
CHARS_PER_TOKEN = 3.5
# Anak tangga max_seq_length yang lazim dipakai konfigurasi training.
LADDER = (1024, 2048, 4096, 8192, 16384, 32768)
PERSENTIL = (50, 75, 90, 95, 99)


def _pesan(row: dict, peran: str) -> str:
    for m in row.get("messages", []):
        if m.get("role") == peran:
            return m.get("content", "")
    return ""


def buat_penghitung(tokenizer_name: str = DEFAULT_TOKENIZER, *, estimate: bool = False):
    """Kembalikan (fungsi hitung token, estimasi: bool, nama yang benar-benar dipakai)."""
    if not estimate:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            return (lambda s: len(tok(s, add_special_tokens=False)["input_ids"]),
                    False, tokenizer_name)
        except Exception as e:  # noqa: BLE001 - transformers/HF melempar tipe beragam
            print(f"[peringatan] tokenizer {tokenizer_name} gagal dimuat "
                  f"({type(e).__name__}: {e}); pakai estimator char/{CHARS_PER_TOKEN}",
                  file=sys.stderr)
    return (lambda s: int(len(s) / CHARS_PER_TOKEN) + 1), True, f"estimasi(char/{CHARS_PER_TOKEN})"


def _persentil(nilai: list[int], p: float) -> int:
    """Persentil nearest-rank. Tanpa numpy, dan tanpa interpolasi token pecahan."""
    if not nilai:
        return 0
    urut = sorted(nilai)
    idx = min(len(urut) - 1, max(0, int(round(p / 100 * len(urut) + 0.5)) - 1))
    return urut[idx]


def ringkas(nilai: list[int]) -> dict:
    if not nilai:
        return {"n": 0}
    out = {"n": len(nilai), "min": min(nilai), "mean": round(sum(nilai) / len(nilai), 1)}
    for p in PERSENTIL:
        out[f"p{p}"] = _persentil(nilai, p)
    out["max"] = max(nilai)
    return out


def rekomendasi_max_seq(p99_total: int) -> int:
    for k in LADDER:
        if k >= p99_total:
            return k
    return LADDER[-1]


def run(path: str | Path, *, tokenizer: str = DEFAULT_TOKENIZER, estimate: bool = False,
        out_json: str | Path | None = None) -> dict:
    """Hitung statistik panjang token satu file ChatML."""
    rows = read_jsonl(path)
    hitung, estimasi, nama = buat_penghitung(tokenizer, estimate=estimate)

    panjang_asisten, panjang_total = [], []
    for r in rows:
        a = hitung(_pesan(r, "assistant"))
        u = hitung(_pesan(r, "user"))
        panjang_asisten.append(a)
        panjang_total.append(a + u)

    st_total = ringkas(panjang_total)
    rekom = rekomendasi_max_seq(st_total.get("p99", 0))
    hasil = {
        "file": str(path),
        "tokenizer": nama,
        "estimasi": estimasi,
        "baris": len(rows),
        "asisten": ringkas(panjang_asisten),
        "total": st_total,
        "max_seq_length_rekomendasi": rekom,
        "terpotong_pada_rekomendasi": sum(1 for t in panjang_total if t > rekom),
    }
    if out_json:
        out_json = Path(out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(hasil, ensure_ascii=False, indent=2), encoding="utf-8")
    return hasil


def cetak(hasil: dict) -> None:
    tag = "  [ESTIMASI -- JANGAN dipakai sebagai angka final]" if hasil["estimasi"] else ""
    print(f"\nfile      : {hasil['file']}")
    print(f"tokenizer : {hasil['tokenizer']}{tag}")
    print(f"baris     : {hasil['baris']}")
    kolom = ["n", "min", "mean"] + [f"p{p}" for p in PERSENTIL] + ["max"]
    print("\n| bagian  | " + " | ".join(kolom) + " |")
    print("|---" * (len(kolom) + 1) + "|")
    for bagian in ("asisten", "total"):
        s = hasil[bagian]
        print(f"| {bagian:7} | " + " | ".join(str(s.get(k, "-")) for k in kolom) + " |")
    print(f"\nmax_seq_length rekomendasi   : {hasil['max_seq_length_rekomendasi']}")
    print(f"contoh terpotong di angka itu : {hasil['terpotong_pada_rekomendasi']} "
          f"({100 * hasil['terpotong_pada_rekomendasi'] / max(hasil['baris'], 1):.2f}%)")


# -------------------------------
# Self-check (CPU, tanpa GPU, tanpa unduh bobot)
# -------------------------------

def self_check() -> bool:
    lolos = True
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "cot.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for i in range(1, 51):
                f.write(json.dumps({"messages": [
                    {"role": "user", "content": "Soal " + "x" * (10 * i)},
                    {"role": "assistant", "content": "Langkah " + "y" * (100 * i) + " \\boxed{7}"},
                ]}, ensure_ascii=False) + "\n")

        h = run(p, estimate=True, out_json=Path(tmp) / "stats.json")
        cetak(h)
        if h["baris"] != 50 or h["asisten"]["max"] <= h["asisten"]["p50"]:
            print("[1] GAGAL: ringkasan panjang tidak monoton"); lolos = False
        else:
            print("[1] ringkasan panjang -> ok")

        if h["total"]["p99"] < h["total"]["p50"] or h["total"]["max"] < h["total"]["p99"]:
            print("[2] GAGAL: urutan persentil salah"); lolos = False
        else:
            print("[2] urutan persentil -> ok")

        if h["max_seq_length_rekomendasi"] < h["total"]["p99"]:
            print("[3] GAGAL: rekomendasi lebih kecil dari p99"); lolos = False
        else:
            print("[3] rekomendasi >= p99 -> ok")

        if not (Path(tmp) / "stats.json").exists():
            print("[4] GAGAL: laporan JSON tidak tertulis"); lolos = False
        else:
            print("[4] laporan JSON -> ok")

        if rekomendasi_max_seq(2001) != 2048 or rekomendasi_max_seq(5000) != 8192:
            print("[5] GAGAL: anak tangga rekomendasi salah"); lolos = False
        else:
            print("[5] anak tangga rekomendasi -> ok")

    print("SELF-CHECK:", "LOLOS" if lolos else "GAGAL")
    return lolos


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Statistik panjang token dataset SFT + rekomendasi max_seq_length")
    ap.add_argument("input", nargs="*",
                    help="file ChatML (default: cot.jsonl dan nocot.jsonl di data/sft/train)")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--estimate", action="store_true",
                    help="lewati transformers, pakai estimator char/token (uji jalur kode saja)")
    ap.add_argument("--out", default=None,
                    help="tulis laporan JSON file pertama ke path ini")
    ap.add_argument("--self-check", action="store_true",
                    help="uji modul ini pada data sintetis (CPU, tanpa GPU)")
    args = ap.parse_args()

    if args.self_check:
        sys.exit(0 if self_check() else 1)

    inputs = args.input or ["data/sft/train/cot.jsonl", "data/sft/train/nocot.jsonl"]
    terproses = 0
    for i, path in enumerate(inputs):
        if not Path(path).exists():
            print(f"[lewati] {path} tidak ada", file=sys.stderr)
            continue
        out = args.out if (args.out and i == 0) else None
        cetak(run(path, tokenizer=args.tokenizer, estimate=args.estimate, out_json=out))
        terproses += 1
    if not terproses:
        sys.exit(1)


if __name__ == "__main__":
    main()
