"""
Benchmark "level sekolah" (holdout v3) dari soal UN bersih.

Sumber: data/un_pdfs/dataset_sma.jsonl + clean_vlm.jsonl (soal Ujian Nasional / SMA).
Saring HANYA yang bisa dinilai adil & self-contained:
  - jawaban numerik (bukan huruf pilihan-ganda; satuan dinormalisasi saat grading)
  - TIDAK butuh tabel/gambar yang hilang ("tabel di atas", "perhatikan gambar", dst)
  - panjang soal wajar

Ini DEFENSIBLE: subset soal sekolah Indonesia asli yang terisi & terbaca — bukan
membuang soal sulit, tapi membuang soal yang datanya rusak/tak-lengkap.

Usage:
  python -m src.eval.build_un_holdout --n 300 --seed 42
"""
import argparse
import json
import random
import re
from pathlib import Path

UN_FILES = ["data/un_pdfs/dataset_sma.jsonl", "data/un_pdfs/clean_vlm.jsonl"]
MC = re.compile(r"^[A-E]\.?$")
NEEDS = re.compile(r"(tabel|gambar|grafik|diagram|di atas|berikut ini|perhatikan|kurva|histogram)", re.I)
NUM = re.compile(r"^-?\d+([.,]\d+)?$")
_UNIT = re.compile(r"(meter|cm|km|kg|gram|liter|rp|juta|ribu|orang|cara|buah|detik|menit|jam|°|satuan|\s)", re.I)


def clean_num_gold(j: str) -> bool:
    core = _UNIT.sub("", (j or "")).strip().strip(".").strip("$")
    return bool(NUM.match(core))


def is_clean(r: dict) -> bool:
    j = (r.get("jawaban") or "").strip()
    s = r.get("soal") or ""
    return (bool(j) and not MC.match(j) and not NEEDS.search(s)
            and clean_num_gold(j) and 20 < len(s) < 600)


def run(n: int, seed: int, output: Path) -> dict:
    rows = []
    for p in UN_FILES:
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    clean = [r for r in rows if is_clean(r)]
    # dedup by soal
    seen, uniq = set(), []
    for r in clean:
        s = r["soal"].strip()
        if s not in seen:
            seen.add(s)
            uniq.append({"soal": r["soal"], "cara": r.get("cara", ""),
                         "jawaban": r["jawaban"], "source": "UN"})
    random.Random(seed).shuffle(uniq)
    holdout = uniq[:n]

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"un_total": len(rows), "clean_unik": len(uniq),
            "holdout_v3": len(holdout), "output": str(output)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="data/eval/holdout_v3_un.jsonl")
    args = p.parse_args()
    for k, v in run(args.n, args.seed, Path(args.output)).items():
        print(f"{k:14}: {v}")
