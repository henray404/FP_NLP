"""
Gabungkan beberapa JSONL (schema soal/cara/jawaban) -> 1 file dedup.

Dedup berdasarkan `soal`. Kalau ada soal kembar, simpan versi PALING LENGKAP:
prioritas = punya jawaban > punya cara > cara terpanjang.

Usage:
  python -m src.preprop.merge_and_dedup \
      --inputs data/merged_dataset.jsonl data/merged_dataset_un_cleanvlm_aimo5000.jsonl \
      --output data/dataset_dedup.jsonl
"""
import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def completeness(o: dict) -> tuple:
    """Skor kelengkapan; makin tinggi makin layak disimpan."""
    jaw = (o.get("jawaban") or "").strip()
    cara = (o.get("cara") or "").strip()
    return (1 if jaw else 0, 1 if cara else 0, len(cara))


def run(inputs: list[Path], output: Path) -> dict:
    all_rows = []
    per_file = {}
    for p in inputs:
        rows = load(p)
        per_file[p.name] = len(rows)
        all_rows.extend(rows)

    best: dict[str, dict] = {}
    for o in all_rows:
        soal = (o.get("soal") or "").strip()
        if not soal:
            continue
        rec = {"soal": soal,
               "cara": (o.get("cara") or "").strip(),
               "jawaban": (o.get("jawaban") or "").strip()}
        if soal not in best or completeness(rec) > completeness(best[soal]):
            best[soal] = rec

    deduped = list(best.values())
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    empty_jaw = sum(1 for r in deduped if not r["jawaban"])
    empty_cara = sum(1 for r in deduped if not r["cara"])
    return {
        "input_files": per_file,
        "total_baris_masuk": len(all_rows),
        "soal_unik_keluar": len(deduped),
        "duplikat_dibuang": len(all_rows) - len(deduped),
        "jawaban_masih_kosong": empty_jaw,
        "cara_masih_kosong": empty_cara,
        "output": str(output),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="data/dataset_dedup.jsonl")
    args = p.parse_args()
    stats = run([Path(x) for x in args.inputs], Path(args.output))
    for k, v in stats.items():
        print(f"{k:22}: {v}")
