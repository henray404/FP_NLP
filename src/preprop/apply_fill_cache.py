"""
Terapkan hasil fill_cache.jsonl (LLM-generated) ke dataset_dedup.jsonl.

Kebijakan "pakai yang dapet aja": hanya entri cache yang BERHASIL (jawaban tidak
kosong) yang dipakai untuk menimpa baris asli. Baris yang gagal di-fill (jawaban
masih kosong) dibiarkan apa adanya.

Usage:
  python -m src.preprop.apply_fill_cache \
      --base data/dataset_dedup.jsonl \
      --cache data/fill_cache.jsonl \
      --output data/dataset_filled.jsonl
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


def run(base: Path, cache: Path, output: Path) -> dict:
    rows = load(base)

    # cache -> map soal->fill, hanya yang jawabannya terisi (last write wins)
    good: dict[str, dict] = {}
    cache_total = 0
    if cache.exists():
        for r in load(cache):
            cache_total += 1
            if (r.get("jawaban") or "").strip():
                good[r["soal"]] = r

    applied = 0
    with open(output, "w", encoding="utf-8") as out:
        for r in rows:
            f = good.get(r["soal"])
            if f is not None:
                rec = {"soal": f["soal"], "cara": f["cara"], "jawaban": f["jawaban"]}
                applied += 1
            else:
                rec = {"soal": r["soal"],
                       "cara": r.get("cara", ""),
                       "jawaban": r.get("jawaban", "")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    final = load(output)
    with_ans = sum(1 for r in final if (r.get("jawaban") or "").strip())
    return {
        "total_soal": len(rows),
        "cache_entries": cache_total,
        "fill_dipakai (ada jawaban)": applied,
        "PUNYA JAWABAN setelah fill": f"{with_ans} ({with_ans/len(rows)*100:.1f}%)",
        "masih kosong": len(rows) - with_ans,
        "output": str(output),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="data/dataset_dedup.jsonl")
    p.add_argument("--cache", default="data/fill_cache.jsonl")
    p.add_argument("--output", default="data/dataset_filled.jsonl")
    args = p.parse_args()
    stats = run(Path(args.base), Path(args.cache), Path(args.output))
    for k, v in stats.items():
        print(f"{k:28}: {v}")
