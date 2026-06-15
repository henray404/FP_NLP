"""
Tag sumber tiap soal: UN (sekolah) vs AIMO (olympiad).

Tag sumber hilang saat merge. Direcover dari provenance asli + cross-check struktur:
  - AIMO pasti  = soal di merged_dataset.jsonl TAPI tidak di file _aimo5000 (= AIMO[5000:20000])
  - UN          = soal yang ada di file sumber UN (dataset_sma.jsonl, clean_vlm.jsonl)
  - sisanya di _aimo5000 tapi bukan UN = AIMO[0:5000]
  - tak cocok semuanya (teks berubah krn normalisasi LaTeX) = unknown

Logika tag tiap soal:
    in AIMO_pasti        -> AIMO
    elif in UN_SET       -> UN
    elif in SUB (aimo5k) -> AIMO         # di B tapi bukan UN => AIMO awal
    else                 -> unknown

Usage:
  python -m src.preprop.tag_source \
      --un data/un_pdfs/dataset_sma.jsonl data/un_pdfs/clean_vlm.jsonl \
      --target data/eval/holdout_v2.jsonl \
      --merged data/merged_dataset.jsonl \
      --sub data/merged_dataset_un_cleanvlm_aimo5000.jsonl
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def soal_set(path: Path) -> set[str]:
    s = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                s.add((json.loads(line).get("soal") or "").strip())
    return s


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def classify(soal: str, aimo_certain: set, un_set: set, sub_set: set) -> str:
    soal = (soal or "").strip()
    if soal in aimo_certain:
        return "AIMO"
    if soal in un_set:
        return "UN"
    if soal in sub_set:
        return "AIMO"
    return "unknown"


def run(un_files: list[Path], target: Path, merged: Path, sub: Path,
        output: Path | None) -> dict:
    un_set = set()
    for p in un_files:
        un_set |= soal_set(p)
    merged_set = soal_set(merged)
    sub_set = soal_set(sub)
    aimo_certain = merged_set - sub_set      # AIMO[5000:20000], provenance pasti

    rows = load_rows(target)
    counts = Counter()
    conflicts = 0
    for r in rows:
        src = classify(r.get("soal", ""), aimo_certain, un_set, sub_set)
        # cross-check: kalau ditandai UN tapi sebenarnya di aimo_certain -> konflik (mustahil di sini
        # karena urutan if, tapi cek eksplisit untuk keamanan)
        if src == "UN" and (r.get("soal") or "").strip() in aimo_certain:
            conflicts += 1
            src = "AIMO"
        r["source"] = src
        counts[src] += 1

    out = output or target
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(rows)
    return {
        "target": str(target),
        "total": n,
        "UN": counts["UN"],
        "AIMO": counts["AIMO"],
        "unknown": counts["unknown"],
        "unknown_pct": round(counts["unknown"] / n * 100, 1) if n else 0,
        "konflik_cross_check": conflicts,
        "un_set_size": len(un_set),
        "aimo_certain_size": len(aimo_certain),
        "output": str(out),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--un", nargs="+", required=True, help="file sumber UN (.jsonl)")
    p.add_argument("--target", required=True, help="jsonl yang di-tag (mis. holdout_v2)")
    p.add_argument("--merged", default="data/merged_dataset.jsonl")
    p.add_argument("--sub", default="data/merged_dataset_un_cleanvlm_aimo5000.jsonl")
    p.add_argument("--output", default=None, help="default: timpa --target")
    args = p.parse_args()
    stats = run([Path(x) for x in args.un], Path(args.target),
                Path(args.merged), Path(args.sub),
                Path(args.output) if args.output else None)
    for k, v in stats.items():
        print(f"{k:22}: {v}")
