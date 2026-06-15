"""
Holdout v2: buang HANYA yang objektif rusak dari holdout v1.
Soal SUSAH tetap dipertahankan (justru itu yang mau diuji).

Dibuang:
  - gold tak-gradeable (bukan angka & tak bisa diparse sympy -> mustahil dinilai)
  - LaTeX patah (delimiter \\( \\) \\[ \\] { } $ tak seimbang -> soal tak terbaca)

Ditambahkan field `hard` (heuristik penanda topik lanjut) UNTUK PELAPORAN per-strata,
BUKAN untuk membuang. Catatan: heuristik ini lemah (under-detect), jadi treat sebagai
indikatif, bukan label pasti.

Usage:
  python -m src.eval.clean_holdout --input data/eval/holdout.jsonl \
      --output data/eval/holdout_v2.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from src.eval.answer_check import _to_expr

_INT = re.compile(r"^-?\d+$")
_NUM = re.compile(r"^-?\d+([./]\d+)?$")
_HARD = re.compile(
    r"(integral|∫|\\int|analiti|akar kesatuan|persamaan fungsi|modulo|kombinator|"
    r"deret tak hingga|\\sum|\\prod|kompleks|matriks|turunan|\\lim)",
    re.IGNORECASE,
)


def gold_gradeable(g: str) -> bool:
    core = (g or "").strip().strip(".").strip("$").strip()
    if _INT.match(core) or _NUM.match(core):
        return True
    return _to_expr(g) is not None


def latex_ok(s: str) -> bool:
    s = s or ""
    return (s.count(r"\(") == s.count(r"\)")
            and s.count(r"\[") == s.count(r"\]")
            and s.count("{") == s.count("}")
            and s.count("$") % 2 == 0)


def run(input_path: Path, output_path: Path) -> dict:
    rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
    kept, bad_gold, bad_latex = [], 0, 0
    for r in rows:
        bg = not gold_gradeable(r.get("jawaban", ""))
        bl = not latex_ok(r.get("soal", ""))
        if bg:
            bad_gold += 1
        if bl:
            bad_latex += 1
        if not bg and not bl:
            r = dict(r)
            r["hard"] = bool(_HARD.search(r.get("soal", "")))
            kept.append(r)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "input": len(rows),
        "dibuang_gold_sampah": bad_gold,
        "dibuang_latex_patah": bad_latex,
        "holdout_v2": len(kept),
        "tag_hard": sum(r["hard"] for r in kept),
        "output": str(output_path),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/eval/holdout.jsonl")
    p.add_argument("--output", default="data/eval/holdout_v2.jsonl")
    args = p.parse_args()
    for k, v in run(Path(args.input), Path(args.output)).items():
        print(f"{k:24}: {v}")
