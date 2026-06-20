"""
SKENARIO 1 — pilih teacher CoT terbaik (gemma vs deepseek vs ernie).

Metrik: seberapa sedikit kandidat HILANG saat difilter judge (candidates -> correct).
Makin sedikit yang berkurang = teacher makin sering benar = makin bagus. Dilaporkan
dalam persen:

    retention% = correct / candidates * 100      (makin TINGGI makin bagus)
    reduction% = 100 - retention%                (makin RENDAH makin bagus)

Dihitung dua level:
- baris   (row): semua kandidat (N/soal) vs kandidat yang lolos    -> acceptance rate.
- soal (problem): soal unik yang DICOBA vs soal unik yang TERJAWAB benar -> coverage.

Pemenang = retention baris tertinggi (tie-break: coverage soal, lalu nama).

Tanpa GPU, tanpa re-judge — cuma baca file `cot_*`/`correct_*` yang sudah ada.

Usage:
    python -m src.cot_synthesis.compare_teachers \
        --teacher gemma   data/cot/easy_gemma/cot_*.jsonl   data/cot/easy_gemma/correct_*.jsonl \
        --teacher deepseek data/cot/easy/cot_*.jsonl        data/cot/easy/correct_*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from .utils import read_jsonl


def _count(path: str | Path) -> tuple[int, set[str]]:
    """Return (jumlah baris, set id soal unik) dari satu jsonl kandidat/correct."""
    rows = read_jsonl(path)
    return len(rows), {r.get("id", "") for r in rows if r.get("id")}


def _resolve(patterns: list[str]) -> list[str]:
    """Expand glob (boleh banyak file per teacher, mis. per-dataset) -> daftar path."""
    out: list[str] = []
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        out.extend(hits or ([pat] if Path(pat).exists() else []))
    return sorted(set(out))


def score_teacher(candidate_paths: list[str], correct_paths: list[str]) -> dict:
    """Hitung metrik retensi untuk satu teacher dari file kandidat + correct-nya."""
    cand_rows, cand_ids = 0, set()
    for p in candidate_paths:
        n, ids = _count(p)
        cand_rows += n
        cand_ids |= ids
    corr_rows, corr_ids = 0, set()
    for p in correct_paths:
        n, ids = _count(p)
        corr_rows += n
        corr_ids |= ids

    retention = (corr_rows / cand_rows * 100) if cand_rows else 0.0
    coverage = (len(corr_ids) / len(cand_ids) * 100) if cand_ids else 0.0
    return {
        "candidates_rows": cand_rows,
        "correct_rows": corr_rows,
        "retention_pct": round(retention, 2),
        "reduction_pct": round(100 - retention, 2),
        "problems_tried": len(cand_ids),
        "problems_solved": len(corr_ids),
        "coverage_pct": round(coverage, 2),
        "candidate_paths": candidate_paths,
        "correct_paths": correct_paths,
    }


def compare(teachers: dict[str, dict]) -> dict:
    """teachers = {nama: {"candidates": [glob...], "correct": [glob...]}}.
    Return metrik per-teacher + pemenang (retention tertinggi)."""
    metrics: dict[str, dict] = {}
    for name, spec in teachers.items():
        cands = _resolve(spec.get("candidates", []))
        corrs = _resolve(spec.get("correct", []))
        if not cands or not corrs:
            print(f"WARNING: {name}: file kurang (candidates={len(cands)}, correct={len(corrs)}) -> skip")
            continue
        metrics[name] = score_teacher(cands, corrs)

    if not metrics:
        raise RuntimeError("tidak ada teacher dengan file lengkap untuk dibandingkan")

    winner = max(metrics, key=lambda n: (metrics[n]["retention_pct"],
                                         metrics[n]["coverage_pct"], n))
    return {"teachers": metrics, "winner": winner}


def render_table(result: dict) -> str:
    """Tabel markdown perbandingan teacher (S1)."""
    rows = ["| teacher | candidates | correct | retention% | reduction% | coverage% |",
            "|---|---|---|---|---|---|"]
    for name, m in sorted(result["teachers"].items(),
                          key=lambda kv: kv[1]["retention_pct"], reverse=True):
        mark = " **(WINNER)**" if name == result["winner"] else ""
        rows.append(f"| {name}{mark} | {m['candidates_rows']} | {m['correct_rows']} | "
                    f"{m['retention_pct']:.2f} | {m['reduction_pct']:.2f} | {m['coverage_pct']:.2f} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="S1: pilih teacher CoT terbaik (retensi tertinggi)")
    ap.add_argument("--teacher", nargs=3, action="append", metavar=("NAME", "CAND_GLOB", "CORRECT_GLOB"),
                    required=True, help="nama teacher + glob kandidat + glob correct (ulang per teacher)")
    ap.add_argument("--out", default="data/eval/s1_teachers.json")
    args = ap.parse_args()

    teachers = {name: {"candidates": [cg], "correct": [rg]} for name, cg, rg in args.teacher}
    result = compare(teachers)

    print(render_table(result))
    print(f"\nWINNER: {result['winner']}  -> pakai sft/ dari teacher ini buat training")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("summary ->", args.out)


if __name__ == "__main__":
    main()
