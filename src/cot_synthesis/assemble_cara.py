"""
Rakit field `cara` untuk soal yang awalnya tak punya solusi (NumGLUE) — Task 1, langkah 5.

Metode (paling bagus utk kualitas+coverage+biaya, faithful ke AIMO-2):
  1. VERIFIED  : ambil solusi dari kandidat teacher yang sudah lolos judge benar
                 (data/cot/correct.jsonl, hasil generate.py -> filter_solutions.py).
                 -> reasoning genuine, terbukti sampai ke gold `jawaban`.
  2. HINTED    : soal yang TIDAK punya kandidat benar -> fallback answer-conditioned:
                 teacher diberi gold `jawaban`, diminta menuliskan langkah yang sampai ke situ.
                 -> coverage 100% (semua soal akhirnya punya `cara`).

Tiap baris ditandai `cara_source` ("verified"/"hinted") demi keterlacakan di laporan.
Resumable: baris yang sudah ada di output dilewati.

Input :
  --problems  data/NumGlue/numglue_{split}_id.jsonl    {soal, jawaban, source}
  --correct   data/cot/numglue_{split}_correct.jsonl   {id, soal, jawaban, text, ...}
Output:
  data/NumGlue/numglue_{split}_final.jsonl             {soal, cara, jawaban, source, cara_source}

Usage:
    python -m src.cot_synthesis.assemble_cara \
        --problems data/NumGlue/numglue_test_id.jsonl \
        --correct  data/cot/numglue_test_correct.jsonl \
        --output   data/NumGlue/numglue_test_final.jsonl
    # tanpa --no-hinted, soal tak-terpecahkan diisi via teacher (butuh GROQ_API_KEY / --backend vllm)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .utils import (get_jawaban, get_soal, openai_client, problem_id, read_jsonl,
                    with_retry)

DEFAULT_API_MODEL = "llama-3.3-70b-versatile"

HINT_PROMPT = (
    "Selesaikan soal matematika berikut. Jawaban akhir yang BENAR sudah diketahui: {jawaban}. "
    "Tuliskan langkah-langkah penyelesaian yang rinci, sistematis, dan logis dalam Bahasa "
    "Indonesia sehingga sampai pada jawaban tersebut. Jangan hanya menuliskan jawaban; "
    "tunjukkan penalarannya. Akhiri dengan jawaban akhir di dalam \\boxed{{}}.\n\n"
    "Soal:\n{soal}"
)


def strip_think(text: str) -> str:
    idx = text.find("</think>")
    return text[idx + 8:].strip() if idx >= 0 else text.strip()


def _best_verified(rows: list[dict]) -> str:
    """Pilih solusi terverifikasi paling ringkas namun lengkap (mengandung \\boxed)."""
    cleaned = [strip_think(r.get("text", "")) for r in rows]
    cleaned = [c for c in cleaned if c.strip()]
    if not cleaned:
        return ""
    boxed = [c for c in cleaned if "\\boxed" in c]
    pool = boxed or cleaned
    return min(pool, key=len)


def _hinted_one(client, model: str, soal: str, jawaban: str, max_tokens: int) -> str:
    def call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user",
                       "content": HINT_PROMPT.format(soal=soal, jawaban=jawaban)}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
    resp = with_retry(call)
    return (resp.choices[0].message.content or "").strip()


def run(problems_path: Path, correct_path: Path | None, output_path: Path, *,
        use_hinted: bool = True, model: str = DEFAULT_API_MODEL,
        max_tokens: int = 2048) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    problems = read_jsonl(problems_path)
    # pid -> daftar kandidat benar
    verified: dict[str, list[dict]] = {}
    if correct_path and Path(correct_path).exists():
        for r in read_jsonl(correct_path):
            verified.setdefault(r["id"], []).append(r)

    done_soal: set[str] = set()
    if output_path.exists():
        for r in read_jsonl(output_path):
            done_soal.add((r.get("soal") or "").strip())

    stats = {"total": len(problems), "verified": 0, "hinted": 0,
             "skipped_done": 0, "no_cara": 0}
    client = None

    with open(output_path, "a", encoding="utf-8") as f:
        for idx, item in enumerate(problems):
            pid = problem_id(item, idx)
            soal, jawaban = get_soal(item), get_jawaban(item)
            if soal.strip() in done_soal:
                stats["skipped_done"] += 1
                continue

            cara, cara_source = "", ""
            if pid in verified:
                cara = _best_verified(verified[pid])
                cara_source = "verified"
                stats["verified"] += 1
            elif use_hinted:
                if client is None:
                    client = openai_client()
                cara = _hinted_one(client, model, soal, jawaban, max_tokens)
                cara_source = "hinted"
                stats["hinted"] += 1
            else:
                stats["no_cara"] += 1
                continue  # tanpa fallback: soal tak terpecahkan dibuang (gaya AIMO-2 ketat)

            rec = {"soal": soal, "cara": cara, "jawaban": jawaban,
                   "source": item.get("source", "numglue"),
                   "cara_source": cara_source}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Rakit cara (verified + hinted fallback)")
    ap.add_argument("--problems", required=True)
    ap.add_argument("--correct", default=None, help="output filter_solutions.py (kandidat benar)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-hinted", action="store_true",
                    help="jangan fallback ke answer-hinted; soal tak-terpecahkan dibuang")
    ap.add_argument("--model", default=DEFAULT_API_MODEL, help="teacher utk fallback hinted (API)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    stats = run(Path(args.problems), Path(args.correct) if args.correct else None,
                Path(args.output), use_hinted=not args.no_hinted,
                model=args.model, max_tokens=args.max_tokens)
    print(f"assemble_cara: {stats}")


if __name__ == "__main__":
    main()
