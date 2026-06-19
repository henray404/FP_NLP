"""
Build the two SFT datasets that power the CoT vs non-CoT experiment.

From the SAME correct solutions (data/cot/correct.jsonl) emit:
- cot.jsonl   : user = CoT prompt   ("tunjukkan langkah"), assistant = reasoning + \\boxed{answer}
- nocot.jsonl : user = non-CoT prompt ("hanya jawaban"),   assistant = \\boxed{answer} only

Both are ChatML-style ({"messages": [...]}) consumable by Unsloth / LLaMA-Factory.
Keeping base + hyperparams identical and varying ONLY this data is what isolates the CoT effect.

Usage:
    python -m src.cot_synthesis.to_chatml data/cot/correct.jsonl --out-dir data/sft
    python -m src.cot_synthesis.to_chatml data/cot/correct.jsonl --out-dir data/sft \
        --keep-think --max-per-problem 4 --max-solution-chars 8000
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from .prompt_wrap import wrap
from .utils import read_jsonl, write_jsonl

_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_WS_RE = re.compile(r"\n{3,}")

# Crude language detector for the CoT reasoning body: the teacher (llama-3.3-70b) leaks English
# despite an Indonesian prompt (~32% of candidates). For an Indonesian-math student we drop
# English-dominant reasoning from the CoT arm so it never learns to reason in English.
_ID_WORDS = re.compile(
    r"\b(adalah|maka|sehingga|dengan|untuk|kita|persamaan|jawaban|nilai|langkah|"
    r"diketahui|jadi|misalkan|karena|bilangan|sehingga|sebuah|pada|dari|yang)\b", re.I)
_EN_WORDS = re.compile(
    r"\b(the|is|we|so|therefore|equation|answer|value|step|since|let|because|first|"
    r"then|thus|need|find|number|given|where|which|that)\b", re.I)


def is_indonesian(text: str) -> bool:
    """True when the body looks Indonesian-dominant (more ID than EN cue words).
    Ties go to Indonesian; near-symbolic bodies with few words also pass."""
    ni, ne = len(_ID_WORDS.findall(text)), len(_EN_WORDS.findall(text))
    return ni >= ne


def clean_solution(text: str, keep_think: bool) -> str:
    """Assistant target for the CoT dataset: full reasoning ending in \\boxed{...}."""
    if not keep_think:
        text = _THINK_TAG_RE.sub("", text)  # drop tags, keep the reasoning content
    text = _WS_RE.sub("\n\n", text)
    return text.strip()


def chatml(user: str, assistant: str) -> dict:
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _solution_rank(solution: str) -> tuple:
    """Sort key for picking the BEST solution per problem (smaller = better):
    Indonesian first, then shorter (cleaner, fits the tiny student's context)."""
    return (0 if is_indonesian(solution) else 1, len(solution))


def run(input_path: str | Path, out_dir: str | Path, *, keep_think: bool = False,
        max_per_problem: int | None = None, max_solution_chars: int | None = None,
        id_only: bool = False, best_per_problem: bool = False) -> dict:
    """Build cot.jsonl + nocot.jsonl from filtered-correct rows (data/cot/correct.jsonl).

    id_only=True          -> drop English-dominant reasoning (keep only Indonesian CoT).
    best_per_problem=True  -> keep ONLY the single best solution per problem (== max_per_problem 1,
                              chosen by _solution_rank instead of candidate order).
    """
    out_dir = Path(out_dir)
    rows = read_jsonl(input_path)
    if best_per_problem:
        max_per_problem = 1

    # 1) Build the pool of eligible (cleaned) solutions, grouped by problem.
    by_problem: dict[str, list[dict]] = {}
    skipped_long = skipped_lang = 0
    for r in rows:
        pred = r.get("pred", "")
        solution = clean_solution(r.get("text", ""), keep_think)
        if not solution or not pred:
            continue
        if max_solution_chars is not None and len(solution) > max_solution_chars:
            skipped_long += 1
            continue
        if id_only and not is_indonesian(solution):
            skipped_lang += 1
            continue
        by_problem.setdefault(r.get("id", ""), []).append(
            {"soal": r.get("soal", ""), "pred": pred, "solution": solution})

    # 2) Per problem, rank best-first and keep up to max_per_problem.
    cot_rows, nocot_rows = [], []
    for pid, cands in by_problem.items():
        cands.sort(key=lambda c: _solution_rank(c["solution"]))
        kept = cands if max_per_problem is None else cands[:max_per_problem]
        for c in kept:
            cot_rows.append(chatml(wrap(c["soal"], "cot"), c["solution"]))
            nocot_rows.append(chatml(wrap(c["soal"], "nocot"), f"\\boxed{{{c['pred']}}}"))

    n_cot = write_jsonl(cot_rows, out_dir / "cot.jsonl")
    n_nocot = write_jsonl(nocot_rows, out_dir / "nocot.jsonl")
    return {
        "input_rows": len(rows),
        "cot_examples": n_cot,
        "nocot_examples": n_nocot,
        "problems": len(by_problem),
        "skipped_long": skipped_long,
        "skipped_lang": skipped_lang,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build CoT and non-CoT ChatML SFT datasets")
    ap.add_argument("input", nargs="?", default="data/cot/correct.jsonl")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--keep-think", action="store_true",
                    help="keep <think> tags verbatim (default: strip tags, keep reasoning text)")
    ap.add_argument("--max-per-problem", type=int, default=None,
                    help="cap kept solutions per problem to balance the set")
    ap.add_argument("--max-solution-chars", type=int, default=None,
                    help="drop solutions longer than this (tiny student has short context)")
    ap.add_argument("--id-only", action="store_true",
                    help="drop English-dominant reasoning from the CoT arm (keep Indonesian only)")
    ap.add_argument("--best-per-problem", action="store_true",
                    help="keep ONLY the single best solution per problem (Indonesian + shortest)")
    args = ap.parse_args()

    stats = run(args.input, args.out_dir, keep_think=args.keep_think,
                max_per_problem=args.max_per_problem, max_solution_chars=args.max_solution_chars,
                id_only=args.id_only, best_per_problem=args.best_per_problem)
    print(f"to_chatml: {stats}")


if __name__ == "__main__":
    main()
