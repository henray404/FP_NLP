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


def run(input_path: str | Path, out_dir: str | Path, *, keep_think: bool = False,
        max_per_problem: int | None = None, max_solution_chars: int | None = None) -> dict:
    out_dir = Path(out_dir)
    rows = read_jsonl(input_path)

    per_problem: dict[str, int] = {}
    cot_rows, nocot_rows = [], []
    skipped_long = 0

    rows.sort(key=lambda r: (r.get("id", ""), r.get("candidate_idx", 0)))
    for r in rows:
        pid = r.get("id", "")
        if max_per_problem is not None and per_problem.get(pid, 0) >= max_per_problem:
            continue
        soal = r.get("soal", "")
        pred = r.get("pred", "")
        solution = clean_solution(r.get("text", ""), keep_think)
        if not solution or not pred:
            continue
        if max_solution_chars is not None and len(solution) > max_solution_chars:
            skipped_long += 1
            continue
        per_problem[pid] = per_problem.get(pid, 0) + 1

        cot_rows.append(chatml(wrap(soal, "cot"), solution))
        nocot_rows.append(chatml(wrap(soal, "nocot"), f"\\boxed{{{pred}}}"))

    n_cot = write_jsonl(cot_rows, out_dir / "cot.jsonl")
    n_nocot = write_jsonl(nocot_rows, out_dir / "nocot.jsonl")
    return {
        "input_rows": len(rows),
        "cot_examples": n_cot,
        "nocot_examples": n_nocot,
        "problems": len(per_problem),
        "skipped_long": skipped_long,
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
    args = ap.parse_args()

    stats = run(args.input, args.out_dir, keep_think=args.keep_think,
                max_per_problem=args.max_per_problem, max_solution_chars=args.max_solution_chars)
    print(f"to_chatml: {stats}")


if __name__ == "__main__":
    main()
