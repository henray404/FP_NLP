"""
Build the final CoT / non-CoT SFT training sets from the WINNING teacher (S1 = DeepSeek).

Pipeline (one model per arm, identical data so only CoT-vs-nonCoT differs):

1. Merge the winning teacher's per-dataset CoT files (numglue + un) into one pool.
2. Derive nocot from the SAME rows -> strip reasoning, keep final \\boxed{pred}. This
   guarantees cot and nocot cover the EXACT same problems (fair comparison).
3. Decontaminate: drop any train row whose `soal` appears in the held-out test sets
   (data/sft/test/*.jsonl). Test is the protected held-out split; leakage is removed
   from TRAIN, never from test.
4. Write data/sft/cot.jsonl + data/sft/nocot.jsonl and report before/after + residual leakage.

The teacher CoT files are already ChatML ({"messages":[user, assistant]}); the `soal` is the
user content with the fixed CoT prompt prefix stripped off.

Usage:
    python -m src.training.build_train
    python -m src.training.build_train \
        --cot data/sft/full_data/numglue/cot_DeepSeek-R1-Distill-Qwen-7B.jsonl \
        --cot data/sft/full_data/vllm_un/cot_DeepSeek-R1-Distill-Qwen-7B.jsonl \
        --test data/sft/test/numglue_test.jsonl --test data/sft/test/easy_test.jsonl \
        --out-dir data/sft/train
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ..cot_synthesis.prompt_wrap import PROMPT_COT, wrap
from ..cot_synthesis.utils import extract_boxed, get_soal, read_jsonl, write_jsonl

# Winning teacher (S1). Default inputs = its per-dataset CoT files.
DEFAULT_COT = [
    "data/sft/full_data/numglue/cot_DeepSeek-R1-Distill-Qwen-7B.jsonl",
    "data/sft/full_data/vllm_un/cot_DeepSeek-R1-Distill-Qwen-7B.jsonl",
]
DEFAULT_TEST = [
    "data/sft/test/numglue_test.jsonl",
    "data/sft/test/easy_test.jsonl",
]

# Fixed CoT prompt prefix (everything before the soal). Stripping it recovers the raw soal.
_COT_PREFIX = PROMPT_COT.format(soal="").rstrip()
_WS_RE = re.compile(r"\s+")


def _norm(soal: str) -> str:
    """Whitespace-collapsed, lowercased key for leakage matching."""
    return _WS_RE.sub(" ", soal).strip().lower()


def _soal_from_chatml(user_content: str) -> str:
    """Recover the raw `soal` from a ChatML user turn by removing the CoT prompt prefix."""
    s = user_content
    if s.startswith(_COT_PREFIX):
        s = s[len(_COT_PREFIX):]
    return s.strip()


def _test_keys(test_paths: list[str]) -> set[str]:
    """Normalized soal of every held-out test row (the decontamination blacklist)."""
    keys: set[str] = set()
    for p in test_paths:
        if not Path(p).exists():
            print(f"WARNING: test set not found, skipped: {p}")
            continue
        for r in read_jsonl(p):
            keys.add(_norm(get_soal(r)))
    return keys


def run(cot_paths: list[str], test_paths: list[str], out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    blacklist = _test_keys(test_paths)

    cot_rows, nocot_rows = [], []
    n_in = leaked = no_box = 0
    seen: set[str] = set()  # dedup across merged files

    for p in cot_paths:
        if not Path(p).exists():
            print(f"WARNING: cot file not found, skipped: {p}")
            continue
        for row in read_jsonl(p):
            n_in += 1
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                continue
            user, assistant = msgs[0]["content"], msgs[1]["content"]
            soal = _soal_from_chatml(user)
            key = _norm(soal)

            if key in blacklist:        # leakage -> drop from TRAIN
                leaked += 1
                continue
            if key in seen:             # duplicate soal across merged files
                continue
            pred = extract_boxed(assistant)
            if not pred:                # nocot needs a final answer; skip if none
                no_box += 1
                continue
            seen.add(key)

            cot_rows.append({"messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]})
            nocot_rows.append({"messages": [
                {"role": "user", "content": wrap(soal, "nocot")},
                {"role": "assistant", "content": f"\\boxed{{{pred}}}"},
            ]})

    n_cot = write_jsonl(cot_rows, out_dir / "cot.jsonl")
    n_nocot = write_jsonl(nocot_rows, out_dir / "nocot.jsonl")

    # sanity: residual leakage must be zero
    residual = sum(1 for r in cot_rows
                   if _norm(_soal_from_chatml(r["messages"][0]["content"])) in blacklist)
    return {
        "rows_in": n_in,
        "leaked_dropped": leaked,
        "no_boxed_dropped": no_box,
        "cot_examples": n_cot,
        "nocot_examples": n_nocot,
        "residual_leakage": residual,
        "test_keys": len(blacklist),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build decontaminated CoT/nonCoT SFT sets (winning teacher)")
    ap.add_argument("--cot", action="append", default=None,
                    help="teacher CoT jsonl (repeat to merge; default = DeepSeek numglue+un)")
    ap.add_argument("--test", action="append", default=None,
                    help="held-out test jsonl for decontamination (repeat; default = numglue+easy)")
    ap.add_argument("--out-dir", default="data/sft/train")
    args = ap.parse_args()

    cot_paths = args.cot or DEFAULT_COT
    test_paths = args.test or DEFAULT_TEST
    stats = run(cot_paths, test_paths, args.out_dir)

    print("build_train:")
    for k, v in stats.items():
        print(f"  {k:18} {v}")
    assert stats["residual_leakage"] == 0, "LEAKAGE REMAINS after decontam!"
    assert stats["cot_examples"] == stats["nocot_examples"], "cot/nocot coverage mismatch!"
    print(f"\nwrote {args.out_dir}/cot.jsonl + nocot.jsonl  (fair: same {stats['cot_examples']} problems)")


if __name__ == "__main__":
    main()
