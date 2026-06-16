"""
Filter teacher candidates down to correct, complete solutions (rejection sampling).

Filters:
1. Completeness -- solution must contain a brace-balanced \\boxed{...} (else generation unfinished).
2. Correctness  -- decided by an **LLM judge** (default), because `jawaban` is a natural-language
   sentence with no \\boxed, so plain string/sympy matching is unreliable. The judge compares the
   teacher's boxed prediction against the gold `jawaban` and answers benar/salah.

Judge backend:
- "api"  : OpenAI-compatible, default **Groq** (set GROQ_API_KEY), model llama-3.1-8b-instant.
           Good for small/dev runs. Free-tier RPD makes it unusable for a full 25k run.
- "vllm" : local model on a GPU (Kaggle/Colab). Use this for the full run -- no API limits.

`--prefilter` runs a free string/math_verify check first to skip obvious matches and save judge
calls; off by default per the "judge only" decision.

Keeps ALL correct candidates per problem (multiple solutions per problem is fine for SFT).

Input:  data/cot/candidates.jsonl   (from generate.py)
Output: data/cot/correct.jsonl
  {id, soal, jawaban, candidate_idx, text, pred}

Usage:
    python -m src.cot_synthesis.filter_solutions data/cot/candidates.jsonl data/cot/correct.jsonl
    python -m src.cot_synthesis.filter_solutions <in> <out> --judge-model llama-3.3-70b-versatile
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .utils import (answers_equivalent, extract_boxed, openai_client, read_jsonl,
                    with_retry, write_jsonl)

DEFAULT_JUDGE_MODEL = "llama-3.1-8b-instant"  # Groq

JUDGE_PROMPT = (
    "Diberikan sebuah soal matematika, jawaban yang diharapkan, dan jawaban prediksi. "
    "Tentukan apakah jawaban prediksi secara matematis ekuivalen dengan jawaban yang "
    "diharapkan. Abaikan perbedaan format/penulisan; nilai hanya kesetaraan nilainya. "
    "Jawab HANYA dengan satu kata: 'benar' atau 'salah'.\n\n"
    "Soal: {soal}\nJawaban yang diharapkan: {gold}\nJawaban prediksi: {pred}\n\n"
    "Apakah kedua jawaban ekuivalen secara matematis? Jawab:"
)


def _make_judge_api(model: str, sleep: float = 0.0):
    """LLM judge over an OpenAI-compatible endpoint (Groq by default). One call per pair."""
    client = openai_client()

    def judge(soal: str, gold: str, pred: str) -> bool:
        prompt = JUDGE_PROMPT.format(soal=soal[:1500], gold=gold, pred=pred)

        def call():
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )

        resp = with_retry(call)
        if sleep:
            time.sleep(sleep)
        return "benar" in (resp.choices[0].message.content or "").strip().lower()

    return judge


def _make_judge_vllm(model: str, tensor_parallel_size: int = 1):
    """Local LLM judge on a GPU (no API limits). For the full run on Kaggle/Colab."""
    import os
    # See generate.py: FlashInfer sampler JIT fails to link libcuda on T4/Kaggle -> use torch sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=0.85, max_model_len=2048,
              tensor_parallel_size=tensor_parallel_size)
    sp = SamplingParams(temperature=0.0, max_tokens=5)

    def judge(soal: str, gold: str, pred: str) -> bool:
        prompt = JUDGE_PROMPT.format(soal=soal[:1500], gold=gold, pred=pred)
        out = llm.generate([prompt], sp)[0].outputs[0].text
        return "benar" in out.strip().lower()

    return judge


def _make_judge(backend: str, model: str, sleep: float = 0.0, tensor_parallel_size: int = 1):
    if backend == "api":
        return _make_judge_api(model, sleep=sleep)
    if backend == "vllm":
        return _make_judge_vllm(model, tensor_parallel_size=tensor_parallel_size)
    raise ValueError(f"judge backend must be 'api' or 'vllm', got {backend!r}")


def run_filter(input_path: str | Path, output_path: str | Path, *,
               judge_backend: str = "api", judge_model: str = DEFAULT_JUDGE_MODEL,
               prefilter: bool = False, sleep: float = 0.0,
               tensor_parallel_size: int = 1) -> dict:
    rows = read_jsonl(input_path)
    judge = _make_judge(judge_backend, judge_model, sleep=sleep,
                        tensor_parallel_size=tensor_parallel_size)

    stats = {"total": len(rows), "no_boxed": 0, "no_gold": 0, "wrong": 0,
             "kept": 0, "by_prefilter": 0, "by_judge": 0}
    kept = []
    for r in rows:
        text = r.get("text", "")
        pred = extract_boxed(text)
        if pred is None:
            stats["no_boxed"] += 1
            continue
        gold = str(r.get("jawaban", "")).strip()
        if not gold:
            stats["no_gold"] += 1
            continue

        ok = False
        if prefilter and answers_equivalent(pred, gold):
            ok = True
            stats["by_prefilter"] += 1
        else:
            ok = judge(r.get("soal", ""), gold, pred)
            if ok:
                stats["by_judge"] += 1

        if not ok:
            stats["wrong"] += 1
            continue
        kept.append({
            "id": r["id"], "soal": r.get("soal", ""), "jawaban": gold,
            "candidate_idx": r.get("candidate_idx", 0), "text": text, "pred": pred,
        })

    stats["kept"] = write_jsonl(kept, output_path)
    stats["problems_covered"] = len({k["id"] for k in kept})
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter teacher candidates to correct solutions (LLM judge)")
    ap.add_argument("input", nargs="?", default="data/cot/candidates.jsonl")
    ap.add_argument("output", nargs="?", default="data/cot/correct.jsonl")
    ap.add_argument("--judge-backend", choices=["api", "vllm"], default="api",
                    help="api=Groq (dev), vllm=local GPU (full run, no API limits)")
    ap.add_argument("--judge-model", default=None,
                    help="default: llama-3.1-8b-instant (api) / Qwen/Qwen2.5-7B-Instruct (vllm)")
    ap.add_argument("--prefilter", action="store_true",
                    help="run free string/math_verify check first to skip obvious matches (saves judge calls)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between judge calls (throttle for Groq free tier)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vllm judge: number of GPUs to shard across (Kaggle 2xT4 -> 2)")
    args = ap.parse_args()

    model = args.judge_model or (
        DEFAULT_JUDGE_MODEL if args.judge_backend == "api" else "Qwen/Qwen2.5-7B-Instruct")
    stats = run_filter(args.input, args.output, judge_backend=args.judge_backend,
                       judge_model=model, prefilter=args.prefilter, sleep=args.sleep,
                       tensor_parallel_size=args.tensor_parallel_size)
    print(f"Filter: {stats}")


if __name__ == "__main__":
    main()
