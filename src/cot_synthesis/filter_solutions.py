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
import json
import os
import time
from pathlib import Path

from .utils import (answers_equivalent, extract_boxed, openai_client, read_jsonl,
                    with_retry)

DEFAULT_JUDGE_MODEL = "llama-3.1-8b-instant"  # Groq

JUDGE_PROMPT = (
    "Diberikan sebuah soal matematika, jawaban yang diharapkan, dan jawaban prediksi. "
    "Tentukan apakah jawaban prediksi secara matematis ekuivalen dengan jawaban yang "
    "diharapkan. Abaikan perbedaan format/penulisan; nilai hanya kesetaraan nilainya. "
    "Jawab HANYA dengan satu kata: 'benar' atau 'salah'.\n\n"
    "Soal: {soal}\nJawaban yang diharapkan: {gold}\nJawaban prediksi: {pred}\n\n"
    "Apakah kedua jawaban ekuivalen secara matematis? Jawab:"
)


# Each judge is a BATCH callable: list[(soal, gold, pred)] -> list[bool]. Batching lets the
# vllm judge process a whole checkpoint chunk in one call (fast); the api judge loops internally.

def _make_judge_api(model: str, sleep: float = 0.0):
    """LLM judge over an OpenAI-compatible endpoint (Groq by default). One call per pair."""
    client = openai_client()

    def judge_batch(triples: list[tuple[str, str, str]]) -> list[bool]:
        out = []
        for soal, gold, pred in triples:
            prompt = JUDGE_PROMPT.format(soal=soal[:1500], gold=gold, pred=pred)

            def call():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                )

            resp = with_retry(call)
            out.append("benar" in (resp.choices[0].message.content or "").strip().lower())
            if sleep:
                time.sleep(sleep)
        return out

    return judge_batch


def _make_judge_vllm(model: str, tensor_parallel_size: int = 1):
    """Local LLM judge on a GPU (no API limits). For the full run on Kaggle/Colab."""
    # See generate.py: notebook uninstalls flashinfer on T4 -> Triton attention + torch sampler.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=0.85, max_model_len=2048,
              tensor_parallel_size=tensor_parallel_size)
    sp = SamplingParams(temperature=0.0, max_tokens=5)

    def judge_batch(triples: list[tuple[str, str, str]]) -> list[bool]:
        if not triples:
            return []
        prompts = [JUDGE_PROMPT.format(soal=s[:1500], gold=g, pred=p) for s, g, p in triples]
        outs = llm.generate(prompts, sp)
        return ["benar" in o.outputs[0].text.strip().lower() for o in outs]

    return judge_batch


def _make_judge(backend: str, model: str, sleep: float = 0.0, tensor_parallel_size: int = 1):
    if backend == "api":
        return _make_judge_api(model, sleep=sleep)
    if backend == "vllm":
        return _make_judge_vllm(model, tensor_parallel_size=tensor_parallel_size)
    raise ValueError(f"judge backend must be 'api' or 'vllm', got {backend!r}")


# -------------------------------
# Checkpoint / resume
# -------------------------------

def _cand_key(row: dict) -> str:
    """Stable per-candidate key: a candidate is one (problem id, candidate_idx) pair."""
    return f"{row.get('id', '')}\t{row.get('candidate_idx', 0)}"


def _processed_keys(output_path: Path, progress_path: Path) -> set[str]:
    """Keys already examined in a previous run = (kept rows in output) U (progress log).
    The progress log records EVERY judged candidate (kept or dropped) so a resume never
    re-judges the same candidate twice."""
    done: set[str] = set()
    if output_path.exists():
        for r in read_jsonl(output_path):
            done.add(_cand_key(r))
    if progress_path.exists():
        with open(progress_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    done.add(line)
    return done


def run_filter(input_path: str | Path, output_path: str | Path, *,
               judge_backend: str = "api", judge_model: str = DEFAULT_JUDGE_MODEL,
               prefilter: bool = False, sleep: float = 0.0,
               tensor_parallel_size: int = 1, batch_size: int = 64,
               resume: bool = True, emit_chatml: bool = False,
               chatml_dir: str | Path = "data/sft", best_per_problem: bool = True,
               id_only: bool = True) -> dict:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_path.with_suffix(output_path.suffix + ".progress")

    if not resume:  # start clean: drop any previous output + checkpoint
        for p in (output_path, progress_path):
            if p.exists():
                p.unlink()

    rows = read_jsonl(input_path)
    done = _processed_keys(output_path, progress_path) if resume else set()

    stats = {"total": len(rows), "resumed": len(done), "no_boxed": 0, "no_gold": 0,
             "wrong": 0, "kept": 0, "by_prefilter": 0, "by_judge": 0, "skipped_done": 0}

    # Pass 1 (cheap, no LLM): completeness + gold + optional prefilter. Anything that still
    # needs the LLM judge is queued; everything else is decided + checkpointed immediately.
    judge = None  # build lazily so a fully-resumed run never loads the GPU judge
    out_f = open(output_path, "a", encoding="utf-8")
    prog_f = open(progress_path, "a", encoding="utf-8")

    def _emit(row: dict, pred: str, gold: str) -> None:
        out_f.write(json.dumps({
            "id": row["id"], "soal": row.get("soal", ""), "jawaban": gold,
            "candidate_idx": row.get("candidate_idx", 0), "text": row.get("text", ""),
            "pred": pred,
        }, ensure_ascii=False) + "\n")

    def _checkpoint(keys: list[str]) -> None:
        for k in keys:
            prog_f.write(k + "\n")
        out_f.flush(); prog_f.flush()
        os.fsync(out_f.fileno()); os.fsync(prog_f.fileno())

    try:
        pending: list[tuple[dict, str, str]] = []  # (row, pred, gold) awaiting judge
        for r in rows:
            key = _cand_key(r)
            if key in done:
                stats["skipped_done"] += 1
                continue
            text = r.get("text", "")
            pred = extract_boxed(text)
            if pred is None:
                stats["no_boxed"] += 1
                _checkpoint([key])
                continue
            gold = str(r.get("jawaban", "")).strip()
            if not gold:
                stats["no_gold"] += 1
                _checkpoint([key])
                continue
            if prefilter and answers_equivalent(pred, gold):
                stats["by_prefilter"] += 1
                stats["kept"] += 1
                _emit(r, pred, gold)
                _checkpoint([key])
                continue
            pending.append((r, pred, gold))

        # Pass 2: judge the queue in batches, flushing output + progress after each batch.
        if pending:
            judge = _make_judge(judge_backend, judge_model, sleep=sleep,
                                tensor_parallel_size=tensor_parallel_size)
            for start in range(0, len(pending), batch_size):
                chunk = pending[start:start + batch_size]
                verdicts = judge([(r.get("soal", ""), gold, pred) for r, pred, gold in chunk])
                batch_keys = []
                for (r, pred, gold), ok in zip(chunk, verdicts):
                    if ok:
                        stats["by_judge"] += 1
                        stats["kept"] += 1
                        _emit(r, pred, gold)
                    else:
                        stats["wrong"] += 1
                    batch_keys.append(_cand_key(r))
                _checkpoint(batch_keys)
                print(f"  checkpoint: judged {min(start + batch_size, len(pending))}/{len(pending)} "
                      f"(kept so far {stats['kept']})", flush=True)
    finally:
        out_f.close(); prog_f.close()

    stats["problems_covered"] = len({r["id"] for r in read_jsonl(output_path)}) if output_path.exists() else 0

    # Final stage: turn the correct solutions straight into ChatML SFT datasets so the CoT
    # pipeline ends in training-ready cot.jsonl/nocot.jsonl (no separate to_chatml call needed).
    if emit_chatml:
        from .to_chatml import run as build_chatml
        cm = build_chatml(output_path, chatml_dir, best_per_problem=best_per_problem,
                          id_only=id_only)
        stats["chatml"] = cm

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
    ap.add_argument("--batch-size", type=int, default=64,
                    help="candidates per checkpoint (flush output+progress after each batch -> resumable)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing output/.progress and re-judge everything from scratch")
    ap.add_argument("--emit-chatml", action="store_true",
                    help="after filtering, build cot.jsonl/nocot.jsonl directly (pipeline ends in ChatML)")
    ap.add_argument("--chatml-dir", default="data/sft", help="output dir for cot.jsonl/nocot.jsonl")
    ap.add_argument("--keep-all-per-problem", action="store_true",
                    help="keep every correct solution per problem (default: 1 best per problem)")
    ap.add_argument("--keep-english", action="store_true",
                    help="keep English-dominant CoT (default: Indonesian-only)")
    args = ap.parse_args()

    model = args.judge_model or (
        DEFAULT_JUDGE_MODEL if args.judge_backend == "api" else "Qwen/Qwen2.5-7B-Instruct")
    stats = run_filter(args.input, args.output, judge_backend=args.judge_backend,
                       judge_model=model, prefilter=args.prefilter, sleep=args.sleep,
                       tensor_parallel_size=args.tensor_parallel_size,
                       batch_size=args.batch_size, resume=not args.no_resume,
                       emit_chatml=args.emit_chatml, chatml_dir=args.chatml_dir,
                       best_per_problem=not args.keep_all_per_problem, id_only=not args.keep_english)
    print(f"Filter: {stats}")


if __name__ == "__main__":
    main()
