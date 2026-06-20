"""
Teacher CoT generation (distillation). Pluggable backend so the teacher is a one-line swap.

Backends:
- "api"  : any OpenAI-compatible endpoint. Default is **Groq** (set GROQ_API_KEY).
           Recommended teacher: deepseek-r1-distill-llama-70b (R1 reasoning, hosted on Groq).
           Writes one candidate at a time and appends immediately -> safe to resume after a
           rate-limit/crash (Groq free tier is ~30 req/min + a daily token cap).
- "vllm" : offline vLLM batch generation on a local GPU (Kaggle T4 / Colab).

For each problem we sample N candidates (default 8, temp 0.7, top-p 0.95) -- diverse solution
paths, as in the AIMO-2 recipe.

Output: data/cot/candidates.jsonl
  {id, soal, jawaban, cara, source, candidate_idx, text}

Usage:
    setx GROQ_API_KEY <key>   # then in a new shell:
    python -m src.cot_synthesis.generate data/merged_dataset.jsonl --backend api -n 8 --limit 50
    python -m src.cot_synthesis.generate data/merged_dataset.jsonl --backend vllm \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B -n 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .prompt_wrap import wrap
from .utils import (get_cara, get_jawaban, get_soal, openai_client, problem_id,
                    read_jsonl, with_retry)

DEFAULT_VLLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# Groq: R1-distill decommissioned. llama-3.3-70b = robust free-tier CoT teacher (high TPM).
# Upgrade to "qwen/qwen3-32b" or "openai/gpt-oss-120b" (reasoning models, lower free TPM) on a paid tier.
DEFAULT_API_MODEL = "llama-3.3-70b-versatile"


# -------------------------------
# Resume support
# -------------------------------

def _already_done(out_path: Path) -> dict[str, int]:
    """Map id -> number of candidates already written, so we can resume."""
    counts: dict[str, int] = {}
    if not out_path.exists():
        return counts
    for row in read_jsonl(out_path):
        counts[row["id"]] = counts.get(row["id"], 0) + 1
    return counts


# -------------------------------
# Backends
# -------------------------------

def _api_one(client, prompt: str, model: str, temperature: float, top_p: float,
             max_tokens: int) -> str:
    """One completion from an OpenAI-compatible endpoint, with R1 reasoning captured."""
    def call():
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    resp = with_retry(call)
    msg = resp.choices[0].message
    text = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None)  # DeepSeek-R1 style
    if reasoning:
        text = f"<think>\n{reasoning}\n</think>\n\n{text}"
    return text


def _build_vllm(model: str, max_tokens: int, tensor_parallel_size: int = 1):
    """Build the vLLM engine ONCE (model load is the expensive part). Reused across batches."""
    import os
    # T4 (compute 7.5): FlashInfer attention compiles but fails at runtime (BatchPrefill "invalid
    # argument"), so the Kaggle notebook uninstalls flashinfer -> vLLM falls back to Triton attention.
    # Also force the native torch sampler (harmless if flashinfer is already gone).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM

    return LLM(model=model, dtype="float16", gpu_memory_utilization=0.95,
               max_model_len=max(4096, max_tokens + 1024), trust_remote_code=True,
               tensor_parallel_size=tensor_parallel_size)


def _vllm_chat(llm, prompts: list[str], n: int, temperature: float, top_p: float,
               max_tokens: int) -> list[list[str]]:
    from vllm import SamplingParams

    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    convos = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(convos, sp)
    return [[o.text for o in out.outputs] for out in outputs]


def generate_vllm(prompts: list[str], model: str, n: int, temperature: float,
                  top_p: float, max_tokens: int, tensor_parallel_size: int = 1) -> list[list[str]]:
    """One-shot helper (builds engine + generates all prompts). For batched/resumable runs
    use run_generate, which builds the engine once and checkpoints per batch."""
    llm = _build_vllm(model, max_tokens, tensor_parallel_size)
    return _vllm_chat(llm, prompts, n, temperature, top_p, max_tokens)


# -------------------------------
# Driver
# -------------------------------

def _in_shard(idx: int, shard: int, num_shards: int) -> bool:
    """Round-robin shard membership. Worker `shard` of `num_shards` owns idx where
    idx % num_shards == shard. Disjoint across workers -> no duplicate work, and because
    problem_id uses the GLOBAL idx, the two output files merge by plain concat."""
    if num_shards <= 1:
        return True
    return idx % num_shards == shard


def run_generate(input_path: str | Path, out_path: str | Path, *, backend: str,
                 model: str, n: int = 8, temperature: float = 0.7, top_p: float = 0.95,
                 max_tokens: int = 4096, limit: int | None = None, sleep: float = 0.0,
                 tensor_parallel_size: int = 1, batch_size: int = 64,
                 shard: int = 0, num_shards: int = 1) -> dict:
    if not 0 <= shard < max(num_shards, 1):
        raise ValueError(f"shard must be in [0, {num_shards}), got {shard}")
    input_path, out_path = Path(input_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(input_path)
    done = _already_done(out_path)

    todo = []
    for idx, item in enumerate(items):
        if not _in_shard(idx, shard, num_shards):
            continue
        pid = problem_id(item, idx)
        soal = get_soal(item)
        if not soal or done.get(pid, 0) >= n:
            continue
        todo.append((pid, item, wrap(soal, mode="cot")))
        if limit is not None and len(todo) >= limit:
            break

    skipped = len(items) - len(todo)
    if not todo:
        return {"problems": len(items), "generated": 0, "skipped": skipped, "resumed": True}

    written = 0
    if backend == "vllm":
        # Build the engine ONCE, then generate batch_size problems at a time and flush after
        # each batch. A flushed batch is a checkpoint: if the Kaggle session dies, rerunning
        # skips finished problems (via _already_done) and resumes from the next batch.
        llm = _build_vllm(model, max_tokens, tensor_parallel_size)
        with open(out_path, "a", encoding="utf-8") as f:
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                cands = _vllm_chat(llm, [p for _, _, p in chunk], n=n, temperature=temperature,
                                   top_p=top_p, max_tokens=max_tokens)
                for (pid, item, _), cs in zip(chunk, cands):
                    for ci, text in enumerate(cs):
                        f.write(json.dumps(_row(pid, item, ci, text), ensure_ascii=False) + "\n")
                        written += 1
                f.flush()
                os.fsync(f.fileno())
                print(f"  checkpoint: {min(start + batch_size, len(todo))}/{len(todo)} problems "
                      f"done ({written} candidates written)", flush=True)
    else:  # api: stream one candidate at a time, append immediately (resumable)
        client = openai_client()
        with open(out_path, "a", encoding="utf-8") as f:
            for pid, item, prompt in todo:
                have = done.get(pid, 0)
                for ci in range(have, n):
                    text = _api_one(client, prompt, model, temperature, top_p, max_tokens)
                    f.write(json.dumps(_row(pid, item, ci, text), ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if sleep:
                        time.sleep(sleep)

    return {"problems": len(items), "generated": written, "todo_problems": len(todo),
            "skipped": skipped}


def _row(pid: str, item: dict, ci: int, text: str) -> dict:
    return {
        "id": pid,
        "soal": get_soal(item),
        "jawaban": get_jawaban(item),
        "cara": get_cara(item),
        "source": item.get("source") or item.get("source_file", ""),
        "candidate_idx": ci,
        "text": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate teacher CoT candidates")
    ap.add_argument("input", help="JSONL with soal (+ jawaban, cara)")
    ap.add_argument("--out", default="data/cot/candidates.jsonl")
    ap.add_argument("--backend", choices=["api", "vllm"], default="api")
    ap.add_argument("--model", default=None, help="teacher model id (backend-specific default)")
    ap.add_argument("-n", "--num-candidates", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most this many NEW problems this run (dev / rate-limit safety)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to wait between api calls (throttle for Groq free tier)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vllm backend: number of GPUs to shard across (Kaggle 2xT4 -> 2)")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="vllm backend: problems per checkpoint (flush after each batch -> resumable)")
    ap.add_argument("--shard", default="0/1",
                    help="split work across workers: 'i/N' (worker i of N owns idx%%N==i). "
                         "Parallel teammates: you --shard 0/2, friend --shard 1/2, then concat outputs.")
    args = ap.parse_args()

    shard_i, num_shards = (int(x) for x in args.shard.split("/"))
    model = args.model or (DEFAULT_API_MODEL if args.backend == "api" else DEFAULT_VLLM_MODEL)
    stats = run_generate(args.input, args.out, backend=args.backend, model=model,
                         n=args.num_candidates, temperature=args.temperature,
                         top_p=args.top_p, max_tokens=args.max_tokens,
                         limit=args.limit, sleep=args.sleep,
                         tensor_parallel_size=args.tensor_parallel_size,
                         batch_size=args.batch_size,
                         shard=shard_i, num_shards=num_shards)
    print(f"Teacher={model} backend={args.backend} | {stats}")


if __name__ == "__main__":
    main()
