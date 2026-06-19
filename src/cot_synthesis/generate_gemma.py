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


def generate_vllm(prompts: list[str], model: str, n: int, temperature: float,
                  top_p: float, max_tokens: int, tensor_parallel_size: int = 1) -> list[list[str]]:
    import os
    # Gemma-2 pakai sliding-window attention (interleaved local/global). vLLM V1 batch-queue
    # scheduler crash di kombinasi ini (T4/XFORMERS): request selesai tapi scheduler cari req_id
    # yg sudah hilang -> "KeyError: '..._...-<hash>'" di update_from_output, EngineCore mati.
    # Paksa V0 engine: stabil buat Gemma-2 di T4, lewati jalur scheduler V1.
    os.environ.setdefault("VLLM_USE_V1", "0")
    # T4 (compute 7.5): FlashInfer attention compiles but fails at runtime (BatchPrefill "invalid
    # argument"), so the Kaggle notebook uninstalls flashinfer -> vLLM falls back to Triton attention.
    # Also force the native torch sampler (harmless if flashinfer is already gone).
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # T4 (sm75) shared memory cap = 64KB/block. Triton attention kernel butuh ~80KB di float32
    # (2x fp16) -> "OutOfResources: shared memory". XFORMERS backend pakai kernel lain yg muat di T4.
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    # Kaggle 2xT4 tidak punya P2P/NVLink -> NCCL all-reduce hang saat TP>1. Matikan P2P + shm
    # transport biar NCCL pakai jalur biasa, dan matikan custom all-reduce vLLM (butuh P2P).
    if tensor_parallel_size > 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_SHM_DISABLE", "1")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="auto", gpu_memory_utilization=0.85,
              max_model_len=max(4096, max_tokens + 1024), trust_remote_code=True,
              tensor_parallel_size=tensor_parallel_size, enforce_eager=True,
              disable_custom_all_reduce=(tensor_parallel_size > 1))
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    convos = [[{"role": "user", "content": p}] for p in prompts]
    outputs = llm.chat(convos, sp)
    return [[o.text for o in out.outputs] for out in outputs]


def generate_hf(prompts: list[str], model: str, n: int, temperature: float,
                top_p: float, max_tokens: int):
    """Plain HuggingFace transformers generation. Robust fallback for Gemma-2 on T4.

    vLLM di T4 (sm75) bermasalah buat Gemma-2: head_dim=256 bikin kernel Triton attention
    minta shared memory > 64KB -> "OutOfResources", dan env-workaround V0/XFORMERS diabaikan
    vLLM versi baru (V0 sudah dihapus). transformers pakai eager/SDPA attention -> tidak ada
    limit shared-mem Triton, jalan stabil. Gemma-2-2b cuma 2B param, muat santai di 1x T4.

    Generator: yield (index, [n teks]) per prompt -> caller bisa tulis incremental (resumable,
    tahan timeout Kaggle) tanpa nunggu semua selesai.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    # Gemma-2 disarankan eager attention (sliding-window interleaved) biar numerik benar.
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="eager")
    m.eval()

    for i, p in enumerate(prompts):
        msgs = [{"role": "user", "content": p}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(m.device)
        with torch.no_grad():
            out = m.generate(
                inputs, do_sample=True, temperature=temperature, top_p=top_p,
                max_new_tokens=max_tokens, num_return_sequences=n,
                pad_token_id=tok.eos_token_id)
        gen = out[:, inputs.shape[1]:]                       # buang prompt
        yield i, tok.batch_decode(gen, skip_special_tokens=True)


# -------------------------------
# Driver
# -------------------------------

def run_generate(input_path: str | Path, out_path: str | Path, *, backend: str,
                 model: str, n: int = 8, temperature: float = 0.7, top_p: float = 0.95,
                 max_tokens: int = 4096, limit: int | None = None, sleep: float = 0.0,
                 tensor_parallel_size: int = 1) -> dict:
    input_path, out_path = Path(input_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = read_jsonl(input_path)
    done = _already_done(out_path)

    todo = []
    for idx, item in enumerate(items):
        pid = problem_id(item, idx)
        soal = get_soal(item)
        if not soal or done.get(pid, 0) >= n:
            continue
        todo.append((pid, item, wrap(soal, mode="cot")))
        if limit is not None and len(todo) >= limit:
            break

    if not todo:
        return {"problems": len(items), "generated": 0, "skipped": len(items)}

    written = 0
    if backend == "vllm":
        all_cands = generate_vllm([p for _, _, p in todo], model=model, n=n,
                                  temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                                  tensor_parallel_size=tensor_parallel_size)
        with open(out_path, "a", encoding="utf-8") as f:
            for (pid, item, _), cands in zip(todo, all_cands):
                for ci, text in enumerate(cands):
                    f.write(json.dumps(_row(pid, item, ci, text), ensure_ascii=False) + "\n")
                    written += 1
    elif backend == "hf":  # transformers: tulis incremental per soal (resumable, tahan timeout)
        with open(out_path, "a", encoding="utf-8") as f:
            for i, cands in generate_hf([p for _, _, p in todo], model=model, n=n,
                                        temperature=temperature, top_p=top_p, max_tokens=max_tokens):
                pid, item, _ = todo[i]
                for ci, text in enumerate(cands):
                    f.write(json.dumps(_row(pid, item, ci, text), ensure_ascii=False) + "\n")
                    written += 1
                f.flush()
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

    return {"problems": len(items), "generated": written, "todo_problems": len(todo)}


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
    ap.add_argument("--backend", choices=["api", "vllm", "hf"], default="api")
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
    args = ap.parse_args()

    model = args.model or (DEFAULT_API_MODEL if args.backend == "api" else DEFAULT_VLLM_MODEL)
    stats = run_generate(args.input, args.out, backend=args.backend, model=model,
                         n=args.num_candidates, temperature=args.temperature,
                         top_p=args.top_p, max_tokens=args.max_tokens,
                         limit=args.limit, sleep=args.sleep,
                         tensor_parallel_size=args.tensor_parallel_size)
    print(f"Teacher={model} backend={args.backend} | {stats}")


if __name__ == "__main__":
    main()
