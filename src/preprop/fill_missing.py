"""
Isi `jawaban`/`cara` yang kosong pakai DeepSeek-R1-Distill-Qwen-7B (LLM-as-generator).
JALAN DI GPU (Kaggle T4x2 / Colab), BUKAN di laptop. Butuh: vllm.

Aturan:
  - Baris yang `jawaban` ATAU `cara`-nya kosong diproses.
  - Kalau `jawaban` SUDAH ada -> dipertahankan; DeepSeek cuma ngisi `cara`.
  - Kalau `jawaban` kosong -> isi \\boxed{} dari output jadi `jawaban`, reasoning jadi `cara`.
  - HANYA untuk train_pool (bukan gold eval), sesuai keputusan proyek.

Resumable: hasil ditulis incremental ke fill_cache.jsonl; rerun = skip yang sudah ada.

Usage (di Kaggle):
  python -m src.preprop.fill_missing \
      --input data/dataset_dedup.jsonl \
      --cache data/fill_cache.jsonl \
      --output data/dataset_filled.jsonl
"""
import argparse
import json
from pathlib import Path

from src.eval.answer_check import extract_boxed

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

PROMPT = (
    "Selesaikan soal matematika berikut. Tunjukkan langkah-langkah penyelesaian "
    "secara rinci dalam Bahasa Indonesia. Pastikan jawaban akhir berada di dalam "
    "\\boxed{{}}.\n\n{soal}"
)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def needs_fill(r: dict) -> bool:
    return not (r.get("jawaban") or "").strip() or not (r.get("cara") or "").strip()


def run(input_path: Path, cache_path: Path, output_path: Path,
        batch_size: int = 16, max_tokens: int = 3072, tp_size: int = 2):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    rows = load_jsonl(input_path)
    todo = [r for r in rows if needs_fill(r)]

    # resume: soal yang sudah ada di cache di-skip
    done: dict[str, dict] = {}
    if cache_path.exists():
        for r in load_jsonl(cache_path):
            done[r["soal"]] = r
    todo = [r for r in todo if r["soal"] not in done]
    print(f"total {len(rows)} | perlu fill total {sum(needs_fill(r) for r in rows)} | "
          f"sudah di cache {len(done)} | sisa diproses {len(todo)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, dtype="float16", tensor_parallel_size=tp_size,
              gpu_memory_utilization=0.90, max_model_len=4096)
    sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as cache_f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "user", "content": PROMPT.format(soal=r["soal"])}],
                    tokenize=False, add_generation_prompt=True)
                for r in batch
            ]
            outs = llm.generate(prompts, sp)
            for r, o in zip(batch, outs):
                gen = o.outputs[0].text
                boxed = extract_boxed(gen)
                jaw = (r.get("jawaban") or "").strip()
                filled = {
                    "soal": r["soal"],
                    "cara": gen.strip(),
                    "jawaban": jaw if jaw else (boxed or ""),
                    "_filled": True,
                    "_fill_ok": bool(jaw) or boxed is not None,
                }
                done[r["soal"]] = filled
                cache_f.write(json.dumps(filled, ensure_ascii=False) + "\n")
                cache_f.flush()
            print(f"  progress {min(i + batch_size, len(todo))}/{len(todo)}")

    # gabung: pakai versi filled kalau ada, else baris asli
    n_fail = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for r in rows:
            f = done.get(r["soal"])
            if f is not None:
                if not f.get("_fill_ok", True):
                    n_fail += 1
                rec = {"soal": f["soal"], "cara": f["cara"], "jawaban": f["jawaban"]}
            else:
                rec = {"soal": r["soal"], "cara": r.get("cara", ""), "jawaban": r.get("jawaban", "")}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    still_empty = sum(1 for r in load_jsonl(output_path) if not (r.get("jawaban") or "").strip())
    print(f"\nDONE -> {output_path}")
    print(f"  fill gagal (no boxed): {n_fail}")
    print(f"  jawaban masih kosong : {still_empty}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/dataset_dedup.jsonl")
    p.add_argument("--cache", default="data/fill_cache.jsonl")
    p.add_argument("--output", default="data/dataset_filled.jsonl")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=3072)
    p.add_argument("--tp-size", type=int, default=2, help="2 utk Kaggle T4x2, 1 utk single GPU")
    args = p.parse_args()
    run(Path(args.input), Path(args.cache), Path(args.output),
        args.batch_size, args.max_tokens, args.tp_size)
