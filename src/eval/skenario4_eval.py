"""
SKENARIO 4 — Model kita vs model lain (eval zero-shot di holdout KITA).

Eval beberapa model di data/eval/holdout.jsonl: prompt minta solusi + jawaban di
\\boxed{}, ekstrak jawaban, cocokkan dengan ground-truth pakai answer_check.
Output: akurasi per-model + detail per-soal.

Model lain (~1.5B, sebanding student kita):
  - nvidia/OpenMath-Nemotron-1.5B   (DARI PAPER 2504.16891)
  - SeaLLMs/SeaLLMs-v3-1.5B-Chat    (ngerti Indonesia)
  - Qwen/Qwen2.5-1.5B-Instruct      (base umum)
Tambah model kita sendiri begitu training non-CoT selesai.

Generation butuh GPU -> jalan di Kaggle (lihat notebooks/skenario4_eval_kaggle.ipynb).
Modul ini bisa juga dijalankan: python -m src.eval.skenario4_eval --holdout ...
"""
import argparse
import json
from pathlib import Path

from src.eval.answer_check import grade

PROMPT = (
    "Selesaikan soal matematika berikut. Tunjukkan langkah-langkah penyelesaian "
    "secara rinci. Pastikan jawaban akhir berada di dalam \\boxed{{}}.\n\n{soal}"
)

DEFAULT_MODELS = {
    "OpenMath-Nemotron-1.5B": "nvidia/OpenMath-Nemotron-1.5B",
    "SeaLLMs-v3-1.5B-Chat": "SeaLLMs/SeaLLMs-v3-1.5B-Chat",
    "Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def load_holdout(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def score_generations(rows: list[dict], generations: list[str]) -> dict:
    """Grade hasil generasi (CPU). rows & generations sejajar. Return ringkasan + detail."""
    details = []
    correct = boxed = 0
    for r, gen in zip(rows, generations):
        g = grade(gen, r.get("jawaban", ""))
        correct += g["correct"]
        boxed += g["has_boxed"]
        details.append({
            "soal": r["soal"][:120],
            "gold": r.get("jawaban", ""),
            "pred": g["pred"],
            "correct": g["correct"],
        })
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "correct": correct,
        "format_ok_rate": round(boxed / n, 4) if n else 0.0,  # % yang ngehasilin \boxed
        "details": details,
    }


def evaluate_model(model_id: str, rows: list[dict],
                   max_new_tokens: int = 2048, batch_size: int = 16) -> dict:
    """Load model (transformers), generate, grade. Butuh GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()

    generations = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(soal=r["soal"])}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(0)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        generations.extend(tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                            skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(rows))}/{len(rows)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_generations(rows, generations)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", default="data/eval/holdout.jsonl")
    p.add_argument("--models", nargs="*", default=None,
                   help="HF model id; default = 3 model pembanding")
    p.add_argument("--out", default="data/eval/skenario4_results.json")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=16)
    args = p.parse_args()

    rows = load_holdout(Path(args.holdout))
    model_ids = args.models or list(DEFAULT_MODELS.values())

    summary = {}
    for mid in model_ids:
        print(f"\n=== {mid} ===")
        res = evaluate_model(mid, rows, args.max_new_tokens, args.batch_size)
        summary[mid] = {k: v for k, v in res.items() if k != "details"}
        print(f"  -> acc {res['accuracy']:.3f} | format_ok {res['format_ok_rate']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n=== RINGKASAN SKENARIO 4 ===")
    for mid, s in summary.items():
        print(f"  {mid:40} acc={s['accuracy']:.3f}  format_ok={s['format_ok_rate']:.3f}")
