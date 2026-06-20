"""
SKENARIO 2 & 3 — eval student kita (base + LoRA adapter) di test set.

S2: model CoT terbaik diuji di beberapa dataset test (numglue, un) -> lihat generalisasi
    lintas-dataset.
S3: model fine-tune CoT vs model fine-tune non-CoT, di test set yang sama -> isolasi efek CoT.

Reuse grader & prompt dari skenario4_eval. Dukung adapter LoRA (PeftModel) di atas base.
Generation butuh GPU -> jalan di Kaggle. Bagian skor/tabel murni CPU (unit-testable).

Usage (programmatic, lihat notebooks/skenario/):
    from src.eval.scenario_eval import eval_specs, render_table, load_sets
    sets = load_sets({"numglue": "data/sft/test/numglue_test.jsonl",
                      "un":      "data/sft/test/un_test.jsonl"})
    specs = {"CoT":   {"model": BASE, "adapter": "/kaggle/working/adapter_cot_1.5b"},
             "nonCoT":{"model": BASE, "adapter": "/kaggle/working/adapter_nocot_1.5b"}}
    res = eval_specs(specs, sets)
    print(render_table(res))
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.answer_check import grade
from src.eval.skenario4_eval import PROMPT, load_holdout, score_generations


def load_sets(paths: dict[str, str]) -> dict[str, list[dict]]:
    """{nama_set: path} -> {nama_set: rows}. Skip set yang filenya tak ada."""
    sets = {}
    for name, p in paths.items():
        if Path(p).exists():
            sets[name] = load_holdout(Path(p))
        else:
            print(f"WARNING: set '{name}' dilewati (tak ketemu: {p})")
    return sets


def _load_model(model_id: str, adapter_dir: str | None):
    """Load base (fp16) + tokenizer, tempel LoRA adapter kalau ada. Butuh GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    # Base Qwen2.5 (non-Instruct) tak punya chat_template -> pasang ChatML biar prompt konsisten.
    if tok.chat_template is None:
        tok.chat_template = (
            "{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + "
            "'<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    return model, tok


def evaluate(model_id: str, rows: list[dict], *, adapter_dir: str | None = None,
             max_new_tokens: int = 2048, batch_size: int = 16) -> dict:
    """Generate greedy + grade satu (model[,adapter]) di satu set soal. Butuh GPU."""
    import torch

    model, tok = _load_model(model_id, adapter_dir)
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
        print(f"    {min(i + batch_size, len(rows))}/{len(rows)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_generations(rows, generations)


def eval_specs(specs: dict[str, dict], sets: dict[str, list[dict]],
               *, max_new_tokens: int = 2048, batch_size: int = 16) -> dict:
    """specs = {label: {"model": id, "adapter": dir|None}}; sets = {nama: rows}.
    Return results[label][set] = ringkasan akurasi (tanpa details)."""
    results: dict[str, dict] = {}
    for label, spec in specs.items():
        results[label] = {}
        for set_name, rows in sets.items():
            print(f"=== {label} @ {set_name} ({len(rows)} soal) ===")
            res = evaluate(spec["model"], rows, adapter_dir=spec.get("adapter"),
                           max_new_tokens=max_new_tokens, batch_size=batch_size)
            results[label][set_name] = {k: v for k, v in res.items() if k != "details"}
            print(f"  -> acc {res['accuracy']:.3f} | format_ok {res['format_ok_rate']:.3f}")
    return results


def render_table(results: dict, *, metric: str = "accuracy") -> str:
    """Tabel markdown: baris = model, kolom = test set (nilai = metric, default accuracy)."""
    labels = list(results)
    set_names = sorted({s for r in results.values() for s in r})
    header = "| model | " + " | ".join(set_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(set_names)) + "|"
    lines = [header, sep]
    for lab in labels:
        cells = []
        for s in set_names:
            v = results[lab].get(s, {}).get(metric)
            cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="S2/S3 eval: student (base+adapter) di test set")
    ap.add_argument("--base", required=True, help="HF base model id (mis. Qwen/Qwen2.5-1.5B)")
    ap.add_argument("--spec", nargs=2, action="append", metavar=("LABEL", "ADAPTER_DIR"),
                    required=True, help="label + path adapter LoRA (ulang per model)")
    ap.add_argument("--set", nargs=2, action="append", metavar=("NAME", "PATH"),
                    required=True, help="nama test set + path jsonl (ulang per set)")
    ap.add_argument("--out", default="data/eval/scenario_results.json")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    sets = load_sets({name: path for name, path in args.set})
    specs = {label: {"model": args.base, "adapter": adir} for label, adir in args.spec}
    results = eval_specs(specs, sets, max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)

    print("\n" + render_table(results))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nsummary ->", args.out)


if __name__ == "__main__":
    main()
