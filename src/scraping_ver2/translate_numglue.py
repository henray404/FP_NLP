"""
Translate NumGLUE (sudah difilter+subsample) EN->ID — Task 1, langkah 4.

Math-safe (lihat translate_core): jawaban numerik di-copy verbatim, soal NL di-translate.
Checkpoint per blok -> aman di-resume (re-run lanjut dari baris terakhir di disk).

Input  : data/NumGlue/numglue_{split}_sample.jsonl   {question, answer, source}
Output : data/NumGlue/numglue_{split}_id.jsonl        {soal, jawaban, source}

Jalankan di lingkungan ber-GPU (Kaggle/Colab). Di CPU sangat lambat.

Usage:
    python -m src.scraping_ver2.translate_numglue --split test
    python -m src.scraping_ver2.translate_numglue --input <in.jsonl> --output <out.jsonl>
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from tqdm.auto import tqdm

from .translate_core import load_model, process_block

FIELD_MAP = [("question", "soal"), ("answer", "jawaban")]
BLOCK_SIZE = 100


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def run(input_path: Path, output_path: Path, *, block_size: int = BLOCK_SIZE,
        batch_size: int = 8, num_beams: int = 2) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = read_jsonl(input_path)

    done = 0
    if output_path.exists():
        done = sum(1 for l in open(output_path, encoding="utf-8") if l.strip())
        print(f"Resume: {done} baris sudah ada, dilewati")

    if done >= len(data):
        print("Sudah lengkap, tidak ada yang dikerjakan")
        return {"total": len(data), "translated": 0, "resumed": True}

    import torch  # noqa: F401  (dipakai via translate_core)
    model, tokenizer, tgt_id, device = load_model()
    print(f"Device: {device}")

    written = 0
    pbar = tqdm(total=len(data), initial=done, desc=f"Translate {input_path.stem}")
    with open(output_path, "a", encoding="utf-8") as f:
        for lo in range(done, len(data), block_size):
            block = data[lo:lo + block_size]
            recs = process_block(block, FIELD_MAP, model, tokenizer, tgt_id, device,
                                 batch_size=batch_size, num_beams=num_beams)
            for src_row, rec in zip(block, recs):
                rec["source"] = src_row.get("source", "numglue")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            f.flush()
            pbar.update(len(block))
            gc.collect()
            if device.type == "cuda":
                import torch
                torch.cuda.empty_cache()
    pbar.close()
    return {"total": len(data), "translated": written, "output": str(output_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate NumGLUE EN->ID (math-safe)")
    ap.add_argument("--split", choices=["dev", "train", "test"], default=None)
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=2)
    args = ap.parse_args()

    if args.split:
        inp = Path(args.input or f"data/NumGlue/numglue_{args.split}_sample.jsonl")
        out = Path(args.output or f"data/NumGlue/numglue_{args.split}_id.jsonl")
    else:
        if not args.input or not args.output:
            ap.error("beri --split, atau --input dan --output")
        inp, out = Path(args.input), Path(args.output)

    stats = run(inp, out, block_size=args.block_size,
                batch_size=args.batch_size, num_beams=args.num_beams)
    print(stats)


if __name__ == "__main__":
    main()
