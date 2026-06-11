"""
Convert openmath_reasoning_20k*.json to Indonesian JSONL with schema: soal, cara, jawaban.

Reads either the already-partially-translated _id.json (problem in ID, rest in EN)
or the raw English JSON. Translates remaining English fields and outputs a clean JSONL.

Field mapping:
  problem            -> soal   (already ID in _id.json; translated if reading EN json)
  generated_solution -> cara   (<think> block stripped; only post-</think> text translated)
  expected_answer    -> jawaban

Usage:
    # Preferred: use _id.json where problem is already translated
    python -m src.scraping_ver2.convert_openmath_to_indo

    # Fall back to raw English JSON (translates all three fields)
    python -m src.scraping_ver2.convert_openmath_to_indo \\
        --input data/hugging_face_AIMO/openmath_reasoning_20k.json

    python -m src.scraping_ver2.convert_openmath_to_indo --batch-size 32
    python -m src.scraping_ver2.convert_openmath_to_indo --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-id"
DEFAULT_ID_JSON = "data/hugging_face_AIMO/openmath_reasoning_20k_id.json"
DEFAULT_EN_JSON = "data/hugging_face_AIMO/openmath_reasoning_20k.json"
OUTPUT_JSONL = "data/hugging_face_AIMO/openmath_reasoning_indo.jsonl"

MAX_SRC_TOKENS = 480


def _strip_think(text: str) -> str:
    """Return only the post-</think> solution; fall back to full text if tag absent."""
    idx = text.find("</think>")
    if idx >= 0:
        return text[idx + 8:].strip()
    return text.strip()


def _protect_latex(text: str) -> tuple[str, dict[str, str]]:
    """Replace $...$ and \\(...\\) with stable placeholders."""
    placeholders: dict[str, str] = {}
    counter = [0]

    def replace(m: re.Match) -> str:
        key = f"__M{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    protected = re.sub(r"\\\(.*?\\\)|\$[^$]+\$", replace, text, flags=re.DOTALL)
    return protected, placeholders


def _restore_latex(text: str, placeholders: dict[str, str]) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def _split_by_words(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> list[str]:
    """Force-split a single oversized segment by words until each fits."""
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []

    for word in words:
        candidate = " ".join(current_words + [word])
        if len(tokenizer.encode(candidate, add_special_tokens=False)) > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
        else:
            current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks or [""]


def _chunk_text(text: str, tokenizer: AutoTokenizer, max_tokens: int) -> list[str]:
    """Split on newlines into chunks that fit within max_tokens.
    If a single paragraph is still too long, sub-split by words."""
    paragraphs = text.split("\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        # Sub-split paragraph if it alone exceeds limit
        para_tokens = len(tokenizer.encode(para, add_special_tokens=False))
        if para_tokens > max_tokens:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_by_words(para, tokenizer, max_tokens))
            continue

        candidate = (current + "\n" + para).strip() if current else para
        token_count = len(tokenizer.encode(candidate, add_special_tokens=False))
        if token_count > max_tokens and current:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks or [""]


def _translate_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_src: int = MAX_SRC_TOKENS,
    max_tgt: int = 512,
) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_src,
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_tgt)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def load_model(device: torch.device) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    print(f"  Loading {TRANSLATION_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)
    model.eval()
    return tokenizer, model


def convert(
    data: list[dict],
    problem_already_translated: bool,
    batch_size: int,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model(device)

    fields_to_translate = ["generated_solution", "expected_answer"]
    if not problem_already_translated:
        fields_to_translate.insert(0, "problem")

    print(f"\n  Fields to translate: {fields_to_translate}")
    print(f"  Entries: {len(data):,}")

    # Build task list: (entry_idx, field, chunks, placeholders)
    tasks: list[tuple[int, str, list[str], dict[str, str]]] = []

    for i, entry in enumerate(data):
        for field in fields_to_translate:
            raw = entry.get(field, "") or ""
            if field == "generated_solution":
                raw = _strip_think(raw)
            if not raw.strip():
                continue
            protected, ph = _protect_latex(raw)
            chunks = _chunk_text(protected, tokenizer, MAX_SRC_TOKENS)
            tasks.append((i, field, chunks, ph))

    # Flatten to (task_idx, chunk_idx, text) for batching
    flat: list[tuple[int, int, str]] = [
        (t_idx, c_idx, chunk)
        for t_idx, task in enumerate(tasks)
        for c_idx, chunk in enumerate(task[2])
    ]

    print(f"  Total chunks to translate: {len(flat):,}")

    chunk_results: dict[tuple[int, int], str] = {}
    for start in tqdm(range(0, len(flat), batch_size), desc="Translating"):
        batch = flat[start : start + batch_size]
        texts = [b[2] for b in batch]
        translated = _translate_batch(texts, tokenizer, model, device)
        for (t_idx, c_idx, _), result in zip(batch, translated):
            chunk_results[(t_idx, c_idx)] = result

    # Reassemble per task
    translations: dict[tuple[int, str], str] = {}
    for t_idx, (i, field, chunks, ph) in enumerate(tasks):
        parts = [chunk_results.get((t_idx, c_idx), "") for c_idx in range(len(chunks))]
        translations[(i, field)] = _restore_latex("\n".join(parts), ph)

    # Build output records
    out: list[dict] = []
    for i, entry in enumerate(data):
        soal = (
            entry.get("problem", "")
            if problem_already_translated
            else translations.get((i, "problem"), entry.get("problem", ""))
        )
        cara = translations.get(
            (i, "generated_solution"),
            _strip_think(entry.get("generated_solution", "")),
        )
        jawaban = translations.get((i, "expected_answer"), entry.get("expected_answer", ""))

        out.append({"soal": soal, "cara": cara, "jawaban": jawaban})

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OpenMath JSON to Indonesian JSONL (soal/cara/jawaban)"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="",
        help=(
            f"Input JSON. Default: use {DEFAULT_ID_JSON} if it exists, "
            f"else fall back to {DEFAULT_EN_JSON}"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_JSONL,
        help=f"Output JSONL (default: {OUTPUT_JSONL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Translation batch size (default: 16; increase on GPU)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only, no translation or write",
    )
    args = parser.parse_args()

    # Resolve input path and whether problem is already translated
    if args.input:
        input_path = Path(args.input)
        problem_already_translated = "_id" in input_path.name
    else:
        id_path = Path(DEFAULT_ID_JSON)
        en_path = Path(DEFAULT_EN_JSON)
        if id_path.exists():
            input_path = id_path
            problem_already_translated = True
            print(f"Using {input_path} (problem already in Indonesian)")
        elif en_path.exists():
            input_path = en_path
            problem_already_translated = False
            print(f"Using {input_path} (will translate all fields)")
        else:
            print(f"ERROR: neither {DEFAULT_ID_JSON} nor {DEFAULT_EN_JSON} found", file=sys.stderr)
            sys.exit(1)

    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    print(f"[1/3] Reading {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    print(f"   Loaded {len(data):,} entries")

    if args.dry_run:
        think_count = sum(1 for d in data if "</think>" in d.get("generated_solution", ""))
        sol_lens = [len(_strip_think(d.get("generated_solution", ""))) for d in data]
        avg_sol = sum(sol_lens) // len(sol_lens) if sol_lens else 0
        print(f"\nDry-run stats:")
        print(f"  Entries            : {len(data):,}")
        print(f"  With </think> tag  : {think_count:,}")
        print(f"  Avg solution length: {avg_sol:,} chars (post-think)")
        print(f"  problem translated : {problem_already_translated}")
        print(f"  Output would be    : {output_path}")
        return

    print(f"\n[2/3] Translating...")
    records = convert(data, problem_already_translated, args.batch_size)

    print(f"\n[3/3] Writing {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Written {len(records):,} entries → {output_path} ({size_mb:.2f} MB)")
    print(f"\nDone! Columns: soal, cara, jawaban")


if __name__ == "__main__":
    main()
