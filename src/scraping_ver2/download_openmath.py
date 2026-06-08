"""
Download OpenMath Reasoning dataset (first 20k samples) from HuggingFace.
Save as JSON in English + Indonesian translation.

Usage:
    python -m src.scraping_ver2.download_openmath
    python -m src.scraping_ver2.download_openmath --output data/hugging_face_AIMO/
    python -m src.scraping_ver2.download_openmath --skip-translate   # English only
    python -m src.scraping_ver2.download_openmath --batch-size 32    # larger GPU batch
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

SAMPLE_SIZE = 20_000
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-en-id"
FIELDS_TO_TRANSLATE = ("problem", "cot_solution", "solution")


def load_openmath_samples(size: int = SAMPLE_SIZE) -> list[dict]:
    """Load first `size` rows from OpenMath Reasoning dataset."""
    print(f"[1/4] Loading first {size:,} rows from OpenMath Reasoning dataset...")
    dataset = load_dataset(
        "nvidia/OpenMathReasoning",
        split="cot",
        streaming=True,
    )

    sample: list[dict] = []
    for row in dataset:
        sample.append(row)
        if len(sample) >= size:
            break
        if len(sample) % 5_000 == 0:
            print(f"   Fetched {len(sample):,} rows...")

    print(f"   Collected {len(sample):,} samples")
    return sample


def save_json(data: list[dict], path: Path, label: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"   Saved {label} -> {path} ({size_mb:.2f} MB)")


def _translate_batch(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
    max_src: int = 512,
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


def translate_samples(
    samples: list[dict],
    batch_size: int = 16,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[3/4] Loading translation model {TRANSLATION_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)
    model.eval()
    print(f"   Model loaded. Translating {len(samples):,} samples (batch={batch_size})...")

    # Build flat list of (sample_idx, field, text) to translate in batches
    tasks: list[tuple[int, str, str]] = []
    for i, s in enumerate(samples):
        for field in FIELDS_TO_TRANSLATE:
            val = s.get(field)
            if isinstance(val, str) and len(val) >= 5:
                tasks.append((i, field, val))

    # Translate in batches
    translations: dict[tuple[int, str], str] = {}
    for start in tqdm(range(0, len(tasks), batch_size), desc="Translating batches"):
        batch = tasks[start : start + batch_size]
        texts = [t[2] for t in batch]
        results = _translate_batch(texts, tokenizer, model, device)
        for (idx, field, _), translated in zip(batch, results):
            translations[(idx, field)] = translated

    # Rebuild samples with translations applied (immutable — new dicts)
    translated_samples: list[dict] = []
    for i, s in enumerate(samples):
        new_sample = {**s}
        for field in FIELDS_TO_TRANSLATE:
            key = (i, field)
            if key in translations:
                new_sample[field] = translations[key]
        translated_samples.append(new_sample)

    return translated_samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download first 20k OpenMath samples + translate to Indonesian"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/hugging_face_AIMO",
        help="Output directory (default: data/hugging_face_AIMO)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=SAMPLE_SIZE,
        help=f"Number of samples (default: {SAMPLE_SIZE:,})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Translation batch size (default: 16; increase on GPU)",
    )
    parser.add_argument(
        "--skip-translate",
        action="store_true",
        help="Skip translation, save English JSON only",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    english_file = output_dir / "openmath_reasoning_20k.json"
    indo_file = output_dir / "openmath_reasoning_20k_id.json"

    # Step 1: download first N rows
    samples = load_openmath_samples(args.size)

    # Step 2: save English
    print("\n[2/4] Saving English JSON...")
    save_json(samples, english_file, "English")

    if args.skip_translate:
        print("\n[--skip-translate] Skipping translation.")
        print(f"\n{'='*50}")
        print("Done!")
        print(f"  English : {english_file}")
        return

    # Step 3: translate
    samples_id = translate_samples(samples, batch_size=args.batch_size)

    # Step 4: save Indonesian
    print("\n[4/4] Saving Indonesian JSON...")
    save_json(samples_id, indo_file, "Indonesian")

    print(f"\n{'='*50}")
    print("Done!")
    print(f"  English    : {english_file}")
    print(f"  Indonesian : {indo_file}")


if __name__ == "__main__":
    main()
