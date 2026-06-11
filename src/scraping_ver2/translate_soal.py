"""
Translate English entries in data/un_pdfs/_all_soal.jsonl to Indonesian.
Entries already in Indonesian are copied unchanged.

Usage:
    python -m src.scraping_ver2.translate_soal
    python -m src.scraping_ver2.translate_soal --input data/un_pdfs/_all_soal.jsonl
    python -m src.scraping_ver2.translate_soal --batch-size 32
    python -m src.scraping_ver2.translate_soal --dry-run   # show stats only, no write
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
FIELDS_TO_TRANSLATE = ("question", "answer", "steps")

# Indonesian tokens: if any appear, text is almost certainly ID
_ID_TOKENS = {
    "dan", "atau", "yang", "dengan", "dari", "untuk", "pada", "adalah",
    "ini", "itu", "akan", "tidak", "jika", "maka", "nilai", "carilah",
    "tentukan", "hitunglah", "buktikan", "diketahui", "misalkan",
    "penyelesaian", "sebuah", "suatu", "bilangan", "persamaan",
}

_LATEX_RE = re.compile(
    r"\$[^$]+\$"               # inline math $...$
    r"|\\[a-zA-Z]+\{[^}]*\}"   # \cmd{...}
    r"|\\[a-zA-Z]+"            # bare \cmd
    r"|[0-9+\-*/^_=().]+"      # numerics / operators
)


def _clean_for_detection(text: str) -> str:
    return _LATEX_RE.sub(" ", text).strip()


def detect_language(text: str) -> str:
    """Return 'id', 'en', or 'unknown'."""
    clean = _clean_for_detection(text).lower()
    words = set(re.findall(r"[a-z]+", clean))
    if not words:
        return "unknown"

    if words & _ID_TOKENS:
        return "id"

    try:
        from langdetect import detect, LangDetectException  # type: ignore
        lang = detect(clean)
        return "en" if lang == "en" else "id"
    except Exception:
        pass

    return "unknown"


def _protect_latex(text: str) -> tuple[str, dict[str, str]]:
    """Replace inline LaTeX $...$ with placeholders to survive translation."""
    placeholders: dict[str, str] = {}
    counter = [0]

    def replace(m: re.Match) -> str:
        key = f"__LATEX{counter[0]}__"
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    protected = re.sub(r"\$[^$]+\$", replace, text)
    return protected, placeholders


def _restore_latex(text: str, placeholders: dict[str, str]) -> str:
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


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


def translate_entries(
    entries: list[dict],
    batch_size: int = 16,
) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/3] Loading {TRANSLATION_MODEL} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL).to(device)
    model.eval()

    # (entry_idx, field, protected_text, placeholders)
    tasks: list[tuple[int, str, str, dict]] = []
    en_indices: set[int] = set()

    for i, entry in enumerate(entries):
        q = entry.get("question", "")
        lang = detect_language(q)
        if lang != "en":
            continue
        en_indices.add(i)
        for field in FIELDS_TO_TRANSLATE:
            val = entry.get(field)
            if isinstance(val, str) and len(val.strip()) >= 4:
                protected, ph = _protect_latex(val)
                tasks.append((i, field, protected, ph))

    print(f"   Detected {len(en_indices):,} English entries → {len(tasks):,} field translations")

    translations: dict[tuple[int, str], str] = {}
    for start in tqdm(range(0, len(tasks), batch_size), desc="Translating"):
        batch = tasks[start : start + batch_size]
        texts = [t[2] for t in batch]
        results = _translate_batch(texts, tokenizer, model, device)
        for (idx, field, _, ph), translated in zip(batch, results):
            translations[(idx, field)] = _restore_latex(translated, ph)

    out: list[dict] = []
    for i, entry in enumerate(entries):
        if i not in en_indices:
            out.append({**entry})
            continue
        new_entry = {**entry, "translated_from": "en"}
        for field in FIELDS_TO_TRANSLATE:
            key = (i, field)
            if key in translations:
                new_entry[field] = translations[key]
        out.append(new_entry)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate English entries in clean_vlm.jsonl to Indonesian"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/un_pdfs/clean_vlm.jsonl",
        help="Input JSONL (default: data/un_pdfs/clean_vlm.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL (default: <input stem>_id.jsonl in same dir)",
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
        help="Print language detection stats only, skip translation and write",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    output_path = (
        Path(args.output)
        if args.output
        else input_path.parent / f"{input_path.stem}_id.jsonl"
    )

    print(f"[1/3] Reading {input_path}...")
    entries: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    print(f"   Loaded {len(entries):,} entries")

    if args.dry_run:
        en = id_ = unk = 0
        for e in entries:
            lang = detect_language(e.get("question", ""))
            if lang == "en":
                en += 1
            elif lang == "id":
                id_ += 1
            else:
                unk += 1
        print(f"\nDry-run stats:")
        print(f"  Indonesian : {id_:,}")
        print(f"  English    : {en:,}")
        print(f"  Unknown    : {unk:,}")
        return

    translated = translate_entries(entries, batch_size=args.batch_size)

    print(f"\n[3/3] Writing {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in translated:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    en_count = sum(1 for e in translated if e.get("translated_from") == "en")
    print(f"   Written {len(translated):,} entries ({en_count:,} translated) → {output_path} ({size_mb:.2f} MB)")
    print("\nDone!")


if __name__ == "__main__":
    main()
