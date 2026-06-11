#!/usr/bin/env python3
"""Merge math datasets into one JSONL with columns: soal, cara, jawaban.

Sources (all already share the soal/cara/jawaban schema):
  - un_pdfs/dataset_sma.jsonl
  - un_pdfs/clean_vlm.jsonl
  - hugging_face_AIMO/parts_v2/part_00000_05000.jsonl

Each output line is a JSON object with exactly those three keys.
Records missing all three fields, or whose soal+jawaban are both empty,
are dropped. Exact-duplicate records are skipped.
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent

SOURCES = [
    BASE / "un_pdfs" / "dataset_sma.jsonl",
    BASE / "un_pdfs" / "clean_vlm.jsonl",
    BASE / "hugging_face_AIMO" / "parts_v2" / "part_00000_05000.jsonl",
    BASE / "hugging_face_AIMO" / "parts_v2" / "part_05000_20000.jsonl",
]

OUTPUT = BASE / "merged_dataset.jsonl"

COLUMNS = ("soal", "cara", "jawaban")


def normalize(obj: dict):
    """Keep only the target columns as stripped strings. None if unusable."""
    rec = {k: (obj.get(k) or "").strip() for k in COLUMNS}
    if not rec["soal"] and not rec["jawaban"]:
        return None
    return rec


def read_jsonl(path: Path):
    """Yield parsed JSON objects from a JSONL file, skipping blank/bad lines."""
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARN {path.name}:{ln} bad JSON, skipped ({e})")


def main():
    total_in = 0
    written = 0
    dropped = 0
    dupes = 0
    seen = set()

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for src in SOURCES:
            if not src.exists():
                print(f"MISSING: {src} — skipped")
                continue
            count = 0
            for obj in read_jsonl(src):
                total_in += 1
                rec = normalize(obj)
                if rec is None:
                    dropped += 1
                    continue
                key = (rec["soal"], rec["cara"], rec["jawaban"])
                if key in seen:
                    dupes += 1
                    continue
                seen.add(key)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
                count += 1
            print(f"  {src.name}: {count} written")

    print(
        f"\nDone -> {OUTPUT.name}\n"
        f"  read   : {total_in}\n"
        f"  written: {written}\n"
        f"  dropped: {dropped} (empty soal+jawaban)\n"
        f"  dupes  : {dupes} (exact match)"
    )


if __name__ == "__main__":
    main()
