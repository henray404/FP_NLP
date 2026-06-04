"""
Merge JSONL files into a single CSV.

Default:
  - Input:  data/extracted/*.jsonl
  - Output: data/extracted_csv/all_extracted.csv

Usage:
  python -m src.scraping.jsonl_to_csv
  python -m src.scraping.jsonl_to_csv data/extracted data/extracted_csv/all_extracted.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


PREFERRED_COLUMNS = [
    "datasource",
    "question",
    "answer",
    "steps",
    "source_file",
    "source_page",
    "source_page_end",
    "question_id",
    "source_bbox",
    "has_image",
    "image_files",
    "image_bboxes",
    "image_text",
    "extraction_method",
    "raw_text",
]


def iter_jsonl_paths(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.jsonl"))
    return [input_path]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def normalize_row(row: dict, datasource: str) -> dict:
    data = dict(row)
    data["datasource"] = data.get("source_file") or datasource
    return data


def build_columns(rows: list[dict]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    ordered = [c for c in PREFERRED_COLUMNS if c in keys]
    extras = sorted(k for k in keys if k not in ordered)
    return ordered + extras


def stringify_value(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = build_columns(rows)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: stringify_value(row.get(k)) for k in columns})


def run(input_path: Path, output_path: Path) -> dict:
    rows: list[dict] = []
    jsonl_paths = list(iter_jsonl_paths(input_path))
    for path in jsonl_paths:
        datasource = path.name
        for row in load_rows(path):
            rows.append(normalize_row(row, datasource))

    if rows:
        write_csv(rows, output_path)

    return {
        "files": len(jsonl_paths),
        "rows": len(rows),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge JSONL files into a CSV")
    parser.add_argument("input", nargs="?", default="data/extracted")
    parser.add_argument("output", nargs="?", default="data/extracted_csv/all_extracted.csv")
    args = parser.parse_args()

    stats = run(Path(args.input), Path(args.output))
    print(f"Files: {stats['files']} | Rows: {stats['rows']} -> {stats['output']}")


if __name__ == "__main__":
    main()
