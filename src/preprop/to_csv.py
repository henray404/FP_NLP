"""
Convert extracted JSONL -> filtered CSV dengan kolom: soal, cara, jawaban.

Filter rule-based (hardcoded):
  - Hapus soal pilihan ganda (pola A/B/C/D/E)
  - Hapus soal true/false
  - Hapus soal yang butuh gambar
  - Hapus soal bahasa Inggris
  - Normalisasi: kalimat biasa = plain text, notasi math = LaTeX

Usage:
    python -m src.preprop.to_csv data/extracted/soal.jsonl data/filtered/soal.csv
    python -m src.preprop.to_csv data/extracted/ data/filtered/soal.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path

from langdetect import detect, LangDetectException


# ─────────────────────────────────────────────────────────────────────────────
# FILTER RULES
# ─────────────────────────────────────────────────────────────────────────────

_MC_RE = re.compile(r"^\s*[A-Ea-e][.)]\s+\S", re.MULTILINE)

_TF_RE = re.compile(
    r"\b("
    r"pernyataan\s+berikut\s+(yang\s+)?(benar|salah)|"
    r"benar\s+atau\s+salah|"
    r"true\s+or\s+false|"
    r"tentukan\s+(mana\s+yang\s+)?(benar|salah)"
    r")\b",
    re.IGNORECASE,
)

_IMAGE_RE = re.compile(
    r"\b("
    r"lihat\s+(gambar|foto|tabel|diagram|grafik)|"
    r"perhatikan\s+(gambar|foto|tabel|diagram|grafik)|"
    r"pada\s+gambar|"
    r"gambar\s+(di\s+)?(atas|bawah|samping|berikut)|"
    r"sesuai\s+(dengan\s+)?gambar|"
    r"berdasarkan\s+gambar|"
    r"ditunjukkan\s+(pada\s+)?gambar|"
    r"gambar\s+\d+|"
    r"pada\s+foto|"
    r"foto\s+di\s+(atas|bawah)|"
    r"yang\s+saya\s+kirim|"
    r"terlampir"
    r")\b",
    re.IGNORECASE,
)


def _is_multiple_choice(text: str) -> bool:
    return len(_MC_RE.findall(text)) >= 3


def _is_true_false(text: str) -> bool:
    return bool(_TF_RE.search(text))


def _needs_image(text: str) -> bool:
    return bool(_IMAGE_RE.search(text))


def _is_indonesian(text: str) -> bool:
    try:
        return detect(text[:500]) == "id"
    except LangDetectException:
        return True  # ragu-ragu: loloskan


def passes_all_rules(soal: str) -> tuple[bool, str]:
    if not soal or len(soal.strip()) < 10:
        return False, "too_short"
    if _is_multiple_choice(soal):
        return False, "multiple_choice"
    if _is_true_false(soal):
        return False, "true_false"
    if _needs_image(soal):
        return False, "needs_image"
    if not _is_indonesian(soal):
        return False, "not_indonesian"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# TEXT + LATEX NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

_FRAC_RE = re.compile(r"(?<!\$)(\d+)\s*/\s*(\d+)(?!\$)")
_POW_RE  = re.compile(r"(?<!\$)([a-zA-Z0-9]+)\^(\d+)(?!\$)")
_SQRT_RE = re.compile(r"(?<!\$)[√∛](\d+)(?!\$)")
_SUB_RE  = re.compile(r"(?<!\$)([a-zA-Z])_(\d+)(?!\$)")


def normalize_math(text: str) -> str:
    """Pertahankan LaTeX existing, konversi pola plain math ke LaTeX inline."""
    protected: dict[str, str] = {}

    def protect(m: re.Match) -> str:
        key = f"__MATH{len(protected)}__"
        protected[key] = m.group(0)
        return key

    # Protect existing LaTeX blocks
    text = re.sub(r"\$\$[\s\S]*?\$\$", protect, text)
    text = re.sub(r"\$[^$\n]+?\$", protect, text)
    text = re.sub(r"\\\[[\s\S]*?\\\]", protect, text)
    text = re.sub(r"\\\([\s\S]*?\\\)", protect, text)

    # Convert plain math patterns
    text = _FRAC_RE.sub(lambda m: f"$\\frac{{{m.group(1)}}}{{{m.group(2)}}}$", text)
    text = _POW_RE.sub(lambda m: f"${m.group(1)}^{{{m.group(2)}}}$", text)
    text = _SQRT_RE.sub(lambda m: f"$\\sqrt{{{m.group(1)}}}$", text)
    text = _SUB_RE.sub(lambda m: f"${m.group(1)}_{{{m.group(2)}}}$", text)

    for key, val in protected.items():
        text = text.replace(key, val)

    return text.strip()


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_math(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLUMNS = ["soal", "cara", "jawaban"]


def load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def process_item(item: dict) -> dict | None:
    """
    Normalize dan filter satu item.
    Input keys: question/soal (wajib), solution/cara/steps (opsional), answer/jawaban (opsional).
    """
    soal    = item.get("question") or item.get("soal") or ""
    cara    = item.get("solution") or item.get("cara") or item.get("steps") or ""
    jawaban = item.get("answer")   or item.get("jawaban") or ""

    soal = clean_text(soal)
    ok, _ = passes_all_rules(soal)
    if not ok:
        return None

    return {
        "soal":    soal,
        "cara":    clean_text(cara),
        "jawaban": clean_text(jawaban),
    }


def _write_csv(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(rows: list[dict], output_path: Path) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run(
    input_path: str | Path,
    output_path: str | Path,
    fmt: str = "jsonl",
    verbose: bool = False,
) -> dict:
    """
    fmt: "jsonl" (default) atau "csv"
    Output extension otomatis disesuaikan jika tidak cocok dengan fmt.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    # Auto-fix extension jika tidak sesuai format
    expected_ext = ".jsonl" if fmt == "jsonl" else ".csv"
    if output_path.suffix != expected_ext:
        output_path = output_path.with_suffix(expected_ext)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        jsonl_files = sorted(input_path.glob("*.jsonl"))
        all_items = []
        for f in jsonl_files:
            all_items.extend(load_jsonl(f))
        if verbose:
            print(f"Loaded {len(all_items)} items from {len(jsonl_files)} files in {input_path}")
    else:
        all_items = load_jsonl(input_path)
        if verbose:
            print(f"Loaded {len(all_items)} items from {input_path}")

    reject_counts: dict[str, int] = {}
    rows: list[dict] = []

    for item in all_items:
        result = process_item(item)
        if result is None:
            soal = clean_text(item.get("question") or item.get("soal") or "")
            _, reason = passes_all_rules(soal)
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
        else:
            rows.append(result)

    if fmt == "csv":
        _write_csv(rows, output_path)
    else:
        _write_jsonl(rows, output_path)

    if verbose:
        print(f"\nFormat  : {fmt.upper()}")
        print(f"Output  : {output_path}")
        print(f"Passed  : {len(rows)}/{len(all_items)}")
        if reject_counts:
            print("Rejected:")
            for k, v in sorted(reject_counts.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v}")

    return {"total": len(all_items), "passed": len(rows), "rejected": reject_counts}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter & convert extracted JSONL -> JSONL atau CSV (soal, cara, jawaban)"
    )
    parser.add_argument("input",  help=".jsonl file atau folder berisi .jsonl")
    parser.add_argument("output", help="Output path (ekstensi otomatis disesuaikan)")
    parser.add_argument(
        "--format", choices=["jsonl", "csv"], default="jsonl",
        help="Output format: jsonl (default) atau csv",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    stats = run(args.input, args.output, fmt=args.format, verbose=args.verbose)
    print(f"Done: {stats['passed']}/{stats['total']} soal -> {args.output}")
