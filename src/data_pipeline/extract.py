"""
Extract soal matematika dari PDF.
Output: data/extracted/*.jsonl  format {"question": str, "answer": str, "source": str}

Usage:
    python -m src.data_pipeline.extract data/raw/osn/soal.pdf
    python -m src.data_pipeline.extract data/raw/osn/
"""
import json
import re
from pathlib import Path

import pdfplumber

EXTRACTED_DIR = Path("data/extracted")


# ─────────────────────────────────────────────────────────────────────────────
# CLEANING
# ─────────────────────────────────────────────────────────────────────────────

_URL_RE            = re.compile(r"https?://\S+", re.IGNORECASE)
_TRAILING_JUNK_RE  = re.compile(r"[ \t]+[a-zA-Z.]{1,2}$", re.MULTILINE)
_UNICODE_NOISE_RE  = re.compile(r"[�\U000e0000-\U000effff]")
_SHORT_LINE_RE     = re.compile(r"^.{1,3}$")


def clean_raw_text(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _UNICODE_NOISE_RE.sub("", text)
    text = _TRAILING_JUNK_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    lines = [l for l in text.splitlines() if not _SHORT_LINE_RE.match(l.strip())]
    text = "\n".join(lines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PARSERS
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# SOAL VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

# Kata kunci yang menandakan ini soal matematika
_SOAL_KEYWORDS_RE = re.compile(
    r"\b(tentukan|hitunglah|carilah|buktikan|tunjukkan|hitung|cari|"
    r"berapakah|berapa|banyaknya|nilai\s+dari|ada\s+berapa|"
    r"adalah|jika|diketahui|diberikan|misalkan|suatu|sebuah)\b",
    re.IGNORECASE,
)

# Pola angka/ekspresi math (indikator soal matematika)
_MATH_INDICATOR_RE = re.compile(
    r"(\d+[+\-*/^=<>]\d+|"       # operasi: 2+3, x=5
    r"\d+[a-zA-Z]|"               # koefisien tanpa spasi: 2x, 3n (bukan "1 Situbondo")
    r"[a-zA-Z]\s*=\s*-?\d+|"     # variabel: x = 5, n = -3
    r"[a-zA-Z]\s*\^\s*\d+|"      # pangkat: x^2
    r"\\[a-zA-Z]+\{|"             # LaTeX command: \frac{
    r"\$[^$\n]+\$)"               # inline math
)


def is_valid_soal(text: str) -> bool:
    """
    True jika teks ini soal matematika valid.
    Filter:
    - Terlalu pendek (< 30 char) -> bukan soal, mungkin judul/strategi
    - Terlalu panjang (> 1500 char) -> pembahasan/materi, bukan soal murni
    - Tidak ada kata kunci soal DAN tidak ada indikator math -> bukan soal
    """
    text = text.strip()
    if len(text) < 30:
        return False
    if len(text) > 1500:
        return False
    has_keyword = bool(_SOAL_KEYWORDS_RE.search(text))
    has_math    = bool(_MATH_INDICATOR_RE.search(text))
    return has_keyword or has_math


# Format A: "SOAL N\n[teks]\nJawaban: [angka]"
_SOAL_BLOCK_RE    = re.compile(r"SOAL\s+\d+\s*\n(.*?)(?=SOAL\s+\d+|\Z)", re.DOTALL | re.IGNORECASE)
_JAWABAN_INLINE_RE = re.compile(r"Jawaban\s*:\s*(.+)", re.IGNORECASE)

# Format B: "1. [teks soal]" numbered list
_NUMBERED_RE = re.compile(r"(?:^|\n)\s*(\d{1,2})\.\s+(.*?)(?=\n\s*\d{1,2}\.\s+|\Z)", re.DOTALL)
_KEY_RE      = re.compile(r"^\s*(\d{1,2})[.)]\s+(.+)$", re.MULTILINE)


def parse_format_a(full_text: str, source: str) -> list[dict]:
    pairs = []
    for m in _SOAL_BLOCK_RE.finditer(full_text):
        block = m.group(1).strip()
        jaw = _JAWABAN_INLINE_RE.search(block)
        jawaban = jaw.group(1).strip() if jaw else ""
        soal = _JAWABAN_INLINE_RE.sub("", block).strip()
        if is_valid_soal(soal):
            pairs.append({"question": soal, "answer": jawaban, "cara": "", "source": source})
    return pairs


def parse_format_b(full_text: str, source: str) -> list[dict]:
    kunci_marker = re.search(
        r"\n(KUNCI|JAWABAN|PEMBAHASAN|PENYELESAIAN)\b", full_text, re.IGNORECASE
    )
    soal_text  = full_text[: kunci_marker.start()] if kunci_marker else full_text
    kunci_text = full_text[kunci_marker.start():] if kunci_marker else ""

    kunci: dict[str, str] = {}
    for km in _KEY_RE.finditer(kunci_text):
        kunci[km.group(1)] = km.group(2).strip()

    pairs = []
    for m in _NUMBERED_RE.finditer(soal_text):
        nomor = m.group(1)
        soal  = re.sub(r"\n+", " ", m.group(2)).strip()
        if is_valid_soal(soal):
            pairs.append({
                "question": soal,
                "answer":   kunci.get(nomor, ""),
                "cara":     "",
                "source":   source,
            })
    return pairs


def parse_qa_pairs(full_text: str, source: str) -> list[dict]:
    if re.search(r"SOAL\s+\d+", full_text, re.IGNORECASE):
        pairs = parse_format_a(full_text, source)
        if pairs:
            return pairs
    return parse_format_b(full_text, source)


# ─────────────────────────────────────────────────────────────────────────────
# PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_to_pairs(pdf_path: str | Path) -> list[dict]:
    pdf_path = Path(pdf_path)
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    full_text = clean_raw_text("\n".join(pages_text))
    pairs = parse_qa_pairs(full_text, pdf_path.stem)
    for p in pairs:
        p["source_file"] = pdf_path.name
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def process_file(pdf_path: Path, out_dir: Path, verbose: bool = False) -> int:
    pairs = extract_pdf_to_pairs(pdf_path)
    if not pairs:
        if verbose:
            print(f"  [  0 soal] {pdf_path.name}")
        return 0
    out_path = out_dir / (pdf_path.stem + ".jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    if verbose:
        print(f"  [{len(pairs):3d} soal] {pdf_path.name} -> {out_path.name}")
    return len(pairs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract QA pairs from OSN math PDFs")
    parser.add_argument("input", nargs="?", default="data/raw/osn",
                        help="PDF file atau folder (default: data/raw/osn)")
    parser.add_argument("--out-dir", default=str(EXTRACTED_DIR))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        pdf_files = sorted(input_path.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDFs in {input_path}")
        total = sum(process_file(f, out_dir, verbose=True) for f in pdf_files)
        print(f"\nTotal: {total} soal extracted -> {out_dir}")
    else:
        count = process_file(input_path, out_dir, verbose=True)
        print(f"Extracted {count} soal -> {out_dir}")
