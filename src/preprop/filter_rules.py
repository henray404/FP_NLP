"""
Rule-based filter — murah, cepat, jalankan pertama.
Input:  JSONL {"question": str, "answer": str, ...}
Output: JSONL soal yang lolos semua rule
"""
import re
import json
from pathlib import Path

from langdetect import detect_langs, LangDetectException


# ── Rules ──────────────────────────────────────────────────────────────────────

MC_PATTERN = re.compile(
    r"(?:^|\s)[AaBbCcDdEe][.)]\s",
    re.MULTILINE,
)

# LaTeX line break "\\" sering dipakai memisah opsi MC dalam satu baris -> ubah jadi newline.
_LATEX_BREAK = re.compile(r"\\\\")

# Untuk deteksi bahasa: buang math/LaTeX/angka supaya langdetect tidak keok di soal padat rumus.
_LATEX_RE = re.compile(
    r"\$[^$]+\$"
    r"|\\[a-zA-Z]+\{[^}]*\}"
    r"|\\[a-zA-Z]+"
    r"|[0-9+\-*/^_=().]+"
)

# Token Indonesia: kalau muncul, teks hampir pasti ID (shortcut sebelum langdetect).
_ID_TOKENS = {
    "dan", "atau", "yang", "dengan", "dari", "untuk", "pada", "adalah",
    "ini", "itu", "akan", "tidak", "jika", "maka", "nilai", "carilah",
    "tentukan", "hitunglah", "buktikan", "diketahui", "misalkan", "hasil",
    "penyelesaian", "sebuah", "suatu", "bilangan", "persamaan", "bentuk",
    "sederhana", "berikut", "tersebut", "banyak", "jumlah", "luas", "panjang",
}

IMAGE_KEYWORDS = re.compile(
    r"(lihat\s+gambar|perhatikan\s+gambar|gambar\s+di\s+(atas|bawah|samping)|"
    r"pada\s+gambar|sesuai\s+gambar|berdasarkan\s+gambar)",
    re.IGNORECASE,
)

TRUE_FALSE_PATTERN = re.compile(
    r"\b(benar|salah|true|false)\b.*\b(benar|salah|true|false)\b",
    re.IGNORECASE,
)


def is_multiple_choice(question: str) -> bool:
    # normalkan LaTeX line break "\\" jadi newline supaya opsi MC inline ikut terdeteksi.
    norm = _LATEX_BREAK.sub("\n", question)
    return len(MC_PATTERN.findall(norm)) >= 3


def needs_image(question: str) -> bool:
    return bool(IMAGE_KEYWORDS.search(question))


def is_true_false(question: str) -> bool:
    return bool(TRUE_FALSE_PATTERN.search(question))


def is_indonesian(text: str) -> bool:
    """Tahan-LaTeX. Buang HANYA kalau yakin bahasa lain (mis. Inggris). Strategi:
    1) buang math/LaTeX dulu, 2) teks terlalu pendek -> keep (tak cukup utk dinilai),
    3) ada token ID -> keep, 4) langdetect: buang hanya jika top bukan 'id' & prob > 0.85."""
    clean = _LATEX_RE.sub(" ", text)
    words = set(re.findall(r"[a-zA-Z]+", clean.lower()))
    if len(words) < 4:
        return True              # mis. "Faktorkan ...", "Bagaimana sifat parabola"
    if words & _ID_TOKENS:
        return True
    try:
        top = detect_langs(clean)[0]
    except (LangDetectException, IndexError):
        return True              # ragu -> jangan buang
    return not (top.lang != "id" and top.prob > 0.85)


def passes_rules(item: dict) -> tuple[bool, str]:
    q = (item.get("question") or item.get("soal") or "")
    if is_multiple_choice(q):
        return False, "multiple_choice"
    if is_true_false(q):
        return False, "true_false"
    if needs_image(q):
        return False, "needs_image"
    if not is_indonesian(q):
        return False, "not_indonesian"
    return True, ""


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_filter(input_path: str | Path, output_path: str | Path) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "passed": 0, "rejected": {}}

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            stats["total"] += 1
            ok, reason = passes_rules(item)
            if ok:
                stats["passed"] += 1
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    return stats


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/sample.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/filtered/after_rules.jsonl"
    stats = run_filter(inp, out)
    print(f"Total: {stats['total']} | Passed: {stats['passed']} | Rejected: {stats['rejected']}")
