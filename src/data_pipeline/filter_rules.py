"""
Rule-based filter — murah, cepat, jalankan pertama.
Input:  JSONL {"question": str, "answer": str, ...}
Output: JSONL soal yang lolos semua rule
"""
import re
import json
from pathlib import Path

from langdetect import detect, LangDetectException


# ── Rules ──────────────────────────────────────────────────────────────────────

MC_PATTERN = re.compile(
    r"^\s*[AaBbCcDdEe][.)\s]",
    re.MULTILINE,
)

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
    return len(MC_PATTERN.findall(question)) >= 3


def needs_image(question: str) -> bool:
    return bool(IMAGE_KEYWORDS.search(question))


def is_true_false(question: str) -> bool:
    return bool(TRUE_FALSE_PATTERN.search(question))


def is_indonesian(text: str) -> bool:
    try:
        return detect(text) == "id"
    except LangDetectException:
        return False


def passes_rules(item: dict) -> tuple[bool, str]:
    q = item.get("question", "")
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
