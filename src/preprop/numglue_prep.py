"""
NumGLUE pre-filter — Task 1, langkah 1 (murah, CPU, jalan duluan sebelum translate/CoT).

Input  : data/NumGlue/NumGLUE_{dev,train,test}.json  (JSONL: {question, answer, type})
Output : data/NumGlue/numglue_{split}_filtered.jsonl  (JSONL: {question, answer, source, source_type})

Membuang "GT jelek" + type non-matematis, lalu menormalkan answer jadi string numerik:
  - Type_1/2/4/8         : jawaban numerik (aritmatika)            -> KEEP
  - Type_5               : dict {number,date,spans}, KEEP hanya kalau `number` terisi
                           (spans/date kosong) -> answer = number
  - Type_3 ("Option N")  : pilihan ganda                          -> DROP (multiple_choice)
  - Type_6 (spans)       : span extraction (reading comprehension) -> DROP (non_math)
  - Type_7 (neutral/..)  : NLI / entailment                        -> DROP (non_math)
  - answer kosong / non-numerik liar                               -> DROP (bad_gt)
  - soal multiple-choice / butuh gambar                            -> DROP (rule)

`type` dipertahankan sebagai `source_type` (dipakai stratified subsample di subsample.py),
lalu DIHAPUS di output akhir setelah subsample. Lihat juga: [[fp-nlp-project]].

Usage:
    python -m src.preprop.numglue_prep                       # ketiga split
    python -m src.preprop.numglue_prep --split dev
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# MC detector dipakai ulang dari filter_rules (language-agnostic: cocokkan "A.", "B)" dst).
from .filter_rules import is_multiple_choice

NON_MATH_TYPES = {"Type_3", "Type_6", "Type_7"}

# answer numerik: int/float/negatif/desimal/persen/pecahan sederhana.
_NUM_RE = re.compile(r"-?\d+(\.\d+)?")
_FRAC_RE = re.compile(r"-?\d+\s*/\s*\d+")

# soal yang merujuk gambar/figur (NumGLUE berbahasa Inggris pada tahap ini).
_IMAGE_EN = re.compile(r"\b(see|shown in|refer to|in)\s+the\s+(figure|image|diagram|picture|graph)\b",
                       re.IGNORECASE)


def normalize_numeric(raw: str) -> str | None:
    """Kembalikan string numerik yang dibersihkan, atau None kalau bukan numerik."""
    s = str(raw).strip()
    if not s:
        return None
    s = s.replace(",", "")                 # 1,000 -> 1000
    s = s.lstrip("$").rstrip("%").strip()  # $5 -> 5 ; 50% -> 50
    if _FRAC_RE.fullmatch(s):
        return s.replace(" ", "")
    m = _NUM_RE.fullmatch(s)
    return m.group(0) if m else None


def classify(item: dict) -> tuple[bool, str, str]:
    """-> (keep, normalized_answer, reason). reason hanya berarti saat keep=False."""
    typ = item.get("type", "")
    ans = item.get("answer")
    q = item.get("question", "") or ""

    if typ in NON_MATH_TYPES:
        return False, "", "non_math_type"
    if is_multiple_choice(q):
        return False, "", "multiple_choice"
    if _IMAGE_EN.search(q):
        return False, "", "needs_image"

    # Type_5: answer berbentuk dict DROP-style.
    if isinstance(ans, dict):
        number = (ans.get("number") or "").strip()
        spans = ans.get("spans") or []
        if number and not spans:
            norm = normalize_numeric(number)
            if norm is not None:
                return True, norm, ""
        return False, "", "bad_gt"  # span/date/number kosong -> bukan GT numerik

    norm = normalize_numeric(ans if ans is not None else "")
    if norm is None:
        return False, "", "bad_gt"
    return True, norm, ""


def run(input_path: Path, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "kept": 0, "rejected": Counter(), "kept_by_type": Counter()}

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            stats["total"] += 1
            keep, norm_ans, reason = classify(item)
            if not keep:
                stats["rejected"][reason] += 1
                continue
            rec = {
                "question": (item.get("question") or "").strip(),
                "answer": norm_ans,
                "source": "numglue",
                "source_type": item.get("type", ""),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            stats["kept_by_type"][item.get("type", "")] += 1

    stats["rejected"] = dict(stats["rejected"])
    stats["kept_by_type"] = dict(sorted(stats["kept_by_type"].items()))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter + normalisasi NumGLUE (Task 1 langkah 1)")
    ap.add_argument("--split", choices=["dev", "train", "test", "all"], default="all")
    ap.add_argument("--in-dir", default="data/NumGlue")
    ap.add_argument("--out-dir", default="data/NumGlue")
    args = ap.parse_args()

    splits = ["dev", "train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        inp = Path(args.in_dir) / f"NumGLUE_{split}.json"
        out = Path(args.out_dir) / f"numglue_{split}_filtered.jsonl"
        stats = run(inp, out)
        print(f"[{split}] total={stats['total']} kept={stats['kept']} "
              f"-> {out}")
        print(f"        rejected={stats['rejected']}")
        print(f"        kept_by_type={stats['kept_by_type']}")


if __name__ == "__main__":
    main()
