"""
Pembersihan data Final (CPU, stdlib only). Berdasarkan audit kualitas:

- NumGLUE: gabung dev+test+train, buang soal yang butuh paragraf konteks (heuristik: TANPA angka
  di soal -> DROP-style reading-comprehension), dedup exact-soal. Output 1 file.
- AIMO   : buang baris dgn GT korup (kode subtitle ASS `{\\cH..}`/`{\\fn..}`, "aku tidak tahu", dst).
- easy   : buang jawaban MC/pernyataan yang nyelip.

Output ditulis ke data/Final/. NumGLUE split lama dihapus dari Final (diganti numglue_clean.jsonl).

Usage:
  python -m src.preprop.clean_final
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FINAL = Path("data/Final")

# --- NumGLUE: soal butuh konteks kalau tak ada satu pun angka ---
_HAS_DIGIT = re.compile(r"\d")

# --- AIMO: pola GT korup ---
_ASS_TAG = re.compile(r"\{\\|\\cH|\\fn|\\fs|\\b1|\\4cH|\\4aH")
_JUNK_ANS = re.compile(r"^\s*(aku tidak tahu|selengkapnya|tidak terhitung|tidak diketahui)\b", re.I)

# --- easy: jawaban MC/pernyataan ---
_MC_ANS = re.compile(r"pernyataan|pilihan ganda|opsi|semua (pernyataan|benar)", re.I)


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def numglue_combine() -> dict:
    files = [FINAL / f"numglue_{s}_final.jsonl" for s in ("dev", "test", "train")]
    rows = []
    for f in files:
        if f.exists():
            rows += load(f)
    total = len(rows)
    seen = set()
    kept = []
    dropped_ctx = dropped_dup = 0
    for r in rows:
        soal = (r.get("soal") or "").strip()
        if not _HAS_DIGIT.search(soal):          # butuh konteks paragraf -> buang
            dropped_ctx += 1
            continue
        if soal in seen:                          # dedup exact lintas-split
            dropped_dup += 1
            continue
        seen.add(soal)
        kept.append({"soal": soal, "cara": r.get("cara", ""),
                     "jawaban": r.get("jawaban", ""), "source": "numglue"})
    out = FINAL / "numglue_clean.jsonl"
    write(kept, out)
    for f in files:                               # ganti split lama dgn file gabungan
        if f.exists():
            f.unlink()
    return {"in": total, "kept": len(kept), "drop_butuh_konteks": dropped_ctx,
            "drop_duplikat": dropped_dup, "out": str(out)}


def filter_file(name: str, bad) -> dict:
    p = FINAL / name
    if not p.exists():
        return {"file": name, "skip": "tak ada"}
    rows = load(p)
    kept = [r for r in rows if not bad(r)]
    write(kept, p)
    return {"file": name, "in": len(rows), "kept": len(kept), "dibuang": len(rows) - len(kept)}


def main() -> None:
    print("== NumGLUE: gabung + buang butuh-konteks + dedup ==")
    print("  ", numglue_combine())

    print("== AIMO: buang GT korup ==")
    print("  ", filter_file("aimo_hard_clean.jsonl",
          lambda r: bool(_ASS_TAG.search(r.get("jawaban", "")) or _JUNK_ANS.search(r.get("jawaban", "")))))

    print("== easy: buang jawaban MC/pernyataan ==")
    print("  ", filter_file("easy_clean.jsonl",
          lambda r: bool(_MC_ANS.search(r.get("jawaban", "")))))

    print("\nFile final di data/Final/:")
    for f in sorted(FINAL.glob("*.jsonl")):
        print(f"  {sum(1 for _ in open(f)):6}  {f.name}")


if __name__ == "__main__":
    main()
