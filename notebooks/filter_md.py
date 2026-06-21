"""Filter MD hasil PDF->Markdown: pisahkan yang BAGUS vs HANCUR.

"Hancur" = ekstraksi rusak sampai konten math tak terpakai:
  - font tak ter-map  -> banyak `(cid:NN)`            (cid_per_kc tinggi)
  - encoding gagal     -> banyak `�` (replacement)  (repl_per_kc tinggi)
  - simbol/akar/pecahan hilang jadi sel kosong         (alpha_ratio rendah)
  - ekstraksi kosong / nyaris kosong                   (chars/words minim)

Output:
  - tabel metrik + verdict ke stdout (urut terburuk dulu)
  - data/filtered/md_quality.csv   (semua file + skor + alasan)
  - data/filtered/md_good.txt      (stem md bagus -> bisa dipakai langsung)
  - data/filtered/md_hancur.txt    (stem -> PDF ini perlu di-VLM ulang)

Jalankan:
  python notebooks/filter_md.py
  python notebooks/filter_md.py --md-dir "G:/.../osn_md" --out data/filtered
"""
import argparse
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MD = Path(
    r"G:/.shortcut-targets-by-id/1Y8RaC_XzbNTbjpMEdlrkNvL3k5GtC-0u/DATA_NLP/raw/0md/osn_md"
)

# ── Ambang (dari kalibrasi 79 md osn) ─────────────────────────────────
CID_PER_KC_MAX  = 3.0    # >3 (cid:NN) per 1000 char  -> font hancur
REPL_PER_KC_MAX = 1.0    # >1 replacement-char per 1000 char -> encoding hancur
ALPHA_MIN       = 0.20   # <20% huruf alfabet -> simbol-soup / math hilang
MIN_CHARS       = 500    # < 500 char -> ekstraksi gagal/kosong
MIN_WORDS       = 50     # < 50 kata  -> ekstraksi gagal/kosong

_CID   = re.compile(r"\(cid:\d+\)")
_REPL  = re.compile("�")
_WORD  = re.compile(r"[A-Za-z]{2,}")


def metrics(text: str) -> dict:
    n = len(text) or 1
    alpha = sum(c.isalpha() for c in text)
    return {
        "chars":       len(text),
        "words":       len(_WORD.findall(text)),
        "cid":         len(_CID.findall(text)),
        "repl":        len(_REPL.findall(text)),
        "cid_per_kc":  len(_CID.findall(text)) / n * 1000,
        "repl_per_kc": len(_REPL.findall(text)) / n * 1000,
        "alpha_ratio": alpha / n,
    }


def verdict(m: dict) -> tuple[str, str]:
    """Return (status, alasan). status in {good, hancur}."""
    reasons = []
    if m["chars"] < MIN_CHARS or m["words"] < MIN_WORDS:
        reasons.append(f"kosong (chars={m['chars']}, words={m['words']})")
    if m["cid_per_kc"] > CID_PER_KC_MAX:
        reasons.append(f"cid {m['cid_per_kc']:.1f}/kc")
    if m["repl_per_kc"] > REPL_PER_KC_MAX:
        reasons.append(f"repl {m['repl_per_kc']:.1f}/kc")
    if m["alpha_ratio"] < ALPHA_MIN:
        reasons.append(f"alpha {m['alpha_ratio']:.2f}")
    return ("hancur", "; ".join(reasons)) if reasons else ("good", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-dir", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "filtered")
    args = ap.parse_args()

    files = sorted(args.md_dir.glob("*.md"))
    if not files:
        print(f"[ERR] tak ada .md di {args.md_dir}")
        return
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        m = metrics(f.read_text(encoding="utf-8", errors="replace"))
        status, why = verdict(m)
        rows.append((f.stem, status, why, m))

    # urut: hancur dulu (cid terburuk), lalu good
    rows.sort(key=lambda r: (r[1] == "good", -r[3]["cid_per_kc"], r[3]["alpha_ratio"]))

    n_good = sum(1 for r in rows if r[1] == "good")
    print(f"{'file':40s} {'status':6s} {'cid/kc':>6s} {'repl/kc':>7s} {'alpha':>5s} {'words':>6s}  alasan")
    for stem, status, why, m in rows:
        print(f"{stem[:40]:40s} {status:6s} {m['cid_per_kc']:6.1f} {m['repl_per_kc']:7.1f} "
              f"{m['alpha_ratio']:5.2f} {m['words']:6d}  {why}")
    print(f"\n{len(rows)} md  ->  {n_good} good  |  {len(rows)-n_good} hancur")

    # tulis manifest
    csv_path = args.out / "md_quality.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["stem", "status", "alasan", "chars", "words",
                    "cid_per_kc", "repl_per_kc", "alpha_ratio"])
        for stem, status, why, m in rows:
            w.writerow([stem, status, why, m["chars"], m["words"],
                        f"{m['cid_per_kc']:.2f}", f"{m['repl_per_kc']:.2f}",
                        f"{m['alpha_ratio']:.3f}"])
    (args.out / "md_good.txt").write_text(
        "\n".join(s for s, st, _, _ in rows if st == "good"), encoding="utf-8")
    (args.out / "md_hancur.txt").write_text(
        "\n".join(s for s, st, _, _ in rows if st == "hancur"), encoding="utf-8")
    print(f"\nmanifest -> {csv_path}")
    print(f"           {args.out / 'md_good.txt'}  (pakai langsung)")
    print(f"           {args.out / 'md_hancur.txt'}  (PDF ini perlu VLM)")


if __name__ == "__main__":
    main()
