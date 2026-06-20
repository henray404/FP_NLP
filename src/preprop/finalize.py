"""
Finalisasi dataset -> data/Final/ (CPU, stdlib only).

Per baris:
  - buang field `cara_source` + key internal (`_ok`, `_fill_ok`, `_id`, dst).
  - sisakan schema {soal, cara, jawaban, source?}.
  - DROP baris yang soal/jawaban/cara kosong. `cara` kosong = kelewat/timeout saat generate
    -> dibuang (sesuai keputusan: lebih baik dibuang daripada bolong).

Cek "ada yang kelewat atau nggak": laporan menghitung berapa baris di-drop karena `cara_kosong`.

Usage:
  python -m src.preprop.finalize                      # auto-discover *_final.jsonl & *_clean.jsonl
  python -m src.preprop.finalize --inputs a.jsonl b.jsonl
  python -m src.preprop.finalize --out-dir data/Final
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

KEEP = ("soal", "cara", "jawaban", "source")           # field yang dipertahankan, urut
DISCOVER_GLOBS = ("data/**/*_final.jsonl", "data/**/*_clean.jsonl")


def clean_row(r: dict) -> tuple[dict | None, str]:
    rec = {k: r[k] for k in KEEP if k in r}
    soal = (rec.get("soal") or "").strip()
    cara = (rec.get("cara") or "").strip()
    jaw = (rec.get("jawaban") or "").strip()
    if not soal:
        return None, "soal_kosong"
    if not jaw:
        return None, "jawaban_kosong"
    if not cara:
        return None, "cara_kosong"        # kelewat / timeout
    # tulis ulang yang sudah di-strip
    rec["soal"], rec["cara"], rec["jawaban"] = soal, cara, jaw
    return rec, ""


def process(inp: Path, out: Path) -> dict:
    rows = [json.loads(l) for l in open(inp, encoding="utf-8") if l.strip()]
    kept: list[dict] = []
    drop = Counter()
    for r in rows:
        rec, reason = clean_row(r)
        if rec is None:
            drop[reason] += 1
        else:
            kept.append(rec)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"file": inp.name, "total": len(rows), "kept": len(kept),
            "dropped": dict(drop), "output": str(out)}


def discover() -> list[Path]:
    hits: list[Path] = []
    for pat in DISCOVER_GLOBS:
        hits += [Path(p) for p in glob.glob(pat, recursive=True)]
    # jangan ikutkan yang sudah di data/Final
    return sorted({h for h in hits if "Final" not in h.parts})


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalisasi -> data/Final (buang cara_source, drop cara kosong)")
    ap.add_argument("--inputs", nargs="*", default=None, help="file .jsonl (default: auto-discover)")
    ap.add_argument("--out-dir", default="data/Final")
    args = ap.parse_args()

    inputs = [Path(x) for x in args.inputs] if args.inputs else discover()
    if not inputs:
        print("Tidak ada file ditemukan (cari *_final.jsonl / *_clean.jsonl di data/). "
              "Beri --inputs secara eksplisit.")
        return

    out_dir = Path(args.out_dir)
    grand = Counter()
    print(f"Ditemukan {len(inputs)} file -> {out_dir}/\n")
    for inp in inputs:
        out = out_dir / inp.name
        st = process(inp, out)
        grand["total"] += st["total"]; grand["kept"] += st["kept"]
        for k, v in st["dropped"].items():
            grand[f"drop_{k}"] += v
        print(f"  {st['file']:32} total={st['total']:6} kept={st['kept']:6} dropped={st['dropped']}")
    print(f"\nTOTAL: {dict(grand)}")
    print(f"Selesai -> {out_dir}/")


if __name__ == "__main__":
    main()
