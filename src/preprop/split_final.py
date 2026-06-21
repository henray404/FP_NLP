"""
Split data/Final jadi train/test (held-out) — CEGAH LEAKAGE (CPU, stdlib only).

Tiap dataset (numglue/easy/aimo_hard) di-carve: TEST_N soal jadi held-out test,
sisanya train. Deterministik (seed). Dedup internal sudah dilakukan sebelumnya, jadi
test tidak punya kembaran di train.

PENTING: SFT/training HANYA boleh pakai split TRAIN. Eval (skenario 1 & 4) pakai TEST.

Output:
  data/Final/train/{ds}_train.jsonl
  data/Final/test/{ds}_test.jsonl

Usage:
  python -m src.preprop.split_final            # TEST_N=300/dataset
  python -m src.preprop.split_final --test-n 250 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FINAL = Path("data/Final")
DATASETS = ("numglue", "easy", "aimo_hard")


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def write(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(test_n: int, seed: int) -> dict:
    stats = {}
    for ds in DATASETS:
        src = FINAL / f"{ds}_clean.jsonl"
        if not src.exists():
            stats[ds] = "tak ada"
            continue
        rows = load(src)
        rng = random.Random(seed)
        rng.shuffle(rows)
        n_test = min(test_n, len(rows) // 2)  # jaga2 dataset kecil: test tak lebih dari separuh
        test, train = rows[:n_test], rows[n_test:]
        write(test, FINAL / "test" / f"{ds}_test.jsonl")
        write(train, FINAL / "train" / f"{ds}_train.jsonl")
        stats[ds] = {"total": len(rows), "train": len(train), "test": len(test)}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Split data/Final -> train/test (anti-leakage)")
    ap.add_argument("--test-n", type=int, default=300, help="jumlah held-out test per dataset")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = run(args.test_n, args.seed)
    for ds, s in stats.items():
        print(f"  {ds:10}: {s}")
    print("\nTRAIN -> data/Final/train/  (buat SFT)")
    print("TEST  -> data/Final/test/   (buat eval skenario 1 & 4)")
    print("INGAT: SFT JANGAN pakai soal di test/.")


if __name__ == "__main__":
    main()
