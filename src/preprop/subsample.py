"""
Stratified subsample — Task 1, langkah 3 (setelah filter + dedup, sebelum translate).

Mengambil N baris dengan mempertahankan proporsi `source_type` (stratified), lalu
MENGHAPUS field `source_type`/`type` di output (sesuai keputusan: type dibuang).
Deterministik via --seed.

Kalau N >= jumlah baris, semua dipakai (cuma type yang dibuang).

Input  : data/NumGlue/numglue_{split}_dedup.jsonl
Output : data/NumGlue/numglue_{split}_sample.jsonl   (type sudah hilang)

Usage:
    python -m src.preprop.subsample --input data/NumGlue/numglue_train_dedup.jsonl \
        --output data/NumGlue/numglue_train_sample.jsonl --n 10000
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DROP_KEYS = ("source_type", "type")


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def stratified_sample(rows: list[dict], n: int, seed: int,
                      strat_key: str = "source_type") -> list[dict]:
    if n >= len(rows):
        return rows
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[r.get(strat_key, "")].append(r)

    total = len(rows)
    picked: list[dict] = []
    # alokasi proporsional dengan floor, lalu bagikan sisa ke bucket terbesar.
    alloc = {}
    for k, items in buckets.items():
        alloc[k] = int(n * len(items) / total)
    remainder = n - sum(alloc.values())
    for k in sorted(buckets, key=lambda k: -len(buckets[k]))[:max(remainder, 0)]:
        alloc[k] += 1

    for k, items in buckets.items():
        take = min(alloc[k], len(items))
        picked.extend(rng.sample(items, take))

    rng.shuffle(picked)
    return picked[:n]


def strip_type(row: dict) -> dict:
    return {k: v for k, v in row.items() if k not in DROP_KEYS}


def run(input_path: Path, output_path: Path, n: int, seed: int) -> dict:
    rows = load(input_path)
    sampled = stratified_sample(rows, n, seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from collections import Counter
    type_dist = Counter(r.get("source_type", "") for r in sampled)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in sampled:
            f.write(json.dumps(strip_type(r), ensure_ascii=False) + "\n")

    return {"input": len(rows), "sampled": len(sampled),
            "type_dist": dict(sorted(type_dist.items())), "output": str(output_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified subsample + drop type")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    stats = run(Path(args.input), Path(args.output), args.n, args.seed)
    for k, v in stats.items():
        print(f"{k:12}: {v}")


if __name__ == "__main__":
    main()
