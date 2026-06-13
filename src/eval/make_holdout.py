"""
Split dataset -> holdout (eval) + train_pool.

Holdout HANYA berisi soal berjawaban "bersih" (integer / angka / ekspresi
tunggal) supaya grading otomatis reliable. Soal berjawaban kalimat/kosong tidak
masuk holdout tapi tetap masuk train_pool.

Output:
  data/eval/holdout.jsonl   -> N soal eval (+ field answer_type)
  data/train_pool.jsonl     -> sisanya

Usage:
  python -m src.eval.make_holdout --input data/filtered/after_rules.jsonl --n 300 --seed 42
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

INT = re.compile(r"^-?\d+$")
NUM = re.compile(r"^-?\d+([./]\d+)?$")
GRADEABLE = {"pure_int", "pure_num", "single_expr"}


def answer_type(jawaban: str) -> str:
    j = (jawaban or "").strip()
    if not j:
        return "empty"
    core = j.strip(".").strip("$").strip()
    if INT.match(core):
        return "pure_int"
    if NUM.match(core):
        return "pure_num"
    if len(re.findall(r"[A-Za-z]{3,}", j)) <= 1:
        return "single_expr"
    return "sentence" if ("$" in j or "\\" in j) else "sentence_plain"


def load(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(input_path: Path, out_dir: Path, n: int, seed: int) -> dict:
    rows = load(input_path)
    for r in rows:
        r["answer_type"] = answer_type(r.get("jawaban", ""))

    gradeable = [r for r in rows if r["answer_type"] in GRADEABLE]
    random.Random(seed).shuffle(gradeable)

    n = min(n, len(gradeable))
    holdout = gradeable[:n]
    holdout_ids = {id(r) for r in holdout}
    train_pool = [r for r in rows if id(r) not in holdout_ids]

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    hpath = eval_dir / "holdout.jsonl"
    tpath = out_dir / "train_pool.jsonl"

    with open(hpath, "w", encoding="utf-8") as f:
        for r in holdout:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(tpath, "w", encoding="utf-8") as f:
        for r in train_pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "total": len(rows),
        "gradeable_pool": len(gradeable),
        "holdout": len(holdout),
        "holdout_komposisi": dict(Counter(r["answer_type"] for r in holdout)),
        "train_pool": len(train_pool),
        "train_pool_berjawaban": sum(1 for r in train_pool if (r.get("jawaban") or "").strip()),
        "holdout_path": str(hpath),
        "train_pool_path": str(tpath),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/filtered/after_rules.jsonl")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    stats = run(Path(args.input), Path(args.out_dir), args.n, args.seed)
    for k, v in stats.items():
        print(f"{k:24}: {v}")
