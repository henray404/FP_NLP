"""
Bangun data SFT non-CoT dari train_pool.jsonl.

Target = (cara natural, kalau pendek) + jawaban di \\boxed{}.
Baris dengan cara terlalu panjang (mis. hasil DeepSeek-fill) -> cara di-drop,
jadi jawaban-aja, supaya tetap "non-CoT" (bukan reasoning raksasa).

Format output (chat) per baris:
  {"messages": [{"role":"user","content": <prompt+soal>},
                {"role":"assistant","content": <cara?>\\n\\n\\boxed{jawaban}}]}

Usage:
  python -m src.training.build_sft --input data/train_pool.jsonl \
      --output data/train_sft.jsonl --max-cara 1200
"""
import argparse
import json
from pathlib import Path

PROMPT = (
    "Selesaikan soal matematika berikut. Tunjukkan langkah-langkah penyelesaian "
    "secara rinci. Pastikan jawaban akhir berada di dalam \\boxed{{}}.\n\n{soal}"
)


def build_target(cara: str, jawaban: str, max_cara: int) -> str:
    cara = (cara or "").strip()
    boxed = "\\boxed{" + jawaban.strip() + "}"
    if cara and len(cara) <= max_cara:
        return cara + "\n\n" + boxed
    return boxed


def run(input_path: Path, output_path: Path, max_cara: int) -> dict:
    rows = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_with_cara = n_answer_only = skipped = 0
    tgt_lens = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for r in rows:
            jaw = (r.get("jawaban") or "").strip()
            if not jaw:                       # tanpa jawaban -> tak bisa jadi target
                skipped += 1
                continue
            cara = (r.get("cara") or "").strip()
            target = build_target(cara, jaw, max_cara)
            if cara and len(cara) <= max_cara:
                n_with_cara += 1
            else:
                n_answer_only += 1
            tgt_lens.append(len(target))
            rec = {"messages": [
                {"role": "user", "content": PROMPT.format(soal=r["soal"])},
                {"role": "assistant", "content": target},
            ]}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    import statistics as st
    return {
        "input": len(rows),
        "ditulis": n_with_cara + n_answer_only,
        "dengan_cara": n_with_cara,
        "jawaban_aja (cara kosong/kepanjangan)": n_answer_only,
        "dilewati (tanpa jawaban)": skipped,
        "panjang_target median/maks": f"{int(st.median(tgt_lens))}/{max(tgt_lens)}",
        "output": str(output_path),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/train_pool.jsonl")
    p.add_argument("--output", default="data/train_sft.jsonl")
    p.add_argument("--max-cara", type=int, default=1200,
                   help="cara lebih panjang dari ini -> di-drop (jadi jawaban-aja)")
    args = p.parse_args()
    stats = run(Path(args.input), Path(args.output), args.max_cara)
    for k, v in stats.items():
        print(f"{k:40}: {v}")
