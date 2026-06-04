"""
LLM-based validity filter (Qwen2.5-7B-Instruct).
Cek: soal well-formed dan bisa diselesaikan dari teks saja?
Input:  data/filtered/after_rules.jsonl
Output: data/filtered/after_validity.jsonl
"""
import json
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE = 32

VALIDITY_PROMPT = """\
Kamu adalah penilai soal matematika. Tentukan apakah soal berikut VALID atau TIDAK VALID.

Soal VALID jika:
1. Soal lengkap dan dapat dipahami dari teks saja (tidak butuh gambar/tabel eksternal)
2. Soal memiliki jawaban numerik atau ekspresi matematika yang pasti
3. Soal ditulis dalam Bahasa Indonesia yang jelas

Soal TIDAK VALID jika:
1. Soal membutuhkan referensi visual yang hilang
2. Soal ambigu atau konteks tidak lengkap
3. Soal bersifat konseptual tanpa jawaban pasti

Jawab hanya dengan satu kata: VALID atau TIDAK_VALID

Soal:
{question}

Jawaban:"""


def parse_validity(output: str) -> bool:
    text = output.strip().upper()
    return "VALID" in text and "TIDAK_VALID" not in text


def run_validity_filter(input_path: str | Path, output_path: str | Path) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    llm = LLM(
        model=MODEL_ID,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)

    prompts = [VALIDITY_PROMPT.format(question=item.get("question") or item.get("soal", "")) for item in items]

    results = []
    for i in range(0, len(prompts), BATCH_SIZE):
        outputs = llm.generate(prompts[i:i+BATCH_SIZE], sampling_params)
        results.extend(outputs)

    stats = {"total": len(items), "passed": 0, "rejected": 0}

    with open(output_path, "w", encoding="utf-8") as fout:
        for item, output in zip(items, results):
            if parse_validity(output.outputs[0].text):
                stats["passed"] += 1
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                stats["rejected"] += 1

    return stats


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/filtered/after_rules.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/filtered/after_validity.jsonl"
    stats = run_validity_filter(inp, out)
    print(f"Total: {stats['total']} | Passed: {stats['passed']} | Rejected: {stats['rejected']}")
