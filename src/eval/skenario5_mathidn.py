"""
SKENARIO 5 — Evaluasi pada tolok ukur EKSTERNAL berbahasa Indonesia (MATH-IDN).

Kenapa ada skenario ini
-----------------------
Skenario 1-4 semuanya diukur di holdout BUATAN SENDIRI (600 soal), jadi angkanya
tidak bisa dibandingkan siapa pun. MATH-IDN (Xiao dkk., Findings of the ACL:
EACL 2026, hal. 4432-4438) adalah 500 soal MATH500 yang diterjemahkan ke Bahasa
Indonesia/Jawa/Sunda/Bugis lalu diverifikasi penutur asli — satu-satunya tolok
ukur penalaran matematika berbahasa Indonesia yang sudah peer-reviewed.

Yang dilaporkan skenario ini BUKAN peringkat melawan model 7B-9B di paper mereka
(soal MATH500 = tingkat kompetisi, jauh di luar domain latih kita yang aritmetika
dasar + UN/OSN). Yang dilaporkan adalah SELISIH terhadap Qwen2.5-1.5B dasar pada
tolok ukur yang sama: apakah distilasi CoT berbahasa Indonesia menaikkan akurasi
out-of-domain, dan apakah efek CoT vs non-CoT bertahan di luar holdout sendiri.

Angka rujukan dari paper MATH-IDN (akurasi, subset Bahasa Indonesia, zero-shot CoT):
    Qwen2.5-7B-Instruct         62.40%
    Qwen2.5-Math-7B-Instruct    61.80%
    Gemma-2-9b-it               58.20%
    Llama-SEA-LION-v3.5-8B-R    57.80%
    LLaMA-3.1-8B-Instruct       48.40%
    Bakpia-V1-1.5B-Javanese     30.20%
    Komodo-7B-Base              20.60%

Sumber data
-----------
Paper mencantumkan https://github.com/aialt/MATH-IND, tetapi per Juli 2026 repo
itu belum dapat diakses publik (404). Dua jalur:

  (a) RESMI  — minta berkasnya ke penulis (Xiao.Xiao@liverpool.ac.uk,
      i.nimah@tue.nl), simpan sebagai JSONL, lalu:
          python -m src.eval.skenario5_mathidn eval --data data/eval/mathidn_id.jsonl ...
      Berkas cukup punya field soal + jawaban (nama kolom otomatis dideteksi).

  (b) REPLIKA — bangun sendiri dari HuggingFaceH4/MATH-500 dengan prompt
      terjemahan yang sama persis seperti Tabel 2 paper MATH-IDN:
          python -m src.eval.skenario5_mathidn build --out data/eval/math500_id.jsonl
      Bedanya dengan MATH-IDN resmi: terjemahan TIDAK melewati penyuntingan
      penerjemah manusia. Kalau jalur ini yang dipakai, di naskah WAJIB ditulis
      sebagai "replikasi protokol MATH-IDN", bukan sebagai MATH-IDN, dan angkanya
      tidak boleh disandingkan sebaris dengan angka paper mereka.

Protokol evaluasi (disamakan dengan MATH-IDN)
---------------------------------------------
  - zero-shot chain-of-thought, system prompt Bahasa Indonesia dari Tabel 1 paper
  - satu sampel per soal, dekode greedy (bukan n=5 seperti Skenario 2-4)
  - jawaban diambil dari \\boxed{} terakhir (protokol penilaian MATH)
  - metrik: accuracy = (1/N) * jumlah jawaban benar

`accuracy` memakai \\boxed{} ketat supaya sebanding dengan MATH. `accuracy_lenient`
memakai fallback angka terakhir; selisih keduanya = berapa banyak skor yang hilang
murni karena pelanggaran format, bukan karena salah nalar.

Generation butuh GPU -> jalankan di Kaggle T4. Bagian skor murni CPU.

Cara menjalankan (Kaggle T4, ~500 soal x 3 model)
--------------------------------------------------
    python -m src.eval.skenario5_mathidn eval \\
        --data data/eval/mathidn_id.jsonl \\
        --base Qwen/Qwen2.5-1.5B \\
        --spec "Qwen2.5-1.5B dasar" - \\
        --spec "Qwen2.5-1.5B CoT"   adapters/adapter_cot_1.5b \\
        --spec "Qwen2.5-1.5B non-CoT" adapters/adapter_nocot_1.5b \\
        --baseline "Qwen2.5-1.5B dasar" \\
        --out data/eval/skenario5_results.json

Kirimkan `data/eval/skenario5_results.json` untuk diisikan ke Tabel VIII naskah.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.eval.answer_check import extract_answer, extract_boxed, is_correct

# ─────────────────────────────────────────────────────────────────────────────
# PROTOKOL MATH-IDN
# ─────────────────────────────────────────────────────────────────────────────

# Tabel 1 paper MATH-IDN, baris "Indonesian" — disalin persis.
SYSTEM_PROMPT_ID = (
    "Kamu adalah ahli matematika. Tolong selesaikan persoalan matematika berikut "
    "dan berikan solusinya secara rinci."
)

# Model kita dilatih untuk menaruh jawaban akhir di \boxed{}; instruksi ini tidak
# ada di paper MATH-IDN, tapi tanpa itu penilaian \boxed ala MATH tidak adil bagi
# SEMUA model yang diuji (termasuk baseline). Diberikan seragam ke semua model.
BOXED_HINT = "Tuliskan jawaban akhir di dalam \\boxed{}."

# Tabel 2 paper MATH-IDN — prompt penerjemahan, dipakai oleh subcommand `build`.
TRANSLATE_INSTRUCTION = (
    "Terjemahkan teks Bahasa Inggris berikut menjadi teks Bahasa Indonesia. "
    "Jangan mengubah teks berformat latex maupun ekspresi matematika di dalamnya."
)
TRANSLATE_ONESHOT_IN = (
    "Convert the point (0, 3) in rectangular coordinates to polar coordinates. "
    "Enter your answer in the form (r, \\theta), where r > 0 and 0 \\le \\theta < 2\\pi."
)
TRANSLATE_ONESHOT_OUT = (
    "Konversikan titik (0, 3) dalam koordinat persegi panjang ke koordinat polar. "
    "Masukkan jawaban Anda dalam bentuk (r, \\theta), di mana r > 0 dan 0 \\le \\theta < 2\\pi."
)

# Akurasi bahasa Indonesia yang DILAPORKAN paper MATH-IDN (untuk konteks di tabel).
MATHIDN_REPORTED_ID = {
    "Qwen2.5-7B-Instruct": 0.6240,
    "Qwen2.5-Math-7B-Instruct": 0.6180,
    "Gemma-2-9b-it": 0.5820,
    "Llama-SEA-LION-v3.5-8B-R": 0.5780,
    "LLaMA-3.1-8B-Instruct": 0.4840,
    "Bakpia-V1-1.5B-Javanese": 0.3020,
    "Komodo-7B-Base": 0.2060,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

_SOAL_KEYS = ("soal", "problem", "question", "indonesian", "problem_id", "text_id")
_GOLD_KEYS = ("jawaban", "answer", "gold", "solution_answer", "final_answer")


def _pick(row: dict, keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_problems(path: Path) -> list[dict]:
    """Baca JSONL/JSON MATH-IDN -> [{soal, jawaban}]. Nama kolom dideteksi otomatis."""
    raw = path.read_text(encoding="utf-8").strip()
    rows = json.loads(raw) if raw.startswith("[") else [
        json.loads(line) for line in raw.splitlines() if line.strip()
    ]
    out = []
    for r in rows:
        soal, gold = _pick(r, _SOAL_KEYS), _pick(r, _GOLD_KEYS)
        if soal and gold:
            out.append({"soal": soal, "jawaban": gold})
    if not out:
        raise SystemExit(
            f"{path}: tidak ada baris dengan pasangan soal+jawaban yang terbaca. "
            f"Kolom yang dicari: {_SOAL_KEYS} dan {_GOLD_KEYS}."
        )
    print(f"  {path.name}: {len(out)}/{len(rows)} soal terbaca")
    return out


def build_math500_id(out_path: Path, *, model: str = "gemini-2.5-flash",
                     limit: int | None = None) -> None:
    """Replika protokol MATH-IDN: MATH-500 (Inggris) -> Bahasa Indonesia via Gemini.

    Bukan MATH-IDN resmi: tanpa tahap penyuntingan penerjemah manusia. Lihat
    docstring modul sebelum memakai angkanya di naskah.
    """
    from datasets import load_dataset
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("set GEMINI_API_KEY (atau GOOGLE_API_KEY) dulu.")
    client = genai.Client(api_key=api_key)

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    if out_path.exists():  # resume
        done = sum(1 for line in out_path.open(encoding="utf-8") if line.strip())
        print(f"  resume: {done} baris sudah ada, lanjut dari sana")

    with out_path.open("a", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i < done:
                continue
            prompt = (
                f"{TRANSLATE_INSTRUCTION}\n\n"
                f"{TRANSLATE_ONESHOT_IN}\nTerjemahan:\n{TRANSLATE_ONESHOT_OUT}\n\n"
                f"{row['problem']}\nTerjemahan:"
            )
            resp = client.models.generate_content(model=model, contents=prompt)
            f.write(json.dumps({
                "soal": (resp.text or "").strip(),
                "jawaban": row["answer"],
                "soal_en": row["problem"],
                "subject": row.get("subject"),
                "level": row.get("level"),
            }, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(ds)}")
    print(f"selesai -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# EVALUASI
# ─────────────────────────────────────────────────────────────────────────────

def score(rows: list[dict], generations: list[str]) -> dict:
    """Nilai satu model. accuracy = protokol MATH (\\boxed ketat)."""
    strict = lenient = boxed = 0
    details = []
    for r, gen in zip(rows, generations):
        gold = r["jawaban"]
        b = extract_boxed(gen)
        ok_strict = is_correct(b, gold)
        ok_lenient = is_correct(extract_answer(gen), gold)
        strict += ok_strict
        lenient += ok_lenient
        boxed += b is not None
        details.append({"soal": r["soal"][:120], "gold": gold, "pred": b,
                        "correct": ok_strict})
    n = len(rows) or 1
    return {
        "n": len(rows),
        "accuracy": round(strict / n, 4),
        "accuracy_lenient": round(lenient / n, 4),
        "format_ok_rate": round(boxed / n, 4),
        "correct": strict,
        "details": details,
    }


def generate(model_id: str, rows: list[dict], *, adapter_dir: str | None = None,
             max_new_tokens: int = 2048, batch_size: int = 8) -> list[str]:
    """Zero-shot CoT greedy, satu sampel per soal. Butuh GPU."""
    import torch

    from src.eval.scenario_eval import _load_model

    model, tok = _load_model(model_id, adapter_dir)
    gens: list[str] = []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT_ID},
             {"role": "user", "content": f"{r['soal']}\n\n{BOXED_HINT}"}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(0)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gens.extend(tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                     skip_special_tokens=True))
        print(f"    {min(i + batch_size, len(rows))}/{len(rows)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return gens


def render_table(results: dict, baseline: str | None = None) -> str:
    """Tabel markdown siap salin ke naskah: akurasi + selisih terhadap baseline."""
    base_acc = results.get(baseline, {}).get("accuracy") if baseline else None
    head = "| Model | Akurasi | Δ vs dasar | Akurasi (lenient) | Kepatuhan format |"
    lines = [head, "|---|---|---|---|---|"]
    for label, s in results.items():
        delta = "–" if base_acc is None else f"{s['accuracy'] - base_acc:+.3f}"
        lines.append(f"| {label} | {s['accuracy']:.3f} | {delta} | "
                     f"{s['accuracy_lenient']:.3f} | {s['format_ok_rate']:.3f} |")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="replika MATH-IDN dari MATH-500 via Gemini")
    b.add_argument("--out", default="data/eval/math500_id.jsonl")
    b.add_argument("--model", default="gemini-2.5-flash")
    b.add_argument("--limit", type=int, default=None)

    e = sub.add_parser("eval", help="evaluasi model pada berkas soal Bahasa Indonesia")
    e.add_argument("--data", required=True, help="JSONL MATH-IDN (atau replika)")
    e.add_argument("--base", default="Qwen/Qwen2.5-1.5B", help="HF base model id")
    e.add_argument("--spec", nargs=2, action="append", required=True,
                   metavar=("LABEL", "ADAPTER_DIR"),
                   help="label + path adapter LoRA; pakai '-' untuk tanpa adapter")
    e.add_argument("--baseline", default=None,
                   help="label yang dipakai sebagai pembanding kolom Δ")
    e.add_argument("--out", default="data/eval/skenario5_results.json")
    e.add_argument("--max-new-tokens", type=int, default=2048)
    e.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    if args.cmd == "build":
        build_math500_id(Path(args.out), model=args.model, limit=args.limit)
        return

    rows = load_problems(Path(args.data))
    results: dict[str, dict] = {}
    for label, adapter in args.spec:
        print(f"\n=== {label} ===")
        gens = generate(args.base, rows,
                        adapter_dir=None if adapter == "-" else adapter,
                        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)
        res = score(rows, gens)
        results[label] = {k: v for k, v in res.items() if k != "details"}
        print(f"  -> acc {res['accuracy']:.3f} (lenient {res['accuracy_lenient']:.3f}) "
              f"| format_ok {res['format_ok_rate']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"protocol": "MATH-IDN zero-shot CoT, 1 sampel, greedy",
                    "data": args.data, "results": results,
                    "mathidn_reported_id": MATHIDN_REPORTED_ID},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + render_table(results, args.baseline))
    print("\nsummary ->", args.out)


if __name__ == "__main__":
    main()
