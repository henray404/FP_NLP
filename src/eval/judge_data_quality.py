"""
Audit kualitas data pakai LLM-as-judge -- penilaian OBJEKTIF, lepas dari heuristik
self-consistency (inspect_data). Judge menyelesaikan/menilai sendiri, bukan sekadar
membandingkan dua angka yang sudah ada.

Beda dengan filter_solutions (yang nilai pred-vs-gold), di sini judge menilai
KUALITAS DATA-nya:

  TEST  (gold {soal, jawaban, cara?}):
    - gold_benar : apakah `jawaban` (gold) BENAR untuk soal? (judge selesaikan sendiri)
    - soal_lengkap: apakah soal bisa dijawab tanpa tabel/teks/data luar yang tak disertakan?
    -> mengukur seberapa bisa-dipercaya gold test (label noise OBJEKTIF, bukan proxy cara).

  TRAIN (chatml {messages:[user,assistant]}):
    - jawaban_benar : apakah jawaban akhir (\\boxed) di solusi benar?
    - penalaran_valid: apakah langkahnya nyambung (bukan ngarang/lompat)?
    -> mengukur kualitas data SFT yang dipelajari student.

Backend judge (reuse infra cot_synthesis):
  - "api"  : OpenAI-compatible (Groq default). Set GROQ_API_KEY. Cocok sampel kecil.
  - "vllm" : model lokal di GPU (Kaggle/Colab). Untuk audit besar tanpa limit API.

Hemat biaya: default --sample (acak, seed tetap) -> audit subset, bukan semua.
Resumable: verdict per-baris ditulis ke <out>/<nama>.verdicts.jsonl, run ulang skip yang sudah.

Usage:
    export GROQ_API_KEY=...   # atau OPENAI_API_KEY / DEEPSEEK_API_KEY
    python -m src.eval.judge_data_quality --sample 80
    python -m src.eval.judge_data_quality \\
        --set test_numglue=data/sft/test/numglue_test.jsonl \\
        --set train_cot=data/sft/train/cot.jsonl --sample 150 --backend api
    python -m src.eval.judge_data_quality --self-check     # uji logika, tanpa API
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

from src.cot_synthesis.utils import openai_client, with_retry
from src.preprop.inspect_data import (
    _assistant_of_chatml,
    _kind,
    _read_jsonl,
    _soal_of_chatml,
)

DEFAULT_MODEL = "llama-3.1-8b-instant"  # Groq; ganti ke model lebih kuat utk audit serius

# ── prompt judge ────────────────────────────────────────────────────────────────
# Minta judge SELESAIKAN sendiri dulu, baru vonis. Output 1 baris label biar gampang parse.

PROMPT_GOLD = (
    "Anda penilai soal matematika yang teliti. Diberikan SOAL dan JAWABAN yang diklaim "
    "benar (gold). Kerjakan soal SENDIRI langkah demi langkah, lalu bandingkan hasilmu "
    "dengan gold.\n\n"
    "Nilai dua hal:\n"
    "1) gold benar secara matematis untuk soal ini?\n"
    "2) soal LENGKAP/self-contained -- bisa dijawab tanpa tabel/grafik/teks/data luar "
    "yang TIDAK disertakan di soal?\n\n"
    "SOAL:\n{soal}\n\nJAWABAN GOLD: {gold}\n{cara}\n"
    "Balas TEPAT format ini di baris terakhir:\n"
    "VERDICT gold=<benar|salah|ragu> lengkap=<ya|tidak>"
)

PROMPT_TRAIN = (
    "Anda penilai solusi matematika yang teliti. Diberikan SOAL dan SOLUSI lengkap "
    "(langkah + jawaban akhir). Verifikasi sendiri.\n\n"
    "Nilai dua hal:\n"
    "1) jawaban akhir solusi benar secara matematis?\n"
    "2) penalaran valid -- langkah nyambung & logis, bukan ngarang/lompat ke jawaban?\n\n"
    "SOAL:\n{soal}\n\nSOLUSI:\n{solusi}\n\n"
    "Balas TEPAT format ini di baris terakhir:\n"
    "VERDICT jawaban=<benar|salah|ragu> penalaran=<valid|cacat|ragu>"
)

_RE = {
    "gold": re.compile(r"gold\s*=\s*(benar|salah|ragu)", re.I),
    "lengkap": re.compile(r"lengkap\s*=\s*(ya|tidak)", re.I),
    "jawaban": re.compile(r"jawaban\s*=\s*(benar|salah|ragu)", re.I),
    "penalaran": re.compile(r"penalaran\s*=\s*(valid|cacat|ragu)", re.I),
}


def parse_verdict(text: str, kind: str) -> dict:
    """Ambil label dari balasan judge. Default 'ragu' kalau tak terbaca (jangan nuduh)."""
    t = text or ""
    if kind == "gold":
        g = _RE["gold"].search(t)
        l = _RE["lengkap"].search(t)
        return {"gold": (g.group(1).lower() if g else "ragu"),
                "lengkap": (l.group(1).lower() if l else "ya")}
    j = _RE["jawaban"].search(t)
    p = _RE["penalaran"].search(t)
    return {"jawaban": (j.group(1).lower() if j else "ragu"),
            "penalaran": (p.group(1).lower() if p else "ragu")}


# ── build item (rid, soal, prompt) per skema ─────────────────────────────────────

def build_items(rows: list[dict], kind: str) -> list[dict]:
    items = []
    for i, r in enumerate(rows):
        if kind == "gold":
            soal = str(r.get("soal", "") or "").strip()
            gold = str(r.get("jawaban", "") or "").strip()
            cara = str(r.get("cara", "") or "").strip()
            cara_blk = f"CARA (solusi dataset, boleh dipakai cek): {cara[:1200]}\n" if cara else ""
            prompt = PROMPT_GOLD.format(soal=soal[:1800], gold=gold or "(kosong)", cara=cara_blk)
            items.append({"rid": i, "soal": soal[:200], "gold": gold[:80], "prompt": prompt})
        else:
            soal = _soal_of_chatml(r)
            sol = _assistant_of_chatml(r)
            prompt = PROMPT_TRAIN.format(soal=soal[:1800], solusi=sol[:3000])
            items.append({"rid": i, "soal": soal[:200], "prompt": prompt})
    return items


# ── judge backends (callable: list[prompt] -> list[text]) ────────────────────────

def make_judge_api(model: str, sleep: float = 0.0):
    client = openai_client()

    def ask(prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            def call():
                return client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": p}],
                    temperature=0.0,
                    max_tokens=300,
                )
            resp = with_retry(call)
            out.append(resp.choices[0].message.content or "")
            if sleep:
                time.sleep(sleep)
        return out

    return ask


def make_judge_vllm(model: str, tensor_parallel_size: int = 1):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=4096, tensor_parallel_size=tensor_parallel_size)
    sp = SamplingParams(temperature=0.0, max_tokens=300)

    def ask(prompts: list[str]) -> list[str]:
        if not prompts:
            return []
        return [o.outputs[0].text for o in llm.generate(prompts, sp)]

    return ask


# ── aggregate ────────────────────────────────────────────────────────────────────

def aggregate(verdicts: list[dict], kind: str) -> dict:
    n = len(verdicts)
    if not n:
        return {"kind": kind, "n": 0}
    if kind == "gold":
        g = [v["gold"] for v in verdicts]
        lengkap = sum(1 for v in verdicts if v["lengkap"] == "ya")
        return {
            "kind": "gold", "n": n,
            "gold_benar_rate": round(g.count("benar") / n, 3),
            "gold_salah_rate": round(g.count("salah") / n, 3),
            "gold_ragu_rate": round(g.count("ragu") / n, 3),
            "soal_lengkap_rate": round(lengkap / n, 3),
        }
    j = [v["jawaban"] for v in verdicts]
    p = [v["penalaran"] for v in verdicts]
    return {
        "kind": "train", "n": n,
        "jawaban_benar_rate": round(j.count("benar") / n, 3),
        "jawaban_salah_rate": round(j.count("salah") / n, 3),
        "penalaran_valid_rate": round(p.count("valid") / n, 3),
        "penalaran_cacat_rate": round(p.count("cacat") / n, 3),
    }


# ── audit satu file ──────────────────────────────────────────────────────────────

def audit(path: str, judge, *, sample: int | None, seed: int, out_dir: str | None,
          batch_size: int = 32, resume: bool = True) -> dict:
    """Return summary dict. `judge` = callable(list[prompt]) -> list[text] (bisa di-mock)."""
    rows = _read_jsonl(path)
    kind = _kind(rows)
    if kind == "unknown":
        return {"path": path, "kind": "unknown", "n": len(rows), "skip": True}

    items = build_items(rows, kind)
    if sample and sample < len(items):
        items = random.Random(seed).sample(items, sample)

    out_path = None
    done: dict[int, dict] = {}
    if out_dir:
        out_path = Path(out_dir) / f"{Path(path).stem}.verdicts.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if resume and out_path.exists():
            for v in _read_jsonl(out_path):
                if "rid" in v:
                    done[v["rid"]] = v

    pending = [it for it in items if it["rid"] not in done]
    new_verdicts: list[dict] = []
    for s in range(0, len(pending), batch_size):
        chunk = pending[s:s + batch_size]
        texts = judge([it["prompt"] for it in chunk])
        for it, txt in zip(chunk, texts):
            v = {"rid": it["rid"], "soal": it["soal"], **parse_verdict(txt, kind)}
            if kind == "gold":
                v["gold_ans"] = it.get("gold", "")
            new_verdicts.append(v)
        if out_path:  # checkpoint tiap batch
            with open(out_path, "a", encoding="utf-8") as f:
                for v in new_verdicts[-len(chunk):]:
                    f.write(json.dumps(v, ensure_ascii=False) + "\n")
        print(f"  {path}: {len(done) + len(new_verdicts)}/{len(items)} dinilai", flush=True)

    all_verdicts = list(done.values()) + new_verdicts
    summary = aggregate(all_verdicts, kind)
    summary["path"] = path
    return summary


def _print_summary(s: dict) -> None:
    if s.get("skip"):
        print(f"  {s['path']}  [skema tak dikenal, n={s['n']}] -- dilewati")
        return
    print(f"\n{s['path']}  [{s['kind']}]  n={s['n']}")
    keys = [k for k in s if k.endswith("_rate")]
    for k in keys:
        print(f"    {s[k]:6.1%}  {k}")


# ── self-check (tanpa API: mock judge) ───────────────────────────────────────────

def _self_check() -> None:
    import tempfile

    assert parse_verdict("blah\nVERDICT gold=salah lengkap=tidak", "gold") == \
        {"gold": "salah", "lengkap": "tidak"}
    assert parse_verdict("VERDICT jawaban=benar penalaran=valid", "train") == \
        {"jawaban": "benar", "penalaran": "valid"}
    # tak terbaca -> default aman
    assert parse_verdict("ngawur", "gold") == {"gold": "ragu", "lengkap": "ya"}
    assert parse_verdict("ngawur", "train") == {"jawaban": "ragu", "penalaran": "ragu"}

    # mock judge: vonis berdasarkan isi prompt (deterministik, tanpa API)
    def mock(prompts):
        out = []
        for p in prompts:
            if "JAWABAN GOLD" in p:  # gold
                bad = "2+2" in p and "5" in p
                out.append("VERDICT gold=" + ("salah" if bad else "benar") + " lengkap=ya")
            else:  # train: solusi buruk ditandai \boxed{9} di data uji
                bad = r"\boxed{9}" in p
                out.append("VERDICT jawaban=benar penalaran=" + ("cacat" if bad else "valid"))
        return out

    with tempfile.TemporaryDirectory() as td:
        gold_p = Path(td) / "g.jsonl"
        gold_p.write_text(
            json.dumps({"soal": "Hitung 2+2 berapa hasilnya", "jawaban": "4"}) + "\n" +
            json.dumps({"soal": "Hitung 2+2 berapa hasilnya", "jawaban": "5"}) + "\n",
            encoding="utf-8")
        s = audit(str(gold_p), mock, sample=None, seed=0, out_dir=None)
        assert s["n"] == 2 and s["gold_benar_rate"] == 0.5, s

        tr_p = Path(td) / "cot.jsonl"
        tr_p.write_text(
            json.dumps({"messages": [{"role": "user", "content": "p\n\nSoal A"},
                                     {"role": "assistant", "content": "langkah benar \\boxed{4}"}]}) + "\n" +
            json.dumps({"messages": [{"role": "user", "content": "p\n\nSoal B"},
                                     {"role": "assistant", "content": "ngarang \\boxed{9}"}]}) + "\n",
            encoding="utf-8")
        s2 = audit(str(tr_p), mock, sample=None, seed=0, out_dir=None)
        assert s2["n"] == 2 and s2["penalaran_valid_rate"] == 0.5, s2

        # resume: tulis verdict, run lagi -> tak panggil judge dua kali
        calls = {"n": 0}
        def counting(prompts):
            calls["n"] += len(prompts)
            return mock(prompts)
        audit(str(gold_p), counting, sample=None, seed=0, out_dir=td)
        first = calls["n"]
        audit(str(gold_p), counting, sample=None, seed=0, out_dir=td)
        assert calls["n"] == first, "resume gagal: judge dipanggil ulang"
    print("self-check OK")


def main() -> None:
    default = [
        "test_numglue=data/sft/test/numglue_test.jsonl",
        "test_easy=data/sft/test/easy_test.jsonl",
        "train_cot=data/sft/train/cot.jsonl",
        "train_nocot=data/sft/train/nocot.jsonl",
    ]
    ap = argparse.ArgumentParser(description="Audit kualitas data via LLM-as-judge (objektif)")
    ap.add_argument("--set", action="append", default=[], metavar="nama=path",
                    help="dataset diaudit (default: 2 test + 2 train)")
    ap.add_argument("--sample", type=int, default=80, help="ambil N baris acak/file (0=semua)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["api", "vllm"], default="api")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sleep", type=float, default=0.0, help="jeda antar call api (anti rate-limit)")
    ap.add_argument("--out", default="data/judge", help="folder verdict per-baris")
    ap.add_argument("--summary-out", default=None, help="tulis ringkasan JSON ke file")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()

    if a.self_check:
        _self_check()
        return

    specs = a.set or default
    sets = dict(s.split("=", 1) for s in specs)
    judge = (make_judge_api(a.model, sleep=a.sleep) if a.backend == "api"
             else make_judge_vllm(a.model))
    sample = a.sample or None

    summaries = []
    for name, path in sets.items():
        if not Path(path).exists():
            print(f"!! {name}: {path} tak ada -- dilewati")
            continue
        print(f"== audit {name} ({path}) ==")
        s = audit(path, judge, sample=sample, seed=a.seed, out_dir=a.out)
        s["name"] = name
        summaries.append(s)
        _print_summary(s)

    print("\n" + "=" * 60 + "\nRINGKASAN (LLM-as-judge, objektif)")
    for s in summaries:
        if s.get("skip"):
            continue
        if s["kind"] == "gold":
            print(f"  {s['name']:14} gold_benar={s['gold_benar_rate']:.1%}  "
                  f"soal_lengkap={s['soal_lengkap_rate']:.1%}  (n={s['n']})")
        else:
            print(f"  {s['name']:14} jawaban_benar={s['jawaban_benar_rate']:.1%}  "
                  f"penalaran_valid={s['penalaran_valid_rate']:.1%}  (n={s['n']})")

    if a.summary_out:
        Path(a.summary_out).write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {a.summary_out}")


if __name__ == "__main__":
    main()
