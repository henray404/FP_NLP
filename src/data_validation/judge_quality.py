"""
Saring `soal`/`jawaban` yang rusak dari data Final/ sebelum sintesis CoT.

Kenapa hanya `soal` dan `jawaban`: pipeline CoT tidak pernah membaca kolom `cara`.
`prompt_wrap.wrap()` hanya mem-format `{soal}`, `generate.py` menyalin `cara` sebagai metadata
passthrough, dan `to_chatml.py` membangun data latih dari keluaran teacher (`text`/`pred`).
Jadi `cara` rusak tidak merusak apa pun -- kolomnya cukup dibuang (default; --keep-cara untuk
mempertahankan). Yang benar-benar merusak: `soal` rusak (teacher membuang N sampel, cakupan
dijamin 0) dan `jawaban` salah (solusi teacher yang BENAR ikut ditolak -> retensi/cakupan turun
palsu, dan kalau bocor ke holdout jadi soal yang mustahil dijawab).

Beda dari `validate_training_data.py` yang sudah ada: modul itu heuristik string murni
(bandingkan angka terakhir `cara` vs `jawaban`) dan banyak false positive. Modul ini menyasar
`soal`/`jawaban` dan memakai LLM judge untuk hal yang tidak bisa diregex.

Dua tahap:
  A. deterministik (CPU)  -- delimiter LaTeX timpang, gold tak-gradeable, sisa pilihan ganda,
                             rujukan gambar, teks terlalu pendek.
  B. LLM judge (GPU/API)  -- Q1 keterjawaban soal, Q2 kelayakan bentuk gold.
     Judge SENGAJA tidak diminta memverifikasi kebenaran matematis: model 7B tidak dapat
     menyelesaikan soal olimpiade dan akan mengarang vonis.

Keluaran 3 keranjang: keep / drop / review (review = judge paling tidak dapat dipercaya, disapu
manual, tidak auto-drop).

Usage:
    # tahap A saja, cepat, tanpa GPU -- lihat dulu berapa yang tersaring
    python -m src.data_validation.judge_quality data/Final/easy_clean_v2.jsonl --no-llm

    # kalibrasi dulu di baris yang sudah dianotasi manual
    python -m src.data_validation.judge_quality data/Final/easy_clean_v2.jsonl \
        --judge-backend vllm --limit 260 --seed 7

    # full run di Kaggle 2xT4
    python -m src.data_validation.judge_quality data/Final/easy_clean_v2.jsonl \
        --judge-backend vllm --tensor-parallel-size 2
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

BARE_LETTER = re.compile(r"^[A-E]$")
FIGURE_REF = re.compile(r"\[Gambar:|\[Tabel:", re.IGNORECASE)
# Suffix versi pada nama berkas: `_v2`, `_v3`, ... Ditanggalkan supaya keluaran selalu
# `<nama>_v3.jsonl`. Sebelumnya hanya `_v2` yang dibuang, sehingga masukan `easy_clean_v3.jsonl`
# menghasilkan `easy_clean_v3_v3.jsonl`.
VERSION_SUFFIX = re.compile(r"_v\d+$")

DEFAULT_VLLM_JUDGE = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_API_JUDGE = "llama-3.1-8b-instant"

Q1_PROMPT = (
    "Anda menilai kelayakan soal matematika untuk dataset pelatihan.\n\n"
    "Soal: {soal}\n\n"
    "Apakah soal di atas merupakan soal matematika yang UTUH dan dapat dijawab hanya dari "
    "teksnya sendiri?\n"
    "Jawab TIDAK jika: teks terpotong atau tidak lengkap, merujuk gambar/tabel/grafik yang tidak "
    "disertakan, merujuk soal lain (misalnya 'soal di atas'), berupa perintah aktivitas atau "
    "diskusi dan bukan soal hitungan, atau teksnya rusak sehingga tidak terbaca.\n\n"
    "Jawab HANYA satu kata: YA atau TIDAK.\nJawab:"
)

Q2_PROMPT = (
    "Anda memeriksa kunci jawaban sebuah soal matematika.\n\n"
    "Soal: {soal}\nKunci jawaban: {jawaban}\n\n"
    "Apakah kunci jawaban di atas berbentuk jawaban akhir yang WAJAR dan BERTIPE BENAR untuk soal "
    "tersebut? Anda TIDAK perlu menghitung atau memastikan nilainya benar -- cukup nilai bentuk "
    "dan tipenya.\n"
    "Jawab TIDAK jika: kunci berupa huruf pilihan ganda saja, berupa kalimat atau paragraf "
    "penjelasan padahal soal meminta nilai, tipenya tidak cocok (misalnya soal meminta koordinat "
    "tetapi kunci berupa satu angka), berupa tabel atau daftar panjang, atau teksnya rusak.\n\n"
    "Jawab HANYA satu kata: YA atau TIDAK.\nJawab:"
)


# -------------------------------
# Tahap A: deterministik (CPU)
# -------------------------------

def stage_a(row: dict) -> str | None:
    """Return alasan drop, atau None kalau lolos ke tahap judge."""
    from src.cot_synthesis.to_chatml import is_indonesian
    from src.eval.clean_holdout import latex_ok
    from src.preprop.filter_rules import is_multiple_choice

    soal = (row.get("soal") or "").strip()
    jawaban = (row.get("jawaban") or "").strip()

    if len(soal) < 15:
        return "soal_terlalu_pendek"
    if not jawaban:
        return "jawaban_kosong"
    if BARE_LETTER.match(jawaban):
        return "jawaban_bare_letter"
    if is_multiple_choice(soal):
        return "sisa_pilihan_ganda"
    if FIGURE_REF.search(soal):
        return "butuh_gambar"
    if not latex_ok(soal):
        return "latex_soal_timpang"
    if not is_indonesian(soal):
        return "soal_bukan_indonesia"
    return None

# Catatan: JANGAN pakai `gold_gradeable()` dari clean_holdout di sini. Fungsi itu untuk memilih
# HOLDOUT (yang butuh gold auto-gradeable oleh answer_check). Untuk train pool, gold berbentuk
# kalimat itu wajar -- `filter_solutions.py` memang memakai LLM judge justru karena `jawaban`
# berupa kalimat natural tanpa \boxed. Dipakai di sini ia membuang 1.002/2.336 baris (43%) yang
# sebenarnya sah, mis. "Rp. 742.500,00", "panjang AC = 24 cm", "4\\sqrt{2}".
# Tambahan: `_to_expr` memanggil sympy `parse_latex` yang butuh paket `antlr4-python3-runtime`;
# tanpa paket itu SEMUA jawaban LaTeX gagal parse dan dianggap tak-gradeable. Kelayakan bentuk
# gold ditangani Q2 di tahap judge.


# -------------------------------
# Tahap B: LLM judge
# -------------------------------

def _is_ya(text: str) -> bool:
    """Vonis judge. 'tidak' dicek lebih dulu supaya 'TIDAK' tidak terbaca memuat 'ya'."""
    t = text.strip().lower()
    if "tidak" in t:
        return False
    return "ya" in t


def _make_judge_vllm(model: str, tensor_parallel_size: int = 1):
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="float16", gpu_memory_utilization=0.85,
              max_model_len=2048, tensor_parallel_size=tensor_parallel_size)
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    def judge_batch(prompts: list[str]) -> list[bool]:
        if not prompts:
            return []
        outs = llm.generate(prompts, sp)
        return [_is_ya(o.outputs[0].text) for o in outs]

    return judge_batch


def _make_judge_api(model: str):
    from src.cot_synthesis.utils import openai_client, with_retry
    client = openai_client()

    def judge_batch(prompts: list[str]) -> list[bool]:
        out = []
        for p in prompts:
            resp = with_retry(lambda: client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": p}],
                temperature=0.0, max_tokens=4))
            out.append(_is_ya(resp.choices[0].message.content or ""))
        return out

    return judge_batch


def _cand_key(row: dict, q: str) -> str:
    return f"{q}\t{row.get('_idx', '')}"


def _load_progress(path: Path) -> dict[str, bool]:
    done: dict[str, bool] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    key, verdict = line.rsplit("\t", 1)
                    done[key] = verdict == "1"
    return done


def run_judge(rows: list[dict], judge, prompt_tpl: str, qtag: str, progress_path: Path,
              batch_size: int, fields: dict[str, str]) -> dict[str, bool]:
    """Jalankan satu pertanyaan judge pada semua baris, resumable per batch.

    fields: {nama_placeholder_prompt: nama_field_row}, mis. {"soal": "soal"}.
    """
    done = _load_progress(progress_path)
    pending = [r for r in rows if _cand_key(r, qtag) not in done]
    print(f"[{qtag}] {len(rows)} baris, {len(done)} sudah dinilai, {len(pending)} sisa")

    with open(progress_path, "a", encoding="utf-8") as pf:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start:start + batch_size]
            prompts = [prompt_tpl.format(**{ph: str(r.get(fld, ""))[:1500]
                                            for ph, fld in fields.items()}) for r in chunk]
            verdicts = judge(prompts)
            for r, ok in zip(chunk, verdicts):
                key = _cand_key(r, qtag)
                done[key] = ok
                pf.write(f"{key}\t{'1' if ok else '0'}\n")
            pf.flush()
            os.fsync(pf.fileno())
            print(f"  [{qtag}] {min(start + batch_size, len(pending))}/{len(pending)}", flush=True)
    return done


# -------------------------------
# Orkestrasi
# -------------------------------

def run(input_path: Path, out_dir: Path, *, use_llm: bool = True,
        judge_backend: str = "vllm", judge_model: str | None = None,
        tensor_parallel_size: int = 1, batch_size: int = 64,
        limit: int | None = None, seed: int = 7, drop_cara: bool = True) -> dict:
    rows = [json.loads(l) for l in open(input_path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):
        r["_idx"] = i
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)
        print(f"MODE KALIBRASI: {limit} baris acak (seed={seed})")

    stem = VERSION_SUFFIX.sub("", input_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep, dropped, review = [], [], []

    # Tahap A
    survivors = []
    for r in rows:
        reason = stage_a(r)
        if reason:
            dropped.append({**r, "_reason": reason, "_stage": "A"})
        else:
            survivors.append(r)
    print(f"Tahap A: {len(dropped)} dibuang, {len(survivors)} lolos ke tahap judge")

    # Tahap B
    if use_llm and survivors:
        model = judge_model or (DEFAULT_VLLM_JUDGE if judge_backend == "vllm" else DEFAULT_API_JUDGE)
        judge = (_make_judge_vllm(model, tensor_parallel_size) if judge_backend == "vllm"
                 else _make_judge_api(model))

        q1 = run_judge(survivors, judge, Q1_PROMPT, "Q1", out_dir / f"{stem}_q1.progress",
                       batch_size, {"soal": "soal"})
        q2 = run_judge(survivors, judge, Q2_PROMPT, "Q2", out_dir / f"{stem}_q2.progress",
                       batch_size, {"soal": "soal", "jawaban": "jawaban"})

        for r in survivors:
            if not q1.get(_cand_key(r, "Q1"), True):
                dropped.append({**r, "_reason": "soal_tak_terjawab", "_stage": "B"})
            elif not q2.get(_cand_key(r, "Q2"), True):
                review.append({**r, "_reason": "gold_meragukan", "_stage": "B"})
            else:
                keep.append(r)
    else:
        keep = survivors

    if drop_cara:
        keep = [{k: v for k, v in r.items() if k != "cara"} for r in keep]

    def _write(rs, suffix):
        p = out_dir / f"{stem}_v3{suffix}.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k != "_idx"},
                                   ensure_ascii=False) + "\n")
        return p

    stats = {
        "input": len(rows), "keep": len(keep), "dropped": len(dropped), "review": len(review),
        "alasan_drop": dict(Counter(r["_reason"] for r in dropped)),
        "keep_path": str(_write(keep, "")),
        "dropped_path": str(_write(dropped, "_dropped")),
        "review_path": str(_write(review, "_review")),
    }
    (out_dir / f"{stem}_v3_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def demo() -> None:
    """Self-check tahap A: baris rusak harus tertangkap, baris wajar harus lolos."""
    assert stage_a({"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": "14"}) is None
    assert stage_a({"soal": "Berapa nilai dari 2 + 2?", "jawaban": "C"}) == "jawaban_bare_letter"
    assert stage_a({"soal": "Hitung luas.", "jawaban": "10"}) == "soal_terlalu_pendek"
    assert stage_a({"soal": "Perhatikan pola berikut. [Gambar: tiga segitiga]",
                    "jawaban": "12"}) == "butuh_gambar"
    assert stage_a({"soal": "Berapa hasil dari 2 + 3 kali 4?", "jawaban": ""}) == "jawaban_kosong"
    assert _is_ya("YA") and not _is_ya("TIDAK") and not _is_ya("Tidak.")

    # Penanggalan suffix versi: berapa pun angkanya, keluaran tidak boleh dobel-suffix.
    def _stem(nama: str) -> str:
        return VERSION_SUFFIX.sub("", Path(nama).stem)

    assert _stem("easy_clean_v2.jsonl") == "easy_clean"
    assert _stem("easy_clean_v3.jsonl") == "easy_clean"
    assert _stem("numglue_clean_v10.jsonl") == "numglue_clean"
    assert _stem("easy_clean.jsonl") == "easy_clean"
    # `_v2` di tengah nama bukan suffix versi -- jangan ikut dibuang.
    assert _stem("easy_v2_clean.jsonl") == "easy_v2_clean"
    print("demo OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Saring soal/jawaban rusak (tahap A + LLM judge)")
    ap.add_argument("input", nargs="?", default="data/Final/easy_clean_v2.jsonl")
    ap.add_argument("--out-dir", default="data/Final")
    ap.add_argument("--no-llm", action="store_true", help="tahap A saja, tanpa GPU")
    ap.add_argument("--judge-backend", choices=["api", "vllm"], default="vllm")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Kaggle 2xT4 -> 2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="mode kalibrasi: N baris acak")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--keep-cara", action="store_true",
                    help="pertahankan kolom cara (default: buang, karena tak dipakai pipeline)")
    ap.add_argument("--demo", action="store_true", help="jalankan self-check tahap A lalu keluar")
    args = ap.parse_args()

    if args.demo:
        demo()
        return

    stats = run(Path(args.input), Path(args.out_dir), use_llm=not args.no_llm,
                judge_backend=args.judge_backend, judge_model=args.judge_model,
                tensor_parallel_size=args.tensor_parallel_size, batch_size=args.batch_size,
                limit=args.limit, seed=args.seed, drop_cara=not args.keep_cara)
    for k, v in stats.items():
        print(f"{k:16}: {v}")


if __name__ == "__main__":
    main()
