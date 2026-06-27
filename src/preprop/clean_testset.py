"""
Perbaikan (cleaning) test set gold -> subset yang ADIL dinilai otomatis.

Buang baris yang bikin akurasi jadi ARTEFAK (bukan ukuran kemampuan model):
  - jawaban_kosong       : gold kosong -> mustahil dinilai
  - soal_rusak           : soal kosong / terlalu pendek
  - mojibake             : teks rusak (encoding)
  - butuh_konteks_luar   : soal nyebut tabel/laporan/teks yg TAK disertakan (unanswerable)
  - cara_vs_gold_mismatch: solusi dataset sendiri tak setuju gold -> LABEL NOISE
  - (opsi) jawaban_teks_bebas: gold kalimat panjang -> auto-grade tak reliable

Deteksi + komparator SAMA dengan eval (answer_check) & inspect_data, jadi keputusan
buang konsisten dgn yang dilihat saat scoring. Murni CPU.

Catatan: buang `cara_vs_gold_mismatch` memakai grader TERBARU. Sisa mismatch setelah
grader diperbaiki -> kemungkinan besar gold-nya yang salah, bukan grader-nya.

Usage:
    python -m src.preprop.clean_testset data/sft/test/numglue_test.jsonl \\
        --out data/sft/test_clean/numglue_test.jsonl
    python -m src.preprop.clean_testset --self-check
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.eval.answer_check import is_correct
from src.preprop.inspect_data import (
    _read_jsonl,
    answer_from_cara,
    answer_shape,
    is_garbled,
    needs_external_context,
)


def drop_reason(r: dict, drop_text: bool = False, drop_mismatch: bool = True) -> str | None:
    """Alasan buang baris (None = simpan). Urutan = prioritas label."""
    soal = str(r.get("soal", "") or "").strip()
    jawab = str(r.get("jawaban", "") or "").strip()
    cara = str(r.get("cara", "") or "")
    if not jawab:
        return "jawaban_kosong"
    if not soal or len(soal) < 20:
        return "soal_rusak"
    if is_garbled(soal) or is_garbled(jawab) or is_garbled(cara):
        return "mojibake"
    if needs_external_context(soal):
        return "butuh_konteks_luar"
    if drop_mismatch and cara:
        pred = answer_from_cara(cara)
        if pred is not None and not is_correct(pred, jawab):
            return "cara_vs_gold_mismatch"
    if drop_text and answer_shape(jawab) == "text":
        return "jawaban_teks_bebas"
    return None


def clean(rows: list[dict], *, drop_text: bool = False, drop_mismatch: bool = True):
    """Return (kept, dropped, report). `dropped` punya field `_drop` (alasan)."""
    kept: list[dict] = []
    dropped: list[dict] = []
    reasons: Counter = Counter()
    for r in rows:
        why = drop_reason(r, drop_text=drop_text, drop_mismatch=drop_mismatch)
        if why is None:
            kept.append(r)
        else:
            reasons[why] += 1
            dropped.append({**r, "_drop": why})
    report = {
        "n_before": len(rows),
        "n_after": len(kept),
        "n_dropped": len(dropped),
        "keep_rate": round(len(kept) / len(rows), 3) if rows else None,
        "dropped_by_reason": dict(reasons.most_common()),
    }
    return kept, dropped, report


def _write_jsonl(rows: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _self_check() -> None:
    rows = [
        # bersih, self-consistent -> SIMPAN
        {"soal": "Satu bilangan dua kali lainnya, jumlah 96. Cari yang kecil.",
         "cara": "x+2x=96 -> x=32. \\boxed{32}", "jawaban": "32", "source": "x"},
        # gold kosong -> buang
        {"soal": "Hitung sesuatu yang panjang sekali kalimatnya di sini.",
         "cara": "...", "jawaban": "", "source": "x"},
        # unanswerable (nyebut laporan) -> buang
        {"soal": "Berapa lama rentang menurut laporan tersebut soal ini?",
         "cara": "\\boxed{15}", "jawaban": "15", "source": "x"},
        # label noise: cara bilang 8 tapi gold 5 -> buang
        {"soal": "Jumlah dua bilangan adalah delapan, berapa hasilnya?",
         "cara": "jadi totalnya \\boxed{8}", "jawaban": "5", "source": "x"},
        # mojibake -> buang
        {"soal": "rumus � rusak tapi panjang kalimatnya cukup",
         "cara": "x", "jawaban": "3", "source": "x"},
    ]
    kept, dropped, rep = clean(rows)
    assert rep["n_before"] == 5 and rep["n_after"] == 1, rep
    reasons = {d["_drop"] for d in dropped}
    assert reasons == {"jawaban_kosong", "butuh_konteks_luar",
                       "cara_vs_gold_mismatch", "mojibake"}, reasons
    assert kept[0]["jawaban"] == "32", kept
    print("self-check OK", rep)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bersihkan test set gold (CPU)")
    ap.add_argument("path", nargs="?", help="file jsonl gold {soal,jawaban,cara}")
    ap.add_argument("--out", help="tulis baris bersih ke sini")
    ap.add_argument("--dropped-out", help="opsional: tulis baris yang dibuang (+alasan)")
    ap.add_argument("--drop-text", action="store_true",
                    help="buang juga gold kalimat-bebas (auto-grade tak reliable)")
    ap.add_argument("--keep-mismatch", action="store_true",
                    help="JANGAN buang baris cara!=gold (default: dibuang)")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    if not a.path or not a.out:
        ap.error("butuh PATH dan --out (atau --self-check)")
    rows = _read_jsonl(a.path)
    kept, dropped, rep = clean(rows, drop_text=a.drop_text,
                               drop_mismatch=not a.keep_mismatch)
    _write_jsonl(kept, a.out)
    if a.dropped_out:
        _write_jsonl(dropped, a.dropped_out)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"-> {rep['n_after']}/{rep['n_before']} disimpan ke {a.out}")


if __name__ == "__main__":
    main()
