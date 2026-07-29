"""
Laporan kualitas data OBJEKTIF -- gabung dua metode jadi satu tabel:

  A. HEURISTIK (deterministik, kuantitatif, tanpa LLM) -- dari inspect_data:
     self-consistency, mojibake, unanswerable, dup, jawaban kosong, jawaban teks-bebas,
     (train) tanpa-boxed / format salah.
  B. LLM-as-JUDGE (objektif, model nilai sendiri) -- dari judge_data_quality:
     gold_benar, soal_lengkap (test); jawaban_benar, penalaran_valid (train).

Heuristik selalu bisa dihitung (CPU). Metrik judge dibaca dari verdict yang sudah ada
(`data/judge/<nama>.verdicts.jsonl`); kalau belum ada -> kolom judge kosong (jalankan
`python -m src.eval.judge_data_quality` dulu).

Usage:
    python -m src.eval.data_quality_report                 # tabel ke stdout (markdown)
    python -m src.eval.data_quality_report --self-check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.judge_data_quality import aggregate
from src.preprop.inspect_data import (
    _kind,
    _mode_from_name,
    _read_jsonl,
    inspect_chatml,
    inspect_gold,
)

# Referensi: APA yang dicek & dinilai. (metric, yang_dicek, metode, baik_bila)
CHECKS = [
    ("self_consistency",   "cara dataset == gold (plafon akurasi)", "heuristik(grader)", "tinggi"),
    ("gold_benar_rate",    "gold benar secara matematis",            "LLM-judge",         "tinggi"),
    ("soal_lengkap_rate",  "soal self-contained (tak butuh data luar)", "LLM-judge",      "tinggi"),
    ("unanswerable_rate",  "soal nyebut tabel/teks luar tak disertakan", "heuristik(regex)", "rendah"),
    ("empty_jawaban_rate", "gold kosong",                            "heuristik",         "rendah"),
    ("text_answer_rate",   "gold kalimat-bebas (susah auto-grade)",  "heuristik",         "rendah"),
    ("mojibake_rate",      "teks rusak (encoding)",                  "heuristik(regex)",  "rendah"),
    ("dup_rate",           "soal duplikat",                          "heuristik",         "rendah"),
    ("jawaban_benar_rate", "(train) jawaban solusi benar",          "LLM-judge",         "tinggi"),
    ("penalaran_valid_rate", "(train) langkah valid, bukan ngarang", "LLM-judge",         "tinggi"),
    ("tanpa_boxed_rate",   "(train) solusi tanpa \\boxed{}",         "heuristik",         "rendah"),
]


def _rate(flags: dict, key: str, n: int) -> float:
    return round(flags.get(key, 0) / n, 3) if n else 0.0


def heuristic_metrics(path: str) -> dict:
    """Metrik deterministik untuk satu dataset (auto gold vs chatml)."""
    rows = _read_jsonl(path)
    kind = _kind(rows)
    stem = Path(path).stem
    if kind == "gold":
        s, _ = inspect_gold(rows)
        n, fl = s["n"], s["flags"]
        return {
            "dataset": stem, "kind": "gold", "n": n,
            "self_consistency": s["self_consistency"],
            "unanswerable_rate": _rate(fl, "butuh_konteks_luar", n),
            "empty_jawaban_rate": _rate(fl, "jawaban_kosong", n),
            "text_answer_rate": round(s["shapes"].get("text", 0) / n, 3) if n else 0.0,
            "mojibake_rate": _rate(fl, "mojibake", n),
            "dup_rate": round(s["dup_soal"] / n, 3) if n else 0.0,
        }
    if kind == "chatml":
        mode = _mode_from_name(path)
        s, _ = inspect_chatml(rows, mode if mode != "auto" else "cot")
        n, fl = s["n"], s["flags"]
        fmt = fl.get("nocot_ada_penalaran", 0) + fl.get("cot_tanpa_penalaran", 0)
        return {
            "dataset": stem, "kind": f"chatml/{mode}", "n": n,
            "tanpa_boxed_rate": _rate(fl, "tanpa_boxed", n),
            "mojibake_rate": _rate(fl, "mojibake", n),
            "dup_rate": round(s["dup_soal"] / n, 3) if n else 0.0,
            "format_issue_rate": round(fmt / n, 3) if n else 0.0,
        }
    return {"dataset": stem, "kind": "unknown", "n": len(rows)}


def judge_metrics(name: str, judge_dir: str = "data/judge") -> dict | None:
    """Baca verdict LLM-judge utk satu dataset. None kalau belum ada."""
    p = Path(judge_dir) / f"{name}.verdicts.jsonl"
    if not p.exists():
        return None
    v = _read_jsonl(p)
    if not v:
        return None
    kind = "gold" if "gold" in v[0] else "train"
    return aggregate(v, kind)


def build_rows(sets: dict[str, str], judge_dir: str = "data/judge") -> list[dict]:
    """sets = {nama: path}. Gabung heuristik + judge per dataset."""
    rows = []
    for name, path in sets.items():
        if not Path(path).exists():
            continue
        row = {"nama": name}
        row.update(heuristic_metrics(path))
        jm = judge_metrics(Path(path).stem, judge_dir) or {}
        for k in ("gold_benar_rate", "soal_lengkap_rate",
                  "jawaban_benar_rate", "penalaran_valid_rate"):
            if k in jm:
                row[k] = jm[k]
        row["judge_n"] = jm.get("n")
        rows.append(row)
    return rows


def render_markdown(rows: list[dict], cols: list[str]) -> str:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.1%}" if 0 <= v <= 1 else f"{v:.3f}"
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def checks_table_md() -> str:
    head = "| metrik | yang dicek | metode | baik bila |"
    sep = "| --- | --- | --- | --- |"
    body = [f"| `{m}` | {d} | {meth} | {good} |" for m, d, meth, good in CHECKS]
    return "\n".join([head, sep, *body])


def _self_check() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        gp = Path(td) / "x_test.jsonl"
        gp.write_text(
            json.dumps({"soal": "Hitung 2+3 hasilnya berapa ya", "cara": "2+3=5 \\boxed{5}", "jawaban": "5"}) + "\n" +
            json.dumps({"soal": "Berapa total menurut tabel berikut soalnya", "cara": "x", "jawaban": ""}) + "\n",
            encoding="utf-8")
        m = heuristic_metrics(str(gp))
        assert m["kind"] == "gold" and m["n"] == 2, m
        assert m["empty_jawaban_rate"] == 0.5, m
        assert m["unanswerable_rate"] == 0.5, m
        assert m["self_consistency"] == 1.0, m   # 1 cara dicek, cocok

        cp = Path(td) / "cot.jsonl"
        cp.write_text(
            json.dumps({"messages": [{"role": "user", "content": "p\n\nSoal A"},
                                     {"role": "assistant", "content": "langkah... \\boxed{4}"}]}) + "\n" +
            json.dumps({"messages": [{"role": "user", "content": "p\n\nSoal B"},
                                     {"role": "assistant", "content": "tanpa kotak"}]}) + "\n",
            encoding="utf-8")
        mc = heuristic_metrics(str(cp))
        assert mc["kind"] == "chatml/cot" and mc["tanpa_boxed_rate"] == 0.5, mc

        assert judge_metrics("x_test", td) is None  # belum ada verdict
        # tulis verdict palsu -> kebaca
        (Path(td) / "x_test.verdicts.jsonl").write_text(
            json.dumps({"rid": 0, "gold": "benar", "lengkap": "ya"}) + "\n" +
            json.dumps({"rid": 1, "gold": "salah", "lengkap": "ya"}) + "\n", encoding="utf-8")
        jm = judge_metrics("x_test", td)
        assert jm["gold_benar_rate"] == 0.5, jm

        rows = build_rows({"test": str(gp)}, judge_dir=td)
        md = render_markdown(rows, ["nama", "n", "self_consistency", "gold_benar_rate"])
        assert "test" in md and "50.0%" in md, md
    assert len(checks_table_md().splitlines()) == len(CHECKS) + 2
    print("self-check OK")


def main() -> None:
    default = {
        "test_numglue": "data/sft/test/numglue_test.jsonl",
        "test_easy": "data/sft/test/easy_test.jsonl",
        "train_cot": "data/sft/train/cot.jsonl",
        "train_nocot": "data/sft/train/nocot.jsonl",
    }
    ap = argparse.ArgumentParser(description="Tabel kualitas data (heuristik + LLM-judge)")
    ap.add_argument("--set", action="append", default=[], metavar="nama=path")
    ap.add_argument("--judge-dir", default="data/judge")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    sets = dict(s.split("=", 1) for s in a.set) if a.set else default
    rows = build_rows(sets, a.judge_dir)

    print("## Apa yang dicek & dinilai\n")
    print(checks_table_md())
    gold_cols = ["nama", "n", "self_consistency", "gold_benar_rate", "soal_lengkap_rate",
                 "unanswerable_rate", "text_answer_rate", "mojibake_rate"]
    train_cols = ["nama", "n", "tanpa_boxed_rate", "mojibake_rate", "format_issue_rate",
                  "jawaban_benar_rate", "penalaran_valid_rate", "judge_n"]
    gold_rows = [r for r in rows if r.get("kind") == "gold"]
    train_rows = [r for r in rows if str(r.get("kind", "")).startswith("chatml")]
    if gold_rows:
        print("\n## TEST (gold)\n")
        print(render_markdown(gold_rows, gold_cols))
    if train_rows:
        print("\n## TRAIN (chatml)\n")
        print(render_markdown(train_rows, train_cols))


if __name__ == "__main__":
    main()
