"""
Re-grade pass@k / maj@k dari CACHE generasi -- tanpa GPU, tanpa generate ulang.

Kapan dipakai: setelah grader (answer_check) diperbaiki, hitung ULANG skor dari output
model yang sudah tersimpan. Cache = folder per (label,set) berisi `pNNNNN.json`
(masing-masing = list[str] N kandidat), format yang ditulis oleh
`sample_eval.eval_specs_sampling(ckpt_dir=...)`.

Skor dihitung via `sampling_metrics.score_samples`, yang memanggil grader terbaru di
`answer_check` -> perbaikan grader langsung kepakai.

Usage:
    # ambil dulu folder cache dari Kaggle (mis. /kaggle/working/ckpt_s3) ke lokal
    python -m src.eval.regrade --ckpt-dir data/eval/ckpt_s3 --label nonCoT \\
        --set numglue=data/sft/test/numglue_test.jsonl \\
        --set easy=data/sft/test/easy_test.jsonl

    python -m src.eval.regrade --self-check
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from src.eval.sample_eval import _slug
from src.eval.sampling_metrics import score_samples
from src.eval.skenario4_eval import load_holdout


def load_cache_dir(cdir: str | Path, n_rows: int) -> tuple[list[list[str]], int]:
    """Baca pNNNNN.json di cdir -> per_problem[i] = list generasi. Return (per_problem, n_missing)."""
    per_problem: list[list[str]] = [[] for _ in range(n_rows)]
    for fp in glob.glob(str(Path(cdir) / "p[0-9]*.json")):
        idx = int(Path(fp).stem[1:])
        if 0 <= idx < n_rows:
            try:
                per_problem[idx] = json.loads(Path(fp).read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    missing = sum(1 for g in per_problem if not g)
    return per_problem, missing


def regrade(ckpt_dir: str, label: str, sets: dict[str, str], *,
            ks_pass=(1, 2, 3), ks_maj=(3, 5)) -> dict:
    """sets = {nama_set: path_gold}. Return {label: {set: metrics}}."""
    out: dict = {label: {}}
    for name, path in sets.items():
        rows = load_holdout(Path(path))
        cdir = Path(ckpt_dir) / f"{_slug(label)}__{_slug(name)}"
        if not cdir.is_dir():
            print(f"!! cache tak ada: {cdir} -- set '{name}' dilewati")
            continue
        per_problem, missing = load_cache_dir(cdir, len(rows))
        if missing:
            print(f"   {name}: {missing}/{len(rows)} soal belum ada di cache "
                  f"(dihitung 0 benar)")
        res = score_samples(rows, per_problem, ks_pass=ks_pass, ks_maj=ks_maj)
        out[label][name] = res
        metrics = [f"pass@{k}" for k in ks_pass] + [f"maj@{k}" for k in ks_maj]
        line = "  ".join(f"{m}={res[m]:.4f}" for m in metrics)
        print(f"{label} @ {name} (n={res['n']}): {line}  format_ok={res['format_ok_rate']:.3f}")
    return out


def _self_check() -> None:
    """Bikin cache palsu di tmp lalu pastikan re-grade hitung benar."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gold = td / "g.jsonl"
        gold.write_text(
            json.dumps({"soal": "a", "jawaban": "4"}) + "\n" +
            json.dumps({"soal": "b", "jawaban": "7/15"}) + "\n", encoding="utf-8")
        cdir = td / f"{_slug('nonCoT')}__{_slug('s')}"
        cdir.mkdir(parents=True)
        # soal0: 3/4 benar (\boxed{4}); soal1: \frac{7}{15} == gold 7/15 (uji patch grader)
        (cdir / "p00000.json").write_text(json.dumps(
            [r"\boxed{4}", r"\boxed{4}", r"\boxed{5}", r"\boxed{4}"]), encoding="utf-8")
        (cdir / "p00001.json").write_text(json.dumps(
            [r"\boxed{\frac{7}{15}}"] * 4), encoding="utf-8")
        res = regrade(str(td), "nonCoT", {"s": str(gold)}, ks_pass=(1,), ks_maj=(3,))
        m = res["nonCoT"]["s"]
        assert m["n"] == 2, m
        assert m["pass@1"] > 0.5, m          # soal1 selalu benar -> rata2 > 0.5
        assert m["maj@3"] == 1.0, m          # mayoritas kedua soal benar
    print("self-check OK")


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-grade metrik dari cache (tanpa GPU)")
    ap.add_argument("--ckpt-dir", help="folder cache (berisi <label>__<set>/pNNNNN.json)")
    ap.add_argument("--label", default="nonCoT", help="label spec saat eval (mis. nonCoT / CoT-terbaik)")
    ap.add_argument("--set", action="append", default=[], metavar="nama=path_gold",
                    help="ulang per test set, mis. numglue=data/sft/test/numglue_test.jsonl")
    ap.add_argument("--out", default=None, help="tulis ringkasan JSON ke file")
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    if not a.ckpt_dir or not a.set:
        ap.error("butuh --ckpt-dir dan minimal satu --set nama=path")
    sets = dict(s.split("=", 1) for s in a.set)
    res = regrade(a.ckpt_dir, a.label, sets)
    if a.out:
        Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
