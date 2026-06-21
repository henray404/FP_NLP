"""
Metrik berbasis SAMPLING untuk eval: pass@k & maj@k (murni CPU, unit-testable).

Tiap soal di-generate N kandidat (sampling, temperature>0). Dari N itu:

- pass@k : peluang minimal 1 dari k sampel benar. Pakai unbiased estimator
           (Chen et al. 2021, HumanEval):  pass@k = 1 - C(n-c, k) / C(n, k)
           dengan n = jumlah sampel, c = jumlah sampel benar. Lebih stabil daripada
           sekadar "ada benar di k pertama".
- maj@k  : self-consistency. Ambil k sampel, voting jawaban (setelah normalisasi),
           jawaban terbanyak dicek benar/salah. k ganjil supaya seri jarang.

format_ok_rate dihitung per-sampel (% generasi yang ngehasilin \\boxed{}).

Butuh n_samples >= max(k) untuk semua k yang diminta.
"""
from __future__ import annotations

from collections import Counter
from math import comb

from src.eval.answer_check import extract_answer, extract_boxed, is_correct, normalize_str


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k. n=jumlah sampel, c=jumlah benar, k<=n (di-clamp)."""
    k = min(k, n)
    if k <= 0 or n <= 0:
        return 0.0
    if c <= 0:
        return 0.0
    if n - c < k:          # tidak cukup yang salah utk mengisi k slot -> pasti kena
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def majority_correct(preds: list[str | None], gold: str, k: int) -> bool:
    """Voting k prediksi pertama (dinormalisasi); jawaban terbanyak dicek ke gold."""
    sub = [normalize_str(p) for p in preds[:k] if p]
    if not sub:
        return False
    winner = Counter(sub).most_common(1)[0][0]
    return is_correct(winner, gold)


def score_samples(rows: list[dict], gens_per_problem: list[list[str]],
                  *, ks_pass=(1, 2, 3), ks_maj=(3, 5)) -> dict:
    """Skor N-sampel/soal. rows & gens_per_problem sejajar (gens_per_problem[i] = list generasi).

    Return ringkasan: {n, pass@k..., maj@k..., format_ok_rate}.
    """
    n_prob = len(rows)
    pass_sum = {k: 0.0 for k in ks_pass}
    maj_sum = {k: 0 for k in ks_maj}
    boxed = total = 0

    for r, gens in zip(rows, gens_per_problem):
        gold = r.get("jawaban", "")
        preds = [extract_answer(g) for g in gens]
        n = len(gens)
        c = sum(1 for p in preds if is_correct(p, gold))
        for k in ks_pass:
            pass_sum[k] += pass_at_k(n, c, k)
        for k in ks_maj:
            maj_sum[k] += 1 if majority_correct(preds, gold, k) else 0
        boxed += sum(1 for g in gens if extract_boxed(g) is not None)
        total += n

    out: dict = {"n": n_prob, "n_samples": (total // n_prob) if n_prob else 0}
    for k in ks_pass:
        out[f"pass@{k}"] = round(pass_sum[k] / n_prob, 4) if n_prob else 0.0
    for k in ks_maj:
        out[f"maj@{k}"] = round(maj_sum[k] / n_prob, 4) if n_prob else 0.0
    out["format_ok_rate"] = round(boxed / total, 4) if total else 0.0
    return out


def render_tables(results: dict, metrics: list[str]) -> str:
    """Satu tabel markdown per metrik (baris=model, kolom=test set)."""
    from src.eval.scenario_eval import render_table  # baris=model x kolom=set
    blocks = []
    for m in metrics:
        blocks.append(f"**{m}**\n\n" + render_table(results, metric=m))
    return "\n\n".join(blocks)
