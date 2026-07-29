"""
Metode KETIGA cek kualitas data: STATISTIK + SEMANTIK (tanpa LLM, tanpa regex per-cacat).

Pelengkap dua metode lain:
  - heuristik aturan  (src.preprop.inspect_data)        -> konsistensi internal, format
  - LLM-as-judge      (src.eval.judge_data_quality)     -> kebenaran matematis
  - STATISTIK+SEMANTIK (file ini)                        -> distribusi & kemiripan teks

Semua deterministik (CPU, bisa diulang persis). Yang dicek:

  lang_id_rate        soal terdeteksi Bahasa Indonesia (langdetect).  rendah = teks asing/sampah
  dup_semantik_rate   soal mirip-dekat soal LAIN di set yg sama (TF-IDF char n-gram cosine
                      >= --dup-thr). nangkep parafrase/near-dup yg lolos cek duplikat persis
  bocor_train_rate    (khusus test) soal test mirip soal TRAIN (cosine >= --leak-thr) ->
                      KONTAMINASI. kalau tinggi, skor eval (pass@k) jadi optimistik palsu.
                      kalau ~0 -> bukti objektif dekontaminasi train/test berhasil
  entropi_jawaban     sebaran nilai gold/boxed (0..1 ternormalisasi). rendah = label
                      degenerate (satu nilai mendominasi -> bisa "ditebak", gampang dicurangi)
  top_jawaban_share   porsi nilai jawaban paling sering. tinggi = sebaran timpang
  len_outlier_rate    panjang soal outlier (robust z, Iglewicz-Hoaglin |z|>3.5) -> kepotong/sampah

TF-IDF char n-gram dipilih karena deterministik & tanpa unduh model. Untuk kemiripan lebih
semantik bisa ganti ke embedding (sentence-transformers), tapi itu butuh model + nondeterministik.

Usage:
    python -m src.eval.data_quality_stats                 # 4 file default -> tabel markdown
    python -m src.eval.data_quality_stats --dup-thr 0.9 --leak-thr 0.85
    python -m src.eval.data_quality_stats --self-check
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from langdetect import DetectorFactory, detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from src.eval.answer_check import extract_boxed, normalize_str
from src.eval.data_quality_report import render_markdown
from src.preprop.inspect_data import _kind, _read_jsonl, _soal_of_chatml, _assistant_of_chatml

DetectorFactory.seed = 0  # langdetect deterministik

DUP_THR = 0.90   # cosine char-ngram utk "near-duplikat" dalam set
LEAK_THR = 0.85  # cosine test<->train utk "kontaminasi"


def _soals(rows: list[dict], kind: str) -> list[str]:
    if kind == "gold":
        return [str(r.get("soal", "") or "").strip() for r in rows]
    return [_soal_of_chatml(r) for r in rows]


def _answers(rows: list[dict], kind: str) -> list[str]:
    """Nilai jawaban ternormalisasi: gold -> 'jawaban', chatml -> boxed(assistant)."""
    out = []
    for r in rows:
        if kind == "gold":
            a = str(r.get("jawaban", "") or "")
        else:
            a = extract_boxed(_assistant_of_chatml(r)) or ""
        a = normalize_str(a).strip()
        if a:
            out.append(a)
    return out


def lang_id_rate(soals: list[str]) -> float:
    """Porsi soal terdeteksi 'id'. Soal sangat pendek dilewati (langdetect tak andal)."""
    ok = tot = 0
    for s in soals:
        if len(s) < 20:
            continue
        tot += 1
        try:
            if detect(s) == "id":
                ok += 1
        except Exception:
            pass
    return round(ok / tot, 3) if tot else 0.0


def _tfidf(corpus: list[str]):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    return vec.fit_transform([s or " " for s in corpus]), vec


def dup_semantic_rate(soals: list[str], thr: float = DUP_THR) -> float:
    """Porsi soal yg punya soal LAIN dgn cosine >= thr (near-duplikat)."""
    if len(soals) < 2:
        return 0.0
    X, _ = _tfidf(soals)
    sim = linear_kernel(X)  # tfidf sudah L2-norm -> linear_kernel = cosine
    n = sim.shape[0]
    dup = 0
    for i in range(n):
        sim[i, i] = -1.0  # buang diri sendiri
        if sim[i].max() >= thr:
            dup += 1
    return round(dup / n, 3)


def leak_rate(test_soals: list[str], train_soals: list[str], thr: float = LEAK_THR) -> float:
    """Porsi soal test yg mirip >= thr ke SOME soal train (kontaminasi)."""
    if not test_soals or not train_soals:
        return 0.0
    _, vec = _tfidf(train_soals)
    Xtr = vec.transform([s or " " for s in train_soals])
    Xte = vec.transform([s or " " for s in test_soals])
    sim = linear_kernel(Xte, Xtr)  # (n_test, n_train)
    hits = (sim.max(axis=1) >= thr).sum()
    return round(int(hits) / len(test_soals), 3)


def answer_dist(answers: list[str]) -> tuple[float, float]:
    """(entropi ternormalisasi 0..1, share nilai terbanyak)."""
    if not answers:
        return 0.0, 0.0
    c = Counter(answers)
    n = len(answers)
    top_share = c.most_common(1)[0][1] / n
    distinct = len(c)
    if distinct < 2:
        return 0.0, round(top_share, 3)
    h = -sum((v / n) * math.log2(v / n) for v in c.values())
    return round(h / math.log2(distinct), 3), round(top_share, 3)


def len_outlier_rate(soals: list[str]) -> float:
    """Outlier panjang soal via robust z (median/MAD), Iglewicz-Hoaglin |z|>3.5."""
    lens = [len(s) for s in soals if s]
    n = len(lens)
    if n < 3:
        return 0.0
    lens.sort()
    med = lens[n // 2] if n % 2 else (lens[n // 2 - 1] + lens[n // 2]) / 2
    abs_dev = sorted(abs(x - med) for x in lens)
    mad = abs_dev[n // 2] if n % 2 else (abs_dev[n // 2 - 1] + abs_dev[n // 2]) / 2
    if mad > 0:
        out = sum(1 for x in lens if abs(0.6745 * (x - med) / mad) > 3.5)
    else:  # MAD=0 (>half identik) -> fallback mean-abs-dev (Iglewicz-Hoaglin sekunder)
        mean_ad = sum(abs_dev) / n
        if mean_ad == 0:
            return 0.0
        out = sum(1 for x in lens if abs((x - med) / (1.253314 * mean_ad)) > 3.5)
    return round(out / n, 3)


def dup_examples(soals: list[str], thr: float = DUP_THR, k: int = 5) -> list[tuple[float, str, str]]:
    """Pasangan soal near-duplikat (cosine>=thr) di set yg sama, top-k cosine."""
    if len(soals) < 2:
        return []
    X, _ = _tfidf(soals)
    sim = linear_kernel(X)
    pairs = []
    for i in range(sim.shape[0]):
        for j in range(i + 1, sim.shape[0]):
            if sim[i, j] >= thr:
                pairs.append((round(float(sim[i, j]), 3), soals[i], soals[j]))
    return sorted(pairs, reverse=True)[:k]


def leak_examples(test_soals: list[str], train_soals: list[str],
                  thr: float = LEAK_THR, k: int = 5) -> list[tuple[float, str, str]]:
    """Soal test paling mirip ke train (cosine>=thr): (cosine, soal_test, soal_train)."""
    if not test_soals or not train_soals:
        return []
    _, vec = _tfidf(train_soals)
    Xtr = vec.transform([s or " " for s in train_soals])
    Xte = vec.transform([s or " " for s in test_soals])
    sim = linear_kernel(Xte, Xtr)
    out = []
    for i in range(sim.shape[0]):
        j = int(sim[i].argmax())
        if sim[i, j] >= thr:
            out.append((round(float(sim[i, j]), 3), test_soals[i], train_soals[j]))
    return sorted(out, reverse=True)[:k]


def stats_metrics(path: str, train_soals: list[str] | None = None,
                  dup_thr: float = DUP_THR, leak_thr: float = LEAK_THR) -> dict:
    rows = _read_jsonl(path)
    kind = _kind(rows)
    soals = _soals(rows, kind)
    ent, top = answer_dist(_answers(rows, kind))
    row = {
        "dataset": Path(path).stem, "kind": kind, "n": len(rows),
        "lang_id_rate": lang_id_rate(soals),
        "dup_semantik_rate": dup_semantic_rate(soals, dup_thr),
        "entropi_jawaban": ent,
        "top_jawaban_share": top,
        "len_outlier_rate": len_outlier_rate(soals),
    }
    if kind == "gold" and train_soals:
        row["bocor_train_rate"] = leak_rate(soals, train_soals, leak_thr)
    return row


# ── runner ───────────────────────────────────────────────────────────────────

def build_rows(paths: list[str], dup_thr: float = DUP_THR, leak_thr: float = LEAK_THR) -> list[dict]:
    """Kumpulkan soal train dulu (jadi acuan kebocoran), lalu hitung tiap set."""
    loaded = {p: _read_jsonl(p) for p in paths if Path(p).exists()}
    train_pool: list[str] = []
    for p, rows in loaded.items():
        if _kind(rows) == "chatml":
            train_pool += _soals(rows, "chatml")
    return [stats_metrics(p, train_pool, dup_thr, leak_thr) for p in loaded]


def _self_check() -> None:
    # bahasa
    assert lang_id_rate(["Hitung dua tambah tiga dikali empat sama dengan berapa hasilnya"]) == 1.0
    assert lang_id_rate(["The quick brown fox jumps over the lazy sleeping dog again now"]) == 0.0

    # near-duplikat: dua soal identik -> keduanya ke-flag
    soals = ["Berapa hasil dari dua tambah tiga sama dengan",
             "Berapa hasil dari dua tambah tiga sama dengan",
             "Sebuah kereta melaju enam puluh kilometer per jam selama dua jam"]
    assert dup_semantic_rate(soals, 0.9) == round(2 / 3, 3), dup_semantic_rate(soals, 0.9)

    # kebocoran: 1 dari 2 soal test identik dgn train
    test = ["Berapa hasil dua tambah tiga", "Soal khas test yang unik sendiri"]
    train = ["Berapa hasil dua tambah tiga", "lain lagi bukan itu sama sekali beda"]
    assert leak_rate(test, train, 0.85) == 0.5, leak_rate(test, train, 0.85)
    lex = leak_examples(test, train, 0.85)
    assert len(lex) == 1 and lex[0][1] == test[0], lex
    assert len(dup_examples(["sama persis ya", "sama persis ya", "beda jauh sekali"], 0.9)) == 1

    # sebaran jawaban
    assert answer_dist(["5", "5", "5", "5"]) == (0.0, 1.0)
    e, t = answer_dist(["1", "2", "3", "4"])
    assert e == 1.0 and t == 0.25, (e, t)

    # outlier panjang: satu soal jauh lebih panjang (panjang bervariasi -> MAD>0)
    lo = len_outlier_rate(["aaaa", "aaaaa", "aaaaaa", "aaaaaaa", "aaaaaaaa", "x" * 500])
    assert lo > 0, lo
    # degenerate: >separuh identik (MAD=0) -> fallback mean-abs-dev tetap nangkep outlier
    assert len_outlier_rate(["ab", "ab", "ab", "ab", "x" * 500]) > 0

    # integrasi pada chatml & gold kecil
    import json, tempfile
    with tempfile.TemporaryDirectory() as td:
        g = Path(td) / "x_test.jsonl"
        g.write_text("\n".join(json.dumps(r) for r in [
            {"soal": "Hitung dua tambah tiga sama dengan berapa ya kira-kira", "jawaban": "5"},
            {"soal": "Hitung dua tambah tiga sama dengan berapa ya kira-kira", "jawaban": "5"},
        ]), encoding="utf-8")
        m = stats_metrics(str(g), train_soals=["Hitung dua tambah tiga sama dengan berapa ya kira-kira"])
        assert m["kind"] == "gold" and m["bocor_train_rate"] == 1.0, m
        assert m["dup_semantik_rate"] == 1.0 and m["top_jawaban_share"] == 1.0, m
    print("self-check OK")


def main() -> None:
    default = [
        "data/sft/test/numglue_test.jsonl", "data/sft/test/easy_test.jsonl",
        "data/sft/train/cot.jsonl", "data/sft/train/nocot.jsonl",
    ]
    ap = argparse.ArgumentParser(description="Cek kualitas data: statistik + semantik (tanpa LLM)")
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--dup-thr", type=float, default=DUP_THR)
    ap.add_argument("--leak-thr", type=float, default=LEAK_THR)
    ap.add_argument("--self-check", action="store_true")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    rows = build_rows(a.paths or default, a.dup_thr, a.leak_thr)
    gold = [r for r in rows if r["kind"] == "gold"]
    train = [r for r in rows if r["kind"] == "chatml"]
    if gold:
        print("## TEST (gold) -- statistik + semantik\n")
        print(render_markdown(gold, ["dataset", "n", "lang_id_rate", "dup_semantik_rate",
              "bocor_train_rate", "entropi_jawaban", "top_jawaban_share", "len_outlier_rate"]))
    if train:
        print("\n## TRAIN (chatml) -- statistik + semantik\n")
        print(render_markdown(train, ["dataset", "n", "lang_id_rate", "dup_semantik_rate",
              "entropi_jawaban", "top_jawaban_share", "len_outlier_rate"]))


if __name__ == "__main__":
    main()
