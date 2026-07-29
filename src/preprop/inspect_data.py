"""
Inspeksi kualitas dataset terstruktur -- diagnosa "garbage in, garbage out".

Dua skema auto-deteksi:
  * GOLD  {soal, jawaban, cara?}     -> test set / data sumber (yang dipakai eval sbg gold)
  * CHATML {messages:[user,assistant]} -> data SFT (cot.jsonl / nocot.jsonl)

Cek pakai komparator yang SAMA dengan eval (src.eval.answer_check) supaya angka
"konsisten/mismatch" benar-benar mencerminkan apa yang dilihat saat scoring.

Pemeriksaan utama (gold):
  - jawaban kosong, soal kosong/terlalu pendek, soal duplikat
  - KONSISTENSI INTERNAL: boxed(cara) vs jawaban -> kalau cara dataset sendiri saja
    sering tak cocok gold, itu kira-kira PLAFON akurasi student (label noise)
  - SOAL TAK-TERJAWAB: nyebut konteks luar (laporan/tabel/grafik/teks/bacaan/...) yang
    tak disertakan di soal -> mustahil dijawab tanpa passage (kasus numglue)
  - bentuk jawaban (numerik / ekspresi / teks-bebas); teks-bebas = sulit dicocokkan
  - mojibake / karakter rusak

Pemeriksaan utama (chatml):
  - assistant kosong / tanpa \boxed{}
  - nocot: assistant harus HANYA \boxed{...}; cot: harus ada penalaran + \boxed{}
  - soal duplikat, mojibake
  - lintas-file cot vs nocot: soal sama -> boxed harus sama (integritas eksperimen)

Usage:
    python -m src.preprop.inspect_data                 # default 4 file sft
    python -m src.preprop.inspect_data data/Final/*.jsonl --out data/inspect
    python -m src.preprop.inspect_data --self-check    # uji logika (CPU, tanpa file)

Baris yang ke-flag ditulis ke `<out>/<nama>.flags.jsonl` (ada field `_flags`) untuk
diperiksa manual. Murni CPU, stdlib + reuse answer_check.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from collections import Counter
from pathlib import Path

from src.eval.answer_check import extract_boxed, is_correct, normalize_str

# ── deteksi soal butuh konteks luar yang tak disertakan ────────────────────────
# numglue & soal bacaan sering merujuk passage/tabel/gambar yang tak ada di `soal`.
_CONTEXT_REF = re.compile(
    r"\b(menurut|berdasarkan|sesuai)\b.*\b(laporan|tabel|grafik|gambar|diagram|teks|"
    r"bacaan|paragraf|data|kutipan|artikel|wacana|cerita)\b|"
    r"\b(tabel|grafik|gambar|diagram|teks|bacaan|paragraf|wacana)\b\s+"
    r"(di\s+atas|berikut|di\s+bawah|tersebut)",
    re.IGNORECASE,
)
# mojibake klasik (UTF-8 dibaca latin-1) + replacement char + control chars.
_MOJIBAKE = re.compile(r"�|Ã[\x80-\xbf]|â€|Â[\x80-\xbf]|[\x00-\x08\x0b\x0c\x0e-\x1f]")
_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")
_FRACTION = re.compile(r"^-?\d+/\d+$")
_INSTRUCTION_SPLIT = "\n\n"     # template prompt diakhiri \n\n sebelum soal asli
_DOLLAR = re.compile(r"\$([^$]+)\$")
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def answer_from_cara(cara: str) -> str | None:
    """Jawaban yang DINYATAKAN solusi dataset. Tak semua `cara` pakai \\boxed{}
    (easy nyatakan di `$...$`/teks), jadi fallback berlapis biar konsistensi
    internal tak salah-vonis: boxed -> span $...$ terakhir -> angka terakhir."""
    b = extract_boxed(cara)
    if b is not None:
        return b
    spans = _DOLLAR.findall(cara or "")
    if spans:
        return spans[-1]
    nums = _NUM.findall(cara or "")
    return nums[-1] if nums else None


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"_parse_error": line[:200]})
    return rows


def _kind(rows: list[dict]) -> str:
    for r in rows:
        if isinstance(r, dict) and "messages" in r:
            return "chatml"
        if isinstance(r, dict) and "soal" in r:
            return "gold"
    return "unknown"


def is_garbled(s: str) -> bool:
    return bool(_MOJIBAKE.search(s or ""))


def needs_external_context(soal: str) -> bool:
    return bool(_CONTEXT_REF.search(soal or ""))


def answer_shape(jawaban: str) -> str:
    """numeric | fraction | expr | text(-bebas, sulit dicocokkan) | empty."""
    s = (jawaban or "").strip()
    if not s:
        return "empty"
    n = normalize_str(s)
    if _NUMERIC.match(n):
        return "numeric"
    if _FRACTION.match(n):
        return "fraction"
    # teks-bebas: ada spasi DAN kata alfabet panjang (bukan sekadar 'x' atau '(3,7)')
    stripped = re.sub(r"\\(?:text|mathrm|boxed)\s*\{[^{}]*\}", "", s)
    if " " in s.strip() and re.search(r"[A-Za-z]{4,}", stripped):
        return "text"
    return "expr"


def _soal_of_chatml(row: dict) -> str:
    """Soal asli dari pesan user terakhir (buang prefix instruksi sebelum \\n\\n)."""
    msgs = row.get("messages") or []
    user = next((m.get("content", "") for m in reversed(msgs)
                 if isinstance(m, dict) and m.get("role") == "user"), "")
    return user.split(_INSTRUCTION_SPLIT, 1)[-1].strip()


def _assistant_of_chatml(row: dict) -> str:
    msgs = row.get("messages") or []
    return next((m.get("content", "") for m in reversed(msgs)
                 if isinstance(m, dict) and m.get("role") == "assistant"), "")


# ── inspeksi per-skema ─────────────────────────────────────────────────────────

def inspect_gold(rows: list[dict]) -> tuple[dict, list[dict]]:
    n = len(rows)
    flagged: list[dict] = []
    soal_counts = Counter()
    shapes = Counter()
    cara_checked = cara_match = 0
    lens = []
    c = Counter()  # nama_flag -> jumlah

    for i, r in enumerate(rows):
        flags = []
        soal = str(r.get("soal", "") or "").strip()
        jawab = str(r.get("jawaban", "") or "").strip()
        cara = str(r.get("cara", "") or "")
        soal_counts[normalize_str(soal)] += 1
        lens.append(len(soal))

        if not jawab:
            flags.append("jawaban_kosong")
        if not soal:
            flags.append("soal_kosong")
        elif len(soal) < 20:
            flags.append("soal_pendek")
        if is_garbled(soal) or is_garbled(jawab) or is_garbled(cara):
            flags.append("mojibake")
        if soal and needs_external_context(soal):
            flags.append("butuh_konteks_luar")

        shape = answer_shape(jawab)
        shapes[shape] += 1
        if shape == "text":
            flags.append("jawaban_teks_bebas")

        # konsistensi internal: jawaban yg dinyatakan cara vs gold (komparator eval)
        if cara and jawab:
            pred = answer_from_cara(cara)
            cara_checked += 1
            if pred is not None and is_correct(pred, jawab):
                cara_match += 1
            elif pred is None:
                flags.append("cara_tanpa_jawaban")
            else:
                flags.append("cara_vs_gold_mismatch")

        for fl in flags:
            c[fl] += 1
        if flags:
            flagged.append({"_i": i, "_flags": flags, "soal": soal[:200],
                            "jawaban": jawab[:80], "cara_ans": (answer_from_cara(cara) or "")[:80]})

    dup = sum(v - 1 for v in soal_counts.values() if v > 1)
    summary = {
        "kind": "gold", "n": n,
        "flags": dict(c.most_common()),
        "dup_soal": dup,
        "shapes": dict(shapes.most_common()),
        "soal_len_median": int(statistics.median(lens)) if lens else 0,
        "cara_checked": cara_checked,
        "cara_match_gold": cara_match,
        # plafon kasar: seberapa sering solusi dataset sendiri == gold
        "self_consistency": round(cara_match / cara_checked, 3) if cara_checked else None,
    }
    return summary, flagged


def inspect_chatml(rows: list[dict], mode: str) -> tuple[dict, list[dict]]:
    """mode: 'cot' | 'nocot' | 'auto' (tebak dari nama file)."""
    n = len(rows)
    flagged: list[dict] = []
    soal_counts = Counter()
    asst_lens = []
    c = Counter()

    for i, r in enumerate(rows):
        flags = []
        soal = _soal_of_chatml(r)
        asst = _assistant_of_chatml(r)
        soal_counts[normalize_str(soal)] += 1
        asst_lens.append(len(asst))

        if not asst.strip():
            flags.append("assistant_kosong")
        boxed = extract_boxed(asst)
        if boxed is None:
            flags.append("tanpa_boxed")
        elif not boxed.strip():
            flags.append("boxed_kosong")
        if is_garbled(soal) or is_garbled(asst):
            flags.append("mojibake")

        if mode == "nocot":
            # nocot harus HANYA \boxed{...}: buang boxed, sisanya harus minim
            residue = re.sub(r"\\boxed\s*\{.*\}", "", asst, flags=re.DOTALL).strip()
            if len(residue) > 15:
                flags.append("nocot_ada_penalaran")
        elif mode == "cot":
            if boxed is not None and len(asst.strip()) - len(boxed) < 40:
                flags.append("cot_tanpa_penalaran")

        for fl in flags:
            c[fl] += 1
        if flags:
            flagged.append({"_i": i, "_flags": flags, "soal": soal[:200],
                            "assistant": asst[:200]})

    dup = sum(v - 1 for v in soal_counts.values() if v > 1)
    summary = {
        "kind": f"chatml/{mode}", "n": n,
        "flags": dict(c.most_common()),
        "dup_soal": dup,
        "asst_len_median": int(statistics.median(asst_lens)) if asst_lens else 0,
    }
    return summary, flagged


def cross_check_cot_nocot(cot_rows: list[dict], nocot_rows: list[dict]) -> dict:
    """Soal yang sama harus punya jawaban (boxed) sama di kedua arm."""
    def index(rows):
        m = {}
        for r in rows:
            key = normalize_str(_soal_of_chatml(r))
            b = extract_boxed(_assistant_of_chatml(r))
            if key and b is not None:
                m[key] = normalize_str(b)
        return m
    a, b = index(cot_rows), index(nocot_rows)
    shared = set(a) & set(b)
    mismatch = [k for k in shared if a[k] != b[k]]
    return {"shared_soal": len(shared), "answer_mismatch": len(mismatch),
            "mismatch_rate": round(len(mismatch) / len(shared), 3) if shared else None}


# ── runner ─────────────────────────────────────────────────────────────────────

def _mode_from_name(path: str) -> str:
    name = Path(path).name.lower()
    if "nocot" in name:
        return "nocot"
    if "cot" in name:
        return "cot"
    return "auto"


def _print_summary(path: str, summary: dict, n_flagged: int) -> None:
    print("=" * 78)
    print(f"{path}   [{summary['kind']}]  n={summary['n']}  flagged={n_flagged}")
    if summary.get("self_consistency") is not None:
        sc = summary["self_consistency"]
        print(f"  SELF-CONSISTENCY cara==gold: {sc:.1%} "
              f"({summary['cara_match_gold']}/{summary['cara_checked']})  "
              f"<- kira-kira PLAFON akurasi student")
    print(f"  dup_soal: {summary['dup_soal']}")
    if "shapes" in summary:
        print(f"  bentuk_jawaban: {summary['shapes']}")
    if summary["flags"]:
        print("  flags:")
        for k, v in summary["flags"].items():
            print(f"    {v:>6}  ({v / max(summary['n'],1):5.1%})  {k}")
    else:
        print("  flags: (none)")


def run(paths: list[str], out_dir: str | None) -> None:
    loaded = {p: _read_jsonl(p) for p in paths}
    cot_rows = nocot_rows = None

    for p, rows in loaded.items():
        kind = _kind(rows)
        if kind == "gold":
            summary, flagged = inspect_gold(rows)
        elif kind == "chatml":
            mode = _mode_from_name(p)
            summary, flagged = inspect_chatml(rows, mode if mode != "auto" else "cot")
            if mode == "cot":
                cot_rows = rows
            elif mode == "nocot":
                nocot_rows = rows
        else:
            print("=" * 78)
            print(f"{p}  [unknown schema] n={len(rows)} -- dilewati")
            continue

        _print_summary(p, summary, len(flagged))
        if out_dir and flagged:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            fp = Path(out_dir) / f"{Path(p).stem}.flags.jsonl"
            with open(fp, "w", encoding="utf-8") as f:
                for row in flagged:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  -> {len(flagged)} baris ke-flag ditulis ke {fp}")

    if cot_rows is not None and nocot_rows is not None:
        x = cross_check_cot_nocot(cot_rows, nocot_rows)
        print("=" * 78)
        print(f"LINTAS-FILE cot vs nocot: soal_sama={x['shared_soal']}  "
              f"jawaban_beda={x['answer_mismatch']} ({x['mismatch_rate']})")


def _self_check() -> None:
    """Assert logika klasifikasi (CPU, tanpa file)."""
    assert answer_shape("15") == "numeric"
    assert answer_shape("3/4") == "fraction"
    assert answer_shape("$(3,7)$") == "expr"
    assert answer_shape("Februari 1998 hingga Desember 1999") == "text"
    assert answer_shape("") == "empty"
    assert needs_external_context("Berapa lama menurut laporan Inggris?")
    assert needs_external_context("Sesuai tabel berikut, berapa total?")
    assert not needs_external_context("Hitung 2 + 3.")
    assert is_garbled("rumus � rusak") and not is_garbled("hitung 2 x 3")
    # fallback jawaban dari cara: boxed -> $...$ -> angka
    assert answer_from_cara("Jadi petanya titik $(3,7)$.") == "(3,7)"
    assert answer_from_cara("\\boxed{32}") == "32"
    assert answer_from_cara("hasilnya 5 lalu 42") == "42"

    # gold: 1 konsisten, 1 mismatch, 1 unanswerable+jawaban kosong
    gold = [
        {"soal": "Satu bilangan dua kali lainnya, jumlah 96. Cari yang kecil.",
         "cara": "x+2x=96 -> x=32. \\boxed{32}", "jawaban": "32"},
        {"soal": "Berapa lama menurut laporan Inggris?",
         "cara": "\\boxed{\\text{Feb 1998 - Des 1999}}", "jawaban": "15"},
        {"soal": "x", "cara": "", "jawaban": ""},
    ]
    s, fl = inspect_gold(gold)
    assert s["cara_match_gold"] == 1 and s["cara_checked"] == 2, s
    flagset = {f for row in fl for f in row["_flags"]}
    assert {"cara_vs_gold_mismatch", "butuh_konteks_luar", "jawaban_kosong",
            "soal_pendek"} <= flagset, flagset

    # chatml nocot: 1 bersih, 1 ada penalaran, 1 tanpa boxed
    nocot = [
        {"messages": [{"role": "user", "content": "prompt\n\nSoal A"},
                      {"role": "assistant", "content": "\\boxed{64}"}]},
        {"messages": [{"role": "user", "content": "prompt\n\nSoal B"},
                      {"role": "assistant", "content": "Pertama kita hitung dulu panjang sekali lalu \\boxed{5}"}]},
        {"messages": [{"role": "user", "content": "prompt\n\nSoal C"},
                      {"role": "assistant", "content": "jawabannya 7"}]},
    ]
    s2, fl2 = inspect_chatml(nocot, "nocot")
    fset2 = {f for row in fl2 for f in row["_flags"]}
    assert {"nocot_ada_penalaran", "tanpa_boxed"} <= fset2, fset2

    # lintas-file: soal sama, jawaban beda -> kedeteksi
    cot = [{"messages": [{"role": "user", "content": "prompt\n\nSoal A"},
                         {"role": "assistant", "content": "...langkah... \\boxed{99}"}]}]
    x = cross_check_cot_nocot(cot, nocot)
    assert x["shared_soal"] == 1 and x["answer_mismatch"] == 1, x
    print("self-check OK")


def main() -> None:
    default = [
        "data/sft/test/numglue_test.jsonl", "data/sft/test/easy_test.jsonl",
        "data/sft/train/cot.jsonl", "data/sft/train/nocot.jsonl",
    ]
    ap = argparse.ArgumentParser(description="Inspeksi kualitas dataset (garbage-in check)")
    ap.add_argument("paths", nargs="*", help="file/glob jsonl (default: 4 file sft)")
    ap.add_argument("--out", default="data/inspect", help="folder output baris ke-flag")
    ap.add_argument("--self-check", action="store_true", help="uji logika lalu keluar")
    a = ap.parse_args()
    if a.self_check:
        _self_check()
        return
    patterns = a.paths or default
    paths = [p for pat in patterns for p in (glob.glob(pat) or [pat]) if Path(p).exists()]
    if not paths:
        ap.error("tak ada file ketemu")
    run(paths, a.out)


if __name__ == "__main__":
    main()
