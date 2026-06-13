"""
Ekstraksi + pencocokan jawaban untuk eval (Skenario 4).

Alur: ambil isi \\boxed{...} terakhir dari output model -> normalisasi ->
cocokkan dengan ground-truth `jawaban`. Pencocokan berlapis:
  1. exact match setelah normalisasi string
  2. ekuivalensi numerik/simbolik via sympy (mis. "0.5" == "1/2", "2x" == "x*2")

Murni CPU, jadi bisa dites lokal tanpa GPU.
"""
import re
from fractions import Fraction

# sympy opsional saat import-time biar modul tetap bisa di-load tanpa sympy,
# tapi memang dibutuhkan untuk pengecekan simbolik.
try:
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _SYMPY = True
    _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
except Exception:  # pragma: no cover
    _SYMPY = False


# ─────────────────────────────────────────────────────────────────────────────
# EKSTRAKSI \boxed{...}
# ─────────────────────────────────────────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """Ambil isi \\boxed{...} TERAKHIR dengan brace matching (handle nested {}).

    Kembalikan None kalau tidak ada \\boxed sama sekali (output dianggap gagal
    format -> di pipeline ini = discarded).
    """
    if not text:
        return None
    marker = r"\boxed"
    starts = [m.end() for m in re.finditer(re.escape(marker), text)]
    if not starts:
        return None
    pos = starts[-1]  # boxed terakhir
    # lewati spasi sampai '{'
    while pos < len(text) and text[pos] != "{":
        if not text[pos].isspace():
            return None
        pos += 1
    if pos >= len(text):
        return None
    depth = 0
    out = []
    for ch in text[pos:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue  # jangan masukkan '{' pembuka
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None  # brace tidak seimbang


# ─────────────────────────────────────────────────────────────────────────────
# NORMALISASI STRING
# ─────────────────────────────────────────────────────────────────────────────

_TEXT_WRAP = re.compile(r"\\(?:text|mathrm|mbox|operatorname)\s*\{([^{}]*)\}")
_LEFT_RIGHT = re.compile(r"\\(?:left|right|,|;|!|quad|qquad)")


def normalize_str(s: str) -> str:
    """Normalisasi ringan supaya exact-match lebih toleran."""
    if s is None:
        return ""
    s = str(s).strip()
    # buang $...$ pembungkus
    s = s.strip("$").strip()
    s = _TEXT_WRAP.sub(r"\1", s)
    s = _LEFT_RIGHT.sub("", s)
    # unit/teks umum di akhir jawaban
    s = re.sub(r"\\(?:cdot|times)", "*", s)
    s = s.replace("\\%", "%").replace("\\$", "")
    s = s.replace(" ", "")
    # ribuan: 1,000 -> 1000 (koma di antara digit)
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    # desimal koma -> titik kalau jelas angka: 1,5 -> 1.5
    s = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = s.rstrip(".")
    return s.lower()


# ─────────────────────────────────────────────────────────────────────────────
# EKUIVALENSI SIMBOLIK (sympy)
# ─────────────────────────────────────────────────────────────────────────────

def _to_expr(s: str):
    """Coba parse string jadi sympy expr (LaTeX dulu, lalu plain)."""
    if not _SYMPY:
        return None
    s = s.strip().strip("$").strip()
    # pecahan a/b sebagai Fraction cepat
    if re.fullmatch(r"-?\d+/\d+", s):
        try:
            return sympy.Rational(s)
        except Exception:
            pass
    for parser in ("latex", "plain"):
        try:
            if parser == "latex":
                expr = parse_latex(s)
            else:
                cleaned = _LEFT_RIGHT.sub("", _TEXT_WRAP.sub(r"\1", s))
                cleaned = cleaned.replace("\\cdot", "*").replace("^", "**")
                expr = parse_expr(cleaned, transformations=_TRANSFORMS, evaluate=True)
            return expr
        except Exception:
            continue
    return None


def symbolic_equal(a: str, b: str) -> bool:
    if not _SYMPY:
        return False
    ea, eb = _to_expr(a), _to_expr(b)
    if ea is None or eb is None:
        return False
    try:
        diff = sympy.simplify(ea - eb)
        return diff == 0
    except Exception:
        try:
            return bool(sympy.nsimplify(ea) == sympy.nsimplify(eb))
        except Exception:
            return False


def _numeric_equal(a: str, b: str, tol: float = 1e-6) -> bool:
    def to_num(x):
        x = x.strip().rstrip("%")
        try:
            return float(Fraction(x))
        except Exception:
            try:
                return float(x)
            except Exception:
                return None
    na, nb = to_num(a), to_num(b)
    if na is None or nb is None:
        return False
    return abs(na - nb) <= tol * max(1.0, abs(nb))


# ─────────────────────────────────────────────────────────────────────────────
# API UTAMA
# ─────────────────────────────────────────────────────────────────────────────

def is_correct(prediction: str | None, gold: str) -> bool:
    """True kalau jawaban prediksi (isi \\boxed) ekuivalen dengan ground-truth.

    `prediction` = hasil extract_boxed (boleh None -> langsung False).
    `gold` = nilai kolom `jawaban` dari dataset.
    """
    if prediction is None:
        return False
    pred_n, gold_n = normalize_str(prediction), normalize_str(gold)
    if not gold_n:
        return False
    if pred_n == gold_n:
        return True
    if _numeric_equal(pred_n, gold_n):
        return True
    if symbolic_equal(prediction, gold):
        return True
    return False


def grade(generation: str, gold: str) -> dict:
    """Grade satu output mentah model. Return dict ringkas untuk logging."""
    pred = extract_boxed(generation)
    return {
        "pred": pred,
        "has_boxed": pred is not None,
        "correct": is_correct(pred, gold),
    }
