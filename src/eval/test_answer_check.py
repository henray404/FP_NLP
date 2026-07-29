"""Tes cepat logika cek jawaban (CPU, tanpa GPU). Jalankan: python -m src.eval.test_answer_check"""
from src.eval.answer_check import extract_boxed, is_correct, grade

# (generation, gold, expected_correct)
CASES = [
    # exact / format dasar
    (r"...jadi \boxed{42}", "42", True),
    (r"\boxed{7}", "8", False),                 # gold beda
    (r"tanpa kotak sama sekali", "5", False),   # no boxed -> False
    # boxed terakhir yang dipakai
    (r"awalnya \boxed{3} tapi ralat \boxed{4}", "4", True),
    # nested braces
    (r"\boxed{\frac{1}{2}}", "1/2", True),
    (r"\boxed{\frac{1}{2}}", "0.5", True),
    # ekuivalensi numerik
    (r"\boxed{0.50}", "1/2", True),
    (r"\boxed{1,000}", "1000", True),           # ribuan
    (r"\boxed{1,5}", "1.5", True),              # desimal koma
    # simbolik
    (r"\boxed{2x}", "x*2", True),
    (r"\boxed{x^2+2x+1}", "(x+1)^2", True),
    # text wrapper / unit
    (r"\boxed{\text{42}}", "42", True),
    # ekuivalensi latex: \(\) wrapper, \dfrac, faktor vs ekspansi
    (r"\boxed{\dfrac{1}{n^2+3n+2}}", r"\(\frac{1}{(n+1)(n+2)}\)", True),
    (r"\boxed{2}", "c=2", True),                 # gold ada prefix var
    # --- patch grader: \frac vs slash, pilihan ganda, satuan, prime, tuple ---
    (r"\boxed{\frac{7}{15}}", "7/15", True),     # \frac{a}{b} == a/b
    (r"\boxed{4}", "A. 4", True),                # gold pilihan ganda "A. 4"
    (r"\boxed{-17}", "B) -17", True),            # "B) -17"
    (r"\boxed{135}", "135 dolar", True),         # angka + satuan
    (r"\boxed{2}", "2 mol", True),
    (r"\boxed{(5,10)}", "A' = (5,10)", True),    # prefix "A' ="
    # salah beneran
    (r"\boxed{10}", "11", False),
    (r"\boxed{tahun 1998}", "15", False),        # jangan match angka nyasar di teks
]


def main():
    passed = 0
    for gen, gold, expected in CASES:
        g = grade(gen, gold)
        ok = g["correct"] == expected
        passed += ok
        mark = "OK " if ok else "XX "
        print(f"{mark} gold={gold!r:8} pred={str(g['pred'])!r:14} correct={g['correct']} (expect {expected})")
    print(f"\n{passed}/{len(CASES)} kasus lolos")
    assert passed == len(CASES), "ADA KASUS GAGAL — cek answer_check.py"


if __name__ == "__main__":
    main()
