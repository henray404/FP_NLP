"""Bandingkan extractor PDF->teks/MD: fitz raw, pymupdf4llm, markitdown.

Potong sub-PDF (page range) per sampel supaya semua tool baca halaman sama,
lalu tulis hasil .md + tabel metrik (chars, waktu, #cid sampah, #simbol math).

Jalankan: python notebooks/compare_pdf2md.py
Output  : data/tool_compare/<sampel>__<tool>.md  + ringkasan ke stdout
"""
import glob
import re
import time
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "tool_compare"
OUT.mkdir(parents=True, exist_ok=True)
G_RAW = Path(r"G:/.shortcut-targets-by-id/1Y8RaC_XzbNTbjpMEdlrkNvL3k5GtC-0u/DATA_NLP/raw")


def _find_local(substr):
    hits = glob.glob(f"{ROOT}/data/raw/**/*{substr}*.pdf", recursive=True)
    return Path(hits[0]) if hits else None


def _find_gdrive(folder):
    d = G_RAW / folder
    hits = sorted(d.glob("*.pdf")) if d.exists() else []
    return hits[0] if hits else None


# (label, path, (start,end) 0-based inklusif) ─ page range konten yang ada math/gambar
SAMPLES = [
    ("osn_budi_digital",  _find_local("MDRMez"),         (0, 1)),   # math simpel, digital
    ("osn_buku_gambar",   _find_local("pmlCix"),         (9, 11)),  # buku + gambar
    ("ebook_kalkulus",    _find_gdrive("kalkulus_1_polsri"), (30, 32)),  # integral/pecahan
    ("ebook_kombinatorik", _find_gdrive("kombinatorika_itb"), (8, 10)),  # rumus/notasi
]

# ── Metrik ────────────────────────────────────────────────────────────
_CID = re.compile(r"\(cid:\d+\)")
_MATHSYM = re.compile(r"[√∫∑∏≤≥≠±∞→∂∆πθλµ²³¼½¾⁄∈∉⊂∪∩]")
_LATEX = re.compile(r"\$[^$]+\$|\\frac|\\sqrt|\\int|\\sum")


def metrics(text):
    return {
        "chars": len(text),
        "cid": len(_CID.findall(text)),          # sampah font tak ter-map (makin kecil makin baik)
        "mathsym": len(_MATHSYM.findall(text)),  # simbol math unicode kebawa
        "latex": len(_LATEX.findall(text)),      # notasi LaTeX/markdown math
    }


# ── Tool runners (semua terima path PDF, return str) ──────────────────
def run_fitz(pdf):
    d = fitz.open(pdf)
    return "\n".join(d[i].get_text() for i in range(d.page_count))


def run_pymupdf4llm(pdf):
    import pymupdf4llm
    return pymupdf4llm.to_markdown(str(pdf), show_progress=False)


def run_markitdown(pdf):
    from markitdown import MarkItDown
    return MarkItDown().convert(str(pdf)).text_content


TOOLS = [("fitz_raw", run_fitz), ("pymupdf4llm", run_pymupdf4llm), ("markitdown", run_markitdown)]


def make_subpdf(src, page_range, dst):
    """Tulis sub-PDF berisi halaman [start..end] dari src."""
    s, e = page_range
    out = fitz.open()
    with fitz.open(src) as d:
        e = min(e, d.page_count - 1)
        out.insert_pdf(d, from_page=s, to_page=e)
    out.save(dst)
    out.close()


def main():
    rows = []
    for label, src, prange in SAMPLES:
        if not src or not Path(src).exists():
            print(f"[skip] {label}: PDF tak ada ({src})")
            continue
        sub = OUT / f"{label}__pages.pdf"
        make_subpdf(src, prange, sub)
        print(f"\n##### {label}  ({Path(src).name}  hal {prange[0]+1}-{prange[1]+1}) #####")
        for tname, fn in TOOLS:
            try:
                t0 = time.perf_counter()
                txt = fn(sub)
                dt = time.perf_counter() - t0
            except Exception as ex:
                print(f"  [ERR] {tname}: {str(ex)[:120]}")
                continue
            (OUT / f"{label}__{tname}.md").write_text(txt, encoding="utf-8")
            m = metrics(txt)
            rows.append((label, tname, dt, m))
            print(f"  {tname:14s} {dt:6.2f}s  chars={m['chars']:6d}  "
                  f"cid={m['cid']:4d}  mathsym={m['mathsym']:4d}  latex={m['latex']:4d}")

    print("\n=== RINGKASAN (cid tinggi=jelek; mathsym/latex tinggi=notasi kebawa) ===")
    print(f"{'sampel':18s} {'tool':14s} {'detik':>6s} {'chars':>7s} {'cid':>5s} {'msym':>5s} {'latex':>6s}")
    for label, tname, dt, m in rows:
        print(f"{label:18s} {tname:14s} {dt:6.2f} {m['chars']:7d} {m['cid']:5d} {m['mathsym']:5d} {m['latex']:6d}")
    print(f"\nFile MD per tool -> {OUT}")
    print("Buka & banding manual: cek apakah pecahan/integral kebaca, soal+jawaban kepisah, gambar disebut.")


if __name__ == "__main__":
    main()
