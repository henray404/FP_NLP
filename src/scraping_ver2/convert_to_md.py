"""
Batch convert semua PDF dan DOCX dalam satu folder ke Markdown menggunakan markitdown.

Output folder: <nama_folder>_md/ (sibling dari folder input)
Output file  : nama file sama, ekstensi diganti .md

Usage:
    python -m src.scraping_ver2.convert_to_md <folder>
    python -m src.scraping_ver2.convert_to_md data/raw/osn_ver2/
    python -m src.scraping_ver2.convert_to_md data/raw/osn_ver2/ --force
    python -m src.scraping_ver2.convert_to_md data/raw/osn_ver2/ --markitdown /path/to/markitdown
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED_EXTS = {".pdf", ".docx"}


def find_markitdown() -> str | None:
    """Cari binary markitdown: dari PATH dulu, lalu venv ML yang diketahui."""
    if path := shutil.which("markitdown"):
        return path
    fallback = Path.home() / "Documents/Kuliah/Sem3/Machine_Learning/.venv/bin/markitdown"
    if fallback.exists():
        return str(fallback)
    return None


def convert_file(src: Path, dst: Path, markitdown_bin: str) -> bool:
    """Convert satu file ke .md. Return True jika berhasil."""
    try:
        result = subprocess.run(
            [markitdown_bin, str(src), "-o", str(dst)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [FAIL] {src.name}: {result.stderr.strip()[:200]}")
            dst.unlink(missing_ok=True)
            return False
        size_kb = dst.stat().st_size // 1024
        print(f"  [OK]   {src.name} -> {dst.name} ({size_kb} KB)")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {src.name}: timeout (>120s)")
        dst.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  [FAIL] {src.name}: {e}")
        dst.unlink(missing_ok=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch convert PDF/DOCX ke Markdown menggunakan markitdown"
    )
    parser.add_argument("folder", help="Folder berisi file PDF/DOCX yang akan dikonversi")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Konversi ulang meski file .md sudah ada",
    )
    parser.add_argument(
        "--markitdown",
        type=str,
        default=None,
        metavar="PATH",
        help="Path ke binary markitdown (opsional, auto-detect jika tidak diisi)",
    )
    args = parser.parse_args()

    input_dir = Path(args.folder).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Folder tidak ditemukan: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"[ERROR] Bukan folder: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Output folder: <foldername>_md di sebelah folder input
    out_dir = input_dir.parent / f"{input_dir.name}_md"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cari markitdown binary
    markitdown_bin = args.markitdown or find_markitdown()
    if not markitdown_bin:
        print(
            "[ERROR] markitdown tidak ditemukan. Install dengan:\n"
            "  pip install markitdown\natau gunakan --markitdown /path/to/binary",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Input  : {input_dir}")
    print(f"Output : {out_dir}")
    print(f"Binary : {markitdown_bin}")

    # Kumpulkan file yang akan dikonversi
    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    )

    if not files:
        print(f"[INFO] Tidak ada file PDF/DOCX di {input_dir}")
        return

    print(f"\nDitemukan {len(files)} file ({', '.join(sorted(SUPPORTED_EXTS))})\n")

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for i, src in enumerate(files, 1):
        dst = out_dir / f"{src.stem}.md"
        prefix = f"[{i}/{len(files)}]"

        if dst.exists() and not args.force:
            print(f"{prefix} SKIP (sudah ada): {dst.name}")
            skip_count += 1
            continue

        print(f"{prefix} Converting: {src.name}")
        ok = convert_file(src, dst, markitdown_bin)
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    print(f"\n{'='*50}")
    print(f"Selesai: {ok_count} berhasil, {skip_count} di-skip, {fail_count} gagal")
    print(f"Output : {out_dir}")


if __name__ == "__main__":
    main()
