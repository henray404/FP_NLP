"""
Scrape soal OSN Matematika SMA dan Perguruan Tinggi dari berbagai sumber baru.
Memastikan tidak ada duplikat dengan memeriksa data/raw/ dan data/extracted/.

Dedup dilakukan di dua level:
  1. PDF level  — cek URL dan Google Drive file_id vs data yang sudah ada di data/raw/
  2. Soal level — cek hash teks pertanyaan vs data/extracted/*.jsonl

Usage:
    python -m src.scraping_ver2.scrape_osn --dry-run
    python -m src.scraping_ver2.scrape_osn
    python -m src.scraping_ver2.scrape_osn --level sma
    python -m src.scraping_ver2.scrape_osn --level kuliah
    python -m src.scraping_ver2.scrape_osn --force
    python -m src.scraping_ver2.scrape_osn --dry-run --save-manifest manifest_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Paths ──────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
OSN_VER2_DIR = RAW_DIR / "osn_ver2"
EXTRACTED_DIR = Path("data/extracted")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
}
DELAY_SECONDS = 2.0
TIMEOUT = 90


# ── Sources ────────────────────────────────────────────────────────────────────
# Setiap entry:
#   slug    : nama folder output di data/raw/osn_ver2/<slug>/
#   level   : "sma" | "kuliah"
#   subject : topik matematika
#   type    : "pdf" (direct) | "gdrive" | "page" (scrape link dari halaman)
#   url     : URL sumber
#   note    : catatan (opsional)

SOURCES: list[dict] = [
    # ── OSN SMA ──────────────────────────────────────────────────────────────
    {
        "slug": "osn_mat_sma_ilmuguru",
        "level": "sma",
        "subject": "Matematika OSN SMA",
        "type": "page",
        "url": "https://www.ilmuguru.org/p/soal-osn-matematika-sma.html",
        "note": "Halaman indeks dengan link Google Drive soal OSN Matematika SMA",
    },
    {
        "slug": "osn_mat_sma_websiteedukasi",
        "level": "sma",
        "subject": "Matematika OSN SMA",
        "type": "page",
        "url": "https://www.websiteedukasi.com/2019/05/soal-osn-matematika-sma-tahun-2019.html",
        "note": "Soal OSN Matematika SMA 2019",
    },
    {
        "slug": "osn_mat_ksmo_aliyah",
        "level": "sma",
        "subject": "Matematika KSM Aliyah",
        "type": "page",
        "url": "https://kemenag.go.id/nasional/kumpulan-soal-kompetisi-sains-madrasah-ksm-matematika-tingkat-aliyah-ixvvv",
        "note": "Soal KSM Matematika tingkat Aliyah (setara SMA)",
    },
    {
        "slug": "osn_mat_sma_mathstudycenter",
        "level": "sma",
        "subject": "Matematika OSN SMA",
        "type": "page",
        "url": "https://matematikastudycenter.blogspot.com/p/soal-osn.html",
        "note": "Arsip soal OSN matematika SMA dari matematikastudycenter",
    },
    {
        "slug": "osn_mat_sma_gurumaju",
        "level": "sma",
        "subject": "Matematika OSN SMA",
        "type": "page",
        "url": "https://www.gurumaju.com/soal-osn-matematika/",
        "note": "Kumpulan soal OSN matematika SMA berbagai tahun",
    },
    {
        "slug": "osn_mat_sma_banksoal",
        "level": "sma",
        "subject": "Matematika OSN SMA",
        "type": "page",
        "url": "https://banksoalku.com/soal-osn-matematika-sma/",
        "note": "Bank soal OSN Matematika SMA",
    },
    # ── OSN / Kompetisi Matematika Perguruan Tinggi ───────────────────────────
    {
        "slug": "kompetisi_mat_fmipa_ui",
        "level": "kuliah",
        "subject": "Kompetisi Matematika PT",
        "type": "page",
        "url": "https://math.ui.ac.id/kompetisi/",
        "note": "Halaman kompetisi matematika FMIPA UI",
    },
    {
        "slug": "kompetisi_mat_itb",
        "level": "kuliah",
        "subject": "Kompetisi Matematika PT",
        "type": "page",
        "url": "https://math.itb.ac.id/kegiatan/kompetisi/",
        "note": "Halaman kompetisi matematika ITB",
    },
    {
        "slug": "kompetisi_mat_ugm",
        "level": "kuliah",
        "subject": "Kompetisi Matematika PT",
        "type": "page",
        "url": "https://math.ugm.ac.id/kompetisi-matematika/",
        "note": "Halaman kompetisi matematika UGM",
    },
    {
        "slug": "olimpiade_mat_pt_blogspot",
        "level": "kuliah",
        "subject": "Olimpiade Matematika PT",
        "type": "page",
        "url": "https://olimpiadematematika.blogspot.com/p/soal-olimpiade-matematika-pt.html",
        "note": "Kumpulan soal olimpiade matematika perguruan tinggi",
    },
    {
        "slug": "himatika_itb_kompetisi",
        "level": "kuliah",
        "subject": "Kompetisi Matematika Mahasiswa",
        "type": "page",
        "url": "https://himatika.itb.ac.id/kompetisi/",
        "note": "Kompetisi matematika mahasiswa HIMATIKA ITB",
    },
]


# ── Google Drive helpers ───────────────────────────────────────────────────────

GDRIVE_FILE_RE = re.compile(r"drive\.google\.com/file/d/([^/?#]+)")
GDRIVE_OPEN_RE = re.compile(r"drive\.google\.com/open\?id=([^&]+)")
GDRIVE_UC_RE = re.compile(r"drive\.google\.com/uc\?.*?id=([^&]+)")


def extract_gdrive_id(url: str) -> str | None:
    for pattern in (GDRIVE_FILE_RE, GDRIVE_OPEN_RE, GDRIVE_UC_RE):
        m = pattern.search(url)
        if m:
            return m.group(1)
    return None


def gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _gdrive_confirm_url(file_id: str, html: str) -> str | None:
    m = re.search(r'href="(/uc\?export=download[^"]+confirm=[^"]+)"', html)
    if m:
        return "https://drive.google.com" + m.group(1).replace("&amp;", "&")
    m = re.search(r'action="(https://drive\.google\.com/uc[^"]+)"', html)
    if m:
        return m.group(1).replace("&amp;", "&") + "&confirm=t"
    return None


# ── Dedup index ────────────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Normalize URL untuk perbandingan: strip trailing slash, lowercase scheme+host."""
    p = urlparse(url.strip())
    return p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower()).geturl().rstrip("/")


def _text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


class ExistingDataIndex:
    """
    Index data yang sudah ada untuk menghindari duplikat.

    Cek dua level:
      - PDF level: URL dan Google Drive file_id dari data/raw/
      - Soal level: hash teks pertanyaan dari data/extracted/*.jsonl
    """

    def __init__(self, raw_dir: Path = RAW_DIR, extracted_dir: Path = EXTRACTED_DIR):
        self.known_urls: set[str] = set()
        self.known_gdrive_ids: set[str] = set()
        self.known_question_hashes: set[str] = set()
        self._load_raw(raw_dir)
        self._load_extracted(extracted_dir)

    def _load_raw(self, raw_dir: Path) -> None:
        if not raw_dir.exists():
            return

        # Cek source.json di setiap subfolder
        for source_json in raw_dir.rglob("source.json"):
            try:
                data = json.loads(source_json.read_text(encoding="utf-8"))
                for key in ("url", "resolved_url", "download_url"):
                    if url := data.get(key):
                        self.known_urls.add(normalize_url(url))
                        gdrive_id = extract_gdrive_id(url)
                        if gdrive_id:
                            self.known_gdrive_ids.add(gdrive_id)
            except Exception:
                pass

        # Cek nama file PDF di folder osn/ (format: year_label_fileid8chars.pdf)
        osn_dir = raw_dir / "osn"
        if osn_dir.exists():
            for pdf_file in osn_dir.glob("*.pdf"):
                # File ID 8 karakter adalah bagian terakhir dari nama file sebelum .pdf
                parts = pdf_file.stem.split("_")
                if parts:
                    candidate_id = parts[-1]
                    if len(candidate_id) == 8:
                        self.known_gdrive_ids.add(candidate_id)

        print(
            f"[DeduP] Raw index: {len(self.known_urls)} URL, "
            f"{len(self.known_gdrive_ids)} GDrive ID dimuat"
        )

    def _load_extracted(self, extracted_dir: Path) -> None:
        if not extracted_dir.exists():
            return

        count = 0
        for jsonl_file in extracted_dir.glob("*.jsonl"):
            try:
                for line in jsonl_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    question = record.get("question", "")
                    if question:
                        self.known_question_hashes.add(_text_hash(question))
                        count += 1
            except Exception:
                pass

        print(f"[DeduP] Extracted index: {count} pertanyaan dimuat")

    def is_url_known(self, url: str) -> bool:
        return normalize_url(url) in self.known_urls

    def is_gdrive_id_known(self, file_id: str) -> bool:
        # Cek full ID dan prefix 8 karakter (sesuai format nama file lama)
        return file_id in self.known_gdrive_ids or file_id[:8] in self.known_gdrive_ids

    def is_question_duplicate(self, question_text: str) -> bool:
        return _text_hash(question_text) in self.known_question_hashes


# ── Page scraper ───────────────────────────────────────────────────────────────

def scrape_links_from_page(url: str) -> list[dict]:
    """Fetch halaman HTML, kumpulkan semua link Google Drive dan PDF langsung."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Gagal fetch halaman {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    found: list[dict] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()

    for a_tag in soup.find_all("a", href=True):
        href: str = a_tag["href"]

        # Google Drive links
        file_id = extract_gdrive_id(href)
        if file_id and file_id not in seen_ids:
            seen_ids.add(file_id)
            label = a_tag.get_text(strip=True) or href
            found.append({
                "label": label,
                "file_id": file_id,
                "gdrive_url": href,
                "download_url": gdrive_download_url(file_id),
            })
            continue

        # Link PDF langsung (bukan GDrive)
        if href.lower().endswith(".pdf") and "drive.google.com" not in href:
            norm = normalize_url(href)
            if norm not in seen_urls:
                seen_urls.add(norm)
                found.append({
                    "label": a_tag.get_text(strip=True) or href,
                    "file_id": None,
                    "direct_url": href,
                    "download_url": href,
                })

    return found


# ── Downloader ─────────────────────────────────────────────────────────────────

def _safe_filename(text: str, maxlen: int = 50) -> str:
    clean = re.sub(r"[^\w\s-]", "", text).strip().replace(" ", "_")
    return re.sub(r"_+", "_", clean)[:maxlen].strip("_")


def _is_pdf_bytes(content_start: bytes) -> bool:
    return content_start[:4] == b"%PDF"


def download_pdf(
    download_url: str,
    out_path: Path,
    file_id: str | None = None,
) -> bool:
    """Download satu PDF ke out_path. Return True jika berhasil."""
    try:
        resp = requests.get(download_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()

        # Handle Google Drive virus-scan confirmation page
        if "text/html" in resp.headers.get("Content-Type", "") and file_id:
            html = resp.text
            confirm_url = _gdrive_confirm_url(file_id, html)
            if not confirm_url:
                print("  [WARN] Tidak bisa menemukan confirm URL dari halaman GDrive")
                return False
            resp = requests.get(confirm_url, headers=HEADERS, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()

        # Stream ke file dan validasi magic bytes
        first_chunk = b""
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if not first_chunk:
                    first_chunk = chunk
                f.write(chunk)

        if first_chunk and not _is_pdf_bytes(first_chunk):
            out_path.unlink(missing_ok=True)
            print("  [FAIL] Bukan file PDF (mungkin login wall atau HTML redirect)")
            return False

        size_kb = out_path.stat().st_size // 1024
        print(f"  [OK] {out_path.name} ({size_kb} KB)")
        return True

    except Exception as e:
        out_path.unlink(missing_ok=True)
        print(f"  [FAIL] {e}")
        return False


def write_source_json(folder: Path, source: dict, download_url: str, status: str,
                      filename: str = "", size_kb: int = 0) -> None:
    record = {
        "slug": source["slug"],
        "level": source.get("level", "unknown"),
        "subject": source.get("subject", ""),
        "url": source["url"],
        "note": source.get("note", ""),
        "status": status,
        "filename": filename,
        "size_kb": size_kb,
        "download_url": download_url,
    }
    (folder / "source.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Main orchestration ─────────────────────────────────────────────────────────

def process_source(
    source: dict,
    index: ExistingDataIndex,
    out_base: Path,
    force: bool = False,
) -> dict:
    """Proses satu source. Return dict statistik: {status, downloaded, skipped_dedup}."""
    slug = source["slug"]
    url = source["url"]
    source_type = source.get("type", "pdf")
    folder = out_base / slug

    result = {"slug": slug, "status": "skip", "downloaded": 0, "skipped_dedup": 0}

    # Cek apakah URL sumber sudah dikenal
    if not force and index.is_url_known(url):
        print(f"[{slug}] SKIP — URL sumber sudah ada di index")
        result["status"] = "dedup_url"
        return result

    print(f"\n[{slug}] level={source.get('level','?')} type={source_type}")
    print(f"  URL: {url}")

    if source_type == "page":
        links = scrape_links_from_page(url)
        print(f"  Ditemukan {len(links)} link dari halaman")

        if not links:
            folder.mkdir(parents=True, exist_ok=True)
            write_source_json(folder, source, url, "no_links_found")
            result["status"] = "no_links"
            return result

        folder.mkdir(parents=True, exist_ok=True)

        for link in links:
            file_id = link.get("file_id")
            dl_url = link.get("download_url", "")
            label = link.get("label", "unknown")

            # Dedup cek
            if file_id and not force and index.is_gdrive_id_known(file_id):
                print(f"    SKIP dedup (GDrive ID): {label[:60]}")
                result["skipped_dedup"] += 1
                continue
            if not force and index.is_url_known(dl_url):
                print(f"    SKIP dedup (URL): {label[:60]}")
                result["skipped_dedup"] += 1
                continue

            safe_label = _safe_filename(label)
            suffix = f"_{file_id[:8]}" if file_id else ""
            filename = f"{safe_label}{suffix}.pdf"
            out_path = folder / filename

            if out_path.exists() and not force:
                print(f"    SKIP (sudah ada): {filename}")
                continue

            print(f"    Download: {label[:60]}")
            ok = download_pdf(dl_url, out_path, file_id)
            if ok:
                result["downloaded"] += 1
                time.sleep(DELAY_SECONDS)

        write_source_json(
            folder, source, url,
            status="ok" if result["downloaded"] > 0 else "partial",
        )
        result["status"] = "ok"

    elif source_type in ("gdrive", "pdf"):
        file_id = extract_gdrive_id(url) if source_type == "gdrive" else None
        dl_url = gdrive_download_url(file_id) if file_id else url

        if file_id and not force and index.is_gdrive_id_known(file_id):
            print(f"  SKIP dedup (GDrive ID): {file_id}")
            result["status"] = "dedup_gdrive"
            return result
        if not force and index.is_url_known(dl_url):
            print(f"  SKIP dedup (URL): {dl_url}")
            result["status"] = "dedup_url"
            return result

        folder.mkdir(parents=True, exist_ok=True)
        filename = f"{slug}.pdf"
        out_path = folder / filename

        if out_path.exists() and not force:
            print(f"  SKIP (sudah ada): {filename}")
            result["status"] = "exists"
            return result

        ok = download_pdf(dl_url, out_path, file_id)
        if ok:
            size_kb = out_path.stat().st_size // 1024
            write_source_json(folder, source, dl_url, "ok", filename, size_kb)
            result["downloaded"] = 1
            result["status"] = "ok"
        else:
            write_source_json(folder, source, dl_url, "failed")
            result["status"] = "failed"

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape soal OSN Matematika SMA + kuliah dari sumber baru (dengan dedup)"
    )
    parser.add_argument("--dry-run", action="store_true", help="List sumber, tanpa download")
    parser.add_argument(
        "--level",
        choices=["sma", "kuliah", "all"],
        default="all",
        help="Filter level: sma | kuliah | all (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download ulang meski sudah ada / sudah di-dedup",
    )
    parser.add_argument(
        "--save-manifest",
        type=str,
        metavar="FILE",
        help="Simpan daftar sumber ke JSON",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(OSN_VER2_DIR),
        help=f"Direktori output (default: {OSN_VER2_DIR})",
    )
    parser.add_argument(
        "--skip-dedup-index",
        action="store_true",
        help="Lewati loading index dedup (lebih cepat, bisa duplikat)",
    )
    args = parser.parse_args()

    out_base = Path(args.out_dir)

    # Filter berdasarkan level
    sources = SOURCES if args.level == "all" else [s for s in SOURCES if s.get("level") == args.level]
    print(f"Sumber yang akan diproses: {len(sources)} (level={args.level})")

    if args.save_manifest:
        Path(args.save_manifest).write_text(
            json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Manifest disimpan ke {args.save_manifest}")

    if args.dry_run:
        print("\n=== DRY RUN — tidak ada download ===")
        for s in sources:
            print(f"  [{s['level']:6}] [{s['type']:6}] {s['slug']}")
            print(f"           {s['url']}")
        return

    # Build dedup index
    if args.skip_dedup_index:
        index = ExistingDataIndex.__new__(ExistingDataIndex)
        index.known_urls = set()
        index.known_gdrive_ids = set()
        index.known_question_hashes = set()
        print("[DeduP] Index dilewati (--skip-dedup-index)")
    else:
        print("\nMemuat index data yang sudah ada...")
        index = ExistingDataIndex()

    # Proses semua sumber
    stats = {"ok": 0, "dedup": 0, "failed": 0, "total_files": 0}
    for source in sources:
        result = process_source(source, index, out_base, force=args.force)
        status = result["status"]
        if status.startswith("dedup"):
            stats["dedup"] += 1
        elif status in ("ok", "no_links", "partial"):
            stats["ok"] += 1
            stats["total_files"] += result["downloaded"]
        elif status == "failed":
            stats["failed"] += 1
        time.sleep(DELAY_SECONDS)

    print(f"\n{'='*50}")
    print(f"Selesai: {stats['ok']} sumber diproses, {stats['dedup']} di-skip (dedup), "
          f"{stats['failed']} gagal")
    print(f"Total file PDF baru: {stats['total_files']}")
    print(f"Output: {out_base.resolve()}")
    print("\nLangkah selanjutnya untuk cek duplikat di level soal:")
    print(f"  python -m src.preprop.extract {out_base}")
    print("  python -m src.preprop.dedup data/filtered/after_validity.jsonl data/filtered/clean.jsonl")


if __name__ == "__main__":
    main()
