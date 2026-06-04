"""
Scrape index soal OSN Matematika SMA dari defantri.com
lalu download semua PDF ke data/raw/osn/

Usage:
    # Dry run -- lihat semua link tanpa download
    python -m src.scraping.scrape_defantri --dry-run

    # Download semua
    python -m src.scraping.scrape_defantri

    # Download tahun tertentu saja
    python -m src.scraping.scrape_defantri --year 2023

    # Simpan manifest link ke JSON
    python -m src.scraping.scrape_defantri --dry-run --save-manifest manifest.json
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://www.defantri.com/2013/04/materi-soal-osk-osp-matematika-sma.html"
RAW_DIR = Path("data/raw/osn")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-scraper/1.0)"}
DELAY_SECONDS = 1.5  # polite delay between downloads


# ── Google Drive helpers ───────────────────────────────────────────────────────

GDRIVE_VIEW_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)/")
GDRIVE_OPEN_RE = re.compile(r"drive\.google\.com/open\?id=([^&]+)")


def extract_file_id(url: str) -> str | None:
    m = GDRIVE_VIEW_RE.search(url) or GDRIVE_OPEN_RE.search(url)
    return m.group(1) if m else None


def gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ── Year / level parser ────────────────────────────────────────────────────────

YEAR_RE = re.compile(r"\b(20\d{2})\b")

LEVEL_MAP = {
    "kabupaten": "kabupaten",
    "kota": "kabupaten",
    "provinsi": "provinsi",
    "nasional": "nasional",
    "semifinal": "nasional_semi",
    "semi final": "nasional_semi",
    "final": "nasional_final",
}


def parse_year(text: str) -> str:
    m = YEAR_RE.search(text)
    return m.group(1) if m else "unknown"


def parse_level(text: str) -> str:
    lower = text.lower()
    for key, val in LEVEL_MAP.items():
        if key in lower:
            return val
    return "unknown"


# ── Scraper ────────────────────────────────────────────────────────────────────

def scrape_links(url: str = INDEX_URL) -> list[dict]:
    """Fetch index page, extract all Google Drive links with metadata."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    entries = []
    current_year = "unknown"

    for tag in soup.find_all(["h4", "h3", "li", "a"]):
        if tag.name in ("h3", "h4"):
            y = parse_year(tag.get_text())
            if y != "unknown":
                current_year = y
            continue

        if tag.name == "a":
            href = tag.get("href", "")
            file_id = extract_file_id(href)
            if not file_id:
                continue
            label = tag.get_text(strip=True)
            year = parse_year(label) or current_year
            entries.append({
                "label": label,
                "year": year,
                "level": parse_level(label),
                "file_id": file_id,
                "gdrive_url": href,
                "download_url": gdrive_download_url(file_id),
            })

    # Deduplicate by file_id
    seen = set()
    unique = []
    for e in entries:
        if e["file_id"] not in seen:
            seen.add(e["file_id"])
            unique.append(e)

    return unique


# ── Downloader ─────────────────────────────────────────────────────────────────

def safe_filename(label: str, year: str, file_id: str) -> str:
    clean = re.sub(r"[^\w\s-]", "", label).strip().replace(" ", "_")
    clean = re.sub(r"_+", "_", clean)[:60]
    return f"{year}_{clean}_{file_id[:8]}.pdf"


def _gdrive_confirm_url(file_id: str, html: str) -> str | None:
    """Extract confirmation URL from Google Drive virus-scan page."""
    m = re.search(r'href="(/uc\?export=download[^"]+confirm=[^"]+)"', html)
    if m:
        return "https://drive.google.com" + m.group(1).replace("&amp;", "&")
    m = re.search(r'action="(https://drive\.google\.com/uc[^"]+)"', html)
    if m:
        return m.group(1).replace("&amp;", "&") + "&confirm=t"
    return None


def download_pdf(entry: dict, out_dir: Path) -> Path | None:
    """Download single PDF. Returns path on success, None on failure."""
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = safe_filename(entry["label"], entry["year"], entry["file_id"])
    out_path = out_dir / filename

    if out_path.exists():
        print(f"  SKIP (exists): {filename}")
        return out_path

    try:
        resp = requests.get(
            entry["download_url"],
            headers=HEADERS,
            timeout=60,
            stream=True,
        )
        resp.raise_for_status()

        # Google Drive large-file virus-scan confirmation
        if "text/html" in resp.headers.get("Content-Type", ""):
            confirm_url = _gdrive_confirm_url(entry["file_id"], resp.text)
            if confirm_url:
                resp = requests.get(confirm_url, headers=HEADERS, timeout=60, stream=True)
                resp.raise_for_status()

        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = out_path.stat().st_size // 1024
        print(f"  OK  {filename} ({size_kb} KB)")
        return out_path

    except Exception as e:
        print(f"  FAIL {filename}: {e}")
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape & download OSN math PDFs from defantri.com")
    parser.add_argument("--dry-run", action="store_true", help="List links, no download")
    parser.add_argument("--year", type=str, help="Filter by year e.g. 2023")
    parser.add_argument("--save-manifest", type=str, help="Save link manifest to JSON")
    parser.add_argument("--out-dir", type=str, default=str(RAW_DIR))
    args = parser.parse_args()

    print(f"Fetching index: {INDEX_URL}")
    entries = scrape_links()
    print(f"Found {len(entries)} unique PDF links")

    if args.year:
        entries = [e for e in entries if e["year"] == args.year]
        print(f"Filtered to year {args.year}: {len(entries)} links")

    if args.save_manifest:
        with open(args.save_manifest, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Manifest saved to {args.save_manifest}")

    if args.dry_run:
        for e in entries:
            print(f"  [{e['year']}] [{e['level']}] {e['label']}")
            print(f"    -> {e['download_url']}")
        return

    out_dir = Path(args.out_dir)
    success, fail = 0, 0
    for i, entry in enumerate(entries):
        print(f"[{i+1}/{len(entries)}] {entry['label']}")
        result = download_pdf(entry, out_dir)
        if result:
            success += 1
        else:
            fail += 1
        if i < len(entries) - 1:
            time.sleep(DELAY_SECONDS)

    print(f"\nDone: {success} downloaded, {fail} failed -> {out_dir}")


if __name__ == "__main__":
    main()
