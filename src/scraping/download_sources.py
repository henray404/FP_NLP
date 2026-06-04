"""
Download buku/materi matematika dari daftar URL ke data/raw/<slug>/.

Setiap link dapat SATU folder berisi:
  - file PDF (jika berhasil di-download)
  - source.json  (url, subject, note, status, filename) untuk traceability

Menangani beberapa tipe URL:
  - URL .pdf langsung               -> stream download
  - Google Drive (file/d/<id>)      -> pakai helper dari scrape_defantri
  - Halaman HTML (scribd / researchgate / katalog kemendikdasmen):
      coba temukan link .pdf di dalam halaman; kalau tidak ketemu -> status "manual"

File hasil download divalidasi magic bytes %PDF; kalau ternyata HTML
(halaman error / login wall) file dihapus dan source ditandai "manual".

Usage:
    python -m src.scraping.download_sources                       # download semua
    python -m src.scraping.download_sources --dry-run             # list saja, tanpa download
    python -m src.scraping.download_sources --only kombinatorika_itb statistika_pendidikan
    python -m src.scraping.download_sources --force              # download ulang walau sudah ada
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests

RAW_DIR = Path("data/raw")
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*",
}
DELAY_SECONDS = 1.5
TIMEOUT = 90


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST — satu entry per link. slug = nama folder di data/raw/
# ─────────────────────────────────────────────────────────────────────────────

SOURCES: list[dict] = [
    {"slug": "kalkulus_1_polsri",          "subject": "Kalkulus",        "note": "tanpa kunci jawaban",
     "url": "http://eprints.polsri.ac.id/15740/1/2024_E-BOOK_KALKULUS%201_Langsung%20Terbit.pdf"},
    {"slug": "kalkulus_2_bbg",             "subject": "Kalkulus",        "note": "sebagian ada jawaban",
     "url": "https://repository.bbg.ac.id/bitstream/764/1/Kalkulus_II.pdf"},
    {"slug": "kalkulus_unipahlawan",       "subject": "Kalkulus/Materi", "note": "tanpa jawaban",
     "url": "https://staff.universitaspahlawan.ac.id/web/upload/materials/1538-materials.pdf"},
    {"slug": "kalkulus_purcell",           "subject": "Kalkulus",        "note": "scribd - kemungkinan butuh login",
     "url": "https://www.scribd.com/document/534445050/Kalkulus-Purcell-Edisi-9-Jilid-1"},
    {"slug": "aljabar_linear_uinsgd",      "subject": "Aljabar Linear",  "note": "",
     "url": "https://digilib.uinsgd.ac.id/26164/1/Buku%20Pengantar%20Aljabar%20Linear.pdf"},
    {"slug": "bilangan_dan_aljabar",       "subject": "Aljabar",         "note": "researchgate - kemungkinan butuh login",
     "url": "https://www.researchgate.net/publication/374875529_BUKU_AJAR_BILANGAN_DAN_ALJABAR_2021"},
    {"slug": "aljabar_linear_dasar",       "subject": "Aljabar Linear",  "note": "",
     "url": "https://wadianforever.wordpress.com/wp-content/uploads/2015/01/aljabar-linear-dasar.pdf"},
    {"slug": "gdrive_1vmka",               "subject": "?",               "note": "Google Drive",
     "url": "https://drive.google.com/file/d/1vmkaNUXp13em7Q_9Sx_xAcbdrbugcI6d/view"},
    {"slug": "gdrive_155ne",               "subject": "?",               "note": "Google Drive",
     "url": "https://drive.google.com/file/d/155neouRfVG-Rhoth6pmJYmeV7JLp16av/view"},
    {"slug": "materi_pembelajaran_uki",    "subject": "?",               "note": "",
     "url": "http://repository.uki.ac.id/1811/1/BUKU%20MATERI%20PEMBELAJARAN.pdf"},
    {"slug": "kombinatorika_itb",          "subject": "Kombinatorika",   "note": "",
     "url": "https://informatika.stei.itb.ac.id/~rinaldi.munir/Matdis/2024-2025/18-Kombinatorika-Bagian1-2024.pdf"},
    {"slug": "kombinatorika_aljabar_itb",  "subject": "Kombinatorika",   "note": "",
     "url": "https://fgb.itb.ac.id/wp-content/uploads/sites/26/2024/11/Ebook-Prof.-Djoko-Suprijanto-Teori-Desain-dan-Teori-Koding-Panorama-Kombinatorika-Aljabar-1.pdf"},
    {"slug": "statistika_pendidikan",      "subject": "Statistika",      "note": "",
     "url": "https://digilib.uinsgd.ac.id/21828/1/buku%20statistika%20pendidikan.pdf"},
    {"slug": "geometri_unri",              "subject": "Geometri",        "note": "Cloudflare bot-block (403) - download manual via browser",
     "url": "https://mashadi.staff.unri.ac.id/files/2018/10/BUKU-GEOMETRI-EDISI-2.pdf"},
    {"slug": "analisa_vektor_uki",         "subject": "Analisa Vektor",  "note": "",
     "url": "http://repository.uki.ac.id/17944/1/BukuAjarAnalisaVektor.pdf"},
    {"slug": "buku_guru_matematika",       "subject": "Matematika SMA",  "note": "",
     "url": "https://repositori.kemendikdasmen.go.id/6962/1/buku%20guru%20matematika.pdf"},
    {"slug": "matematika_kelas_xii",       "subject": "Matematika SMA",  "note": "katalog SPA - resolve via API getDetails",
     "url": "https://buku.kemendikdasmen.go.id/katalog/Buku-Guru-Matematika-Kelas-XII"},
    {"slug": "matematika_tingkat_lanjut_xii", "subject": "Matematika SMA", "note": "katalog SPA - resolve via API getDetails",
     "url": "https://buku.kemendikdasmen.go.id/katalog/buku-panduan-guru-matematika-tingkat-lanjut-untuk-smama-kelas-xii"},
]


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE (mirror dari scrape_defantri; di-inline supaya modul ini hanya
# butuh `requests` dan bisa jalan tanpa bs4 / dependency lain)
# ─────────────────────────────────────────────────────────────────────────────

_GDRIVE_VIEW_RE = re.compile(r"drive\.google\.com/file/d/([^/]+)/")
_GDRIVE_OPEN_RE = re.compile(r"drive\.google\.com/open\?id=([^&]+)")


def extract_file_id(url: str) -> str | None:
    m = _GDRIVE_VIEW_RE.search(url) or _GDRIVE_OPEN_RE.search(url)
    return m.group(1) if m else None


def gdrive_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def _gdrive_confirm_url(file_id: str, html: str) -> str | None:
    """Ambil URL konfirmasi dari halaman virus-scan Google Drive (file besar)."""
    m = re.search(r'href="(/uc\?export=download[^"]+confirm=[^"]+)"', html)
    if m:
        return "https://drive.google.com" + m.group(1).replace("&amp;", "&")
    m = re.search(r'action="(https://drive\.google\.com/uc[^"]+)"', html)
    if m:
        return m.group(1).replace("&amp;", "&") + "&confirm=t"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_gdrive(url: str) -> bool:
    return "drive.google.com" in url


# Halaman katalog buku.kemendikdasmen.go.id adalah SPA React; URL PDF asli
# diambil dari API getDetails (field "attachment").
KEMDIK_API = "https://api.buku.cloudapp.web.id/api/catalogue/getDetails?slug="


def is_kemendik_katalog(url: str) -> bool:
    p = urlparse(url)
    return "buku.kemendikdasmen.go.id" in p.netloc and "/katalog/" in p.path


def resolve_kemendik_attachment(url: str, session: requests.Session) -> str | None:
    slug = url.rstrip("/").split("/katalog/")[-1]
    resp = get_with_retry(session, KEMDIK_API + slug, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, dict):
        return results.get("attachment")
    return None


def headers_for(url: str) -> dict:
    """Header dasar + Referer = origin URL (banyak server menolak tanpa Referer)."""
    p = urlparse(url)
    h = dict(HEADERS)
    h["Referer"] = f"{p.scheme}://{p.netloc}/"
    return h


def get_with_retry(session: requests.Session, url: str, retries: int = 1, **kwargs) -> requests.Response:
    """GET dengan satu kali retry untuk error koneksi transient (RemoteDisconnected dll)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return session.get(url, **kwargs)
        except requests.exceptions.ConnectionError as e:
            last_exc = e
            if attempt < retries:
                time.sleep(2)
    raise last_exc  # type: ignore[misc]


def url_looks_like_pdf(url: str) -> bool:
    return urlparse(url).path.lower().endswith(".pdf")


def filename_from(resp: requests.Response, url: str, slug: str) -> str:
    """Tentukan nama file dari Content-Disposition, lalu URL, lalu slug."""
    cd = resp.headers.get("Content-Disposition", "")
    m = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)", cd, re.IGNORECASE)
    name = unquote(m.group(1)) if m else unquote(Path(urlparse(resp.url).path).name)
    if not name.lower().endswith(".pdf"):
        name = f"{slug}.pdf"
    name = re.sub(r"[^\w.\-]+", "_", name).strip("_")
    return name or f"{slug}.pdf"


def find_pdf_link(base_url: str, html: str) -> str | None:
    """Cari link .pdf di dalam halaman HTML landing page (butuh bs4; opsional)."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("    (bs4 tidak terpasang -> tidak bisa parse landing page; pip install beautifulsoup4)")
        return None
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if ".pdf" in a["href"].lower():
            return urljoin(base_url, a["href"])
    for tag in soup.find_all(["iframe", "embed"], src=True):
        if ".pdf" in tag["src"].lower():
            return urljoin(base_url, tag["src"])
    meta = soup.find("meta", attrs={"name": "citation_pdf_url", "content": True})
    if meta:
        return urljoin(base_url, meta["content"])
    return None


def save_pdf(resp: requests.Response, out_path: Path) -> bool:
    """Stream ke file, validasi magic bytes %PDF. Hapus & return False jika bukan PDF."""
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    with open(out_path, "rb") as f:
        head = f.read(5)
    if not head.startswith(b"%PDF"):
        out_path.unlink(missing_ok=True)
        return False
    return True


def fetch_gdrive(url: str, session: requests.Session) -> requests.Response:
    fid = extract_file_id(url)
    if not fid:
        raise ValueError("Google Drive file id tidak ditemukan")
    resp = get_with_retry(session, gdrive_download_url(fid), headers=HEADERS, timeout=TIMEOUT, stream=True)
    resp.raise_for_status()
    if "text/html" in resp.headers.get("Content-Type", ""):
        confirm = _gdrive_confirm_url(fid, resp.text)
        if confirm:
            resp = get_with_retry(session, confirm, headers=HEADERS, timeout=TIMEOUT, stream=True)
            resp.raise_for_status()
    return resp


def fetch_pdf_response(url: str, session: requests.Session) -> tuple[requests.Response | None, str]:
    """
    Kembalikan (response_siap_di-stream, resolved_url).
    Untuk landing page HTML, ikuti ke link .pdf di dalamnya.
    Raise / return (None, ...) kalau tidak ada PDF.
    """
    if is_gdrive(url):
        return fetch_gdrive(url, session), url

    if is_kemendik_katalog(url):
        pdf_url = resolve_kemendik_attachment(url, session)
        if not pdf_url:
            return None, url
        resp = get_with_retry(session, pdf_url, headers=headers_for(pdf_url), timeout=TIMEOUT, stream=True, allow_redirects=True)
        resp.raise_for_status()
        return resp, pdf_url

    resp = get_with_retry(session, url, headers=headers_for(url), timeout=TIMEOUT, stream=True, allow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "").lower()

    if "pdf" in ctype or url_looks_like_pdf(resp.url):
        return resp, resp.url

    if "html" in ctype:
        pdf_url = find_pdf_link(resp.url, resp.text)
        if not pdf_url:
            return None, resp.url
        resp2 = get_with_retry(session, pdf_url, headers=headers_for(pdf_url), timeout=TIMEOUT, stream=True, allow_redirects=True)
        resp2.raise_for_status()
        return resp2, pdf_url

    # tipe lain (octet-stream dll) — coba saja, save_pdf akan memvalidasi
    return resp, resp.url


def write_meta(folder: Path, source: dict, status: str, **extra) -> None:
    meta = {
        "slug": source["slug"],
        "subject": source.get("subject", ""),
        "url": source["url"],
        "note": source.get("note", ""),
        "status": status,
        **extra,
    }
    (folder / "source.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PER-SOURCE
# ─────────────────────────────────────────────────────────────────────────────

def process(source: dict, session: requests.Session, force: bool, raw_dir: Path) -> str:
    folder = raw_dir / source["slug"]
    folder.mkdir(parents=True, exist_ok=True)

    existing = [p for p in folder.glob("*.pdf")]
    if existing and not force:
        print(f"  SKIP (exists): {source['slug']}/{existing[0].name}")
        return "skip"

    try:
        resp, resolved = fetch_pdf_response(source["url"], session)
    except Exception as e:
        print(f"  FAIL {source['slug']}: {e}")
        write_meta(folder, source, "failed", error=str(e))
        return "failed"

    if resp is None:
        print(f"  MANUAL {source['slug']}: tidak ada link PDF di halaman (download manual)")
        write_meta(folder, source, "manual", reason="no_pdf_link", resolved_url=resolved)
        return "manual"

    filename = filename_from(resp, source["url"], source["slug"])
    out_path = folder / filename

    ok = save_pdf(resp, out_path)
    if not ok:
        print(f"  MANUAL {source['slug']}: konten bukan PDF (login wall / halaman HTML)")
        write_meta(folder, source, "manual", reason="not_a_pdf", resolved_url=resolved)
        return "manual"

    size_kb = out_path.stat().st_size // 1024
    print(f"  OK   {source['slug']}/{filename} ({size_kb} KB)")
    write_meta(folder, source, "ok", filename=filename, size_kb=size_kb, resolved_url=resolved)
    return "ok"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Download buku/materi matematika ke data/raw/<slug>/")
    parser.add_argument("--dry-run", action="store_true", help="List sources tanpa download")
    parser.add_argument("--only", nargs="+", help="Hanya proses slug tertentu")
    parser.add_argument("--force", action="store_true", help="Download ulang walau file sudah ada")
    parser.add_argument("--out-dir", default=str(RAW_DIR))
    args = parser.parse_args()

    raw_dir = Path(args.out_dir)

    sources = SOURCES
    if args.only:
        wanted = set(args.only)
        sources = [s for s in SOURCES if s["slug"] in wanted]
        missing = wanted - {s["slug"] for s in sources}
        if missing:
            print(f"Slug tidak dikenal: {', '.join(sorted(missing))}")

    print(f"{len(sources)} sources -> {raw_dir}/<slug>/\n")

    if args.dry_run:
        for s in sources:
            print(f"  [{s.get('subject','?'):16}] {s['slug']:28} {s['url']}")
        return

    session = requests.Session()
    counts: dict[str, int] = {}
    for i, s in enumerate(sources):
        print(f"[{i+1}/{len(sources)}] {s['slug']}")
        status = process(s, session, args.force, raw_dir)
        counts[status] = counts.get(status, 0) + 1
        if i < len(sources) - 1:
            time.sleep(DELAY_SECONDS)

    summary = " | ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
    print(f"\nDone -> {summary}")


if __name__ == "__main__":
    main()
