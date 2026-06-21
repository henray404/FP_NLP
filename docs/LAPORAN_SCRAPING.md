# Laporan Scraping & Akuisisi Data — IndoMathReason

Laporan ini menjelaskan proses pengumpulan data soal matematika Indonesia untuk
proyek IndoMathReason, dari percobaan pertama sampai pendekatan yang dipakai
sekarang. Sumber laporan: riwayat commit, script `src/scraping/`, modul
`src/data_pipeline/`, dan notebook ekstraksi.

## 1. Tujuan

Mengumpulkan soal matematika Indonesia (OSN SMA dan buku ajar SMA/PT) beserta
jawaban dan pembahasan sebagai korpus latih untuk fine-tuning Qwen2.5-0.5B-Base.
Proses berkembang melalui enam fase. Setiap fase memperbaiki masalah pada fase
sebelumnya.

## 2. Ringkasan Fase

| Fase | Pendekatan | Tool / Model | Hasil | Masalah utama |
|------|-----------|--------------|-------|---------------|
| 0 | Pengumpulan manual | — | Data awal di-commit | Tidak terskala, tanpa metadata |
| 1 | Scraper OSN defantri.com | requests, BeautifulSoup | 80 PDF OSN | Tahun tak terbaca, filename `unknown_*` |
| 2 | Scraper multi-sumber | requests, BeautifulSoup | 18 sumber buku ajar | Sumber berdinding (login, 403) |
| 3 | Ekstraksi teks | pdfplumber, Tesseract OCR | `clean.jsonl`, 1781 record | 82% jawaban kosong, watermark |
| 4 | Banding tool PDF→MD | fitz, pymupdf4llm, markitdown | Pilih extractor | Notasi math rusak |
| 5 | Filter mutu MD | Heuristik metrik | Pisah MD `good`/`hancur` | Banyak MD hancur |
| 6 | Ekstraksi VLM | Gemini Flash, PyMuPDF | Pipeline aktif | Rate limit API (429/503) |

## 3. Detail per Fase

### Fase 0 — Pengumpulan Manual

Commit `25611f8`, `9a46684`, `9ab15d2`, `cff4e77`. Data di-commit langsung ke
repo. Soal teks dan soal gambar sudah dipisah, tapi soal gambar belum diubah jadi
teks.

**Masalah:** tanpa otomatisasi dan metadata sumber, proses tidak terskala.

### Fase 1 — Scraper OSN `defantri.com`

File: `src/scraping/scrape_defantri.py`. Output: 80 PDF di `data/raw/osn/`.

**Tool:** `requests` (HTTP), `beautifulsoup4` (parsing HTML). Sumber tunggal:
index OSN Matematika SMA di `defantri.com`.

**Cara kerja:**
1. Ambil halaman index, telusuri tag `h3/h4/li/a`. Heading tahun jadi konteks
   `current_year`.
2. Ekstrak semua link Google Drive (regex `file/d/<id>` dan `open?id=`), parse
   metadata `{label, year, level, file_id}`. Pemetaan level: kabupaten/kota,
   provinsi, nasional/semifinal/final.
3. Deduplikasi berdasarkan `file_id`.
4. Download via `uc?export=download&id=`. Untuk file besar, tangani halaman
   konfirmasi virus-scan Google Drive dengan mengekstrak URL konfirmasi dari HTML.
5. User-Agent kustom, timeout 60 detik, jeda 1.5 detik antar download, skip file
   yang sudah ada (resume).

**Masalah:** metadata tahun tidak terbaca dari heading HTML, jadi filename keluar
sebagai `unknown_Download_File_<id8>.pdf`. Solusi: lacak tahun lewat manifest
JSON, bukan nama file.

### Fase 2 — Scraper Multi-Sumber

File: `src/scraping/download_sources.py`. Manifest 18 sumber buku ajar (kalkulus,
aljabar linear, kombinatorika, statistika, geometri, vektor).

**Tool:** `requests` (Session + retry), `beautifulsoup4`. Manifest-routed:
scraper memilih jalur download otomatis per tipe URL.

**Empat jalur download:**
1. PDF langsung — stream download.
2. Google Drive — pakai ulang helper gdrive dari Fase 1.
3. Katalog SPA `buku.kemendikdasmen.go.id` — halaman React, URL PDF asli di-resolve
   lewat API `getDetails` (field `attachment`).
4. Landing page HTML — cari link `.pdf` di dalam halaman (`<a>`, `<iframe>`,
   `<embed>`, meta `citation_pdf_url`).

**Pengaman:** validasi magic bytes `%PDF`; jika hasil download ternyata HTML
(login wall), file dihapus dan sumber ditandai `manual`. Header `Referer` di-set
ke origin. Retry sekali untuk error koneksi transient. Tiap sumber disimpan ke
`data/raw/<slug>/` dengan `source.json` (url, subject, status, filename).

**Masalah:** tiga sumber berdinding harus di-download manual: Scribd (login),
ResearchGate (login), `mashadi.staff.unri.ac.id` (Cloudflare bot-block 403).
Catatan: 14 dari 18 sumber adalah level PT, lebih luas dari scope OSN/SMA, jadi
manifest menandai `level` dan `topic` untuk filter di hilir.

### Fase 3 — Ekstraksi Teks (pdfplumber)

File: `src/data_pipeline/extract.py`, `extract_questions.py`, `filter_rules.py`,
`to_csv.py`, `dedup.py`.

**Tool / model:**
- `pdfplumber` — ekstraksi teks per-halaman.
- `pytesseract` (Tesseract OCR, bahasa `ind+eng`) — fallback untuk halaman scan.
- Opsional VLM caption gambar: `Salesforce/blip-image-captioning-base`.
- Split soal via pola `SOAL N` atau nomor; crop gambar dengan traceability
  per-halaman (`source_page`, `question_id`).

**Filter dan dedup hilir:**
- `filter_rules.py` — regex pilihan ganda, kata kunci gambar, true/false,
  deteksi bahasa (`langdetect`).
- `filter_validity.py` — LLM judge `Qwen2.5-7B-Instruct` via vLLM, greedy decode,
  cek label `VALID` / `TIDAK_VALID`.
- `dedup.py` — embedding `paraphrase-multilingual-MiniLM-L12-v2` +
  FAISS `IndexFlatIP` (cosine). Threshold dedup 0.92, dekontaminasi 0.90.

**Masalah** (audit `clean.jsonl`, 1781 record):
1. **Schema drift** — kunci `soal/jawaban/cara` di data lama tidak cocok dengan
   `question/answer` yang dibaca `filter_rules.py` dan `dedup.py`. `dedup.py`
   menghasilkan `KeyError`.
2. **82% jawaban kosong** (1461 dari 1781). Ekstraktor gagal memasangkan soal
   dengan jawaban.
3. **Watermark dan footer** tersisip karakter-per-karakter ke dalam teks soal.
   Sekitar 9% record tanpa sinyal soal sama sekali.

Kesimpulan: ekstraksi teks merusak notasi matematika dan mencampur soal dengan
jawaban. Ini memicu pindah ke pendekatan berbasis gambar.

### Fase 4 — Banding Tool PDF→Markdown

File: `notebooks/compare_pdf2md.py`.

Bandingkan tiga tool konversi PDF→Markdown pada sampel halaman math-berat (OSN
digital, buku bergambar, integral kalkulus, notasi kombinatorika). Tiap sampel
dipotong jadi sub-PDF supaya semua tool baca halaman sama.

| Tool | Catatan |
|------|---------|
| `fitz_raw` (PyMuPDF) | Ekstraksi teks mentah |
| `pymupdf4llm` | PDF→Markdown |
| `markitdown` | PDF→Markdown |

Metrik per output: `chars` (volume), `cid` (jumlah `(cid:NN)`, font gagal map,
makin kecil makin baik), `mathsym` (simbol math unicode kebawa), `latex` (notasi
LaTeX yang dipertahankan). Output ditulis ke `data/tool_compare/` untuk
inspeksi manual.

**Masalah:** beberapa PDF (font non-standar) menghasilkan banyak `(cid:NN)` di
semua tool, jadi notasi math hilang.

### Fase 5 — Filter Mutu Markdown

File: `notebooks/filter_md.py`. Output: `data/filtered/md_quality.csv`,
`md_good.txt`, `md_hancur.txt`.

Klasifikasi tiap MD jadi `good` atau `hancur` berdasarkan ambang yang dikalibrasi
dari 79 MD OSN:

| Metrik | Ambang `hancur` | Arti |
|--------|-----------------|------|
| `cid_per_kc` | > 3.0 per 1000 char | Font tak ter-map |
| `repl_per_kc` | > 1.0 per 1000 char | Encoding gagal (`�`) |
| `alpha_ratio` | < 0.20 | Simbol-soup, math hilang |
| `chars` / `words` | < 500 / < 50 | Ekstraksi kosong |

MD `good` masuk jalur teks murah (Fase 6); MD `hancur` dikirim ke VLM image.

**Masalah:** sebagian besar PDF buku ajar masuk kategori `hancur`, jadi tidak
bisa dipakai langsung dan menambah beban VLM.

### Fase 6 — Ekstraksi VLM Gemini (Sekarang)

File: `notebooks/extract_vlm_gemini.ipynb`. Output: `data/extracted_vlm/`.

**Tool / model:**
- `google-genai` (Gemini API) — backend default.
- `PyMuPDF` (fitz) — render tiap halaman PDF jadi gambar PNG (tanpa poppler).
- Backend alternatif: Gemini CLI (OAuth) atau `Qwen2.5-VL-7B-Instruct` lokal/Kaggle
  via GPU (`transformers`, `bitsandbytes`).

**Model dan parameter (routing hybrid):**

| Route | Model | Thinking | DPI |
|-------|-------|----------|-----|
| Teks bersih | `gemini-2.5-flash-lite` | 0 (off) | — |
| Halaman ribet | `gemini-3.1-flash-lite` | -1 (high/dinamis) | 200 |

Parameter global: `TEMPERATURE=0.0`, render default `DPI=150`, `MAX_RETRIES=6`,
`MAX_WORKERS=4`. Harga `gemini-3.1-flash-lite` sekitar $0.10 / $0.40 per juta
token (input / output).

**Cara kerja:** tiap halaman di-render jadi gambar, lalu Gemini mengekstrak hasil
terstruktur via satu prompt JSON Indonesia. Prompt memisahkan `soal` vs `materi`,
memisahkan `question`/`answer`/`steps`, membuang noise (header, footer, watermark,
URL, nama penulis), dan mempertahankan notasi math sebagai LaTeX inline `$...$`.

**Routing per-halaman** (`route_page`, menghemat token):
- Halaman teks bersih memakai model murah, thinking off.
- Halaman ribet (cakupan gambar > 12% luas halaman, atau > 30% baris ≤3 char
  seperti pecahan vertikal, atau teks < 40 char) di-render gambar DPI 200 dan
  memakai model kuat + thinking, supaya model lemah tidak berhalusinasi membuat
  soal palsu.
- Halaman scan tanpa text-layer (`SKIP_SCAN`) dibuang, tidak dikirim.
- Halaman tanpa marker soal di-skip, kecuali folder `osn` (kumpulan soal murni).

**Ketahanan operasional:** resume per-halaman (append ke `.raw.jsonl`, lanjut dari
halaman terakhir), eksekusi paralel 4 worker, client per-thread, retry yang
menghormati `retryDelay`, dan batch mode (1 job async, 50% lebih murah, tanpa 503).

**Filter pipeline terintegrasi:** `_all_soal.jsonl` melewati rule-based (buang
pilihan ganda, true/false, non-Indonesia), validity, dedup (MiniLM + FAISS,
threshold 0.92), lalu reformat ke `{soal, cara, jawaban}` → `clean_vlm.jsonl`.

**Masalah:** kuota API Gemini sering kena rate limit (429) dan model overloaded
(503). Diatasi dengan batch mode, penurunan worker dari 8 ke 4, dan retry. Saat
kuota habis, validity check diganti heuristik teks murni (CPU, gratis) yang
menolak soal yang merujuk objek visual tak terdeskripsi.

## 4. Ringkasan Tool & Model

| Tahap | Tool / Model |
|-------|--------------|
| Scraping | `requests`, `beautifulsoup4` |
| Ekstraksi teks | `pdfplumber`, Tesseract OCR (`ind+eng`), BLIP (opsional) |
| PDF→Markdown | `fitz`, `pymupdf4llm`, `markitdown` |
| Render halaman | `PyMuPDF` (DPI 150/200) |
| VLM ekstraksi | `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite`; alt `Qwen2.5-VL-7B-Instruct` |
| Validity judge | `Qwen2.5-7B-Instruct` (vLLM) atau heuristik teks |
| Dedup | `paraphrase-multilingual-MiniLM-L12-v2` + FAISS `IndexFlatIP` |
| CoT synthesis (rencana) | `DeepSeek-R1-Distill-Qwen-7B` |
| Base fine-tune | `Qwen2.5-0.5B-Base` |

## 5. Status Data Sekarang

```
data/raw/osn/         80 PDF OSN (fase 1)
data/raw/<slug>/      18 sumber buku ajar + source.json (fase 2)
data/extracted/       80 JSONL pdfplumber + images/ (fase 3)
data/extracted_vlm/   hasil VLM Gemini (fase 6)
data/tool_compare/    output banding 3 tool PDF→MD (fase 4)
data/filtered/        clean.jsonl, md_quality.csv, md_good.txt, md_hancur.txt
```

Data mentah tidak di-commit (ukuran besar). Tersimpan di Google Drive `DATA_NLP`.

## 6. Langkah Berikutnya

1. Migrasi `clean.jsonl` lama ke schema kanonik dan strip watermark.
2. Jalankan VLM batch penuh atas 80 OSN dan 18 buku ajar.
3. Lanjut ke CoT synthesis (`DeepSeek-R1-Distill-Qwen-7B`) untuk mengisi jawaban
   yang kosong via teacher model.
