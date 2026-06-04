# data/

Isi folder `data/` **tidak** disimpan di git (terlalu besar). Datanya ada di Google Drive.

## Link Google Drive

<!-- TODO: tempel link Google Drive kamu di sini -->
(belum diisi)

## Struktur yang diharapkan

```
data/
  raw/            # PDF mentah (per folder sumber) — download dari Drive
  extracted/      # hasil pdfplumber lama (jsonl + images)
  extracted_vlm/  # hasil notebooks/extract_vlm_gemini.ipynb
  filtered/       # hasil filter_rules -> filter_validity -> dedup
```

Download isi dari link Drive di atas, lalu taruh sesuai struktur ini sebelum menjalankan pipeline.
