# Language Mixing pada Model Reasoning (untuk Skenario 1 & Pembahasan)

**Rujukan paling penting untuk paper: menjelaskan masalah yang sudah kalian alami tapi belum
kalian namai.**

## Rujukan utama

> "Language Mixing in Reasoning Language Models: Patterns, Impact, and Internal Causes."
> arXiv:2505.14815, 2025.

`[Terverifikasi]` judul + ID. `[Perlu cek]` penulis — belum dibuka.

Kutipan kunci dari ringkasan pencarian
`[Perlu cek — verifikasi kalimat persisnya sebelum dikutip di paper]`:

> "DeepSeek-R1 distilled models consistently exhibit **higher language mixing entropy** than their
> original backbones, suggesting distillation **amplifies language mixing when the input language
> is neither English nor Chinese**."

## Kenapa penting

Input kalian Bahasa Indonesia — bukan Inggris, bukan Mandarin. Teacher lama
(`DeepSeek-R1-Distill-Qwen-7B`) justru dari keluarga yang paling rawan.

**Gejalanya sudah tercatat di paper kalian sendiri:** Bab III.B.2 melaporkan sekitar **58% langkah
hasil pengisian terdeteksi bercampur Bahasa Inggris**. Selama ini ditulis sebagai catatan lewat;
dengan rujukan di atas, bisa naik jadi temuan yang punya dasar literatur.

## Kenapa mahal (bukan sekadar estetika)

`to_chatml.py` dijalankan `id_only=True` → CoT dominan Inggris **dibuang** (`is_indonesian()`,
baris 39-43). Artinya:

> Solusi yang **benar secara matematis tetapi berbahasa Inggris tetap terbuang.**

Jadi retensi 37,08% dan cakupan 66,61% di Tabel X adalah nilai **sebelum** filter bahasa. Yield
efektif yang benar-benar menjadi data latih lebih rendah, dan paper belum pernah melaporkan
selisihnya.

## Rujukan pendukung

### DeepSeek-R1 (sumber masalah; sudah jadi ref [4] paper kalian)
> DeepSeek-AI dkk. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement
> Learning." arXiv:2501.12948, 2025.

Paper aslinya mengakui language mixing sebagai keterbatasan yang belum tuntas: upaya menambahkan
*language consistency reward* saat RL sudah dicoba tetapi pencampuran tetap ada
`[Perlu cek — konfirmasi ke bagian Limitations aslinya]`.

### Qwen3 — kandidat pembanding multibahasa
> "Qwen3 Technical Report." arXiv:2505.09388, 2025.

`[Terverifikasi]` judul + ID. Mengklaim dukungan **119 bahasa dan dialek**, Indonesia termasuk di
benchmark multibahasanya.

### Catatan tambahan `[Perlu cek]`
Memaksa keluaran ke bahasa dengan konsistensi rendah dilaporkan bisa mendorong model ke pola
*slow-thinking* yang lebih lemah dan justru **menurunkan performa**. Artinya trade-off
"akurasi vs bahasa" itu nyata dan harus **diukur**, bukan diasumsikan.

## Dipakai di Skenario 1 sebagai apa

| metrik | rumus | kenapa |
|---|---|---|
| `format_%` | proporsi kandidat memuat `\boxed{}` | tanpa `\boxed{}` langsung dieliminasi `filter_solutions.py` |
| `indonesia_%` | proporsi kandidat lolos `is_indonesian()` | kandidat Inggris dibuang `to_chatml(id_only=True)` |
| **skor efektif** | `cakupan_% × indonesia_% / 100` | cakupan mentah menyesatkan |

## Dua hipotesis yang diuji

**H1 — spesialis matematika gagal di format/bahasa Indonesia.**
Bukti dari data kalian sendiri: Tabel XI/XII mencatat `OpenMath-Nemotron-1.5B` punya kepatuhan
format **0,147** (numglue) dan **0,219** (easy) — terburuk dari semua model yang diuji.

**H2 — R1-Distill paling parah language mixing-nya.**
Dasar: arXiv:2505.14815 + catatan 58% campur Inggris di paper kalian.

**Kalau H1 dan H2 benar:**

> Untuk distilasi CoT bahasa non-Inggris, model generalis multibahasa mengungguli spesialis
> matematika — karena akurasi matematis tidak berguna bila keluarannya terbuang di filter
> bahasa/format.

Klaim itu jauh lebih menarik daripada "kami mengganti teacher dengan yang lebih besar".
