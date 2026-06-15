# Laporan Progres — FP NLP: Math Reasoning Bahasa Indonesia

Replikasi-mini paper **AIMO-2 / OpenMathReasoning** (arXiv:2504.16891) untuk melatih
student model kecil (Qwen2.5-0.5B/1.5B) menyelesaikan soal matematika **Bahasa Indonesia**.

> **Konvensi laporan ini:** tiap tahap dicatat dengan format **Apa → Kenapa → File**.
> Prinsip integritas: angka hasil dinaikkan HANYA lewat (1) grading yang adil,
> (2) pembersihan data rusak, (3) training on-domain — **bukan** dengan membuang soal
> sulit agar terlihat bagus.

---

## Ringkasan alur

```
2 file mentah ─► dedup ─► isi-kosong(LLM) ─► rule-filter ─► split
  35.380         24.223     24.223            19.080         ├─ holdout  300  (eval)
                                                             └─ train    18.780 ─► SFT 18.297
```

| Tahap | Output | Jumlah |
|---|---|---|
| Gabung 2 file mentah | `merged` | 25.190 |
| Dedup by-soal | `dataset_dedup.jsonl` | 24.223 |
| Isi jawaban/cara kosong (LLM) | `dataset_filled.jsonl` | 24.223 (95,7% berjawaban) |
| Rule-filter (buang MC/TF/gambar/non-Indo) | `filtered/after_rules.jsonl` | 19.080 |
| Split eval | `eval/holdout.jsonl` | 300 |
| Split train | `train_pool.jsonl` → `train_sft.jsonl` | 18.780 → 18.297 |

---

## Tahap 1 — Gabung + dedup

**Apa:** Menggabungkan `merged_dataset.jsonl` (25.190) + `merged_dataset_un_cleanvlm_aimo5000.jsonl` (10.190), lalu buang duplikat berdasarkan `soal`.

**Kenapa:** Cek menunjukkan file kedua **100% subset** file pertama (0 soal baru), jadi gabung-mentah malah munculin 10.190 duplikat. Selain itu di file pertama ada 967 soal kembar internal. Dedup **by-soal** (bukan exact) dipilih supaya satu soal = satu baris; saat ada kembaran, disimpan versi **paling lengkap** (prioritas: ada jawaban > ada cara > cara terpanjang) agar tidak membuang baris yang justru terisi.

**File:** `src/preprop/merge_and_dedup.py` → menghasilkan `data/dataset_dedup.jsonl`.

## Tahap 2 — Isi jawaban/cara kosong pakai LLM

**Apa:** 1.765 soal punya `jawaban`/`cara` kosong. Diisi dengan **DeepSeek-R1-Distill-Qwen-7B** (sesuai arahan dosen: LLM-as-generator).

**Kenapa & aturan integritas:** Hasil LLM **hanya untuk train_pool**, **bukan** untuk kunci holdout — kalau kunci eval hasil tebakan LLM, angka benchmark tidak bisa dipercaya. Soal yang sudah punya jawaban → jawaban asli dipertahankan, LLM hanya mengisi langkah. Kebijakan akhir: **"pakai yang dapat saja"** — hanya fill yang menghasilkan jawaban yang dipakai (705 baris), sisanya dibiarkan. Hasil: **95,7% berjawaban**.

**Catatan:** ~58% `cara` hasil DeepSeek tercampur Inggris (model reasoning) — perlu di-spot-check; sebagian besar yang panjang nanti dibuang di Tahap 5.

**File:** `notebooks/fill_missing_kaggle.ipynb` (jalan di Kaggle, transformers) + `src/preprop/apply_fill_cache.py` (gabung cache → `dataset_filled.jsonl`). Catatan teknis: vllm gagal di Kaggle (flashinfer/-lcuda) → pindah ke `transformers`.

## Tahap 3 — Rule-based filter (preprocessing pipeline)

**Apa:** Buang soal pilihan-ganda, true/false, butuh-gambar, non-Indonesia; normalisasi LaTeX. 24.223 → **19.080 lolos**.

**Kenapa:** Tahap deterministik & murah dari pipeline paper (sebelum validity-check/embedding yang butuh GPU). Yang dibuang: 4.261 non-Indonesia (langdetect, sebagian soal LaTeX-berat ke-flag — ini jadi knob Skenario 1), 449 pilihan-ganda, 319 kependekan, 110 butuh-gambar, 4 true/false.

**File:** `src/preprop/to_csv.py` → `data/filtered/after_rules.jsonl`.

## Tahap 4 — Split holdout (eval) + train_pool

**Apa:** Pisah 300 soal **holdout** untuk evaluasi + sisanya (18.780) untuk training.

**Kenapa 300, bukan 70/30:** Holdout = **benchmark evaluasi** (seperti GSM8K/MATH), bukan test-split rasio. (1) Eval LLM mahal (generate reasoning panjang per soal); 30% = ~7.000 soal × banyak model = puluhan jam, tak terkejar Kaggle gratis. (2) Benchmark math standar berukuran tetap ratusan–ribuan, bukan rasio (GSM8K test=1.319, AIME=30). (3) Data training jangan disia-siakan. (4) 300 sampel → margin error akurasi ~±3%, cukup untuk membandingkan model. Holdout diambil **hanya dari jawaban bersih** (integer/angka/ekspresi tunggal) supaya grading otomatis reliable; jawaban kalimat/kosong tetap masuk train_pool.

**File:** `src/eval/make_holdout.py` → `data/eval/holdout.jsonl` + `data/train_pool.jsonl`.

## Tahap 5 — Bangun data SFT non-CoT

**Apa:** Format chat untuk training. Target = `cara` (kalau ≤3000 char) + `\boxed{jawaban}`. 18.297 baris.

**Kenapa cap 3000 char:** Sebaran panjang `cara` median 1.931 (solusi UN/OSN asli), tapi ekor 5.000–32.000 char = reasoning DeepSeek raksasa. Cap di **3.000** (p90≈2.974) menjaga solusi natural masuk (89%) tapi membuang reasoning monster — supaya tetap "non-CoT", bukan kebawa CoT panjang.

**File:** `src/training/build_sft.py` → `data/train_sft.jsonl`.

---

## Tahap 6 — Eval harness (cek jawaban)

**Apa:** Logika ekstraksi `\boxed{}` + pencocokan jawaban berlapis (exact → numerik → simbolik sympy).

**Kenapa berlapis:** Jawaban bisa beda bentuk tapi sama nilai (`0.5`=`1/2`, `2x`=`x*2`, `(n+1)(n+2)`=`n²+3n+2`). Pencocokan string saja akan salah-vonis banyak jawaban benar.

**File:** `src/eval/answer_check.py` (+ `src/eval/test_answer_check.py`, 15 kasus uji lolos).

## Tahap 7 — Skenario 4: model lain vs data kita (zero-shot)

**Apa:** Eval OpenMath-Nemotron-1.5B (dari paper), SeaLLMs-v3-1.5B, Qwen2.5-1.5B-Instruct di holdout.

**Hasil awal (run pertama) sangat rendah** dan setelah diselidiki ternyata **didominasi artefak**, bukan kemampuan. Tiga masalah ditemukan & diperbaiki:

1. **Truncation** — `format_ok` OpenMath cuma 8% (model reasoning, `max_new_tokens=2048` kekecilan, jawaban kepotong sebelum `\boxed`). **Fix:** naikkan ke 4096 + re-run OpenMath saja.
2. **Grading meleset** — gold `\(\frac{1}{(n+1)(n+2)}\)` vs pred `\dfrac{1}{n^2+3n+2}` adalah SAMA tapi divonis salah. **Fix:** `answer_check` ditambah strip `\(\)\[\]`, `\dfrac→\frac`, prefix `var=`.
3. **Format tidak dipatuhi** — banyak model (SeaLLMs) menjawab **tanpa `\boxed`** → skor 0 padahal menjawab. **Fix:** `grade()` diberi **fallback** ambil angka terakhir kalau tak ada `\boxed`. (SeaLLMs 0% → 8,2%.)

**Hasil setelah grading adil** (subset 147 jawaban integer bersih):

| Model | acc (int-147) | Catatan |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 8,2% | valid |
| SeaLLMs-v3-1.5B | 8,2% | valid (dengan fallback) |
| OpenMath-Nemotron-1.5B | 6,8% | under-estimate, sedang di-rerun 4096 token |

**Temuan kritis (penting untuk metodologi):** Pemeriksaan manual 14 kasus salah menunjukkan holdout **brutal + berisik**: (a) banyak soal olympiad keras beneran (integral, persamaan fungsional, complex analysis), (b) banyak soal **hasil translate mesin yang rusak** ("Biarkan f Dan g menjadi analitis"), (c) sebagian gold **sampah** (`'10 ^\ sekitar'`, `'f ((x) = kx'`). Akibatnya semua model mentok ~8% (lantai) → **daya pembeda rendah**: susah menunjukkan student lebih baik nanti.

**File:** `src/eval/skenario4_eval.py` (modul), `notebooks/skenario-4.ipynb` (Kaggle), output `data/output/skenario4_results_regraded.json` + `gen_*.json`.

## Tahap 8 — Holdout v2 (pembersihan, BUKAN mempermudah)

**Apa:** Buang dari holdout v1 HANYA yang objektif rusak: 14 gold tak-gradeable (sampah seperti `'10 ^ sekitar'`) + 1 LaTeX patah → **285 soal**. Soal sulit **dipertahankan** (justru itu yang diuji). Ditambah tag `hard` untuk pelaporan per-strata.

**Kenapa (integritas):** Membuang soal yang **mustahil dinilai** (gold bukan angka & tak terparse) atau **tak terbaca** (LaTeX tak seimbang) itu data-quality yang sah. Membuang soal sulit TIDAK dilakukan — tujuan benchmark justru menguji apakah model bisa soal sulit.

**Temuan jujur:** Cleaning hanya membuang **15/300** → skor hampir tak berubah (Qwen 7,4%, OpenMath 6,7%, SeaLLM 4,9%). **Kesimpulan: skor ~7–8% itu NYATA** — soal memang sulit (didominasi olympiad AIMO-translate), bukan artefak data. Tagging `hard` keyword lemah (hanya 34 ter-deteksi; banyak soal sulit lain tak ter-tag). Implikasi: kontribusi "hasil bagus" harus bersifat **relatif** (student on-domain vs baseline) atau **per-sumber** (UN/sekolah vs AIMO/olympiad), bukan akurasi absolut tinggi.

**File:** `src/eval/clean_holdout.py` → `data/eval/holdout_v2.jsonl`.

## Tahap 9 — Tag sumber (UN vs AIMO) + diagnosis akar skor rendah

**Apa:** File sumber asli ditemukan (`data/un_pdfs/`: `dataset_sma.jsonl`+`clean_vlm.jsonl`=UN 5.190;
`part_*.jsonl`=AIMO 19.033). Holdout di-tag per provenance: **255 AIMO + 39 UN + 6 unknown**.

**Hasil per-sumber (semua model):** UN ~5–9%, AIMO ~5–7% — **tidak diskriminatif**.

**Diagnosis akar (penting):** Skor rendah NYATA & punya 2 sebab berbeda:
1. **AIMO** = soal kompetisi (AoPS/olympiad) ditranslate kasar → memang sangat sulit untuk model 1.5B.
2. **UN** = soal sekolah (modus, peluru, bunga) TAPI **kualitas data buruk**: 724/3.570 UN = pilihan
   ganda yang opsinya hilang saat ekstraksi (gold tinggal huruf `D`/`B`), banyak butuh tabel/gambar
   yang tak ter-ekstrak ("tabel di atas"), satuan/teks di gold. Jadi UN tampak sesulit AIMO padahal
   karena format jawaban & data hilang, bukan kesulitan murni.

**Kesimpulan:** Dataset mentah tidak bisa memberi angka baseline tinggi secara jujur. Verifikasi
manual mengonfirmasi eval & grading SUDAH benar (soal bersih yang terjawab ter-grade benar).

**Implikasi keputusan:** angka "bagus" yang defensible harus dari (a) **subset UN bersih** (soal
sekolah numerik & self-contained — in-domain, sah) sebagai benchmark "level sekolah", dan/atau
(b) **peningkatan relatif** student vs baseline. Bukan dari memaksa angka absolut tinggi.

**File:** tagging via `src/preprop/tag_source.py` (pakai file di `data/un_pdfs/`).

---

## Inventaris file

### `src/preprop/` — pembersihan data
| File | Fungsi |
|---|---|
| `merge_dataset.py` | gabung sumber awal (script lama) |
| `merge_and_dedup.py` | **gabung 2 file + dedup by-soal** (Tahap 1) |
| `fill_missing.py` | isi kosong via vllm (deprecated di Kaggle) |
| `apply_fill_cache.py` | **terapkan hasil fill ke dataset** (Tahap 2) |
| `to_csv.py` | **rule-filter + normalisasi LaTeX** (Tahap 3) |
| `filter_rules.py` | aturan filter (dipakai to_csv) |
| `filter_validity.py` | validity-check via Qwen-7B (GPU, belum dijalankan) |
| `dedup.py` | dedup embedding + dekontaminasi (GPU, belum) |
| `extract.py`, `extract_questions.py` | ekstraksi soal dari PDF (tahap awal) |

### `src/eval/` — evaluasi
| File | Fungsi |
|---|---|
| `answer_check.py` | **ekstraksi `\boxed` + cek ekuivalensi jawaban** (inti grading) |
| `make_holdout.py` | **split holdout/train** (Tahap 4) |
| `skenario4_eval.py` | **modul eval Skenario 4** |
| `clean_holdout.py` | **buat holdout v2** (buang gold-sampah/LaTeX-patah) |
| `test_answer_check.py` | unit test grading (15 kasus) |

### `src/training/` — pelatihan
| File | Fungsi |
|---|---|
| `build_sft.py` | **bangun data SFT non-CoT** (Tahap 5) |

### `notebooks/` — dijalankan di Kaggle (self-contained, transformers)
| File | Fungsi | Upload |
|---|---|---|
| `fill_missing_kaggle.ipynb` | isi jawaban kosong (DeepSeek-7B) | `dataset_dedup.jsonl` |
| `skenario-4.ipynb` | eval model lain | `holdout.jsonl` |
| `skenario3_train_noncot_kaggle.ipynb` | training QLoRA 0.5B+1.5B | `train_sft.jsonl` |

### `data/`
| File | Isi |
|---|---|
| `merged_dataset.jsonl` | mentah gabungan (25.190) |
| `dataset_dedup.jsonl` | setelah dedup (24.223) |
| `dataset_filled.jsonl` | setelah isi-kosong (95,7% berjawaban) |
| `filtered/after_rules.jsonl` | setelah rule-filter (19.080) |
| `eval/holdout.jsonl` | benchmark eval v1 (300) |
| `eval/holdout_v2.jsonl` | benchmark eval bersih (285, gold-sampah dibuang) |
| `train_pool.jsonl` / `train_sft.jsonl` | data training (18.780 / 18.297) |
| `output/*.json` | hasil + generasi mentah Skenario 4 |

---

## Keputusan terbuka & langkah berikutnya

1. **Re-run OpenMath** (4096 token) — sedang dikerjakan, untuk angka fair model paper.
2. **Holdout v2** ✅ dibuat (285, buang 15 sampah). Temuan: skor rendah NYATA (soal sulit), bukan
   artefak. **Lever sebenarnya untuk benchmark diskriminatif = pisah sumber** (UN/sekolah vs
   AIMO/olympiad) — tapi tag sumber hilang saat merge. Perlu keputusan: (a) recover sumber dari file
   asli di Drive (`un_pdfs/*`, `hugging_face_AIMO/*`) lalu re-merge dengan tag, atau (b) terima &
   laporkan kontribusi secara relatif (student vs baseline).
3. **Training non-CoT** (Skenario 3) — notebook siap, menghasilkan "model kita".
4. **Skenario 1** (variasi preprocessing) & **Skenario 2** (CoT, dikerjakan rekan).
