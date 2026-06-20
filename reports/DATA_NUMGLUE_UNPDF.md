# Penambahan & Pembersihan Data: NumGLUE + un_pdfs

> Sesi kerja 2026-06-19. Mendokumentasikan keputusan + pipeline untuk (1) menambah data NumGLUE
> (Inggris → diterjemahkan + di-generate `cara`-nya) dan (2) membersihkan data di `data/un_pdfs/`.
> Schema kanonik dataset: `{soal, cara, jawaban}` (lihat `src/cot_synthesis/utils.py`).

## Keputusan (defensible)

| Topik | Keputusan | Alasan |
|---|---|---|
| Skala NumGLUE | train **10.000**, dev **2.000**, test **3.000** | Translate + CoT untuk 91rb baris penuh tidak realistis (mahal/lama, Groq free-tier). Subsample tetap representatif. |
| Metode generate `cara` | **blind-generate + verify, fallback answer-hinted** | Faithful ke AIMO-2 (generate banyak solusi → filter by correctness). Fallback answer-hinted menjamin coverage 100% ("biar ada caranya semua"). Pure answer-conditioned berisiko *post-hoc rationalization*; pure rejection-sampling bikin sebagian soal tak punya `cara`. |
| Filter "GT jelek" NumGLUE | drop type non-matematis + answer non-numerik | Lihat tabel type di bawah. |
| Field `type` | **dihapus** di output akhir (dipakai sementara utk stratified subsample) | Sesuai instruksi. |
| `filter_validity` (LLM) | **diikutkan** | Konfirmasi user. |
| GT kosong di un_pdfs | **di-generate** pakai `fill_missing.py` (LLM-as-generator, DeepSeek-R1), bukan dibuang | Instruksi user: jangan buang clean_vlm, generate GT yang tak ada. |
| Translate | NLLB-200 distilled-1.3B, **math-safe** (math di-splice verbatim, tak pernah masuk model) | Sama seperti pipeline AIMO (`openmath_translate_v2.py`). |

### Aturan filter type NumGLUE (`src/preprop/numglue_prep.py`)

| Type | Isi | Keputusan |
|---|---|---|
| Type_1/2/4/8 | jawaban numerik (aritmatika) | keep |
| Type_5 | dict `{number,date,spans}` | keep **hanya** kalau `number` terisi & `spans` kosong → answer = number |
| Type_3 | `"Option N"` | drop (multiple_choice) |
| Type_6 | span extraction | drop (non_math) |
| Type_7 | `neutral/entailment/contradiction` (NLI) | drop (non_math) |

## Temuan

- **`clean_vlm.jsonl`: 1616 dari 2558 baris (63%) `jawaban` KOSONG** → **di-generate** via
  `fill_missing.py` (DeepSeek-R1 solve → `\boxed` jadi jawaban, reasoning jadi cara), TIDAK dibuang.
  Easy merged total: 5190 baris (1620 jawaban + 1292 cara kosong perlu di-fill).
- Setelah filter NumGLUE, **Type_5 mendominasi (~92%)** karena memang proporsi aslinya begitu
  (DROP-style numeric reading-comprehension). Stratified subsample mempertahankan proporsi ini.
  Kalau ingin lebih beragam, bisa di-cap per type (belum dilakukan).
- Hasil filter NumGLUE: dev 10185→5710, train 71281→40060, test 10583→5404 (cukup untuk target subsample).

## Pipeline

### Notebook (final)

| Notebook | Platform | Cakupan |
|---|---|---|
| `notebooks/numglue_pipeline_colab.ipynb` | Colab | NumGLUE A1 (dedup+subsample) + A2 (translate, NLLB/transformers) |
| `notebooks/numglue_cot_kaggle.ipynb` | Kaggle vLLM | NumGLUE A3 (generate + correctness + assemble `cara`) |
| `notebooks/unpdf_pipeline_colab.ipynb` | Colab transformers 4-bit | un_pdfs B1 fill + B2 validity + B3 dedup (easy & aimo_hard) |

**Pembagian platform (paralel):** A3 (NumGLUE) di **Kaggle vLLM**; Track B (un_pdfs) di **Colab
transformers 4-bit** → jalan bareng tanpa rebutan kuota Kaggle. Tak pakai Groq API karena free-tier
batas **100k token/hari** (tak cukup ribuan soal). Track B pakai transformers (bukan vLLM) supaya
robust di Colab (vLLM kena CUDA mismatch); 4-bit biar muat T4, model dibebaskan antar-tahap (tanpa restart).
**NumGLUE A3 hemat 1 model:** correctness dicek `math_verify`/string (jawaban numerik), jadi
**tanpa judge LLM**; hinted fallback pakai model generate yang sama. un_pdfs tetap pakai judge
(Qwen) karena jawabannya kalimat.

### Task 1 — NumGLUE (per split: dev/train/test)

```bash
# 1. filter + normalisasi answer + drop type non-matematis (CPU, cepat) — SUDAH dijalankan
python -m src.preprop.numglue_prep --split all
#    -> data/NumGlue/numglue_{split}_filtered.jsonl  {question, answer, source, source_type}

# 2. dedup internal + dekontaminasi vs holdout (GPU ringan; sentence-transformers + faiss)
python -m src.preprop.dedup data/NumGlue/numglue_dev_filtered.jsonl data/NumGlue/numglue_dev_dedup.jsonl
#    (decontam vs data/eval/holdout_v3_un.jsonl bisa ditambah lewat benchmark_paths)

# 3. stratified subsample + drop type (CPU)
python -m src.preprop.subsample --input data/NumGlue/numglue_dev_dedup.jsonl \
    --output data/NumGlue/numglue_dev_sample.jsonl --n 2000
#    train --n 10000 ; test --n 3000
#    -> {question, answer, source}   (type sudah hilang)

# 4. translate EN->ID math-safe (GPU; Kaggle/Colab)
python -m src.scraping_ver2.translate_numglue --split dev    # train/test idem
#    -> data/NumGlue/numglue_{split}_id.jsonl  {soal, jawaban, source}

# 5a. blind CoT teacher (AIMO-2 style): n kandidat per soal
python -m src.cot_synthesis.generate data/NumGlue/numglue_dev_id.jsonl \
    --out data/cot/numglue_dev_cand.jsonl --backend api -n 4
# 5b. verifikasi correctness (LLM judge benar/salah)
python -m src.cot_synthesis.filter_solutions data/cot/numglue_dev_cand.jsonl \
    data/cot/numglue_dev_correct.jsonl
# 5c. rakit cara: verified + fallback answer-hinted (coverage 100%)
python -m src.cot_synthesis.assemble_cara \
    --problems data/NumGlue/numglue_dev_id.jsonl \
    --correct  data/cot/numglue_dev_correct.jsonl \
    --output   data/NumGlue/numglue_dev_final.jsonl
#    -> {soal, cara, jawaban, source, cara_source}
```

### Task 2 — un_pdfs

> **Colab: langkah GPU (B1 fill → B2 validity → B3 dedup) untuk `easy` DAN `aimo_hard` ada di
> `notebooks/unpdf_pipeline_colab.ipynb`** (transformers 4-bit; tanpa restart antar-model).
> Langkah CPU (merge + filter_rules) sudah dijalankan. AIMO (`part_00000`) = `aimo_hard`, ikut pipeline
> yang sama (fill nyaris no-op, tetap kena validity+dedup).

```bash
# A. dataset_sma + clean_vlm = soal full-Indo MUDAH -> merge + dedup-by-soal (PERTAHANKAN GT kosong)
python -m src.preprop.merge_and_dedup \
    --inputs data/un_pdfs/dataset_sma.jsonl data/un_pdfs/clean_vlm.jsonl \
    --output data/un_pdfs/easy_merged.jsonl
#    SUDAH dijalankan: 5190 unik (1620 jawaban + 1292 cara kosong)

# B. part_00000_05000 = soal AIMO SUSAH -> merge + dedup-by-soal
python -m src.preprop.merge_and_dedup \
    --inputs data/un_pdfs/part_00000_05000.jsonl \
    --output data/un_pdfs/aimo_hard.jsonl
#    SUDAH dijalankan: 5000 -> 4929 unik (3 jawaban kosong)

# C. filter_rules (CPU) -- duluan, supaya fill tidak buang compute ke soal yang kena rule
python -m src.preprop.filter_rules data/un_pdfs/easy_merged.jsonl data/un_pdfs/easy_after_rules.jsonl
python -m src.preprop.filter_rules data/un_pdfs/aimo_hard.jsonl   data/un_pdfs/aimo_hard_after_rules.jsonl
#    SUDAH dijalankan: easy 5190->3417 (MC 1045, English 637, needs_image 87, T/F 4)
#                      aimo 4929->4872 (not_id 38, T/F 14, MC 5)

# D. GENERATE GT/cara yang kosong (LLM-as-generator, GPU/vLLM) -- BUKAN dibuang
python -m src.preprop.fill_missing \
    --input data/un_pdfs/easy_after_rules.jsonl \
    --cache data/un_pdfs/easy_fill_cache.jsonl \
    --output data/un_pdfs/easy_filled.jsonl
#    aimo_hard_after_rules.jsonl idem (kalau perlu isi GT kosong)

# E. preproc lanjutan (sama seperti NumGLUE):
python -m src.preprop.filter_validity data/un_pdfs/easy_filled.jsonl data/un_pdfs/easy_after_valid.jsonl  # GPU/vLLM
python -m src.preprop.dedup           data/un_pdfs/easy_after_valid.jsonl data/un_pdfs/easy_clean.jsonl   # embedding dedup
```

**Catatan filter_rules:** deteksi bahasa & MC diperbaiki agar tahan LaTeX — sebelumnya soal
Indonesia padat rumus salah dibuang sebagai `not_indonesian`, dan MC dengan opsi dipisah `\\`
(satu baris) lolos deteksi. Sekarang: strip LaTeX dulu, shortcut token ID, buang hanya jika
langdetect yakin bahasa lain (prob>0.85); MC menormalkan `\\` jadi newline.

## File yang ditamb/diubah

- **baru**: `src/preprop/numglue_prep.py`, `src/preprop/subsample.py`,
  `src/scraping_ver2/translate_core.py`, `src/scraping_ver2/translate_numglue.py`,
  `src/cot_synthesis/assemble_cara.py`
- **diubah**: `src/preprop/dedup.py` & `src/preprop/filter_rules.py` (dukung field `soal`
  selain `question`), `src/preprop/merge_and_dedup.py` (flag `--drop-empty-jawaban`)

## Catatan eksekusi

- Langkah 1–3 (Task 1) dan A–B (Task 2) **CPU**, sudah dijalankan/teruji lokal.
- Langkah 4–5 (translate, CoT) dan filter_validity butuh **GPU/API** → jalankan di Kaggle/Colab.
- Translate, generate, filter_solutions, assemble_cara semuanya **resumable** (checkpoint per blok/baris).
- **Dekontaminasi (vs holdout) dimatikan** (`DECONTAM=False` di A1 & B3) — holdout final dibuat
  BARU dari pool bersih ini nanti, jadi dekontaminasi vs holdout lama percuma. **Dedup internal
  tetap jalan** (buang near-duplicate; penting supaya saat holdout di-carve nanti tak ada kembaran
  yang bocor ke train). Workflow bebas-bocor: finalize + dedup internal → sample holdout dari pool
  → train = pool − holdout (disjoint by construction).
