# Rencana Build Notebook — 2026-08-08

---

## BACA INI DULU SEBELUM NGODING APA PUN

**JANGAN langsung bikin notebook. Lakukan urutan ini dulu.**

### Langkah 1 — Baca ulang (wajib, jangan dilewat)

| File | Kenapa harus dibaca |
|---|---|
| `docs/references/README.md` | indeks + aturan sitasi |
| `docs/references/01_multi_teacher_distillation.md` | desain Skenario 6, apa yang wajib disitir |
| `docs/references/02_preference_optimization.md` | desain Skenario 7, kenapa KTO dipilih |
| `docs/references/03_language_mixing.md` | alasan sumbu bahasa di Skenario 1 |
| `docs/references/04_teacher_models.md` | kandidat model + status verifikasi repo ID |
| `docs/references/05_positioning.md` | klaim mana yang boleh, mana yang wajib sitir |
| `docs/superpowers/specs/2026-08-07-novelty-brainstorm-notes.md` | temuan bug data + rencana judge |
| `reports/Final Project Kelompok 5-2.pdf` | paper versi sekarang — tabel mana saja yang berubah |
| `2504.16891v1.pdf` (akar repo) | paper AIMO-2, resep yang sedang direplikasi |

> **Koreksi 2026-08-08:** baris terakhir tabel ini dulu menyebut `KrocohMasStanis.pdf`.
> **File itu tidak pernah ada di repo.** Hanya dua PDF di atas yang ada
> (`find . -iname "*.pdf"` — dua hasil, keduanya sudah tercantum).

### Langkah 2 — Brainstorm dulu, jangan langsung eksekusi

Sebelum menulis sel notebook pertama, jawab dulu bersama user:

1. **Sisa waktu berapa hari?** Menentukan berhenti di Fase 5 atau lanjut Fase 6-7.
2. **Data sudah benar-benar beku?** Kalau `easy_clean_v3` / `numglue_clean_v3` masih mungkin
   berubah, JANGAN mulai Fase 2 — semua hilirnya batal.
3. **Tahap B LLM judge dijalankan atau dilewat?** Kalau dilewat: hemat ~1 jam, tapi kehilangan aset
   kalibrasi untuk paper.
4. **Berapa kandidat teacher?** 3 (tier 14B saja) atau 5 (plus tier 7B).
5. ~~**Repo ID berstatus `[Perlu cek]` sudah diverifikasi?**~~ **SELESAI 2026-08-08.** Semua repo ID
   di `04_teacher_models.md` sudah dicek lewat HF API, semuanya HTTP 200, kolom lisensi sudah
   ditambahkan. Yang tersisa untuk dijawab: **lisensi mana yang mengikat.** OpenMath-Nemotron
   `cc-by-4.0` (wajib atribusi kalau menang bake-off), Qwen2.5-3B `qwen-research` (non-komersial).

**Kalau ada yang belum terjawab, tanya user. Jangan menebak.**

### Langkah 3 — Baru bangun, satu fase per satu

Jangan bikin semua notebook sekaligus. Bangun per fase, jalankan, verifikasi hasilnya, baru lanjut.

---

## Status saat ini

### Sudah selesai

| Item | Detail |
|---|---|
| Data v3 | `easy_clean_v3.jsonl` 2.254 + `numglue_clean_v3.jsonl` 3.762 = **6.016**, kolom `cara` sudah dibuang |
| `src/data_validation/judge_quality.py` | Tahap A jalan (buang 82 dari easy, 69 dari numglue). Tahap B belum dijalankan |
| `notebooks/cot/cot_pipeline_a6000.ipynb` | 12 sel, dry-run logika di CPU lolos. **Belum diuji di GPU** |
| `src/cot_synthesis/generate.py` | tambah param `dtype` (default `float16`; A6000 pakai `bfloat16`). 7 test lolos |
| `src/training/configs/cot_3b.yaml` | student `Qwen/Qwen2.5-3B` |
| `docs/references/*` | 6 file rujukan riset |

### Belum dibuat

| File | Isi | Risiko |
|---|---|---|
| `src/training/configs/nocot_3b.yaml` | pasangan `cot_3b.yaml`; beda hanya `dataset`, `output_dir`, `mode` | nol |
| `src/cot_synthesis/merge_teachers.py` | union `correct.jsonl` lintas teacher (concat + dedup) | rendah |
| `src/training/build_preference.py` | `candidates` + `correct` → `dpo.jsonl` + `kto.jsonl` | rendah, murni CPU |
| `src/training/train_pref.py` | TRL `DPOTrainer`/`KTOTrainer` + QLoRA, pola niru `train_sft.py` | **tertinggi** |

---

## Keputusan yang sudah dikunci

| Hal | Keputusan |
|---|---|
| Student | `Qwen/Qwen2.5-3B` **base** (bukan Instruct, bukan 7B, bukan Qwen3) |
| Dataset | `easy_clean` + `numglue_clean` saja. **`aimo_hard` TIDAK dipakai** |
| Kolom `cara` | dibuang — seluruh CoT dari teacher |
| GPU | A6000 48 GB single, `bfloat16`, `tensor_parallel_size=1` |
| U0 | **tetap dipakai** sebagai baseline Skenario 6 (= arm A / SFT-CoT) |

---

## Lengan training (5 run, semua QLoRA 3B, hyperparameter identik)

| # | Arm | Data | Skenario |
|---|---|---|---|
| A | SFT-CoT | CoT teacher pemenang | S3 utama, basis D & E, **= U0** |
| B | SFT-nonCoT | `\boxed{}` saja, soal identik | S3 kontrol |
| C | SFT-CoT-union | union 2 teacher terbaik | S6 (= U1) |
| D | SFT-CoT + DPO | pasangan chosen/rejected | S7 |
| E | SFT-CoT + KTO | label biner tak-berpasangan | S7 |

D dan E **bertumpuk di atas A**, bukan training dari nol.

---

## Tujuh skenario

| # | Skenario | Biaya | Baru? |
|---|---|---|---|
| S1 | Bake-off teacher + sumbu `format%` & `indonesia%` | 1-2 jam | metrik baru |
| S2 | Generalisasi lintas subset | **gratis** — belah tabel S3/S4 per subset | — |
| S3 | CoT vs non-CoT terkontrol | 2× train | — |
| S4 | vs baseline publik (+ `Qwen2.5-3B-Instruct`) | 1-2 jam | baseline direvisi |
| S5 | MATH-IDN eksternal | ~30 menit | — |
| S6 | U0 single vs U1 union-2 (+ U2 metrik data saja) | ~gratis + 1 train | **novelty** |
| S7 | SFT vs +DPO vs +KTO | data gratis + 2 train | **novelty** |

---

## Urutan eksekusi

```
Fase 1  bekukan data           (~1 jam)   <- GATE: setelah ini data TIDAK BOLEH berubah
Fase 2  S1 bake-off teacher    (1-2 jam)
Fase 3  sintesis CoT penuh     (4-8 jam)  <- BOTTLENECK
Fase 4  train A + B            (2-4 jam)  -> S3
Fase 5  eval S2/S3/S4/S5       (2-4 jam)
=========================================  <- PAPER SUDAH UTUH DI SINI
Fase 6  S6 union -> train C     (~2 jam)
Fase 7  S7 DPO+KTO -> train D,E (~3 jam)
```

**Garis batas itu penting.** S1-S5 = paper lengkap dan bisa disubmit. S6-S7 = nilai tambah.
Kalau DPO/KTO gagal konvergen di hari terakhir, paper tetap selamat. Kalau dikerjakan duluan lalu
gagal, dua-duanya hilang.

Semua estimasi waktu **belum diukur** — perkiraan kasar.

---

## Detail per fase

### Fase 1 — bekukan data
1. Kalibrasi judge di 260 baris (seed=7) → bandingkan dengan anotasi manual
2. Tahap B judge penuh → `*_v4.jsonl` (kalau diputuskan jalan)
3. Split holdout 300/subset + dekontaminasi

Dry-run CPU sudah memberi angka: holdout **600**, train_pool **5.416**
(numglue 3.462 + easy 1.954), dekontaminasi buang **0**.

### Fase 2 — S1 bake-off
Sudah ada di `notebooks/cot/cot_pipeline_a6000.ipynb` sel 4-7.
Metrik: retensi, cakupan, `format_%`, `indonesia_%`, skor efektif = cakupan × indonesia / 100.

### Fase 3 — sintesis CoT penuh
5.416 soal × 8 sampel ≈ 43k generasi → rejection sampling → ChatML.
**Kalau waktu mepet: `n=8` → `n=4`**, waktu separuh, cakupan turun sedikit.

Output ke `data/sft/train/{cot,nocot}.jsonl` — path ini yang dibaca `src/training/configs/*.yaml`.

### Fase 4 — train A + B
`train_sft.py --config src/training/configs/cot_3b.yaml` dan `nocot_3b.yaml`.
Hyperparameter **wajib identik** antar keduanya — itu yang mengisolasi efek CoT.

### Fase 5 — evaluasi
- S3: holdout 600 × 5 sampel, temp 0.7, top-p 0.95
- S4: zero-shot; tambahkan `Qwen2.5-3B-Instruct` sebagai baseline utama
- S5: MATH-IDN 500 soal, greedy, 1 sampel, system prompt Indonesia
- S2: gratis, belah tabel S3/S4 per subset

### Fase 6 — S6 union
1. `merge_teachers.py`: concat `correct_*.jsonl` 2 teacher terbaik + dedup
2. `to_chatml(best_per_problem=True)` otomatis pilih terbaik lintas teacher (Indonesia dulu)
3. Train arm C, bandingkan dengan A
4. U2 (union semua): **hanya laporkan metrik data**, jangan dilatih

### Fase 7 — S7 DPO + KTO
1. `build_preference.py` → `dpo.jsonl` + `kto.jsonl` (nol generasi baru)
2. Train D (DPO) dan E (KTO) di atas checkpoint A
3. KTO: rasio data ≈ 1 : 1,7 → titik awal `λ_D ≈ 1.7`, `λ_U = 1.0`

---

## Unsloth — catatan penting untuk semua tahap training

`src/training/train_sft.py` **sudah memakai Unsloth** (`FastLanguageModel` + trl `SFTTrainer`).
Jangan diganti; tapi ada beberapa jebakan yang harus diingat.

### Aturan yang tidak boleh dilanggar

1. **`import unsloth` HARUS paling pertama**, sebelum `transformers`/`peft`/`trl`. Unsloth
   melakukan monkeypatch saat import; kalau kebalik, patch-nya tidak kena dan jalur cepatnya mati.
   Di `train_sft.py:166` urutannya sudah benar — jangan diubah.
2. **Unsloth hanya memakai SATU GPU.** Di A6000 single GPU ini justru tidak masalah (dulu di
   Kaggle 2×T4 hanya 1 yang terpakai).
3. **bf16 di A6000**: `is_bfloat16_supported()` akan mengembalikan `True` (Ampere sm_86), jadi
   `SFTConfig` otomatis pakai bf16. Ini yang kita mau — jangan dipaksa fp16.
4. **Qwen2.5 base tanpa `chat_template`**: sudah ditangani `train_sft.py:183-188` lewat
   `get_chat_template(tok, chat_template="qwen-2.5")`. Kalau ganti keluarga model, cek ulang bagian
   ini.
5. **`train_on_responses_only`** dari `unsloth.chat_templates` dipakai untuk masking loss
   (`train_sft.py:219-224`). Ini yang bikin loss hanya dihitung di token asisten — jangan dimatikan,
   itu bagian dari desain eksperimen.

### Konflik environment: Unsloth vs vLLM

vLLM dan Unsloth sama-sama rewel soal versi `torch`/`transformers`. Selama ini aman karena keduanya
**dipakai di notebook terpisah**:

| Tahap | Framework |
|---|---|
| Fase 1-3 (judge, bake-off, sintesis CoT) | vLLM |
| Fase 4, 6, 7 (training) | Unsloth |

**Saran: jangan install keduanya di environment yang sama** kalau bisa dihindari. Kalau terpaksa
satu environment, install vLLM dulu baru Unsloth, dan verifikasi `torch.__version__` tidak berubah
setelahnya.

### Unsloth untuk DPO dan KTO (Fase 7) — RISIKO

- **DPO**: Unsloth menyediakan `PatchDPOTrainer()` yang harus dipanggil **sebelum** membuat
  `DPOTrainer`. `[Perlu cek]` — nama fungsi dan cara pakainya belum diverifikasi ke dokumentasi
  Unsloth versi yang terpasang.
- **KTO**: **belum diverifikasi apakah Unsloth punya patch untuk `KTOTrainer`.** Kalau ternyata
  tidak ada, ada dua jalan keluar:
  1. jalankan KTO dengan `peft` + `trl` biasa tanpa Unsloth (lebih lambat, tapi 3B di A6000 masih
     sangat muat), atau
  2. pakai Unsloth hanya untuk memuat model 4-bit, lalu serahkan `KTOTrainer` ke trl polos

**Cek ini DULUAN sebelum Fase 7 dimulai**, jangan pas lagi training. Ini bagian paling rawan dari
seluruh rencana.

### Reproduksibilitas

Catat versi `unsloth`, `trl`, `transformers`, `torch` di log training. Paper melaporkan
hyperparameter (Tabel VIII) tapi belum melaporkan versi framework — reviewer bisa menanyakannya,
dan Unsloth cukup sering berubah perilakunya antar versi.

---

## Jebakan `.gitignore` — file data bisa hilang diam-diam

`.gitignore:9` mengabaikan `data/*`. Untuk `data/Final/` polanya lebih halus:

```
!data/Final/          # .gitignore:16 — foldernya di-un-ignore
data/Final/*          # .gitignore:17 — TAPI seluruh isinya diabaikan lagi
!data/Final/*_v3.jsonl          # .gitignore:18 — hanya v3 yang lolos
!data/Final/*_v3_dropped.jsonl  # :19
!data/Final/*_v3_review.jsonl   # :20
!data/Final/*_v3_report.json    # :21
```

Artinya **hanya berkas ber-infiks `_v3` yang benar-benar ter-track**. Semua nama lain di
`data/Final/` tidak ikut commit dan lenyap begitu direktori kerja bersih.

**Ini sudah pernah menggigit:** `easy_clean_v2.jsonl` hilang karena pola ini, dan kalibrasi judge
sempat terputus karena file acuannya tidak ada lagi.

> ⚠️ **Begitu v4 dibuat (Fase 1, output Tahap B judge), tambahkan dulu ke `.gitignore`:**
> ```
> !data/Final/*_v4*
> ```
> Kalau langkah ini dilewat, `easy_clean_v4.jsonl` dan `numglue_clean_v4.jsonl` akan hilang dengan
> cara yang persis sama — dan seluruh hilir Fase 2-7 kehilangan input.

---

## Kolom `cara` sudah dibuang di v3 — plafon evaluasi harus diambil dari file asli

v3 membuang kolom `cara` (`src/data_validation/judge_quality.py:256` menyaring kunci `cara` dari
setiap baris yang disimpan). Konsekuensinya dua metrik **tidak bisa lagi dihitung dari v3**:

| metrik | kode | butuh |
|---|---|---|
| `self_consistency` — "cara dataset == gold", dipakai sebagai **plafon akurasi** | `src/eval/data_quality_report.py:35,63` | kolom `cara` |
| cek `cara_vs_gold_mismatch` — deteksi *label noise* | `src/preprop/clean_testset.py:56` | kolom `cara` |

**Angka plafon evaluasi wajib diambil dari `easy_clean.jsonl` / `numglue_clean.jsonl` yang asli**,
bukan dari v3/v4. Menjalankan `data_quality_report.py` di atas v3 akan menghasilkan
`self_consistency` yang menyesatkan, bukan error yang kelihatan.

> Catatan jalur: issue #12 menyebut `data_quality_report.py` dan `clean_testset.py:53`. Jalur
> sebenarnya di repo adalah `src/eval/data_quality_report.py` dan `src/preprop/clean_testset.py`,
> dan barisnya **56**, bukan 53. Angka di tabel atas sudah dikoreksi.

---

## Utang paper yang sudah diketahui (`reports/LAPORAN_AKHIR.md`)

Diverifikasi terhadap kode pada 2026-08-08. Semua ini **belum diperbaiki** — dicatat supaya tidak
ditemukan ulang dari nol.

| lokasi | isi dokumen | isi kode | catatan |
|---|---|---|---|
| §3.4 butir 2 | *n*=8 kandidat/soal | `N_CANDIDATES = 4` di **ketiga** notebook Kaggle yang menghasilkan angka paper (`cot_pipeline_kaggle.ipynb`, `_gemma`, `_qwenmath`) | `src/cot_synthesis/generate.py:215` memang *default* 8, dan `notebooks/cot/cot_pipeline_a6000.ipynb` sel 8 memakai `n=8` — **tapi notebook itu belum pernah jalan di GPU**. Angka yang sudah ada di paper berasal dari run N=4 |
| §3.8 | pass@1, **pass@4, maj@4** | `n_samples=5, ks_pass=(1,2,3), ks_maj=(3,5)` — `src/eval/sample_eval.py:52,119` dan ketiga notebook skenario | Tidak ada satu pun `k=4` di kode. Yang benar: pass@1/2/3 dan maj@3/5 |
| §3.7 | 4 skenario | rencana sekarang **7 skenario** (S1-S7, lihat tabel di atas) | Daftar di paper ketinggalan; S5-S7 belum tercantum sama sekali |
| §3.6 | model dasar Qwen2.5-**0.5B/1.5B** | keputusan terkunci: `Qwen/Qwen2.5-3B` base (`src/training/configs/cot_3b.yaml`) | Ditemukan saat verifikasi §3.4/§3.7/§3.8; **di luar lingkup issue #12**, dicatat saja |

---

## Konsekuensi yang sering kelupaan

**Data berubah → semua tabel paper berubah.** Bukan hanya skenario yang dijalankan ulang:
Tabel IV, V, IX, X, XI, XII, XIII **semuanya** harus dihitung ulang. Ini biaya terbesarnya,
bukan skenarionya.

---

## Masih perlu diverifikasi

1. ~~Repo ID berstatus `[Perlu cek]` di `04_teacher_models.md`~~ — **selesai 2026-08-08**, semua
   HTTP 200 via HF API. Yang tersisa: **angka** benchmark AceMath dan Qwen3 (masih dari ringkasan
   sekunder, bukan repo ID)
2. Apakah baris `jawaban` bare-letter bocor ke holdout lama — cek
   `re.match(r'^[A-E]$', jawaban)` pada file holdout di Kaggle/Drive
3. Apakah environment Kaggle saat `clean_holdout.py` dijalankan punya `antlr4-python3-runtime`
   (tanpa itu `sympy.parse_latex` gagal → semua jawaban LaTeX dianggap tak-gradeable)
4. Kutipan persis arXiv:2505.14815 dan angka KTO vs DPO di GSM8K — keduanya masih dari ringkasan
   sekunder
