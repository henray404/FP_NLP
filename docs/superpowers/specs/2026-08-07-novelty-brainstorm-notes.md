# Novelty Brainstorm — IndoMathReason Paper (WIP notes, belum final)

**Date:** 2026-08-07
**Status:** brainstorming, belum di-lock scope-nya
**Paper acuan:** `KrocohMasStanis.pdf` (draft submisi, Skenario 1-4 sudah punya angka riil,
Skenario 5/MATH-IDN masih `[BELUM DIISI]`)

## Masalah

Novelty paper saat ini = replikasi pipeline gaya AIMO-2 (rejection sampling CoT distillation),
dipindah ke Bahasa Indonesia, model dikecilkan (Qwen2.5 0.5B/1.5B), dilatih di 1 GPU T4 16GB
pakai QLoRA. Tabel II (posisi penelitian) sendiri mengonfirmasi semua baris berbunyi "teknik
sama, bahasa/skala beda" — bukan kontribusi metode baru. Perlu delta metodologis, bukan cuma
transfer bahasa/skala.

## Konteks teknis yang relevan

Dari `docs/superpowers/specs/2026-06-19-cot-compare-select-design.md`:
- `src.cot_synthesis.filter_solutions.run_filter` sudah ada dan jalan (bukan "belum diimplementasi"
  seperti tertulis di CLAUDE.md — CLAUDE.md itu basi soal ini).
- Tiap generator run menghasilkan `candidates.jsonl` (SEMUA kandidat, benar+salah) dan
  `correct.jsonl` (subset yang lolos rejection sampling), dengan skema:
  - `candidates.jsonl`: `{id, soal, jawaban, cara, source, candidate_idx, text}`
  - `correct.jsonl`: `{id, soal, jawaban, candidate_idx, text, pred}`
- Artinya pasangan **chosen (dari correct.jsonl) vs rejected (candidates.jsonl minus correct.jsonl,
  match by id+candidate_idx)** bisa dibentuk tanpa generate ulang apa pun. Ini infrastruktur siap
  pakai buat opsi DPO/ensemble di bawah.
- Sudah ada preseden notebook pembanding 2 generator (Gemma-2 vs DeepSeek-R1) berbasis metrik
  coverage — pola yang sama bisa diperluas jadi union/ensemble alih-alih pilih 1 pemenang.

## Opsi yang dibahas

### 1. Ensemble multi-teacher rejection sampling
Union solusi benar dari 3 teacher (DeepSeek-R1, Qwen2.5-Math-7B, ERNIE-4.5-0.3B) yang sudah
di-generate untuk Skenario 1, bukan cuma pakai pemenang tunggal. Soal yang gagal di teacher A bisa
kejawab teacher B → cakupan soal naik dari 66,61% (Tabel X), train-ready set membesar.
**Biaya: nyaris nol** — data kandidat sudah ada, tinggal union + rebuild ChatML + retrain QLoRA
sekali + re-eval Skenario 2-4.

**Update pasca deep research (lihat bagian Riset Pembanding):** teknik ini SUDAH ADA di literatur
(TwT/DCRS, TinyLLM) — bukan orisinal, wajib disitasi. TinyLLM juga nemuin performa bisa TURUN
kalau makin banyak teacher digabung (knowledge conflict). Karena ERNIE-4.5-0.3B retensinya cuma
3,08% (jauh di bawah 2 teacher 7B lain, lihat Tabel X paper), **rekomendasi: exclude ERNIE dari
union**, cuma gabungin DeepSeek-R1 + Qwen2.5-Math-7B. Laporkan sebagai ablasi 3 baris: single-best
(DeepSeek-R1 saja) vs union-2 (tanpa ERNIE) vs union-3 (semua) — sekalian jadi bukti kuantitatif
paper ini sadar risiko yang disebut TinyLLM, bukan cuma niru mentah-mentah.

### 2. Augmentasi/curriculum nembak kelemahan subset easy
Temuan 1 (Skenario 2) di paper: gap easy vs numglue (pass@1 0,127 vs 0,231) itu linguistik, bukan
matematis — konsisten sama pivot-language finding MATH-IDN. Paper diagnosis lalu berhenti (dilempar
ke bagian Saran). Dua sub-opsi:
- **2a. Paraphrase augmentation** (mahal: perlu generate ulang soal+CoT via LLM)
- **2b. Curriculum ordering by solve-rate** (nyaris gratis: reorder training pakai proxy kesulitan
  dari jumlah kandidat benar per soal di Skenario 1, tanpa data baru)

### 3. Budget-aware CoT (length-controlled)
Tambah varian training antara full-CoT dan non-CoT (CoT dipotong/dipadatkan), petakan kurva
akurasi-vs-token. Nyambung ke framing "perangkat spek rendah" di pendahuluan. Murah (reuse data
CoT yang ada, tinggal truncate), tapi novelty-nya lebih lemah — sifatnya ablasi bukan mekanisme baru.

### 4. DPO (Direct Preference Optimization) — kandidat novelty utama
Bukan PPO/full-RLHF (terlalu berat buat 1 GPU 16GB + waktu mepet) dan bukan GRPO/RLVR ala
DeepSeek-R1 (butuh rollout live tiap step, mahal). **DPO** cocok karena:
- Preference pair (chosen/rejected) sudah bisa dibentuk gratis dari `candidates.jsonl` +
  `correct.jsonl` yang sudah ada (lihat konteks teknis di atas).
- Cuma 1 model policy di memori (+ reference via adapter swap) — `TRL DPOTrainer` sudah support
  PEFT/QLoRA out of the box, ringan di T4.
- Nutup limitation yang paper sendiri sudah akui di bagian Saran ("eksplorasi format-aware reward").
- Tidak ada satu pun rujukan di Tabel II yang pakai preference optimization/RL untuk problem ini —
  semua prompting atau SFT-only. Ini pembeda metodologis paling tajam dibanding opsi 1-3.

Desain kasar: SFT (sudah ada) → tahap kedua DPO pakai pasangan chosen/rejected dari rejection
sampling → bandingkan SFT-only vs SFT+DPO di holdout yang sama (skenario baru, pola mirip Skenario 3).

**Trade-off:** risiko lebih tinggi dari opsi 1 — perlu setup training baru (TRL, hyperparameter
DPO beta, dst) yang belum pernah disentuh di pipeline ini.

## Referensi buat DPO

1. Rafailov, R., dkk., "Direct Preference Optimization: Your Language Model is Secretly a Reward
   Model," *NeurIPS*, 2023, arXiv:2305.18290. — paper dasar, wajib dikutip.
2. Khaki, S., Li, J., Ma, L., Yang, L., Ramachandra, P., "RS-DPO: A Hybrid Rejection Sampling and
   Direct Preference Optimization Method for Alignment of Large Language Models," *Findings of
   ACL: NAACL 2024*, hal. 1665–1680. — kombinasi rejection sampling + DPO, persis pola pipeline ini.
3. Hwang, H., Kim, D., Kim, S., Ye, S., Seo, M., "Self-Explore: Enhancing Mathematical Reasoning in
   Language Models with Fine-grained Rewards," arXiv:2404.10346, 2024. — bukti DPO efektif spesifik
   di domain reasoning matematis (GSM8K/MATH), bukan cuma alignment umum.

## Rekomendasi sequencing (belum final, masih dibrainstorm)

1 (ensemble, murah, boost data) → 4/DPO (novelty utama) → 2b (curriculum, opsional kalau sisa waktu).
Skip 2a (paraphrase) — kalah prioritas per unit effort dibanding DPO. 3 (budget-CoT) jadi cadangan
kalau semua di atas kelar duluan.

## Riset pembanding (deep research, 2026-08-07)

Verdict: **kedua teknik (ensemble multi-teacher, DPO-setelah-rejection-sampling) itu sendiri BUKAN
hal baru** di literatur global. Yang belum ketemu di pencarian adalah kombinasinya buat bahasa
low-resource / Bahasa Indonesia. Novelty paper harus diframe di level kombinasi+bahasa+batasan
sumber daya, bukan diklaim sebagai teknik baru — supaya nggak ketauan reviewer yang paham literatur
RLHF/distillation.

**Precedent buat ensemble multi-teacher (Opt 1):**
1. Xu, J., Zhou, M., Liu, W., Liu, H., Han, S., Zhang, D., "TwT: Thinking without Tokens by
   Habitual Reasoning Distillation with Multi-Teachers' Guidance," arXiv:2503.24198, 2025. —
   *Dual-Criteria Rejection Sampling (DCRS)*: generate dataset distilasi dari banyak teacher model
   sekaligus. Paling mirip sama rencana Opt 1.
2. Tian, Y., dkk., "TinyLLM: Learning a Small Student from Multiple Large Language Models,"
   arXiv:2402.04616, 2024 (WSDM 2025). — distilasi dari 2+ teacher LLM ke satu student; nemuin
   performa bisa TURUN kalau teacher makin banyak (knowledge conflict antar rationale).
3. Chen, H., Wu, S., Quan, X., Wang, R., Yan, M., Zhang, J., "MCC-KD: Multi-CoT Consistent
   Knowledge Distillation," arXiv:2310.14747, 2023. — multi-CoT consistency dari beberapa rationale
   teacher, preseden lebih lama.

**Precedent buat DPO/RS-DPO di math reasoning (Opt 4):**
Area aktif — varian lain yang perlu disebut di related work: M-DPO (multi-turn math DPO, 2025),
Step-controlled DPO (2025), rStar-Math (2025, MCTS self-evolution), TINA (LoRA reasoning kecil,
2025), Phi-4-mini-reasoning (2025), KEPO (2026). RS-DPO (Khaki dkk. 2024, sudah dicatat di atas)
tetap jadi rujukan paling dekat pola pipeline ini (rejection sampling → preference pair → DPO).

**Celah asli yang ditemukan (dasar klaim novelty):** tidak ada paper yang ditemukan
menggabungkan CoT distillation + rejection sampling + DPO khusus untuk bahasa low-resource/non-
Inggris, atau menerapkan DPO/preference optimization untuk math reasoning Bahasa Indonesia / bahasa
Asia Tenggara low-resource lain. Paper Indonesia/SEA yang ada (SEA-LION, SEA-HELM, NusaMT) itu NLP
umum, bukan math reasoning + RL/DPO.

**Implikasi framing paper:** ganti kalimat kontribusi dari "kami mengusulkan teknik X" jadi "teknik
X sudah terbukti efektif di domain umum/Bahasa Inggris [cite TwT/TinyLLM/RS-DPO]; penelitian ini
yang pertama menerapkan dan mengevaluasinya secara terkontrol untuk penalaran matematika Bahasa
Indonesia di bawah batasan satu GPU 16GB." Venue (buletin pagelaran mahasiswa nasional, bukan
ACL/NeurIPS) realistis cukup dengan novelty level aplikasi ini, asal jujur nyitir kerja terkait.

## Breakdown pipeline asli (dari kode, bukan narasi paper)

CLAUDE.md basi soal status implementasi — `src/cot_synthesis` dan `src/training` sudah jalan penuh.

1. **Akuisisi & ekstraksi** — `src/scraping/scrape_defantri.py`, `src/scraping_ver2/scrape_osn.py`
   → PDF mentah. `notebooks/extract_vlm_gemini.ipynb` + `src/preprop/extract_questions.py` →
   `data/extracted_vlm*/*.jsonl`
2. **Merge & dedup mentah** — `src/preprop/merge_dataset.py`, `merge_and_dedup.py`, `dedup.py`,
   `tag_source.py`
3. **Isi jawaban kosong** — `src/preprop/fill_missing.py` (+ `apply_fill_cache.py`),
   `notebooks/preprop/fill_missing_kaggle.ipynb`
4. **Filter aturan** — `src/preprop/filter_rules.py`
5. **Filter validitas LLM** — `src/preprop/filter_validity.py`,
   `notebooks/preprop/filter_validity_kaggle.ipynb`
6. **Split holdout + dekontaminasi** — `src/preprop/split_final.py`, `clean_testset.py`,
   `src/eval/make_holdout.py`, `build_un_holdout.py`, `clean_holdout.py` →
   `train_pool.jsonl` + holdout 600 soal
7. **Generate kandidat teacher (CoT)** — `src/cot_synthesis/generate.py` (+ `generate_gemma.py`),
   `notebooks/cot/cot_pipeline_kaggle*.ipynb` (varian gemma/deepseek/qwenmath) →
   `data/cot/candidates.jsonl` `{id, soal, jawaban, cara, source, candidate_idx, text}` — SEMUA
   kandidat benar+salah, per teacher terpisah.
8. **Rejection sampling (judge)** — `src/cot_synthesis/filter_solutions.py` (`run_filter`) →
   `data/cot/correct.jsonl` `{id, soal, jawaban, candidate_idx, text, pred}` — subset lolos, per
   teacher terpisah.
9. **Skenario 1 — pilih teacher** — `src/cot_synthesis/compare_teachers.py` (`compare_teachers`) +
   `notebooks/skenario/s1_select_teacher.ipynb`. Baca `candidates.jsonl`+`correct.jsonl` tiap
   teacher, hitung retention%/coverage%, pilih 1 pemenang. **Titik suntik Opt 1 (ganti jadi
   union, bukan pilih 1).**
10. **Bangun ChatML SFT** — `src/cot_synthesis/to_chatml.py` (`run`) → dari `correct.jsonl`
    pemenang, bikin `data/sft/cot.jsonl` + `data/sft/nocot.jsonl` sekaligus (best-per-problem,
    filter Indonesia-only).
11. **Training QLoRA** — `src/training/train_sft.py` (`build_and_train`, config YAML di
    `src/training/configs/`), Unsloth+trl `SFTTrainer`, `notebooks/train/train_sft_kaggle.ipynb`
    → adapter LoRA di `outputs/{cot,nocot}_{0.5b,1.5b}/`
12. **Evaluasi Skenario 2-4** — `src/eval/scenario_eval.py`, `skenario4_eval.py`, `vllm_eval.py`,
    `answer_check.py`, `sampling_metrics.py`, `notebooks/skenario/s2_eval_crossdata.ipynb`,
    `s3_cot_vs_noncot*.ipynb`, `skenario-4.ipynb`
13. **Laporan kualitas data (Tabel IX)** — `src/eval/data_quality_report.py`,
    `data_quality_stats.py`, `judge_data_quality.py`, `notebooks/revisi/03_04_kualitas_data*.ipynb`
14. **Belum di-commit, status belum jelas** — `src/data_validation/validate_training_data.py` +
    `notebooks/revisi/06_repair_cara.ipynb` (lihat `git status`) — perlu dicek isinya.

### Titik suntik novelty

- **Opt 1 (ensemble):** ganti langkah #9. Perlu skrip baru (mis. `merge_teachers.py`, pola mirip
  `compare_teachers.py`) yang **union** `correct.jsonl` dari 2 teacher (exclude ERNIE) by `id`,
  hasil union masuk ke langkah #10 (`to_chatml.py`) tanpa ubah apa pun di #10-12.
- **Opt 4 (DPO):** nyisip antara #8 dan #11. Perlu skrip baru (`build_dpo.py`): **chosen** = baris
  `correct.jsonl`, **rejected** = baris `candidates.jsonl` yang id+candidate_idx-nya TIDAK ada di
  `correct.jsonl` (gagal judge). Skrip training baru (`train_dpo.py`) niru struktur `TrainConfig`
  di `train_sft.py`, ganti `SFTTrainer`→`DPOTrainer` (trl), load dari checkpoint hasil #11.

## Temuan kualitas data `easy_clean.jsonl` (2026-08-07)

Ditemukan saat sampling manual sebelum mulai eksperimen novelty. **Ini blocker: harus dibereskan
dulu sebelum opt-1/DPO**, karena rejection sampling akan menilai kandidat terhadap gold standard
yang sebagian rusak → angka retensi/cakupan (Tabel X) jadi tidak sahih.

### Sudah dibersihkan (notebook `notebooks/revisi/07_bersihkan_easy_clean.ipynb`)

Input 2.686 baris → output `data/Final/easy_clean_v2.jsonl` 2.336 baris (87,0%), dibuang 350 (13,0%):

| Alasan | Jumlah | Tindakan |
|---|---|---|
| `jawaban` cuma huruf bare A-E (nilai hilang total) | 278 | buang |
| soal berbahasa Inggris (deteksi di field `soal`) | 50 | buang |
| `cara` berisi frasa LLM bingung/mengarang | 22 | buang |
| `jawaban` = "X. \<nilai\>" (prefix pilgan) | 107 | **diperbaiki**, prefix di-strip |

Akar masalah bug pilgan: `is_multiple_choice()` di `src/preprop/filter_rules.py:33` butuh ≥3 baris
opsi **di dalam teks soal**; soal yang opsinya sudah hilang saat ekstraksi VLM tapi kunci jawabannya
tersalin apa adanya ("C") lolos filter. Dua filter holdout juga tidak menangkapnya:
`make_holdout.py:36` (regex `[A-Za-z]{3,}` tidak match huruf tunggal → diklasifikasi `single_expr`,
masuk GRADEABLE) dan `clean_holdout.py:33` (sympy mem-parse "C" sebagai simbol matematika yang sah →
dianggap gradeable). **Perlu dicek apakah baris bare-letter ini bocor ke holdout 600 soal** — file
holdout tidak ada di repo lokal (ada di Kaggle/Drive), cek dengan `re.match(r'^[A-E]$', jawaban)`.

### BELUM dibersihkan — hasil review manual 260 baris acak dari v2 (11% sampel, dibaca satu-satu)

Pembersih di atas hanya menyasar 3 bug spesifik; kualitas keseluruhan lebih buruk dari angka 87%.

| # | Kategori | Rate di sampel | Estimasi di 2.336 |
|---|---|---|---|
| 1 | Frasa fabrikasi **varian lain** (daftar detektor terlalu sempit) | 11/260 (4,2%) | ±98 |
| 2 | Korupsi teks/LaTeX dari OCR (huruf/kurung hilang jadi karakter acak) | 6/260 (2,3%) | ±54 |
| 3 | `cara` vs `jawaban` mismatch (kesimpulan beda dari field jawaban) | 6/260 (2,3%) | ±54 |
| 4 | Content-bleed: isi `cara` milik soal LAIN | 5/260 (1,9%) | ±45 |
| 5 | `cara` berbahasa Inggris (detektor lama hanya cek field `soal`) | 4/260 (1,5%) | ±35 |

Contoh per kategori (indeks sampel, seed=7):
- (1) varian lolos: "saya perlu tahu detail soalnya", "kita perlu lebih jelas tentang",
  "tidak jelas mengenai", "karena informasi tersebut tidak disediakan",
  "diserahkan kepada pembaca/pemirsa" (placeholder dari buku sumber, bukan LLM) — #53, #108, #111,
  #140, #203, #215, #220
- (2) #25 `"$& \le 8 - 16$"` + cara `"$B -4)B -4) \le 0$"`; #144 `"$9s$ membagi $33rr$"`;
  #222 `"K = \text{c s ses}"`
- (3) #164 cara→(11,9) vs jawaban→"(7,8)"; #187 cara→...=9 vs jawaban→...=16;
  #169 cara→1 vs jawaban→160π; #190 cara→159 vs jawaban→149
- (4) #48 soal tanya $a+b+c+d+e$, cara membahas $x^2-4x-1$ (topik lain); #80, #230, #254
- (5) #58, #103, #124, #207 — cara dibuka "Okay, so I have this problem..."

**Estimasi gabungan (ada overlap): 15-20% dari 2.336 baris v2 masih cacat.** Artinya fraksi
benar-benar bersih dari 2.686 baris asli ≈ **70-75%**, bukan 87%. Angka 87% hanya mengukur 3 bug
yang diburu, bukan mutu keseluruhan.

Kelima kategori tidak bisa diandalkan ke regex (frasa terlalu bervariasi, korupsi tak berpola,
content-bleed hanya terlihat kalau isi dibaca) → perlu tahap **LLM judge**, lihat bagian berikut.

## Rencana LLM judge + pembersihan v3

### Temuan pengubah kalkulasi: `cara` tidak dipakai pipeline

Verifikasi kode:
- `src/cot_synthesis/prompt_wrap.py:15-32` — prompt teacher hanya `.format(soal=...)`. Hanya `soal`
  yang masuk konteks model.
- `src/cot_synthesis/generate.py:195` — `"cara": get_cara(item)` hanya menyalin ke
  `candidates.jsonl` sebagai metadata passthrough, tidak pernah masuk prompt.
- `src/cot_synthesis/to_chatml.py:86` — data latih dibangun dari `r["text"]` (keluaran teacher) dan
  `r["pred"]`, bukan `cara` sumber.
- `notebooks/cot/cot_pipeline_kaggle.ipynb` — input langsung
  `data/Final/{numglue_clean, aimo_hard_clean, easy_clean}.jsonl`.

Jadi yang berpengaruh hanya **`soal`** (masuk prompt teacher) dan **`jawaban`** (gold rejection
sampling). Kategori cacat 1, 2-bagian-cara, dan 5 dari review manual **tidak merusak pipeline**;
kategori 3 dan 4 hanya berguna sebagai *sinyal* bahwa `jawaban` patut dicurigai.

**Konsekuensi praktis: `cara` tidak perlu dibersihkan, cukup dihapus kolomnya.** ±190 baris cacat
hilang sebagai kelas, nol panggilan judge, nol baris hilang. (`easy_nocara.jsonl` yang sudah ada
tampaknya hasil kesimpulan yang sama.)

### Tahap A — pre-pass deterministik (CPU, tanpa GPU)

Pakai helper yang sudah ada, jangan tulis ulang:
- `latex_ok()` — `src/eval/clean_holdout.py:40`, delimiter `$ { } \( \[` tak seimbang = teks korup
- `gold_gradeable()` — `src/eval/clean_holdout.py:33`, `jawaban` bisa diparse angka/sympy
- `is_multiple_choice()` — `src/preprop/filter_rules.py:32`
- `is_indonesian()` — `src/cot_synthesis/to_chatml.py:39`
- tambahan: `soal` memuat `[Gambar:`, terlalu pendek, atau `jawaban` huruf bare A-E

### Tahap B — LLM judge, 2 pertanyaan terpisah

**Prinsip: jangan minta judge memverifikasi kebenaran matematika.** Qwen-7B tidak bisa
menyelesaikan soal OSN; kalau dipaksa ia mengarang vonis dan baris bagus ikut terbuang. Hanya
tanyakan penilaian teks:

- **Q1 keterjawaban** (input: `soal` saja) — soal utuh dan bisa dijawab dari teksnya sendiri?
  TIDAK jika terpotong, merujuk gambar/tabel yang hilang, merujuk soal lain, berupa perintah
  aktivitas, atau teks rusak.
- **Q2 kelayakan gold** (input: `soal` + `jawaban`) — `jawaban` berbentuk wajar dan **bertipe
  benar**? Sengaja bukan "benar nilainya". Menangkap: kunci berupa huruf, esai padahal diminta
  angka, tipe tak cocok, tabel/daftar panjang.
- **Q3 opsional (sinyal lunak)** — kesimpulan `cara` sejalan dengan `jawaban`? Hasil **tidak**
  auto-drop, hanya untuk memprioritaskan baris yang gold-nya dicurigai.

Dipisah per pertanyaan (bukan satu prompt multi-kriteria) karena 7B menurun akurasinya pada
keluaran terstruktur banyak bagian. Satu panggilan = satu pertanyaan, `max_tokens≈4`.

Model: `Qwen2.5-7B-Instruct` (sudah dipakai `filter_solutions.py` + `filter_validity.py` — konsisten,
satu hal saja yang dijelaskan di metodologi). Kaggle 2×T4 → `--tensor-parallel-size 2`.
Checkpoint/resume: salin pola `.progress` dari `filter_solutions.py:112-131`.

### Hasil Tahap A (sudah dijalankan, `src/data_validation/judge_quality.py`)

`python -m src.data_validation.judge_quality data/Final/easy_clean_v2.jsonl --no-llm`

Dari 2.336 baris: **82 dibuang (3,5%)**, 2.254 lolos ke tahap judge.

| Alasan | Jumlah | Contoh |
|---|---|---|
| `butuh_gambar` | 77 | soal geometri ber-`[Gambar:` |
| `latex_soal_timpang` | 4 | `"...invers matriks A adalah...(Pilihan A-E tersedia)"`, `"Perhatikan tabel berikut! ...(Tabel nilai 5-9...)"` |
| `soal_terlalu_pendek` | 1 | `"Hitunglah P65."` |

**Dua bug ketemu saat menjalankannya** (justru bukti kenapa kalibrasi wajib):

1. **`gold_gradeable()` salah dipakai untuk train pool.** Versi pertama Tahap A memakainya dan
   membuang **1.002/2.336 baris (43%)** — hampir semua false positive: `"Rp. 742.500,00"`,
   `"panjang AC = 24 cm"`, `"4\sqrt{2}"`. Fungsi itu dirancang untuk memilih **holdout** (yang
   butuh gold auto-gradeable oleh `answer_check`). Untuk train pool, gold berbentuk kalimat itu
   wajar — docstring `filter_solutions.py` sendiri menyatakan LLM judge dipakai *justru karena*
   `jawaban` berupa kalimat natural tanpa `\boxed`. Sudah dihapus dari Tahap A; kelayakan bentuk
   gold diserahkan ke Q2.
2. **`sympy.parse_latex` tidak berfungsi tanpa paket `antlr4-python3-runtime`.** Muncul
   `UserWarning: antlr4.error.ErrorListener module is not installed` → `_to_expr()` gagal untuk
   SEMUA jawaban LaTeX → dianggap tak-gradeable. Ini memperbesar false positive di atas.
   **Perlu dicek**: apakah environment Kaggle saat `clean_holdout.py` dijalankan punya antlr4?
   Bukti tidak langsung menyatakan aman — Tabel V paper melaporkan 198/300 gold holdout easy
   berupa ekspresi simbolik, jadi `_to_expr` tampaknya bekerja di sana. Tetap layak diverifikasi
   karena kalau tidak, holdout jadi bias ke jawaban numerik murni.

### Kalibrasi (wajib, jangan dilewat)

Jalankan judge pada **260 baris yang sudah dianotasi manual** (seed=7, lihat bagian temuan di atas),
bandingkan dengan label manual, hitung presisi/recall per pertanyaan. Kalau presisi vonis "TIDAK"
rendah → judge membuang baris bagus → perketat prompt atau turunkan ke review-only. Jangan jalankan
2.336 baris buta.

Catatan kejujuran soal label: review manual dilakukan untuk mencari cacat secara umum, bukan
pelabelan Q1/Q2 sistematis. Baris yang **ditandai cacat** = label kuat (untuk recall); baris yang
tidak ditandai = "dianggap wajar" = label lemah (untuk false-positive rate). Laporkan asimetri ini
apa adanya, jangan diklaim sebagai anotasi penuh.

Nilai tambah: kalibrasi ini **aset paper** — "validasi LLM judge terhadap anotasi manual" menjawab
pertanyaan reviewer "bagaimana Anda tahu filter Anda bekerja?". Paper saat ini nol validasi manusia
untuk seluruh judge LLM-nya.

### Kebijakan pembersihan: 3 keranjang

| Keranjang | Isi | Aksi |
|---|---|---|
| DROP | Q1=TIDAK, atau gold tak-gradeable (Tahap A) | buang |
| REVIEW | Q2=TIDAK, atau Q3 mismatch | tulis ke `*_v3_review.jsonl` + alasan, **jangan** auto-drop |
| KEEP | sisanya | simpan, kolom `cara` dibuang |

REVIEW diperkirakan 50-100 baris — bisa disapu manual <1 jam, dan justru di situ judge paling tidak
dapat dipercaya.

### Dua keputusan yang masih terbuka

1. **Re-run hilir atau bekukan data sekarang?** Kalau jumlah baris berubah, seluruh hilir ikut
   berubah: generate teacher, training, eval, dan angka Tabel IV/V/IX/X/XI/XII harus dihitung ulang.
   Dengan sisa 4 hari, ini biaya terbesar — bukan judge-nya.
2. **`aimo_hard_clean.jsonl` (3.606 baris) ikut dibersihkan?** File itu ada di `DATASETS` notebook
   CoT jadi **sudah masuk pipeline**, tapi tidak muncul di Tabel V paper (yang hanya numglue + easy
   = 6.517). Perlu klarifikasi: eksperimen yang belum dilaporkan, atau nyasar masuk?

## Belum diputuskan

- Scope final: berapa banyak dari opsi 1/2b/3/4 yang benar-benar dikerjakan sebelum deadline.
- Apakah DPO training dijalankan di kedua ukuran model (0.5B & 1.5B) atau cuma 1.5B (student utama).
- Bagaimana isi Skenario 5 (MATH-IDN) yang masih `[BELUM DIISI]` disinkronkan dengan eksperimen baru.
