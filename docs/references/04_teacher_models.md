# Kandidat Teacher & Student Model (untuk Skenario 1 & Metodologi)

Ukuran VRAM = perkiraan bf16 (params × 2 byte), belum termasuk KV cache.
Target perangkat: **A6000 48 GB, single GPU**.

## Kandidat teacher — tier 14B (inti eksperimen)

Dipilih supaya perbandingan **terkontrol**: parameter sama, filosofi training berbeda.

| tag | repo | basis | filosofi | VRAM |
|---|---|---|---|---|
| `qwen3-14b` | `Qwen/Qwen3-14B` | Qwen3 | generalis multibahasa | ~28 GB |
| `r1-distill-14b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | Qwen2.5-14B | distilasi jejak reasoning R1 | ~28 GB |
| `openmath-14b` | `nvidia/OpenMath-Nemotron-14B` | Qwen2.5-14B | spesialis matematika | ~28 GB |

## Kandidat teacher — tier 7B (kontinuitas Skenario 1 lama)

| tag | repo | catatan | VRAM |
|---|---|---|---|
| `r1-distill-7b` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | juara lama: retensi 37,08%, cakupan 66,61% | ~15 GB |
| `qwen25-math-7b` | `Qwen/Qwen2.5-Math-7B-Instruct` | runner-up lama: retensi 30,82%, cakupan 61,94% | ~15 GB |

## Status verifikasi repo ID

| repo | status |
|---|---|
| `Qwen/Qwen3-14B` | `[Terverifikasi]` halaman HF muncul di hasil pencarian |
| `nvidia/OpenMath-Nemotron-14B` | `[Terverifikasi]` halaman HF muncul |
| `nvidia/OpenMath-Nemotron-32B` | `[Terverifikasi]` halaman HF muncul |
| `nvidia/OpenMath-Nemotron-14B-Kaggle` | `[Terverifikasi]` halaman HF muncul |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `[Terverifikasi]` sudah dipakai di repo ini |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `[Perlu cek]` bagian keluarga distill, belum dibuka |
| `Qwen/Qwen2.5-Math-7B-Instruct` | `[Perlu cek]` disebut di paper lama, belum dibuka ulang |
| `Qwen/Qwen3-32B-AWQ` | **`[Belum diverifikasi — tebakan pola nama]`. Jangan dipakai tanpa dicek.** |

## OpenMath-Nemotron — kenapa penting secara metodologis

> Gitman, I. dkk. "AIMO-2 Winning Solution: Building State-of-the-Art Mathematical Reasoning Models
> with the OpenMathReasoning Dataset." arXiv:2504.16891, 2025.

Ini **ref [5] paper kalian** — pendekatan yang sedang direplikasi.

`OpenMath-Nemotron-32B` = fine-tune `Qwen/Qwen2.5-32B` pada OpenMathReasoning;
`OpenMath-Nemotron-14B` = dari `Qwen/Qwen2.5-14B`. Varian `-14B-Kaggle` dipakai pada submisi juara
pertama AIMO-2. `[Terverifikasi]` dari halaman HF.

**Memakai teacher dari karya asalnya membuat klaim replikasi jauh lebih kuat** daripada memakai
model yang kebetulan jago matematika.

**Bukti tandingan dari data kalian sendiri:** Tabel XI/XII mencatat `OpenMath-Nemotron-1.5B` punya
kepatuhan format hanya **0,147** (numglue) dan **0,219** (easy) — terburuk dari semua model yang
diuji. Lihat `03_language_mixing.md` (hipotesis H1).

## Model matematika lain yang sempat diriset

| model | catatan | status |
|---|---|---|
| Qwen2.5-Math (7B/72B) | arXiv:2409.12122 | `[Terverifikasi]` ID |
| AceMath | arXiv:2412.15084, Findings ACL 2025 | `[Terverifikasi]` ID/venue |
| `nvidia/AceMath-RL-Nemotron-7B` | dilaporkan 69,0% pass@1 AIME 2024 | `[Perlu cek]` angka dari ringkasan |

## Angka benchmark — HATI-HATI

Dari Qwen3 Technical Report via ringkasan pencarian:
> Qwen3-4B: **97,0** MATH-500 / **73,8** AIME'24
> DeepSeek-R1-Distill-Qwen-32B: **94,3** MATH-500 / **72,6** AIME'24

`[Perlu cek]` — **self-reported vendor**, thinking mode aktif. Kalau masuk paper, wajib ditandai
sebagai angka yang dilaporkan pengembang, bukan pengukuran independen.

## Student model — keputusan: Qwen2.5-3B

**Bukan 7B.** Alasan:

1. **Narasi paper selamat.** Judul menyebut "Model Bahasa Kecil", intisari menyebut "perangkat
   berspesifikasi rendah". Student 7B membatalkan klaim itu → judul + abstrak + pendahuluan harus
   ditulis ulang. 3B dalam 4-bit ≈ 2 GB, masih jujur disebut kecil.
2. **Klaim vs baseline jadi lebih kuat.** Paper sekarang berdalih "meskipun Phi-3.5 berukuran lebih
   dari dua kali lipat (3,8B)". Pada 3B vs 3,8B dalih itu hilang.
3. **Kurva kapasitas 0,5B → 1,5B → 3B** dalam satu keluarga, tanpa confound ganti keluarga.
4. QLoRA 3B di A6000 sepele.

**Kenapa bukan Qwen3-4B** meski math-nya lebih kuat: Qwen3 punya thinking mode bawaan — base-nya
cenderung mengeluarkan CoT sendiri. Itu **mengontaminasi Skenario 3**: lengan "non-CoT" tidak lagi
benar-benar non-CoT. Qwen2.5 base adalah LM polos → eksperimen terkontrol lebih bersih.

**Wajib base, bukan `-Instruct`.** Varian Instruct sudah punya kemampuan CoT bawaan sehingga efek
data CoT kalian tidak terukur. Catatan teknis: Qwen2.5 base tidak membawa `chat_template` (hanya
`-Instruct` yang punya); `train_sft.py:183-188` sudah menanganinya dengan memasang template ChatML.

## Baseline Skenario 4 — perlu direvisi

Student naik ke 3B, komposisi baseline lama tidak lagi adil:

| baseline | ukuran | status |
|---|---|---|
| `Qwen2.5-3B-Instruct` | 3B | **tambahkan** — base identik, post-training beda. Paling informatif: mengisolasi nilai data kalian di atas instruction-tuning biasa |
| Phi-3.5-mini-instruct | 3,8B | pertahankan — kini perbandingan setara |
| SmolLM2-1.7B-Instruct | 1,7B | turunkan jadi konteks skala |
| OpenMath-Nemotron-1.5B | 1,5B | turunkan jadi konteks skala |

Menang atas model yang lebih kecil tidak membuktikan apa pun.
