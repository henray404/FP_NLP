# Posisi & Klaim Kebaruan Paper

Apa yang **boleh** diklaim, apa yang **wajib** disitir, dan bagaimana membingkainya.

## Masalah awalnya

Paper versi `KrocohMasStanis.pdf` lemah kebaruannya. Tabel II (posisi penelitian) hampir setiap
barisnya berbunyi "teknik sama, bahasa/skala berbeda". Itu terbaca sebagai **replikasi**, bukan
kontribusi metode.

## Hasil riset: semua teknik yang dipakai sudah ada duluan

| Teknik | Sudah ada di | Boleh diklaim baru? |
|---|---|---|
| Distilasi CoT + rejection sampling | DeepSeek-R1, AIMO-2 | tidak |
| Multi-teacher / union | TwT, TinyLLM, MCC-KD | tidak |
| DPO setelah rejection sampling | RS-DPO | tidak |
| KTO | paper KTO | tidak |
| QLoRA | paper QLoRA | tidak |

**Jangan sekali-kali menulis "kami mengusulkan teknik X".** Reviewer yang paham literatur
RLHF/distillation akan menangkapnya, dan kredibilitas seluruh paper ikut jatuh.

## Celah yang benar-benar ditemukan

Pencarian tidak menemukan paper yang:
1. menggabungkan CoT distillation + rejection sampling + DPO/KTO **khusus untuk bahasa
   low-resource / non-Inggris**, atau
2. menerapkan preference optimization untuk penalaran matematis **Bahasa Indonesia atau bahasa
   Asia Tenggara low-resource lain**

Paper Indonesia/SEA yang ada (SEA-LION, SEA-HELM, NusaMT) semuanya NLP umum, bukan penalaran
matematis + RL/preference optimization.

`[Perlu cek]` — ini hasil pencarian, bukan survei sistematis. Tulis sebagai **"sepanjang
penelusuran penulis"**, bukan pernyataan absolut.

## Bingkai kontribusi yang bertahan

Kontribusi ditempatkan di **dimensi pengukuran**, bukan di algoritma:

> Teknik-teknik ini sudah terbukti efektif di domain umum/Bahasa Inggris [sitir TwT, TinyLLM,
> RS-DPO, KTO]. Penelitian ini yang pertama menerapkan dan **mengevaluasinya secara terkontrol**
> untuk penalaran matematika Bahasa Indonesia, di mana terdapat sumbu kegagalan yang tidak muncul
> pada setting Bahasa Inggris.

### Sumbu kegagalan itu, konkretnya

Di Bahasa Inggris, satu-satunya pertanyaan saat menilai kandidat solusi: **"benar atau salah?"**

Di Bahasa Indonesia ada dua pertanyaan tambahan yang menentukan apakah solusi terpakai:
1. **Formatnya patuh?** Tanpa `\boxed{}` → dieliminasi `filter_solutions.py`
2. **Bahasanya Indonesia?** Dominan Inggris → dibuang `to_chatml(id_only=True)`

Akibatnya **solusi yang benar secara matematis bisa tetap terbuang.** Cakupan mentah menyesatkan.
Karena itu dipakai `skor efektif = cakupan × rasio Indonesia`.

## Tiga klaim spesifik yang bisa dipertahankan

### 1. Pemilihan teacher untuk distilasi non-Inggris (Skenario 1)
> Untuk distilasi CoT bahasa non-Inggris, model generalis multibahasa dapat mengungguli spesialis
> matematika, karena akurasi matematis tidak berguna bila keluarannya terbuang di filter
> bahasa/format.

Diuji lewat H1/H2 di `03_language_mixing.md`. **Kalau hipotesisnya salah pun tetap layak
dilaporkan** — artinya spesialis matematika ternyata tangguh lintas bahasa, dan itu juga temuan.

### 2. Union multi-teacher yang sadar-bahasa (Skenario 6)
> Penggabungan multi-teacher pada bahasa berdaya-rendah punya trade-off yang tidak ada di literatur
> Inggris: menambah teacher menaikkan cakupan matematis tetapi berisiko memasukkan solusi berbahasa
> Inggris yang nantinya terbuang. Seleksi union karena itu harus sadar-bahasa.

Detail: `01_multi_teacher_distillation.md`.

### 3. KTO memanfaatkan limbah rejection sampling (Skenario 7)
> KTO memanfaatkan ~33% soal yang dibuang total oleh rejection sampling (soal tanpa satu pun solusi
> benar), menyerang langsung bottleneck cakupan 66,61% yang dilaporkan penelitian ini sendiri.

Detail + tabel perbandingan SFT/DPO/KTO: `02_preference_optimization.md`.

## Kontribusi metodologis tambahan: validasi judge oleh manusia

Paper saat ini memakai LLM judge di beberapa tahap (validitas soal, kebenaran solusi) **tanpa satu
pun validasi manusia**. Reviewer wajar bertanya: "bagaimana Anda tahu filter Anda bekerja?"

Sesi ini menghasilkan anotasi manual atas **260 baris acak** (seed=7) dari `easy_clean_v2.jsonl`,
bisa dipakai mengukur presisi/recall LLM judge.

**Catatan kejujuran yang wajib ditulis:** review manual dilakukan untuk mencari cacat secara umum,
**bukan** pelabelan Q1/Q2 sistematis. Baris yang ditandai cacat = label kuat (untuk recall); baris
yang tidak ditandai = "dianggap wajar" = label lemah (untuk false-positive rate). Laporkan asimetri
ini apa adanya, jangan diklaim sebagai anotasi penuh.

## Temuan kualitas data yang juga layak dilaporkan

Detail lengkap: `docs/superpowers/specs/2026-08-07-novelty-brainstorm-notes.md`.

- **278 baris (10,3%)** `jawaban` hanya berisi huruf pilihan ganda tanpa nilai — soal secara
  harfiah tidak bisa dijawab
- **107 baris** `jawaban` berprefiks huruf pilihan ganda
- Akar masalah: `is_multiple_choice()` butuh ≥3 baris opsi **di dalam teks soal**; soal yang
  opsinya sudah hilang saat ekstraksi VLM tetapi kunci jawabannya tersalin ("C") lolos filter
- Dua filter holdout juga tidak menangkapnya (`make_holdout.py:36`, `clean_holdout.py:33`)

Memperkuat bagian Keterbatasan dan menunjukkan penjaminan mutu yang serius — asalkan ditulis
sebagai perbaikan, bukan disembunyikan.

## Yang TIDAK jadi dikerjakan

- **PPO / full RLHF** — terlalu berat, butuh reward model + value model terpisah
- **GRPO / RLVR** — butuh rollout live tiap step, jauh lebih mahal dari SFT
- **Paraphrase augmentation** — kalah prioritas per satuan usaha dibanding KTO
