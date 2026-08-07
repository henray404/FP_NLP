# Referensi Paper — IndoMathReason

Kumpulan rujukan hasil riset sesi 2026-08-07/08, dipilah menurut kegunaannya untuk paper.

## Isi

| File | Untuk bagian paper |
|---|---|
| [01_multi_teacher_distillation.md](01_multi_teacher_distillation.md) | Skenario 6 (union multi-teacher), Tinjauan Pustaka, Tabel II |
| [02_preference_optimization.md](02_preference_optimization.md) | Skenario 7 (DPO/KTO), Tinjauan Pustaka, Tabel II |
| [03_language_mixing.md](03_language_mixing.md) | Skenario 1 (sumbu bahasa), Pembahasan, Keterbatasan |
| [04_teacher_models.md](04_teacher_models.md) | Skenario 1 (kandidat teacher), Metodologi |
| [05_positioning.md](05_positioning.md) | Pendahuluan, Kontribusi, Tabel II |

## Aturan pakai

1. **Jangan sitir apa pun dari sini tanpa membuka sumber aslinya dulu.** File-file ini hasil
   pencarian web, bukan hasil baca paper penuh. Beberapa angka berasal dari ringkasan sekunder.
2. Setiap klaim diberi label keyakinan:
   - `[Terverifikasi]` — judul/penulis/venue/ID terlihat langsung di halaman resmi (arXiv/ACL/HF)
   - `[Perlu cek]` — dari ringkasan sekunder, belum dikonfirmasi ke sumber primer
3. Angka benchmark yang dilaporkan vendor (Qwen, NVIDIA) **selalu** ditandai self-reported kalau
   masuk paper.

## Ringkasan posisi paper (versi singkat)

Teknik yang dipakai — distilasi CoT, rejection sampling, multi-teacher, DPO/KTO — **semuanya sudah
ada di literatur**. Yang belum ada: penerapan + pengukurannya pada **bahasa non-Inggris
berdaya-rendah**, di mana ada sumbu kegagalan tambahan yang tidak dipikirkan literatur Inggris:
model bisa **benar secara matematis tapi keluarannya berbahasa Inggris**, sehingga terbuang filter
bahasa.

Kontribusi dibingkai di **dimensi pengukuran**, bukan di algoritma. Detail: `05_positioning.md`.
