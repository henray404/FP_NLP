# Distilasi Multi-Teacher (untuk Skenario 6)

**Intinya: menggabungkan beberapa teacher BUKAN ide baru. Wajib disitir, jangan diklaim.**

## Rujukan wajib

### TwT — paling dekat dengan yang kita lakukan
> Xu, J., Zhou, M., Liu, W., Liu, H., Han, S., Zhang, D. "TwT: Thinking without Tokens by Habitual
> Reasoning Distillation with Multi-Teachers' Guidance." arXiv:2503.24198, 2025.

`[Terverifikasi]` — judul + daftar penulis dibuka langsung dari arXiv.

Punya teknik bernama **Dual-Criteria Rejection Sampling (DCRS)**: membangkitkan dataset distilasi
dari **beberapa teacher model sekaligus** untuk mendapat kandidat yang lebih beragam dan bermutu.
Praktis sama dengan rencana Skenario 6 kita.

### TinyLLM — sumber peringatan "kebanyakan guru malah jelek"
> "Beyond Answers: Transferring Reasoning Capabilities to Smaller LLMs Using Multi-Teacher
> Knowledge Distillation." arXiv:2402.04616, 2024. Dipublikasikan di WSDM 2025,
> DOI 10.1145/3701551.3703577.

`[Terverifikasi]` judul/ID/venue. `[Perlu cek]` daftar penulis lengkap — belum dibuka.

Temuan yang relevan: **performa student menurun ketika jumlah teacher terus ditambah**, karena
*knowledge conflict* — rationale antar teacher saling bertentangan (halusinasi, jalur penalaran
berbeda, domain keahlian berbeda).

**Ini alasan teknis kenapa kita berhenti di union-2 dan tidak melatih union-all.**

### MCC-KD — preseden lebih lama
> Chen, H., Wu, S., Quan, X., Wang, R., Yan, M., Zhang, J. "MCC-KD: Multi-CoT Consistent Knowledge
> Distillation." arXiv:2310.14747, 2023.

`[Terverifikasi]` — judul + penulis dibuka langsung dari arXiv.

Membangkitkan banyak rationale per soal lalu memaksa konsistensi antar prediksi lewat KL-divergence
dua arah. Preseden bahwa "banyak jalur penalaran per soal" sudah lama digarap.

## Yang BOLEH diklaim

Bukan tekniknya. Yang belum ada di ketiga paper di atas:

> Ketiganya bekerja pada data berbahasa Inggris. Di setting itu, satu-satunya pertanyaan saat
> menggabungkan teacher adalah **"jawabannya benar atau tidak?"**
>
> Pada bahasa berdaya-rendah muncul sumbu kedua: teacher bisa **benar secara matematis tetapi
> menghasilkan penalaran berbahasa Inggris**, yang kemudian dibuang filter bahasa
> (`to_chatml.py`, `id_only=True`). Karena itu penggabungan harus **sadar-bahasa** — memilih solusi
> Indonesia lebih dulu, bukan sekadar memilih yang benar.

`_solution_rank()` di `src/cot_synthesis/to_chatml.py:61-64` sudah melakukannya (urutan: Indonesia
dulu, lalu terpendek). Jadi tugas kita **mengukur dan menamai**, bukan membangun.

## Desain Skenario 6

| Arm | Isi | Dilatih? |
|---|---|---|
| **U0** | teacher pemenang saja (= arm A / SFT-CoT) | sudah ada, tetap dipakai |
| **U1** | union 2 teacher skor efektif tertinggi | ya, 1 run |
| **U2** | union semua teacher | tidak — metrik data saja |

U2 sengaja tidak dilatih; dipakai sebagai bukti kuantitatif atas peringatan TinyLLM.

### Metrik tanpa training (gratis, langsung setelah bake-off)
- cakupan soal U0 / U1 / U2
- jumlah soal yang **hanya** terpecahkan teacher non-pemenang (mengukur saling melengkapi)
- rasio bahasa Indonesia setelah seleksi
- kemiripan semantik antar solusi terpilih (proxy *knowledge conflict*)

### Metrik dengan training (hanya U1)
pass@1, maj@k, kepatuhan format di holdout — dibanding U0.

## Catatan implementasi

- Union = concat `correct.jsonl` antar teacher + dedup. `to_chatml(best_per_problem=True)` otomatis
  memilih 1 solusi terbaik per soal lintas teacher.
- ID soal cocok lintas teacher karena `problem_id()` memakai indeks global `train_pool`, dan semua
  teacher dijalankan pada file + urutan yang sama.
- Karena hanya 1 solusi disimpan per soal, dataset **tidak** membengkak oleh duplikasi; pertambahan
  murni dari soal yang gagal di teacher pemenang tetapi berhasil di teacher lain.
