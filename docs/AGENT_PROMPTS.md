# Prompt & Prosedur Menjalankan 12 Agen

Dipakai bersama issue #1–#12 di `henray404/FP_NLP`.

**Aturan operasional:** konteks dikosongkan setiap ganti agen. Konsekuensinya:

1. **Thread issue = memori.** Tiap agen wajib menutup dengan komentar berisi hasil + angka. Agen berikutnya membaca itu, bukan mengingatnya.
2. **Prasyarat diverifikasi dari filesystem, bukan diasumsikan.** Agen yang baru lahir tidak tahu apakah agen sebelumnya benar-benar selesai.
3. **Branch di-merge sebelum agen berikutnya mulai.** Kalau tidak, cek prasyarat akan gagal padahal pekerjaannya sudah ada.

---

## Prompt universal

Ganti `<N>` dengan nomor issue, dan `<PRASYARAT>` dengan baris dari tabel di bawah.

```
Kamu mengerjakan SATU issue di repo FP_NLP. Konteksmu kosong — semua yang kamu
butuhkan ada di issue dan di dalam repo, bukan di ingatanmu.

Repo  : D:\Main Storage\Vscode\Coursework\FP_NLP\FP_NLP
Issue : #<N>

LANGKAH WAJIB, berurutan:

1. Jalankan `gh issue view <N>`. Baca penuh. Itu spesifikasimu, lengkap dengan
   acceptance criteria.

2. Baris pertama body memuat "Blokir: #X". Baca komentar di issue #X — hasil dan
   angka dari agen sebelumnya ada di sana.

3. VERIFIKASI PRASYARAT dari filesystem. Jangan berasumsi:
   <PRASYARAT>
   Kalau belum terpenuhi: JANGAN mulai bekerja. Tulis komentar di issue #<N>
   menyebutkan apa yang kurang, lalu berhenti.

4. Buat branch `issue-<N>` lalu kerjakan.

5. Uji. Setiap modul src/ baru wajib punya `--self-check` atau `--demo` yang
   jalan di CPU tanpa GPU.

6. Commit ke branch tersebut. JANGAN push ke main.

7. Tulis komentar di issue #<N> berisi:
   - apa yang dikerjakan
   - angka hasil (jumlah baris, metrik, apa pun yang terukur)
   - file yang dibuat/diubah
   - APA YANG HARUS DIKETAHUI AGEN BERIKUTNYA
   Ini satu-satunya cara kamu mewariskan konteks. Tanpa ini pekerjaanmu hilang.

8. Centang acceptance criteria yang BENAR-BENAR terpenuhi. Biarkan kosong yang
   tidak. Jangan centang demi terlihat rapi.

ATURAN REPO:
- Logika di src/, notebook tipis — hanya memanggil dan memvisualkan. Contoh:
  notebooks/cot/cot_pipeline_a6000.ipynb, 12 sel untuk satu pipeline penuh.
- Docstring dan komentar Bahasa Indonesia. Ikuti gaya
  src/data_validation/judge_quality.py.
- data/* masuk .gitignore kecuali data/Final/*_v3.*. Jangan menaruh sesuatu yang
  harus bertahan di dalam data/ tanpa menambah pengecualian .gitignore.
- Jangan mengubah file di luar lingkup issue-mu.

KALAU MENEMUKAN MASALAH DI LUAR LINGKUP: jangan perbaiki diam-diam. Buka issue
baru atau tulis di komentar. Perbaikan senyap membuat agen lain bekerja di atas
asumsi yang salah.

JUJUR SOAL HASIL: kalau tes gagal, tulis gagal beserta outputnya. Kalau sebagian
tidak selesai, sebutkan bagian mana dan kenapa. Jangan melaporkan selesai kalau
belum.
```

---

## Prasyarat per agen

| # | `<PRASYARAT>` |
|---|---|
| 1 | `src/data_validation/judge_quality.py` ada. Tidak ada prasyarat lain — ini tiket paling awal. |
| 2 | `src/data_validation/calibration_labels.py` ada dan `--self-check`-nya lolos. |
| 3 | `notebooks/revisi/08_kalibrasi_judge.ipynb` ada DAN komentar issue #2 memuat angka presisi/recall Q1. Tanpa angka itu kamu tidak tahu apakah judge layak dilepas. |
| 4 | `data/Final/easy_clean_v4.jsonl` dan `numglue_clean_v4.jsonl` ada, jumlah barisnya tercatat di komentar issue #3. |
| 5 | File holdout dan `train_pool` ada, jumlahnya tercatat di komentar issue #4. |
| 6 | Komentar issue #5 menyebut satu teacher pemenang beserta skor efektifnya. |
| 7 | `data/sft/train/cot.jsonl` dan `nocot.jsonl` ada, DAN komentar issue #6 mengonfirmasi kedua file punya himpunan soal identik. |
| 8 | Adapter di `outputs/cot_3b/` dan `outputs/nocot_3b/` ada. |
| 9 | Komentar issue #5 menyebut DUA teacher teratas, dan `correct_<teacher>.jsonl` untuk keduanya ada. |
| 10 | `candidates_<teacher>.jsonl` dan `correct_<teacher>.jsonl` ada untuk teacher pemenang. |
| 11 | `dpo.jsonl` dan `kto.jsonl` ada, DAN komentar issue #10 menyebut λ_D hasil hitungan. Jangan pakai angka 1,7 dari dokumen. |
| 12 | Tidak ada. Bisa dijalankan kapan saja. |

---

## Urutan eksekusi

Label `siap-jalan` menandai yang tidak diblokir.

```
Putaran 1   agen #1  +  agen #12       <- paralel, dua-duanya CPU, tak bersinggungan file
Putaran 2   agen #2
Putaran 3   agen #3      GPU
Putaran 4   agen #4                    <- GERBANG: setelah ini data BEKU
Putaran 5   agen #5      GPU
Putaran 6   agen #6      GPU, 4-8 jam  <- BOTTLENECK
Putaran 7   agen #7      GPU
Putaran 8   agen #8      GPU           <== PAPER SUDAH UTUH DI SINI
Putaran 9   agen #9  +  agen #10       <- paralel: #9 GPU, #10 CPU
Putaran 10  agen #11     GPU
```

Hanya dua titik yang benar-benar paralel: putaran 1 dan putaran 9. Sisanya berurutan
karena tiap tahap memakan keluaran tahap sebelumnya.

### Antara dua putaran, lakukan ini

```bash
# 1. pastikan agen menutup dengan komentar
gh issue view <N> --comments

# 2. periksa hasilnya sebelum merge
git diff main..issue-<N> --stat

# 3. merge supaya agen berikutnya lolos cek prasyarat
git checkout main && git merge issue-<N>

# 4. tutup issue
gh issue close <N>

# 5. buka label agen berikutnya
gh issue edit <M> --remove-label menunggu --add-label siap-jalan
```

Langkah 3 tidak boleh dilewat. Agen berikutnya memeriksa filesystem; kalau branch
belum di-merge, ia akan menyimpulkan prasyaratnya belum ada lalu berhenti.

---

## Kalau agen berhenti karena prasyarat kurang

Itu perilaku yang benar, bukan kegagalan. Periksa:

1. Branch agen sebelumnya sudah di-merge?
2. Agen sebelumnya sudah menulis komentar berisi angka?
3. Kalau agen sebelumnya berhenti di tengah — baca komentarnya, perbaiki
   penyebabnya, jalankan ulang agen itu dengan prompt yang sama. Prompt-nya
   idempoten: langkah 1-3 hanya membaca.
