# Lembar Konfirmasi 11 False Positive Q1

Judge menandai 20 baris sebagai soal tak-terjawab. 10 cocok anotasi manual lama.
11 sisanya di bawah ini — belum pernah dianotasi manusia.

## Cara mengisi

Untuk tiap baris, jawab SATU pertanyaan: **apakah soal ini rusak menurut kriteria Q1?**

Kriteria Q1 (dari `judge_quality.py:58-67`) — jawab RUSAK kalau soal:
- teks terpotong atau tidak lengkap
- merujuk gambar/tabel/grafik yang tidak disertakan
- merujuk soal lain (mis. 'soal di atas')
- berupa perintah aktivitas atau diskusi, bukan soal hitungan
- teksnya rusak sehingga tidak terbaca

Ganti `[ ]` jadi `[x]` pada baris keputusanmu. Kalau ragu, pilih RAGU — jangan dipaksa.

Kolom 'usul Claude' adalah **hipotesis yang belum divalidasi**. Abaikan kalau tidak setuju.

---

## idx 9

```
SOAL   : Tentukan banyaknya pasangan $(x, y, z)$ jika $H + G + \text{ } = 6$ dengan
a. $1 \le H, G, \text{ } \le 5$
b. $H, G$, dan $\text{ }$ adalah bilangan bulat tak negatif
JAWABAN: 28
```

usul Claude: _variabel ketiga kosong (\text{ })_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Variabel ketiga pada persamaan H+G+__=6 hilang, sehingga domain/identitas z tidak dapat diketahui. Karena ini termasuk teks tidak lengkap/rusak, Q1 terpenuhi.

---

## idx 41

```
SOAL   : Silahkan selesaikan sendiri, jawab soal ini adalah $x \equiv 24 \pmod{49}$.
JAWABAN: $x \equiv 24 \pmod{49}$
```

usul Claude: _bukan soal, sudah memuat jawaban_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: 
Tidak ada pertanyaan/permasalahan yang harus diselesaikan; teks hanya berupa instruksi dan sudah mencantumkan jawaban.
---

## idx 57

```
SOAL   : Tentukan lama waktu untuk mengisi daya baterai yang kosong hingga menjadi 90\% penuh dengan $k = 0{,}02$.
JAWABAN: ≈ $115{,}13$ menit.
```

usul Claude: _rujuk konteks soal sebelumnya_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Rumus/model pengisian baterai tidak tersedia dalam soal ini; nilai jawaban 115,13 menit tidak dapat diturunkan hanya dari informasi yang diberikan.

---

## idx 64

```
SOAL   : Hitung banyaknya juring berhimpit dengan warna sama, di semua $2n$ kemungkinan himpitan.
JAWABAN: 2n
```

usul Claude: _n tak terdefinisi, rujuk soal lain_

- [ ] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [x] RAGU  — jangan dipakai di metrik

catatan: Kemungkinan membutuhkan konteks visual/diagram atau definisi dari soal sebelumnya untuk menentukan apa yang dimaksud dengan juring berhimpit dan warna yang sama.

---

## idx 68

```
SOAL   : sehingga berbentuk barisan. Berapa probabilitas bahwa banyaknya kartu yang dijajarkan dari kiri ke kanan dan ditempatkan pada tempat ke-$i$ akan lebih besar atau sama dengan $i$ untuk setiap $i$ dengan $1 \le i \le 5$?
JAWABAN: \frac{1}{120}
```

usul Claude: _terpotong, mulai di tengah kalimat_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Soal dimulai di tengah kalimat (“sehingga berbentuk barisan”), sehingga konteks dan informasi awal yang diperlukan untuk memahami probabilitas tidak tersedia.

---

## idx 84

```
SOAL   : Mengapa tidak ada apotema yang bersesuaian dengan diameter?
JAWABAN: Karena diameter lingkaran adalah garis melintasi lingkaran, bukan sisi bangun datar, dan apotema didefinisikan untuk bangun datar yang memiliki sisi, bukan lingkaran.
```

usul Claude: _pertanyaan diskusi, bukan hitungan_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Pertanyaan bersifat konseptual dan meminta penjelasan, bukan perhitungan matematis.

---

## idx 129

```
SOAL   : Jika ada soal lain dengan kemiringan 1, bagaimana cara menyelesaikannnya?
JAWABAN: y = x + 1
```

usul Claude: _rujuk soal lain_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Merujuk secara eksplisit pada “soal lain” dan informasi soal yang dirujuk tidak tersedia. Jawaban y=x+1 juga tidak dapat ditentukan hanya dari kemiringan 1.

---

## idx 141

```
SOAL   : Panjang garis bagi sudut $AA'$ juga dapat dihitung. Perhitungan ini diserahkan kepada pembaca sebagai latihan.
JAWABAN: \frac{5\sqrt{7}}{3}
```

usul Claude: _bukan soal ('diserahkan kepada pembaca')_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Bukan soal mandiri; hanya menyatakan bahwa perhitungan panjang garis bagi sudut AA
′
 diserahkan sebagai latihan kepada pembaca, tanpa memberikan instruksi/data soal yang lengkap.

---

## idx 142

```
SOAL   : Nilai untuk $i(6,7)$ adalah ...
JAWABAN: 3
```

usul Claude: _fungsi i tak terdefinisi_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Fungsi i tidak didefinisikan dalam soal, sehingga informasi yang diperlukan untuk menghitung i(6,7) kemungkinan hilang atau berada pada konteks sebelumnya.

---

## idx 195

```
SOAL   : Bentuklah kelompok yang terdiri atas 3-4 siswa. Carilah peta kota yang dilengkapi dengan tempat-tempat penting seperti rumah kalian, tempat ibadah, sekolah, puskesmas, pos kamling, toko, dan lain-lain. Tentukan suatu objek titik asal $(0, 0)$. Gambarkan dalam koordinat Kartesius. Tentukan koordinat titik-titik yang menunjukkan lokasi tempat-tempat penting tersebut. Tentukan koordinat titik-titik rumah kalian. Buat laporan dan paparkan hasilnya.
JAWABAN: \begin{aligned}
&\text{Rumah: } (0, 0) \\
&\text{Tempat Ibadah: } (5, 10) \\
&\text{Sekolah: } (15, 5) \\
&\text{Puskesmas: } (8, 15)
\end{aligned}
```

usul Claude: _perintah aktivitas_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Soal berupa instruksi aktivitas kelompok dan membutuhkan tindakan/observasi dunia nyata (mencari peta dan menentukan lokasi), bukan soal hitungan mandiri.

---

## idx 233

```
SOAL   : Andaikan banyak data genap misal 15, 25, 35, 45, 55, 65, 75, 85. Oleh karena tidak ada data yang berada tepat di tengah, maka kita tentukan dengan menjumlah data keempat dan kelima kemudian dibagi dua, yaitu:
JAWABAN: 50
```

usul Claude: _teks penjelasan, bukan soal_

- [x] RUSAK — masukkan ke Q1_POSITIF
- [ ] WAJAR — judge salah, ini false positive sungguhan
- [ ] RAGU  — jangan dipakai di metrik

catatan: Teks berupa penjelasan cara menentukan median untuk jumlah data genap, bukan soal mandiri. Tidak ada instruksi/pertanyaan yang perlu diselesaikan.

---

## Setelah selesai

Kumpulkan indeks yang kamu tandai RUSAK, mis. `[9, 41, 68, 141, 195]`.
Tempel daftar itu ke prompt sesi berikutnya di bagian
`Indeks FP yang DIKONFIRMASI rusak:`.

Total baris di lembar ini: 11