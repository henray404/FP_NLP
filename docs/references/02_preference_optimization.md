# Preference Optimization: DPO & KTO (untuk Skenario 7)

**Intinya: DPO dan KTO bukan temuan kita. Yang bisa diklaim adalah ALASAN memilih KTO — karena
struktur limbah pipeline kita cocok dengan bentuk data yang KTO butuhkan.**

## Rujukan wajib

### DPO — paper dasar
> Rafailov, R., dkk. "Direct Preference Optimization: Your Language Model is Secretly a Reward
> Model." *Advances in Neural Information Processing Systems (NeurIPS)*, 2023. arXiv:2305.18290.

`[Terverifikasi]` judul/venue/ID. Melatih model menyukai respons *chosen* dibanding *rejected*
langsung dari pasangan preferensi, tanpa reward model terpisah seperti PPO.

### KTO — paper dasar
> "KTO: Model Alignment as Prospect Theoretic Optimization." arXiv:2402.01306, 2024.

`[Terverifikasi]` judul + ID. `[Perlu cek]` daftar penulis — belum dibuka ke halaman aslinya.

Berbasis *prospect theory* Kahneman–Tversky. Beda utama dari DPO: **hanya butuh label biner
(layak / tidak layak) per respons, tidak perlu berpasangan.**

### RS-DPO — pola pipeline paling dekat
> Khaki, S., Li, J., Ma, L., Yang, L., Ramachandra, P. "RS-DPO: A Hybrid Rejection Sampling and
> Direct Preference Optimization Method for Alignment of Large Language Models." *Findings of the
> ACL: NAACL 2024*, hal. 1665–1680. arXiv:2402.10038.

`[Terverifikasi]` — BibTeX lengkap terlihat di ACL Anthology.

Menggabungkan rejection sampling + DPO: bangun model SFT, generate banyak respons, pilih pasangan
kontras berdasar distribusi reward. Persis pola pipeline kita → rujukan metodologis terkuat.

### Self-Explore — bukti DPO efektif khusus penalaran matematis
> Hwang, H., Kim, D., Kim, S., Ye, S., Seo, M. "Self-Explore: Enhancing Mathematical Reasoning in
> Language Models with Fine-grained Rewards." arXiv:2404.10346, 2024.

`[Terverifikasi]` judul + penulis. Berguna menjawab "kenapa DPO, bukan sekadar SFT lebih banyak" di
Tinjauan Pustaka.

### Varian lain yang bisa disebut sekilas
M-DPO (multi-turn math), Step-controlled DPO, rStar-Math, TINA, Phi-4-mini-reasoning, KEPO.
`[Perlu cek]` — semuanya dari ringkasan pencarian, belum dibuka satu pun.
**Jangan sitir sebelum diverifikasi.**

## Kenapa KTO cocok untuk pipeline kita — argumen utama

Angka dari Tabel X paper lama (teacher DeepSeek-R1-Distill-Qwen-7B):

```
26.068 kandidat  →   9.666 benar   (retensi 37,08%)
                 →  16.402 salah
cakupan soal 66,61%  →  33,4% soal TIDAK punya satu pun solusi benar
```

Konsekuensi per metode:

| Kelompok soal | SFT | DPO | KTO |
|---|---|---|---|
| punya ≥1 benar **dan** ≥1 salah | dipakai | dipakai | dipakai |
| semua kandidat benar (8/8) | dipakai | **terbuang** (tak ada rejected) | dipakai (positif) |
| semua kandidat salah (0/8) — **33,4% soal** | **terbuang** | **terbuang** | **dipakai (negatif)** |

**Klaim yang bisa dibawa ke paper:**

> KTO memanfaatkan ~33% soal yang selama ini dibuang total oleh rejection sampling. Ini menyerang
> langsung bottleneck yang dilaporkan penelitian ini sendiri (cakupan 66,61%), bukan sekadar
> menambahkan satu metode alignment.

## Rasio data & hiperparameter

Rasio kita ≈ **9.666 : 16.402 = 1 : 1,7** (layak : tidak layak) — timpang ringan.

Panduan dari ringkasan literatur KTO `[Perlu cek — ringkasan sekunder]`:
- Default `λ_D = λ_U = 1`
- Untuk data timpang, setel λ agar **rasio efektif mendekati 1:1 sampai 4:3**
- KTO dilaporkan sanggup menangani ketimpangan hingga 1:10 dengan penyetelan λ
- Efeknya **asimetris**: mengurangi contoh *desirable* jauh lebih merusak daripada mengurangi
  *undesirable*

Titik awal untuk kita: `λ_D ≈ 1.7`, `λ_U = 1.0`.

## Angka yang TIDAK boleh masuk paper tanpa verifikasi

- "KTO +13,5 poin di atas DPO pada GSM8K" — dari ringkasan sekunder, **belum dibuka ke paper
  aslinya**. Kalau mau dipakai: buka arXiv:2402.01306, kutip tabelnya langsung.

## Konstruksi data — nol generasi baru

Diturunkan dari dua file yang sudah ada:
- **chosen / desirable** = baris di `data/cot/correct_<teacher>.jsonl`
- **rejected / undesirable** = baris di `data/cot/candidates_<teacher>.jsonl` yang pasangan
  `(id, candidate_idx)`-nya tidak ada di `correct`

Format keluaran (TRL):
- DPO → `{"prompt": "...", "chosen": "...", "rejected": "..."}`
- KTO → `{"prompt": "...", "completion": "...", "label": true}`

## Desain Skenario 7

| Arm | Training |
|---|---|
| SFT-CoT (arm A) | baseline |
| SFT-CoT + DPO | tahap kedua di atas arm A |
| SFT-CoT + KTO | tahap kedua di atas arm A |

Keduanya **bertumpuk di atas arm A**, bukan training dari nol.

## Risiko

Item **paling berisiko** di seluruh rencana: TRL `DPOTrainer`/`KTOTrainer` belum pernah dipakai di
repo ini. Kerjakan **paling akhir**, setelah Skenario 1–5 terkunci, supaya kegagalan di sini tidak
menenggelamkan paper.
