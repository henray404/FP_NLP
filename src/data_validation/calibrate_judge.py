"""
Kalibrasi LLM judge `judge_quality.py` terhadap 260 baris beranotasi manual.

Pertanyaan yang dijawab modul ini: *"bagaimana Anda tahu filter Anda bekerja?"* — sebelum judge
dilepas ke 2.336 baris penuh, vonisnya diadu dulu dengan label manusia di
`calibration_labels.py`.

Alur:
  1. bangun ulang `easy_clean_v2.jsonl` (`src/preprop/clean_easy_v2.py`, deterministik)
  2. ambil sampel 260 baris seed=7 (`calibration_labels.sampel_kalibrasi`)
  3. jalankan Q1 & Q2 lewat `judge_quality.run_judge` — prompt, parser vonis, dan berkas
     `.progress` yang sama persis dengan produksi. Modul ini TIDAK menulis ulang logika judge;
     kalau prompt di `judge_quality.py` berubah, kalibrasi ini ikut berubah dengan sendirinya.
  4. hitung presisi/recall + interval kepercayaan Wilson

BATAS SAHIH — dibaca sebelum mengutip angkanya:

  * **Q1 hanya punya 10 positif kuat.** Dengan n=10, interval 95% recall lebarnya 28-53 poin
    persen tergantung titik estimasinya (paling lebar di sekitar 0,5; paling sempit di 0 atau
    1) — kira-kira ±25 poin persen di sekitar 0,7. Karena itu modul ini selalu mencetak
    interval Wilson, bukan hanya titik estimasi. Titik estimasi sendirian akan menyesatkan.
  * **Q2 hanya punya 1 positif pasti (+2 ragu).** Itu TIDAK cukup untuk angka apa pun. Presisi
    dan recall Q2 dicetak dengan penanda `TIDAK SAHIH` dan tidak boleh masuk paper sebagai
    klaim performa.
  * **Baris yang tidak ditandai manual adalah label lemah**, artinya "tampak wajar saat disapu",
    bukan "diverifikasi benar". Anotasi manual menyasar baris rusak, bukan menilai 260 baris
    satu per satu. Konsekuensinya: false positive yang terhitung adalah **batas atas** — sebagian
    di antaranya bisa jadi memang rusak tetapi luput saat anotasi. Jadi presisi yang dilaporkan
    adalah **batas bawah**.

Usage:
    python -m src.data_validation.calibrate_judge --self-check          # CPU, vonis palsu
    python -m src.data_validation.calibrate_judge --judge-backend hf    # GPU lokal (1 kartu)
    python -m src.data_validation.calibrate_judge --judge-backend vllm  # sama dgn produksi
    python -m src.data_validation.calibrate_judge --judge-backend api
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from src.data_validation import calibration_labels as labels
from src.data_validation.judge_quality import (
    DEFAULT_API_JUDGE,
    DEFAULT_VLLM_JUDGE,
    Q1_PROMPT,
    Q2_PROMPT,
    _cand_key,
    _is_ya,
    _make_judge_api,
    _make_judge_vllm,
    run_judge,
    stage_a,
)

# Judge default untuk backend `hf` (satu GPU konsumen). SENGAJA lebih kecil dari
# `DEFAULT_VLLM_JUDGE` (7B) karena 7B fp16 butuh ~15 GB — tidak muat di kartu 8 GB. Kalau
# kalibrasi dijalankan dengan model ini, angkanya berlaku untuk model ini, bukan untuk 7B;
# sebutkan modelnya setiap kali mengutip hasil.
DEFAULT_HF_JUDGE = "Qwen/Qwen2.5-1.5B-Instruct"

Z_95 = 1.959963984540054  # kuantil normal dua sisi untuk 95%


# -------------------------------
# Statistik
# -------------------------------

def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Interval kepercayaan Wilson score untuk proporsi k/n.

    Dipakai alih-alih interval normal (Wald) karena n di sini kecil (10) dan proporsinya dekat
    0 atau 1 — dua kondisi yang membuat Wald melenceng, bahkan bisa keluar dari [0, 1].
    n=0 mengembalikan (0.0, 1.0): tidak ada informasi, bukan nol.
    """
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"k={k} harus di dalam 0..n={n}")
    denom = n + z * z
    center = (k + z * z / 2) / denom
    half = (z / denom) * math.sqrt(k * (n - k) / n + z * z / 4)
    return max(0.0, center - half), min(1.0, center + half)


def metrik(prediksi_positif: set[int], acuan_positif: set[int]) -> dict:
    """Presisi/recall + interval Wilson. 'Positif' = baris divonis RUSAK.

    Presisi: dari yang judge tandai rusak, berapa yang memang ditandai manusia — penyebutnya
    jumlah tanda judge. Recall: dari yang manusia tandai, berapa yang judge tangkap —
    penyebutnya jumlah anotasi manual (10 untuk Q1).
    """
    tp = len(prediksi_positif & acuan_positif)
    fp = len(prediksi_positif - acuan_positif)
    fn = len(acuan_positif - prediksi_positif)

    n_presisi = tp + fp
    n_recall = tp + fn
    presisi = tp / n_presisi if n_presisi else float("nan")
    recall = tp / n_recall if n_recall else float("nan")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "presisi": presisi, "presisi_n": n_presisi, "presisi_ci": wilson_ci(tp, n_presisi),
        "recall": recall, "recall_n": n_recall, "recall_ci": wilson_ci(tp, n_recall),
    }


def _fmt(m: dict, nama: str, *, sahih: bool = True) -> list[str]:
    """Baris-baris laporan untuk satu pertanyaan judge."""
    tag = "" if sahih else "   <== TIDAK SAHIH, n positif terlalu kecil"
    lo_p, hi_p = m["presisi_ci"]
    lo_r, hi_r = m["recall_ci"]
    lebar_r = (hi_r - lo_r) * 100
    return [
        f"{nama}{tag}",
        f"  TP {m['tp']:3d}   FP {m['fp']:3d}   FN {m['fn']:3d}",
        f"  presisi {m['presisi']:.3f}  (n={m['presisi_n']:3d})  CI95 Wilson "
        f"[{lo_p:.3f}, {hi_p:.3f}]   <- BATAS BAWAH (lihat catatan label lemah)",
        f"  recall  {m['recall']:.3f}  (n={m['recall_n']:3d})  CI95 Wilson "
        f"[{lo_r:.3f}, {hi_r:.3f}]   <- lebar {lebar_r:.0f} poin persen",
    ]


CATATAN_LABEL_LEMAH = (
    "CATATAN LABEL: baris yang tidak ditandai manual = label LEMAH ('dianggap wajar saat "
    "disapu'), bukan label negatif terverifikasi. Maka FP yang terhitung adalah BATAS ATAS "
    "dan presisi yang dilaporkan adalah BATAS BAWAH."
)


# -------------------------------
# Judge
# -------------------------------

def _make_judge_hf(model: str, batch_ukuran_maks: int = 8):
    """Judge lewat transformers di satu GPU lokal — jalan keluar kalau vLLM tak terpasang.

    Greedy (`do_sample=False`, 4 token) supaya sepadan dengan `SamplingParams(temperature=0.0,
    max_tokens=4)` di backend vLLM. Vonis tetap dibaca `_is_ya` dari `judge_quality.py`.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, device_map="cuda:0")
    llm.eval()

    def judge_batch(prompts: list[str]) -> list[bool]:
        hasil: list[bool] = []
        for mulai in range(0, len(prompts), batch_ukuran_maks):
            potongan = prompts[mulai:mulai + batch_ukuran_maks]
            teks = [tok.apply_chat_template([{"role": "user", "content": p}],
                                            tokenize=False, add_generation_prompt=True)
                    for p in potongan]
            enc = tok(teks, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(llm.device)
            with torch.no_grad():
                out = llm.generate(**enc, max_new_tokens=4, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            baru = out[:, enc["input_ids"].shape[1]:]
            hasil += [_is_ya(t) for t in tok.batch_decode(baru, skip_special_tokens=True)]
        return hasil

    return judge_batch


def buat_judge(backend: str, model: str | None, tensor_parallel_size: int = 1):
    """Kembalikan (fungsi_judge, nama_model) sesuai backend."""
    if backend == "hf":
        nama = model or DEFAULT_HF_JUDGE
        return _make_judge_hf(nama), nama
    if backend == "vllm":
        nama = model or DEFAULT_VLLM_JUDGE
        return _make_judge_vllm(nama, tensor_parallel_size), nama
    if backend == "api":
        nama = model or DEFAULT_API_JUDGE
        return _make_judge_api(nama), nama
    raise ValueError(f"backend tak dikenal: {backend}")


# -------------------------------
# Orkestrasi
# -------------------------------

def vonis_ke_posisi(sampel: list[dict], done: dict[str, bool], qtag: str) -> dict[int, bool]:
    """Petakan hasil `run_judge` (berkunci `_idx` berkas) ke posisi 0..259 di sampel.

    Label manual di `calibration_labels.py` 0-indexed terhadap sampel, bukan terhadap berkas —
    pemetaan ini yang menjembatani keduanya.
    """
    return {pos: done[_cand_key(r, qtag)]
            for pos, r in enumerate(sampel) if _cand_key(r, qtag) in done}


def evaluasi(sampel: list[dict], q1: dict[int, bool], q2: dict[int, bool],
             *, sertakan_ragu: bool = False) -> dict:
    """Bandingkan vonis judge dengan label manual. Vonis TIDAK (=rusak) dihitung positif."""
    rusak_q1 = {pos for pos, ok in q1.items() if not ok}
    rusak_q2 = {pos for pos, ok in q2.items() if not ok}

    acuan_q1 = set(labels.Q1_POSITIF)
    acuan_q2 = set(labels.q2_positif(sertakan_ragu=sertakan_ragu))

    return {
        "n_sampel": len(sampel),
        "n_dinilai_q1": len(q1),
        "n_dinilai_q2": len(q2),
        "n_ditandai_judge_q1": len(rusak_q1),
        "n_ditandai_judge_q2": len(rusak_q2),
        "q1": metrik(rusak_q1, acuan_q1),
        "q2": metrik(rusak_q2, acuan_q2),
        "q2_sahih": len(acuan_q2) >= 5,   # 1-3 positif tidak akan pernah lolos ambang ini
        "q2_n_acuan": len(acuan_q2),
        "meleset_q1": sorted(acuan_q1 - rusak_q1),
        "tertangkap_q1": sorted(acuan_q1 & rusak_q1),
    }


def ringkas_stage_a(sampel: list[dict]) -> dict:
    """Tahap A (CPU) pada sampel yang sama — konteks: sebagian rusak sudah tersaring regex."""
    alasan: dict[int, str] = {}
    for pos, r in enumerate(sampel):
        sebab = stage_a(r)
        if sebab:
            alasan[pos] = sebab
    acuan_q1 = set(labels.Q1_POSITIF)
    return {
        "n_drop": len(alasan),
        "tumpang_tindih_q1": sorted(set(alasan) & acuan_q1),
        "alasan": alasan,
    }


def laporan(hasil: dict, stage: dict, model: str) -> str:
    """Laporan teks siap cetak — juga dipakai notebook 08 sebagai artefak paper."""
    baris = [
        "KALIBRASI JUDGE vs ANOTASI MANUAL",
        "=" * 62,
        f"model judge     : {model}",
        f"sampel          : {hasil['n_sampel']} baris (seed={labels.SEED})",
        f"dinilai Q1/Q2   : {hasil['n_dinilai_q1']} / {hasil['n_dinilai_q2']}",
        f"ditandai judge  : Q1 {hasil['n_ditandai_judge_q1']}, Q2 {hasil['n_ditandai_judge_q2']}",
        "",
        f"Tahap A (regex, sebelum judge): {stage['n_drop']} baris terbuang, "
        f"{len(stage['tumpang_tindih_q1'])} di antaranya juga positif manual Q1",
        "",
    ]
    baris += _fmt(hasil["q1"], f"Q1 keterjawaban soal — {hasil['q1']['recall_n']} positif manual")
    baris += [
        f"  tertangkap: {hasil['tertangkap_q1']}",
        f"  meleset   : {hasil['meleset_q1']}",
        "",
    ]
    baris += _fmt(hasil["q2"], f"Q2 kelayakan bentuk gold — {hasil['q2_n_acuan']} positif manual",
                  sahih=hasil["q2_sahih"])
    if not hasil["q2_sahih"]:
        baris += [
            "  PERINGATAN: acuan Q2 hanya "
            f"{hasil['q2_n_acuan']} baris positif. Angka presisi/recall di atas TIDAK layak "
            "dikutip sebagai performa Q2 — n-nya terlalu kecil untuk membedakan judge bagus "
            "dari judge asal. Perlu anotasi Q2 tambahan sebelum klaim apa pun.",
        ]
    baris += [
        "",
        CATATAN_LABEL_LEMAH,
        "CATATAN n KECIL: recall Q1 dihitung dari 10 positif. Interval Wilson di atas itulah "
        "hasil sesungguhnya; titik estimasinya sendiri tidak bermakna.",
    ]
    return "\n".join(baris)


def run(input_v2: Path, out_dir: Path, *, judge_backend: str = "hf",
        judge_model: str | None = None, tensor_parallel_size: int = 1,
        batch_size: int = 16, regenerasi: bool = True,
        sertakan_ragu: bool = False) -> dict:
    """Pipa penuh: regenerasi v2 -> sampel 260 -> judge Q1/Q2 -> metrik + laporan."""
    from src.preprop import clean_easy_v2

    if regenerasi or not input_v2.exists():
        stats_v2 = clean_easy_v2.run(clean_easy_v2.DEFAULT_INPUT, input_v2)
        print(f"v2 dibangun ulang: {stats_v2['input']} -> {stats_v2['bersih']} baris")

    sampel = labels.sampel_kalibrasi(input_v2)
    out_dir.mkdir(parents=True, exist_ok=True)

    judge, model = buat_judge(judge_backend, judge_model, tensor_parallel_size)
    done_q1 = run_judge(sampel, judge, Q1_PROMPT, "Q1", out_dir / "kalibrasi_q1.progress",
                        batch_size, {"soal": "soal"})
    done_q2 = run_judge(sampel, judge, Q2_PROMPT, "Q2", out_dir / "kalibrasi_q2.progress",
                        batch_size, {"soal": "soal", "jawaban": "jawaban"})

    hasil = evaluasi(sampel,
                     vonis_ke_posisi(sampel, done_q1, "Q1"),
                     vonis_ke_posisi(sampel, done_q2, "Q2"),
                     sertakan_ragu=sertakan_ragu)
    stage = ringkas_stage_a(sampel)
    teks = laporan(hasil, stage, model)
    print(teks)

    (out_dir / "kalibrasi_judge_report.txt").write_text(teks, encoding="utf-8")
    (out_dir / "kalibrasi_judge_report.json").write_text(
        json.dumps({"model": model, "hasil": hasil, "stage_a": stage["n_drop"]},
                   ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    return hasil


# -------------------------------
# Self-check (CPU, tanpa judge)
# -------------------------------

def self_check() -> None:
    """Verifikasi aritmetika presisi/recall/Wilson dengan vonis PALSU. Tidak butuh GPU."""
    # Wilson dibandingkan dengan nilai terbitan yang sudah dikenal (z=1,96).
    lo, hi = wilson_ci(5, 10)
    assert abs(lo - 0.2366) < 5e-4 and abs(hi - 0.7634) < 5e-4, (lo, hi)
    lo, hi = wilson_ci(10, 10)
    assert abs(lo - 0.7225) < 5e-4 and hi == 1.0, (lo, hi)
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and abs(hi - 0.2775) < 5e-4, (lo, hi)
    assert wilson_ci(0, 0) == (0.0, 1.0)              # n=0 -> tak ada informasi, bukan nol
    lo, hi = wilson_ci(7, 10)
    assert lo < 0.7 < hi, "interval wajib memuat titik estimasi"
    # Interval menyempit saat n naik pada proporsi yang sama.
    lebar_100 = wilson_ci(50, 100)[1] - wilson_ci(50, 100)[0]
    lebar_10 = wilson_ci(5, 10)[1] - wilson_ci(5, 10)[0]
    assert lebar_100 < lebar_10, (lebar_100, lebar_10)

    # Metrik: judge sempurna.
    m = metrik({1, 2, 3}, {1, 2, 3})
    assert (m["tp"], m["fp"], m["fn"]) == (3, 0, 0)
    assert m["presisi"] == 1.0 and m["recall"] == 1.0
    assert m["presisi_ci"][1] == 1.0 and m["presisi_ci"][0] < 1.0, "n=3 tak boleh CI sempit"

    # Metrik: judge menangkap 2 dari 4, dengan 3 tanda meleset.
    m = metrik({1, 2, 8, 9, 10}, {1, 2, 3, 4})
    assert (m["tp"], m["fp"], m["fn"]) == (2, 3, 2)
    assert abs(m["presisi"] - 2 / 5) < 1e-12 and abs(m["recall"] - 2 / 4) < 1e-12
    assert m["presisi_n"] == 5 and m["recall_n"] == 4

    # Metrik: judge diam total -> presisi NaN (0/0), recall 0, bukan crash.
    m = metrik(set(), {1, 2})
    assert m["presisi"] != m["presisi"], "presisi 0/0 harus NaN"
    assert m["recall"] == 0.0 and m["presisi_ci"] == (0.0, 1.0)

    # Jalur penuh dengan vonis palsu: judge menangkap 7 dari 10 positif Q1, plus 5 salah tanda.
    sampel_palsu = [{"soal": f"soal ke-{i} yang cukup panjang untuk lolos", "jawaban": str(i),
                     "_idx": i} for i in range(labels.UKURAN_SAMPEL)]
    tertangkap = set(labels.Q1_POSITIF[:7])
    salah_tanda = {200, 201, 202, 204, 205}
    assert not (salah_tanda & set(labels.Q1_POSITIF)), "kontrol tes bocor ke acuan"
    q1_palsu = {pos: pos not in (tertangkap | salah_tanda)
                for pos in range(labels.UKURAN_SAMPEL)}
    q2_palsu = {pos: pos != labels.Q2_POSITIF[0] for pos in range(labels.UKURAN_SAMPEL)}

    hasil = evaluasi(sampel_palsu, q1_palsu, q2_palsu)
    assert hasil["q1"]["tp"] == 7 and hasil["q1"]["fn"] == 3 and hasil["q1"]["fp"] == 5
    assert abs(hasil["q1"]["recall"] - 0.7) < 1e-12
    assert abs(hasil["q1"]["presisi"] - 7 / 12) < 1e-12
    lo, hi = hasil["q1"]["recall_ci"]
    # 7/10 -> Wilson [0,397, 0,892], lebar 0,495. Pagar 0,45 menjaga klaim "interval lebar"
    # tetap benar tanpa mengunci angka pastinya.
    assert (hi - lo) > 0.45, f"CI recall n=10 wajib lebar, dapat {(hi - lo):.3f}"
    assert hasil["q2"]["tp"] == 1 and hasil["q2_sahih"] is False, "Q2 tak boleh ditandai sahih"
    assert len(hasil["meleset_q1"]) == 3

    teks = laporan(hasil, {"n_drop": 12, "tumpang_tindih_q1": [53, 108]}, "model-palsu")
    assert "TIDAK SAHIH" in teks and "BATAS BAWAH" in teks and "label LEMAH" in teks

    print("self-check OK: 6 kasus Wilson, 4 kasus metrik, 1 jalur evaluasi penuh (vonis palsu)")
    print(f"  contoh: recall Q1 7/10 = 0,700 dengan CI95 Wilson "
          f"[{lo:.3f}, {hi:.3f}] -> lebar {(hi - lo) * 100:.0f} poin persen")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kalibrasi judge terhadap anotasi manual 260 baris")
    ap.add_argument("--input", default="data/Final/easy_clean_v2.jsonl")
    ap.add_argument("--out-dir", default="reports/kalibrasi_judge")
    ap.add_argument("--judge-backend", choices=["hf", "vllm", "api"], default="hf")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-regen", action="store_true",
                    help="pakai berkas v2 yang ada, jangan bangun ulang")
    ap.add_argument("--sertakan-ragu", action="store_true",
                    help="masukkan kandidat Q2 yang belum pasti sebagai acuan (batas atas)")
    ap.add_argument("--self-check", action="store_true",
                    help="verifikasi aritmetika dengan vonis palsu, tanpa judge")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    run(Path(args.input), Path(args.out_dir), judge_backend=args.judge_backend,
        judge_model=args.judge_model, tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size, regenerasi=not args.no_regen,
        sertakan_ragu=args.sertakan_ragu)


if __name__ == "__main__":
    main()
