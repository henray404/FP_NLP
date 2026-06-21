# Setup Inference vLLM — S3 CoT vs non-CoT (run di PC server)

Tujuan: jalankan `s3_cot_vs_noncot.ipynb` di PC server pakai **vLLM** (bukan HF
`model.generate`) supaya inference jauh lebih cepat.

Beban kerja: 600 soal (easy 300 + numglue 300) × 5 sampel × 2 model (CoT, nonCoT)
= **6000 generasi**. Di vLLM (PagedAttention + continuous batching) ini selesai
hitungan menit, bukan jam.

---

## 1. Yang perlu disiapin di server

| Item | Detail | Cek |
|------|--------|-----|
| GPU NVIDIA | `nvidia-smi` jalan, catat nama + VRAM | `nvidia-smi` |
| OS Linux | vLLM resmi cuma Linux | `uname -a` |
| Python 3.9–3.12 | venv terpisah | `python --version` |
| **Adapter LoRA** | folder `adapter_cot_1.5b/` + `adapter_nocot_1.5b/`, tiap-tiap isi `adapter_config.json` + `adapter_model.safetensors` | — |
| **Test set** | `data/sft/test/easy_test.jsonl` + `numglue_test.jsonl` (600 soal) — **`data/` TIDAK di git**, copy manual | — |
| Base model | `Qwen/Qwen2.5-1.5B` auto-download dari HF (~3 GB) — butuh internet / cache | — |
| Rank adapter | buka `adapter_config.json`, lihat field `r` (mis. 16/32/64) → set `max_lora_rank` | — |

> Yang HARUS dicopy manual ke server: **2 folder adapter** + **2 file test jsonl**.
> Repo code (`src/`) ikut `git clone`. `data/` & model TIDAK ikut git.

Saran taruh adapter di `models/`:
```
FP_NLP/
  models/adapter_cot_1.5b/{adapter_config.json, adapter_model.safetensors}
  models/adapter_nocot_1.5b/{adapter_config.json, adapter_model.safetensors}
  data/sft/test/{easy_test.jsonl, numglue_test.jsonl}
```

---

## 2. Install environment

```bash
git clone https://github.com/henray404/FP_NLP.git
cd FP_NLP
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -U vllm        # narik torch + CUDA runtime otomatis

# verifikasi GPU kebaca:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

**Kalau GPU server = RTX 50xx / Blackwell (sm_120):** `pip install vllm` default mungkin
torch-nya belum support sm_120 → error `sm_120 not compatible`. Perlu build CUDA 12.8
(torch cu128). Ikut petunjuk Blackwell di docs.vllm.ai (GPU install). GPU server lama
(Ampere/Ada: 30xx/40xx/A100/T4) → `pip install vllm` biasa cukup.

> `peft` TIDAK diperlukan lagi — vLLM load adapter LoRA sendiri lewat `LoRARequest`.

---

## 3. Perubahan notebook `s3_cot_vs_noncot.ipynb`

Notebook sekarang dirancang buat **Kaggle + HF generate**. 4 cell perlu diganti.
Yang lain (markdown, scoring, save) tetap.

### Cell install (yang `!pip install peft ... torchao`) → ganti jadi:
```python
# vLLM sudah diinstall di venv (lihat docs/SETUP_INFERENCE_VLLM.md). Tak perlu peft.
```
(atau kosongkan; jangan jalankan uninstall torchao — itu hack khusus jalur HF/peft.)

### Cell repo (`REPO = '/kaggle/working/FP_NLP'` …) → ganti jadi:
```python
import os, sys
from pathlib import Path
REPO = os.path.abspath('.')        # notebook dijalankan dari dalam repo di server
sys.path.insert(0, REPO); os.chdir(REPO)
print('repo:', REPO)
```

### Cell config (yang `find_adapter('...')` glob `/kaggle/input`) → ganti jadi:
```python
from pathlib import Path
BASE = 'Qwen/Qwen2.5-1.5B'
ADAPTER_COT   = 'models/adapter_cot_1.5b'      # <-- sesuaikan path server
ADAPTER_NOCOT = 'models/adapter_nocot_1.5b'
assert Path(ADAPTER_COT, 'adapter_config.json').exists(), ADAPTER_COT
assert Path(ADAPTER_NOCOT, 'adapter_config.json').exists(), ADAPTER_NOCOT

SET_PATHS = {'numglue': 'data/sft/test/numglue_test.jsonl',
             'easy':    'data/sft/test/easy_test.jsonl'}
print('CoT:', ADAPTER_COT, '| nonCoT:', ADAPTER_NOCOT)
print('sets:', SET_PATHS)
```

### Cell eval (yang import `eval_specs_sampling`) → ganti import + 1 baris call:
```python
from src.eval.scenario_eval import load_sets
from src.eval.vllm_eval import eval_specs_sampling_vllm     # <-- backend vLLM
from src.eval.sampling_metrics import render_tables
import json
from pathlib import Path

sets = load_sets(SET_PATHS)
specs = {'CoT':    {'model': BASE, 'adapter': ADAPTER_COT},
         'nonCoT': {'model': BASE, 'adapter': ADAPTER_NOCOT}}
METRICS = ['pass@1', 'pass@2', 'pass@3', 'maj@3', 'maj@5']

results = eval_specs_sampling_vllm(
    specs, sets, n_samples=5, ks_pass=(1, 2, 3), ks_maj=(3, 5),
    temperature=0.7, top_p=0.95, max_new_tokens=2048,
    max_lora_rank=64,        # <-- HARUS >= r di adapter_config.json
)

print('\n=== SKENARIO 3: CoT vs non-CoT (sampling, vLLM) ===')
print(render_tables(results, METRICS))

print('\ndelta (CoT - nonCoT):')
for m in METRICS:
    for s in sorted(sets):
        c = results['CoT'][s][m]; n = results['nonCoT'][s][m]
        print(f'  {m:7} {s:10} CoT={c:.3f} nonCoT={n:.3f} delta={c-n:+.3f}')

Path('data/eval').mkdir(parents=True, exist_ok=True)
Path('data/eval/s3_cot_vs_noncot.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print('\nsummary -> data/eval/s3_cot_vs_noncot.json')
```

> Modul baru `src/eval/vllm_eval.py` **sudah ada di repo** (commit + push dulu dari lokal
> biar ikut `git clone` di server). Output `results` formatnya identik dgn versi HF →
> scoring/tabel/save tak berubah.

---

## 4. Jalankan

```bash
jupyter notebook    # atau jupyter lab, lalu run cell berurutan
# atau headless: jupyter nbconvert --to notebook --execute s3_cot_vs_noncot.ipynb
```

**Smoke test dulu** (hemat waktu): sebelum 600 soal penuh, coba 2 soal:
```python
mini = {k: v[:2] for k, v in sets.items()}
_ = eval_specs_sampling_vllm(specs, mini, n_samples=5, max_lora_rank=64)
```
Kalau lolos tanpa error → lanjut full.

### Estimasi waktu (6000 generasi, kasar — tergantung panjang output)
| GPU | Estimasi |
|-----|----------|
| A100 / H100 | ~2–4 min |
| RTX 4090 / 3090 | ~3–6 min |
| RTX 50xx (Blackwell) | ~3–6 min (setelah build cu128 beres) |
| T4 / GPU lama 16GB | ~10–20 min |

Plus load base + adapter ~1 menit di awal.

---

## 5. Troubleshooting

| Gejala | Fix |
|--------|-----|
| `CUDA out of memory` | turunkan `gpu_memory_utilization=0.80`, atau `max_model_len=3072`; jangan turunkan `n_samples` di bawah 5 (butuh maj@5) |
| `max_lora_rank too small` / rank mismatch | set `max_lora_rank` = `r` di `adapter_config.json` (atau lebih besar) |
| `sm_120 not compatible` (RTX 50xx) | install torch/vLLM build CUDA 12.8 (lihat §2) |
| GPU tak support bf16 (lama) | tambah arg `dtype='float16'` di call `eval_specs_sampling_vllm` |
| base lama download | set `export HF_HOME=/path/cache` atau pre-download `huggingface-cli download Qwen/Qwen2.5-1.5B` |
| format_ok rendah / `\boxed{}` jarang | normal kalau model belum kuat; cek prompt = `src/eval/skenario4_eval.PROMPT` |

---

## 6. Ringkasan perubahan

- **Baru:** `src/eval/vllm_eval.py` — `eval_specs_sampling_vllm()`, drop-in pengganti
  `sample_eval.eval_specs_sampling()`, output identik, backend vLLM. Base di-load
  sekali; CoT & nonCoT dikirim sbg `LoRARequest` (tak reload base).
- **Notebook:** 4 cell (install, repo, config, eval) — lihat §3.
- **Tak berubah:** scoring (`sampling_metrics.score_samples`), tabel, save JSON,
  prompt, grader (`answer_check`).
- **Commit + push** `src/eval/vllm_eval.py` + edit notebook sebelum clone di server.
