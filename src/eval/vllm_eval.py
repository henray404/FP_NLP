"""
Eval SAMPLING via vLLM (pengganti cepat sample_eval.py yang berbasis HF generate).

Sama persis kontraknya dengan src.eval.sample_eval.eval_specs_sampling:
  - input  : specs = {label: {"model": base_id, "adapter": dir|None}}, sets = {nama: rows}
  - output : results[label][set] = {n, pass@k..., maj@k..., format_ok_rate}  (lewat score_samples)

Beda: generation pakai vLLM (PagedAttention + continuous batching + n sampel native),
jadi 5-20x lebih cepat daripada loop model.generate HF. Butuh GPU + vLLM (Linux).

Optimasi kunci:
  - base di-load SEKALI; tiap adapter dikirim sbg LoRARequest (tak reload base).
  - n_samples dipetakan ke SamplingParams(n=...) -> vLLM yang batch, bukan kita.
  - prompt ChatML dibangun manual (base Qwen2.5 non-Instruct tak punya chat_template).

Usage (lihat notebooks/s3_cot_vs_noncot.ipynb):
    from src.eval.vllm_eval import eval_specs_sampling_vllm
    res = eval_specs_sampling_vllm(specs, sets, n_samples=5,
                                   ks_pass=(1,2,3), ks_maj=(3,5),
                                   temperature=0.7, top_p=0.95, max_new_tokens=2048)
"""
from __future__ import annotations

from src.eval.skenario4_eval import PROMPT
from src.eval.sampling_metrics import score_samples

# ChatML manual: base Qwen2.5-1.5B (non-Instruct) tak punya chat_template.
# Sama dgn yang dipasang scenario_eval._load_model.
_CHATML = "<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"


def _build_prompts(rows: list[dict]) -> list[str]:
    return [_CHATML.format(body=PROMPT.format(soal=r["soal"])) for r in rows]


def eval_specs_sampling_vllm(
    specs: dict[str, dict],
    sets: dict[str, list[dict]],
    *,
    n_samples: int = 5,
    ks_pass=(1, 2, 3),
    ks_maj=(3, 5),
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 2048,
    max_model_len: int = 4096,
    max_lora_rank: int = 64,
    gpu_memory_utilization: float = 0.90,
    dtype: str = "bfloat16",
) -> dict:
    """Generate N sampel/soal pakai vLLM lalu skor pass@k & maj@k. Butuh GPU + vLLM.

    Asumsi SEMUA spec share base model yang sama (kasus S3: CoT vs nonCoT di atas
    Qwen2.5-1.5B). Base di-load sekali; tiap adapter dikirim sbg LoRARequest.

    max_lora_rank HARUS >= rank `r` di adapter_config.json (cek; default 64).
    max_model_len >= panjang_prompt + max_new_tokens. OOM -> turunkan
    gpu_memory_utilization atau max_model_len atau n_samples.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    need = max([*ks_pass, *ks_maj])
    assert n_samples >= need, f"n_samples ({n_samples}) < k terbesar ({need})"

    bases = {s["model"] for s in specs.values()}
    assert len(bases) == 1, f"semua spec harus base sama, dapat: {bases}"
    base = bases.pop()
    has_lora = any(s.get("adapter") for s in specs.values())

    llm = LLM(
        model=base,
        dtype=dtype,
        enable_lora=has_lora,
        max_lora_rank=max_lora_rank,
        max_loras=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    sp = SamplingParams(
        n=n_samples, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens,
    )

    results: dict[str, dict] = {}
    for lora_idx, (label, spec) in enumerate(specs.items(), start=1):
        adapter = spec.get("adapter")
        lora_req = LoRARequest(label, lora_idx, adapter) if adapter else None
        results[label] = {}
        for set_name, rows in sets.items():
            print(f"=== {label} @ {set_name} ({len(rows)} soal x {n_samples} sampel) ===")
            prompts = _build_prompts(rows)
            outs = llm.generate(prompts, sp, lora_request=lora_req)
            # outs[i].outputs = list n CompletionOutput; .text = teks generasi.
            per_problem = [[o.text for o in ro.outputs] for ro in outs]
            res = score_samples(rows, per_problem, ks_pass=ks_pass, ks_maj=ks_maj)
            results[label][set_name] = res
            print(f"  -> pass@1 {res.get('pass@1'):.3f} | format_ok {res['format_ok_rate']:.3f}")
    return results
