"""
Eval berbasis SAMPLING untuk S2/S3/S4: generate N kandidat/soal lalu hitung pass@k & maj@k.

Beda dgn scenario_eval (greedy, pass@1 saja): di sini do_sample=True, num_return_sequences=N,
sehingga bisa pass@1/2/3 dan maj@3/5 (lihat sampling_metrics). Satu pintu untuk:
  - S2: model CoT terbaik (base+adapter) lintas-dataset -> pass@1,2,3 + maj@3,5
  - S3: CoT vs nonCoT (base+adapter) -> pass@1,2,3
  - S4: model pembanding (HF id, adapter=None) -> samakan dgn S2

Butuh GPU. Bagian skor (sampling_metrics) murni CPU.

Usage (lihat notebooks/skenario/):
    from src.eval.sample_eval import eval_specs_sampling
    from src.eval.sampling_metrics import render_tables
    sets = {"numglue": rows1, "easy": rows2}
    specs = {"CoT": {"model": BASE, "adapter": ADAPTER_COT}}
    res = eval_specs_sampling(specs, sets, n_samples=5, ks_pass=(1,2,3), ks_maj=(3,5))
    print(render_tables(res, ["pass@1","pass@2","pass@3","maj@3","maj@5"]))
"""
from __future__ import annotations

from src.eval.scenario_eval import _load_model
from src.eval.skenario4_eval import PROMPT
from src.eval.sampling_metrics import score_samples


def evaluate_sampling(model_id: str, rows: list[dict], *, adapter_dir: str | None = None,
                      n_samples: int = 5, ks_pass=(1, 2, 3), ks_maj=(3, 5),
                      temperature: float = 0.7, top_p: float = 0.95,
                      max_new_tokens: int = 2048, batch_size: int = 4) -> dict:
    """Generate N sampel/soal (sampling) lalu skor pass@k & maj@k. Butuh GPU.

    n_samples harus >= max(ks_pass + ks_maj). batch_size dihitung per-SOAL; tiap soal
    memekarkan jadi n_samples sekuens -> kecilkan batch_size kalau OOM.
    """
    import torch

    need = max([*ks_pass, *ks_maj])
    assert n_samples >= need, f"n_samples ({n_samples}) < k terbesar ({need})"

    model, tok = _load_model(model_id, adapter_dir)
    per_problem: list[list[str]] = [[] for _ in rows]
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(soal=r["soal"])}],
            tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(0)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p,
                                 num_return_sequences=n_samples, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        dec = tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        # urutan HF: [b0s0..b0s(N-1), b1s0..]; kelompokkan per soal.
        for j in range(len(batch)):
            per_problem[i + j] = dec[j * n_samples:(j + 1) * n_samples]
        print(f"    {min(i + batch_size, len(rows))}/{len(rows)}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_samples(rows, per_problem, ks_pass=ks_pass, ks_maj=ks_maj)


def eval_specs_sampling(specs: dict[str, dict], sets: dict[str, list[dict]], *,
                        n_samples: int = 5, ks_pass=(1, 2, 3), ks_maj=(3, 5),
                        temperature: float = 0.7, top_p: float = 0.95,
                        max_new_tokens: int = 2048, batch_size: int = 4) -> dict:
    """specs = {label: {"model": id, "adapter": dir|None}}; sets = {nama: rows}.
    Return results[label][set] = {pass@k..., maj@k..., format_ok_rate, n}."""
    results: dict[str, dict] = {}
    for label, spec in specs.items():
        results[label] = {}
        for set_name, rows in sets.items():
            print(f"=== {label} @ {set_name} ({len(rows)} soal x {n_samples} sampel) ===")
            res = evaluate_sampling(
                spec["model"], rows, adapter_dir=spec.get("adapter"),
                n_samples=n_samples, ks_pass=ks_pass, ks_maj=ks_maj,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, batch_size=batch_size)
            results[label][set_name] = res
            pass1 = res.get("pass@1")
            print(f"  -> pass@1 {pass1:.3f} | format_ok {res['format_ok_rate']:.3f}")
    return results
