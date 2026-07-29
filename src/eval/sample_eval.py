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

import json
import re
from pathlib import Path

from src.eval.scenario_eval import _load_model
from src.eval.skenario4_eval import PROMPT
from src.eval.sampling_metrics import score_samples


def _cache_path(cdir: Path, idx: int) -> Path:
    return cdir / f"p{idx:05d}.json"


def _load_cache(cdir: Path, n_problems: int, n_samples: int) -> list[list[str]]:
    """Baca cache per-soal. Soal yang punya >= n_samples kandidat -> dipakai (ambil n_samples
    pertama); sisanya kosong -> perlu digenerate ulang."""
    per_problem: list[list[str]] = [[] for _ in range(n_problems)]
    for idx in range(n_problems):
        fp = _cache_path(cdir, idx)
        if fp.exists():
            try:
                cand = json.loads(fp.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(cand, list) and len(cand) >= n_samples:
                per_problem[idx] = cand[:n_samples]
    return per_problem


def evaluate_sampling(model_id: str, rows: list[dict], *, adapter_dir: str | None = None,
                      n_samples: int = 5, ks_pass=(1, 2, 3), ks_maj=(3, 5),
                      temperature: float = 0.7, top_p: float = 0.95,
                      max_new_tokens: int = 2048, batch_size: int = 4,
                      ckpt_dir: str | None = None, cache_tag: str | None = None) -> dict:
    """Generate N sampel/soal (sampling) lalu skor pass@k & maj@k. Butuh GPU.

    n_samples harus >= max(ks_pass + ks_maj). batch_size dihitung per-SOAL; tiap soal
    memekarkan jadi n_samples sekuens -> kecilkan batch_size kalau OOM.

    Checkpoint/resume: kalau `ckpt_dir` diisi, kandidat tiap soal dicache ke
    `ckpt_dir/<cache_tag>/pNNNNN.json`. Run ulang nge-skip soal yang sudah lengkap,
    jadi model cuma di-load kalau masih ada soal yang harus digenerate.
    """
    import torch

    need = max([*ks_pass, *ks_maj])
    assert n_samples >= need, f"n_samples ({n_samples}) < k terbesar ({need})"

    cdir: Path | None = None
    if ckpt_dir:
        cdir = Path(ckpt_dir) / (cache_tag or "default")
        cdir.mkdir(parents=True, exist_ok=True)
        per_problem = _load_cache(cdir, len(rows), n_samples)
    else:
        per_problem = [[] for _ in rows]

    todo = [i for i, c in enumerate(per_problem) if len(c) < n_samples]
    if cdir is not None:
        print(f"    cache: {len(rows) - len(todo)}/{len(rows)} soal sudah ada, {len(todo)} digenerate")
    if not todo:
        return score_samples(rows, per_problem, ks_pass=ks_pass, ks_maj=ks_maj)

    model, tok = _load_model(model_id, adapter_dir)
    for b in range(0, len(todo), batch_size):
        idxs = todo[b:b + batch_size]
        batch = [rows[i] for i in idxs]
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
        for j, idx in enumerate(idxs):
            cand = dec[j * n_samples:(j + 1) * n_samples]
            per_problem[idx] = cand
            if cdir is not None:
                _cache_path(cdir, idx).write_text(
                    json.dumps(cand, ensure_ascii=False), encoding="utf-8")
        print(f"    {min(b + batch_size, len(todo))}/{len(todo)} (soal baru)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score_samples(rows, per_problem, ks_pass=ks_pass, ks_maj=ks_maj)


def _slug(s: str) -> str:
    """Bikin nama folder cache aman dari label/set (buang karakter non-alnum)."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s).strip("_") or "x"


def eval_specs_sampling(specs: dict[str, dict], sets: dict[str, list[dict]], *,
                        n_samples: int = 5, ks_pass=(1, 2, 3), ks_maj=(3, 5),
                        temperature: float = 0.7, top_p: float = 0.95,
                        max_new_tokens: int = 2048, batch_size: int = 4,
                        ckpt_dir: str | None = None) -> dict:
    """specs = {label: {"model": id, "adapter": dir|None}}; sets = {nama: rows}.
    Return results[label][set] = {pass@k..., maj@k..., format_ok_rate, n}.

    Kalau `ckpt_dir` diisi, generasi tiap soal dicache per (label, set) di
    `ckpt_dir/<label>__<set>/` -> run ulang nge-skip yang sudah selesai."""
    results: dict[str, dict] = {}
    for label, spec in specs.items():
        results[label] = {}
        for set_name, rows in sets.items():
            print(f"=== {label} @ {set_name} ({len(rows)} soal x {n_samples} sampel) ===")
            res = evaluate_sampling(
                spec["model"], rows, adapter_dir=spec.get("adapter"),
                n_samples=n_samples, ks_pass=ks_pass, ks_maj=ks_maj,
                temperature=temperature, top_p=top_p,
                max_new_tokens=max_new_tokens, batch_size=batch_size,
                ckpt_dir=ckpt_dir,
                cache_tag=f"{_slug(label)}__{_slug(set_name)}" if ckpt_dir else None)
            results[label][set_name] = res
            pass1 = res.get("pass@1")
            print(f"  -> pass@1 {pass1:.3f} | format_ok {res['format_ok_rate']:.3f}")
    return results
