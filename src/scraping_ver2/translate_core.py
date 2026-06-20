"""
Inti translate EN->ID yang math-safe (dipakai ulang dari openmath_translate_v2.py).

Prinsip: model TIDAK PERNAH melihat satu simbol matematika pun.
  1. Pisah tiap field jadi span NL vs span math (\\(...\\), \\[...\\], $...$, $$...$$, \\boxed{}).
  2. Translate HANYA span NL (NLLB-200, en->id kuat).
  3. Splice math asli kembali verbatim -> nol korupsi variabel/simbol, nol LaTeX hilang.

Dedup span NL antar-entry untuk throughput GPU. Field-map diparametrisasi supaya bisa
dipakai untuk skema apa pun (AIMO: problem/solution/answer; NumGLUE: question/answer).
"""
from __future__ import annotations

import re

import torch

# ── Segmentasi: pisah teks jadi NL vs math; math tak pernah masuk model ──
# Urutan penting: $$ sebelum $, display sebelum inline.
MATH_PATTERN = re.compile(
    r"(\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\\(.*?\\\)"
    r"|\$[^$\n]*?\$"
    r"|\\boxed\{(?:[^{}]|\{[^{}]*\})*\}"
    r")",
    re.DOTALL,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

MAX_SPAN_CHARS = 600  # span NL lebih panjang dari ini dipecah per-kalimat dulu


def strip_think(text: str) -> str:
    idx = text.find("</think>")
    return text[idx + 8:].strip() if idx >= 0 else text.strip()


def has_alpha(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _sentence_chunks(s: str, max_chars: int = MAX_SPAN_CHARS):
    if len(s) <= max_chars:
        return [s]
    out, cur = [], ""
    for sent in SENT_SPLIT.split(s):
        cand = (cur + " " + sent).strip() if cur else sent
        if len(cand) > max_chars and cur:
            out.append(cur); cur = sent
        else:
            cur = cand
    if cur:
        out.append(cur)
    return out or [s]


def build_template(text: str, max_span_chars: int = MAX_SPAN_CHARS):
    """List segmen [kind, content], kind in {'math','txt'}. Span 'txt' panjang dipecah dulu."""
    parts = MATH_PATTERN.split(text)
    segs = []
    for i, p in enumerate(parts):
        if i % 2 == 1:        # math tertangkap
            segs.append(["math", p])
        elif p:               # teks biasa
            if len(p) > max_span_chars:
                for piece in _sentence_chunks(p, max_span_chars):
                    segs.append(["txt", piece])
            else:
                segs.append(["txt", p])
    return segs


def reattach_ws(original: str, translated: str) -> str:
    """Model membuang whitespace tepi; kembalikan supaya splice dgn math tetap rapi."""
    lead = original[: len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + translated.strip() + trail


def translate_batch(model, tokenizer, tgt_id, device, texts,
                    num_beams: int = 2, max_src: int = 256, max_tgt: int = 400):
    if not texts:
        return []
    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_src).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                             max_length=max_tgt, num_beams=num_beams)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def load_model(model_name: str = "facebook/nllb-200-distilled-1.3B",
               src_lang: str = "eng_Latn", tgt_lang: str = "ind_Latn",
               use_fp16: bool = True):
    """Muat NLLB + resolve forced BOS target. Return (model, tokenizer, tgt_id, device)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = src_lang
    dtype = torch.float16 if (use_fp16 and device.type == "cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    if tgt_id is None or tgt_id == tokenizer.unk_token_id:
        tgt_id = tokenizer.lang_code_to_id[tgt_lang]
    return model, tokenizer, tgt_id, device


def process_block(entries: list[dict], field_map: list[tuple[str, str]],
                  model, tokenizer, tgt_id, device, *, batch_size: int = 8,
                  num_beams: int = 2, strip_think_fields: set[str] | None = None):
    """Translate satu blok entry sesuai field_map [(src_key, out_key), ...].

    Span NL unik di-batch lalu di-translate sekali; math di-splice balik verbatim.
    Return list record {out_key: teks_id}.
    """
    strip_think_fields = strip_think_fields or set()
    templates = []          # per entry: {out_key: [segments]}
    uniq: dict[str, int] = {}
    pool: list[str] = []
    refs = []               # (entry_i, out_key, seg_i, pool_i)

    for ei, entry in enumerate(entries):
        tmpl = {}
        for src_key, out_key in field_map:
            raw = entry.get(src_key, "") or ""
            if src_key in strip_think_fields:
                raw = strip_think(raw)
            raw = str(raw).strip()
            segs = build_template(raw)
            for si, (kind, content) in enumerate(segs):
                if kind == "txt" and has_alpha(content):
                    if content not in uniq:
                        uniq[content] = len(pool); pool.append(content)
                    refs.append((ei, out_key, si, uniq[content]))
            tmpl[out_key] = segs
        templates.append(tmpl)

    out = [None] * len(pool)
    for i in range(0, len(pool), batch_size):
        out[i:i + batch_size] = translate_batch(
            model, tokenizer, tgt_id, device, pool[i:i + batch_size], num_beams=num_beams)

    for ei, out_key, si, pi in refs:
        seg = templates[ei][out_key][si]
        seg[1] = reattach_ws(seg[1], out[pi])

    out_keys = [k for _, k in field_map]
    records = []
    for tmpl in templates:
        records.append({k: "".join(s[1] for s in tmpl[k]) for k in out_keys})
    return records
