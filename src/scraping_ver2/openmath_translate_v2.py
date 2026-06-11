#!/usr/bin/env python3
"""OpenMath -> Indonesian (v2 — math-safe segmentation).

The model never sees a single math symbol:
  1. Split each field into NL spans vs math spans (\\(...\\), \\[...\\], $...$, $$...$$, stray \\boxed{}).
  2. Translate ONLY the natural-language spans (NLLB-200, strong en->id).
  3. Splice the original math back verbatim -> zero variable/symbol corruption, zero dropped LaTeX.
  4. expected_answer is usually pure LaTeX -> 0 NL spans -> copied verbatim into 'jawaban' (never empty).

Batches NL spans across many entries for GPU throughput. Checkpoints per block. Resumable:
re-run the script to continue from the last checkpoint (clears VRAM each run).

Install (your torch env):
    pip install transformers sentencepiece sacremoses accelerate tqdm
"""

import json
import re
import gc
from pathlib import Path

from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ── CONFIG ──────────────────────────────────────────────────────
INPUT_PATH  = "hugging_face_AIMO/openmath_reasoning_20k.json"
OUTPUT_DIR  = "hugging_face_AIMO/parts_v2"

START       = 5000    # inclusive  (continue right after the 0:5000 chunk)
END         = 20000   # exclusive  (chunk 2: entries 5000..19999)

MODEL_NAME  = "facebook/nllb-200-distilled-1.3B"  # strong en->id; ~2.6 GB fp16, fits 4060 8 GB
# fallback if VRAM/quality tradeoff needed: "facebook/nllb-200-distilled-600M"
SRC_LANG    = "eng_Latn"
TGT_LANG    = "ind_Latn"

RUN_CHUNK   = 500     # entries to process THIS run, then stop. Re-run the script
                      # to continue from the last checkpoint (clears VRAM each run).
BLOCK_SIZE  = 100     # entries processed + checkpointed per block
BATCH_SIZE  = 8       # NL spans per GPU forward pass; auto-halves on OOM
NUM_BEAMS   = 2       # 1 = fastest, 4 = best; 2 = balanced
MAX_SRC_TOK = 256
MAX_TGT_TOK = 400
MAX_SPAN_CHARS = 600  # NL spans longer than this get sentence-split before translation
USE_FP16    = True
# ─────────────────────────────────────────────────────────


# ── Segmentation: split text into NL vs math, never feed math to the model ──

# Order matters: $$ before $, display before inline.
MATH_PATTERN = re.compile(
    r"(\$\$.*?\$\$"                              # $$ ... $$
    r"|\\\[.*?\\\]"                              # \[ ... \]
    r"|\\\(.*?\\\)"                              # \( ... \)
    r"|\$[^$\n]*?\$"                             # $ ... $
    r"|\\boxed\{(?:[^{}]|\{[^{}]*\})*\}"        # stray \boxed{...}
    r")",
    re.DOTALL,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def strip_think(text: str) -> str:
    idx = text.find("</think>")
    return text[idx + 8:].strip() if idx >= 0 else text.strip()


def has_alpha(s: str) -> bool:
    return any(c.isalpha() for c in s)


def _sentence_chunks(s: str, max_chars: int):
    """Split a long NL span into <=max_chars pieces on sentence boundaries."""
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


def build_template(text: str):
    """Return list of [kind, content] segments. kind in {'math','txt'}.
    Long 'txt' segments are pre-split so each translatable piece fits the model."""
    parts = MATH_PATTERN.split(text)
    segs = []
    for i, p in enumerate(parts):
        if i % 2 == 1:            # captured math
            segs.append(["math", p])
        elif p:                   # plain text
            if len(p) > MAX_SPAN_CHARS:
                for piece in _sentence_chunks(p, MAX_SPAN_CHARS):
                    segs.append(["txt", piece])
            else:
                segs.append(["txt", p])
    return segs


def translate_batch(model, tokenizer, tgt_id, device, texts):
    """Translate a list of NL strings en->id. Math never reaches here."""
    if not texts:
        return []
    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_SRC_TOK).to(device)
        gen = model.generate(**enc, forced_bos_token_id=tgt_id,
                             max_length=MAX_TGT_TOK, num_beams=NUM_BEAMS)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def reattach_ws(original: str, translated: str) -> str:
    """Model strips edge whitespace; restore it so splice with math stays clean."""
    lead = original[: len(original) - len(original.lstrip())]
    trail = original[len(original.rstrip()):]
    return lead + translated.strip() + trail


# source field -> output key
FIELD_MAP = [("problem", "soal"),
             ("generated_solution", "cara"),
             ("expected_answer", "jawaban")]
OUT_KEYS = [k for _, k in FIELD_MAP]


def process_block(entries, model, tokenizer, tgt_id, device):
    """Build templates for a block, batch-translate all unique NL spans, reassemble."""
    templates = []                 # per entry: {out_key: [segments]}
    uniq = {}                      # nl string -> index in pool (dedup)
    pool = []                      # unique strings to translate
    refs = []                      # (entry_i, out_key, seg_i, pool_i)

    for ei, entry in enumerate(entries):
        tmpl = {}
        for field, out_key in FIELD_MAP:
            raw = entry.get(field, "") or ""
            if field == "generated_solution":
                raw = strip_think(raw)
            raw = raw.strip()
            segs = build_template(raw)
            for si, (kind, content) in enumerate(segs):
                if kind == "txt" and has_alpha(content):
                    key = content
                    if key not in uniq:
                        uniq[key] = len(pool); pool.append(content)
                    refs.append((ei, out_key, si, uniq[key]))
            tmpl[out_key] = segs
        templates.append(tmpl)

    # translate unique pool in batches
    out = [None] * len(pool)
    for i in range(0, len(pool), BATCH_SIZE):
        out[i:i + BATCH_SIZE] = translate_batch(
            model, tokenizer, tgt_id, device, pool[i:i + BATCH_SIZE])

    # write translations back (preserving edge whitespace)
    for ei, out_key, si, pi in refs:
        seg = templates[ei][out_key][si]
        seg[1] = reattach_ws(seg[1], out[pi])

    # reassemble — math spliced back verbatim
    records = []
    for tmpl in templates:
        records.append({k: "".join(s[1] for s in tmpl[k]) for k in OUT_KEYS})
    return records


def flush(part_file, records):
    with open(part_file, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("GPU :", torch.cuda.get_device_name(0))
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("WARNING: no CUDA — running on CPU (slow)")

    out_dir = Path(OUTPUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)
    part_file = out_dir / f"part_{START:05d}_{END:05d}.jsonl"
    print("Output:", part_file)

    # quick self-test of segmentation
    _demo = r"The square root of \(z = a+bi\) is \[\boxed{\sqrt{r}\,e^{i\theta/2}}\] which is real."
    for k, c in build_template(_demo):
        print(f"  {k:4} | {c}")

    # load model
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.src_lang = SRC_LANG
    dtype = torch.float16 if (USE_FP16 and device.type == "cuda") else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
    model.eval()

    # resolve forced BOS for target language (handles both old/new tokenizer APIs)
    tgt_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    if tgt_id is None or tgt_id == tokenizer.unk_token_id:
        tgt_id = tokenizer.lang_code_to_id[TGT_LANG]
    print("forced_bos_token_id:", tgt_id)
    if device.type == "cuda":
        print(f"Model VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # read slice
    print(f"Reading {INPUT_PATH} (slice {START}:{END}) ...")
    with open(INPUT_PATH, encoding="utf-8") as f:
        all_data = json.load(f)
    data = all_data[START:END]
    del all_data; gc.collect()
    print(f"Slice size: {len(data):,} entries")

    # resume from checkpoint
    done = 0
    if part_file.exists():
        with open(part_file, encoding="utf-8") as f:
            done = sum(1 for line in f if line.strip())
        print(f"Resuming: {done:,} records already on disk, skipping them")
    else:
        print("No checkpoint, starting fresh")

    # main loop: block by block, checkpoint after each block
    total = len(data)
    pbar = tqdm(total=total, initial=done, desc=f"Translating [{START}:{END}]")

    for lo in range(done, total, BLOCK_SIZE):
        hi = min(lo + BLOCK_SIZE, total)
        records = process_block(data[lo:hi], model, tokenizer, tgt_id, device)
        flush(part_file, records)
        pbar.update(hi - lo)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    pbar.close()
    print(f"\nDone. Records in {part_file.name}: "
          f"{sum(1 for l in open(part_file) if l.strip()):,}")

    # sanity check: verify math preserved + jawaban non-empty
    with open(part_file, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    print(f"Total lines: {len(lines):,}")
    empty_jawaban = sum(1 for l in lines if not json.loads(l)["jawaban"].strip())
    print(f"Empty jawaban: {empty_jawaban} / {len(lines)}")
    for label, line in [("FIRST", lines[0]), ("LAST", lines[-1])]:
        r = json.loads(line)
        print(f"\n── {label} ──")
        print("soal   :", r["soal"][:300])
        print("cara   :", r["cara"][:300])
        print("jawaban:", r["jawaban"])


if __name__ == "__main__":
    main()
