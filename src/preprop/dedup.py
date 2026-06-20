"""
Embedding-based deduplication + decontamination vs eval benchmarks.
- Dedup: cosine similarity > 0.92 -> hapus duplikat
- Decontamination: cosine > 0.90 vs benchmark -> hapus
Input:  data/filtered/after_validity.jsonl
Output: data/filtered/clean.jsonl
"""
import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEDUP_THRESHOLD = 0.92
DECONTAM_THRESHOLD = 0.90
BATCH_SIZE = 128


def get_text(item: dict) -> str:
    """Field soal/pertanyaan, apa pun schema-nya (un_pdfs pakai `soal`, NumGLUE `question`)."""
    return (item.get("question") or item.get("soal") or "").strip()


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # L2 normalize -> cosine = dot product
        show_progress_bar=True,
    ).astype(np.float32)


def dedup(embeddings: np.ndarray, threshold: float) -> list[int]:
    """Return indices to KEEP (first occurrence wins)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    keep = []
    for i in range(len(embeddings)):
        if index.ntotal > 0:
            D, _ = index.search(embeddings[i:i+1], k=1)
            if D[0][0] >= threshold:
                continue
        index.add(embeddings[i:i+1])
        keep.append(i)
    return keep


def decontaminate(
    train_embeddings: np.ndarray,
    benchmark_embeddings: np.ndarray,
    threshold: float,
) -> list[int]:
    """Return indices of train items NOT overlapping with benchmark."""
    index = faiss.IndexFlatIP(benchmark_embeddings.shape[1])
    index.add(benchmark_embeddings)
    keep = []
    for i, emb in enumerate(train_embeddings):
        D, _ = index.search(emb.reshape(1, -1), k=1)
        if D[0][0] < threshold:
            keep.append(i)
    return keep


def run_dedup(
    input_path: str | Path,
    output_path: str | Path,
    benchmark_paths: list[str | Path] | None = None,
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    model = SentenceTransformer(EMBED_MODEL)
    questions = [get_text(item) for item in items]
    embeddings = embed_texts(questions, model)

    keep_idx = dedup(embeddings, DEDUP_THRESHOLD)
    print(f"After internal dedup: {len(keep_idx)} / {len(items)}")

    if benchmark_paths:
        bench_items = []
        for bpath in benchmark_paths:
            with open(bpath, encoding="utf-8") as f:
                for line in f:
                    bench_items.append(json.loads(line))
        bench_emb = embed_texts([get_text(b) for b in bench_items], model)
        safe = decontaminate(embeddings[keep_idx], bench_emb, DECONTAM_THRESHOLD)
        keep_idx = [keep_idx[i] for i in safe]
        print(f"After decontamination: {len(keep_idx)}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for i in keep_idx:
            fout.write(json.dumps(items[i], ensure_ascii=False) + "\n")

    return {"total_input": len(items), "final": len(keep_idx)}


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "data/filtered/after_validity.jsonl"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/filtered/clean.jsonl"
    stats = run_dedup(inp, out)
    print(stats)
