# CoT Generator Comparison & Selection — Design

**Date:** 2026-06-19
**Artifact:** `notebooks/cot_compare_select.ipynb`

## Problem

Two teacher generators produce CoT candidate datasets for the IndoMathReason SFT corpus:

- **Gemma-2** (`notebooks/cot_pipeline_kaggle_gemma.ipynb`, backend `hf`)
- **DeepSeek-R1-Distill-Qwen-7B** (`notebooks/cot_pipeline_kaggle.ipynb`, backend `vllm`)

Both pipelines already judge their own candidates with the **same** LLM judge
(`Qwen/Qwen2.5-7B-Instruct` via vLLM, inside `src.cot_synthesis.filter_solutions.run_filter`)
and each emits `cot/correct.jsonl`. We need to decide **which generator's CoT dataset is more
correct**, and feed that winner into the SFT fine-tune pipeline.

## Decisions

- **Input granularity:** aggregate the existing `correct.jsonl` from both runs. The judge
  (Qwen2.5-7B vLLM) already ran upstream with identical model + prompt, so the verdicts are
  directly comparable. This notebook does **not** load a GPU or re-judge.
- **Winner metric:** *coverage* — number of unique soal `id` with ≥1 correct solution. More soal
  covered = broader SFT corpus. Secondary reported metrics: total correct solutions, acceptance
  rate.
- **Match key:** problem `id` (`problem_id`). Assumes both runs consumed the same `train_pool`
  ordering. A mismatch warning is printed if pools diverge.
- **Final SFT artifact:** **copy** the winning run's already-emitted `sft/cot.jsonl` +
  `sft/nocot.jsonl`. Both pipelines build ChatML in their filter cell with the same policy
  (`best_per_problem=True, id_only=True`), so the winner's `sft/` is already correct — no rebuild.
- **Unequal pools:** if the two runs attempted different soal sets (e.g. one ran `LIMIT=50`, the
  other full), print a warning and decide the winner on the **intersection** of attempted soal —
  fair head-to-head. Raw counts are also reported for context.

## Non-goals

- No re-judging, no GPU, no new model loads.
- No new code in `src/` — reuse `utils.read_jsonl`.
- Not training; only selecting + emitting the SFT-ready dataset.

## Notebook structure

Kaggle notebook (matches existing `cot_pipeline_*` convention: clone repo, read `/kaggle/input`,
write `/kaggle/working`).

1. **Setup** — clone `FP_NLP` repo, `sys.path.insert`. No vLLM install; pure-python aggregation.
2. **Locate inputs** — `rglob('correct.jsonl')` under `/kaggle/input`. Tag each hit with a source
   label (gemma / deepseek) inferred from its dataset folder name; manual path override variables
   (`GEMMA_CORRECT`, `DEEPSEEK_CORRECT`) for ambiguous cases. Also try to locate each run's
   `candidates.jsonl` to recover the attempted-soal pool (optional).
3. **Aggregate** — for each generator build:
   - `solved_ids` = set of `id` with ≥1 correct row,
   - `total_correct` = row count,
   - `pool_ids` = set of attempted `id` (from `candidates.jsonl` if found, else `solved_ids` with a
     warning that acceptance rate is unavailable).
4. **Fairness check** — compare `pool_ids` across generators. If unequal, print a warning and
   compute `common = pool_gemma ∩ pool_deepseek`; coverage for ranking is measured on `common`.
5. **Compare + pick winner** — print a table: generator | coverage (raw) | coverage (intersection) |
   total correct | acceptance rate. Winner = max intersection-coverage; tie-break by total correct.
   Store winner label + winning `correct.jsonl` path.
6. **Copy winner SFT** — locate the winning run's `sft/` (sibling of its `correct.jsonl`, or
   `rglob('cot.jsonl')` under its dataset dir) and copy `cot.jsonl` + `nocot.jsonl` into
   `/kaggle/working/sft/`. If the winner's `sft/` is absent (only `correct.jsonl` uploaded), abort
   with a clear message telling the user to upload it.
7. **Download** — write `compare_summary.json` (per-generator metrics + winner) and zip `sft/` +
   the summary for download / Save Version.

## Data contracts

`correct.jsonl` rows (from `filter_solutions._emit`):
`{id, soal, jawaban, candidate_idx, text, pred}`.

`candidates.jsonl` rows (optional, for pool denominator):
`{id, soal, jawaban, cara, source, candidate_idx, text}`.

`compare_summary.json` (new):
```json
{
  "generators": {
    "gemma":    {"coverage_raw": int, "coverage_common": int, "total_correct": int,
                 "pool_size": int, "acceptance_rate": float | null},
    "deepseek": {"...": "..."}
  },
  "common_pool_size": int,
  "pools_equal": bool,
  "winner": "gemma" | "deepseek",
  "winner_correct_path": str
}
```

## Edge cases

- Only one `correct.jsonl` found → print error, abort (nothing to compare).
- `candidates.jsonl` missing → acceptance rate = null, pool = solved set, warn.
- Empty intersection → fall back to raw coverage with a loud warning.
- Tie on intersection-coverage → tie-break total correct, then alphabetical label.

## Testing

Manual / notebook-level (no test suite in repo). Sanity asserts in-cell: both inputs found,
winner is non-null, `sft/cot.jsonl` non-empty after copy.
