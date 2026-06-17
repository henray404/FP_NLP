"""CPU unit tests for CoT-pipeline checkpoint/resume (no GPU, no API).

Run: python -m pytest src/cot_synthesis/test_checkpoint.py -q
"""
import json

from src.cot_synthesis import filter_solutions as fs
from src.cot_synthesis.generate import _already_done, _in_shard


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# -------------------------------
# generate resume bookkeeping
# -------------------------------

def test_already_done_counts_candidates_per_id(tmp_path):
    out = tmp_path / "candidates.jsonl"
    _write_jsonl(out, [
        {"id": "q1", "candidate_idx": 0, "text": "a"},
        {"id": "q1", "candidate_idx": 1, "text": "b"},
        {"id": "q2", "candidate_idx": 0, "text": "c"},
    ])
    done = _already_done(out)
    assert done == {"q1": 2, "q2": 1}
    assert _already_done(tmp_path / "missing.jsonl") == {}


def test_shards_are_disjoint_and_cover_everything():
    n_items, num_shards = 23, 3
    owned = [set(i for i in range(n_items) if _in_shard(i, s, num_shards))
             for s in range(num_shards)]
    union = set().union(*owned)
    assert union == set(range(n_items))            # every item owned by someone
    for a in range(num_shards):                     # no item owned twice
        for b in range(a + 1, num_shards):
            assert owned[a].isdisjoint(owned[b])
    assert all(_in_shard(i, 0, 1) for i in range(n_items))  # single worker owns all


# -------------------------------
# filter checkpoint helpers
# -------------------------------

def test_cand_key_is_per_candidate():
    a = {"id": "q1", "candidate_idx": 0}
    b = {"id": "q1", "candidate_idx": 1}
    assert fs._cand_key(a) != fs._cand_key(b)
    assert fs._cand_key(a) == "q1\t0"


def test_processed_keys_unions_output_and_progress(tmp_path):
    out = tmp_path / "correct.jsonl"
    prog = tmp_path / "correct.jsonl.progress"
    _write_jsonl(out, [{"id": "q1", "candidate_idx": 0}])
    prog.write_text("q2\t0\nq2\t1\n", encoding="utf-8")
    keys = fs._processed_keys(out, prog)
    assert keys == {"q1\t0", "q2\t0", "q2\t1"}


# -------------------------------
# filter run + resume (fake judge -> no GPU/API)
# -------------------------------

CANDS = [
    {"id": "q1", "candidate_idx": 0, "soal": "2+2", "jawaban": "4", "text": r"\boxed{4}"},
    {"id": "q1", "candidate_idx": 1, "soal": "2+2", "jawaban": "4", "text": r"\boxed{5}"},  # wrong
    {"id": "q2", "candidate_idx": 0, "soal": "3+3", "jawaban": "6", "text": "no box here"},  # no_boxed
    {"id": "q3", "candidate_idx": 0, "soal": "1+1", "jawaban": "2", "text": r"\boxed{2}"},
]


def _install_fake_judge(monkeypatch):
    """Judge that says 'benar' iff boxed pred == gold. Records how many it judged."""
    calls = {"n": 0}

    def fake_make(backend, model, sleep=0.0, tensor_parallel_size=1):
        def judge_batch(triples):
            calls["n"] += len(triples)
            return [pred.strip() == gold.strip() for (_soal, gold, pred) in triples]
        return judge_batch

    monkeypatch.setattr(fs, "_make_judge", fake_make)
    return calls


def test_run_filter_keeps_correct_and_checkpoints(tmp_path, monkeypatch):
    calls = _install_fake_judge(monkeypatch)
    inp = tmp_path / "candidates.jsonl"
    out = tmp_path / "correct.jsonl"
    _write_jsonl(inp, CANDS)

    stats = fs.run_filter(inp, out, judge_backend="api", judge_model="x", batch_size=2)

    kept = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept_keys = {(r["id"], r["candidate_idx"]) for r in kept}
    assert kept_keys == {("q1", 0), ("q3", 0)}
    assert stats["kept"] == 2
    assert stats["no_boxed"] == 1
    assert stats["wrong"] == 1
    assert calls["n"] == 3          # q1#0, q1#1, q3#0 judged; no_boxed skipped before judge
    # progress file recorded every examined candidate (kept + dropped)
    prog = out.with_suffix(out.suffix + ".progress")
    assert len(prog.read_text(encoding="utf-8").splitlines()) == 4


def test_run_filter_resume_does_not_reprocess(tmp_path, monkeypatch):
    calls = _install_fake_judge(monkeypatch)
    inp = tmp_path / "candidates.jsonl"
    out = tmp_path / "correct.jsonl"
    _write_jsonl(inp, CANDS)

    fs.run_filter(inp, out, judge_backend="api", judge_model="x", batch_size=2)
    first = calls["n"]
    out_after_first = out.read_text(encoding="utf-8")

    # second run: everything already checkpointed -> judge called 0 more times, output unchanged
    stats2 = fs.run_filter(inp, out, judge_backend="api", judge_model="x", batch_size=2)
    assert calls["n"] == first              # no re-judging
    assert stats2["skipped_done"] == 4
    assert stats2["kept"] == 0              # nothing new kept this run
    assert out.read_text(encoding="utf-8") == out_after_first  # no duplicate rows


def test_run_filter_no_resume_starts_clean(tmp_path, monkeypatch):
    _install_fake_judge(monkeypatch)
    inp = tmp_path / "candidates.jsonl"
    out = tmp_path / "correct.jsonl"
    _write_jsonl(inp, CANDS)

    fs.run_filter(inp, out, judge_backend="api", judge_model="x")
    stats = fs.run_filter(inp, out, judge_backend="api", judge_model="x", resume=False)
    assert stats["skipped_done"] == 0       # ignored previous checkpoint
    kept = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(kept) == 2                   # clean rewrite, not appended/duplicated
