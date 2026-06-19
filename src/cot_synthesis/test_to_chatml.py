"""CPU unit tests for to_chatml (ChatML build, language filter, best-per-problem).

Run: python -m pytest src/cot_synthesis/test_to_chatml.py -q
"""
import json

from src.cot_synthesis import to_chatml as tc


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _cand(pid, ci, text, pred="4", soal="2+2?"):
    return {"id": pid, "candidate_idx": ci, "soal": soal, "jawaban": pred,
            "text": text, "pred": pred}


# -------------------------------
# language detector
# -------------------------------

def test_is_indonesian():
    assert tc.is_indonesian("Diketahui persamaan, maka jawaban adalah 4 sehingga selesai")
    assert not tc.is_indonesian("First we find the equation, then the answer value is 4 since")
    # near-symbolic / tie -> Indonesian (so we don't drop clean math)
    assert tc.is_indonesian("x = 2 + 2 = 4")


# -------------------------------
# run(): chatml shape + both arms
# -------------------------------

def test_run_emits_both_arms_chatml(tmp_path):
    inp = tmp_path / "correct.jsonl"
    _write(inp, [_cand("q1", 0, "Jadi jawabannya adalah \\boxed{4}", pred="4")])
    stats = tc.run(inp, tmp_path / "sft")
    cot = [json.loads(l) for l in (tmp_path / "sft" / "cot.jsonl").read_text(encoding="utf-8").splitlines()]
    nocot = [json.loads(l) for l in (tmp_path / "sft" / "nocot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats["cot_examples"] == 1 and stats["nocot_examples"] == 1
    # chatml structure
    assert cot[0]["messages"][0]["role"] == "user"
    assert cot[0]["messages"][1]["role"] == "assistant"
    # cot keeps reasoning; nocot is boxed-only
    assert "Jadi jawabannya" in cot[0]["messages"][1]["content"]
    assert nocot[0]["messages"][1]["content"] == "\\boxed{4}"


# -------------------------------
# best_per_problem: 1 best, ranked
# -------------------------------

def test_best_per_problem_keeps_one_shortest_indonesian(tmp_path):
    inp = tmp_path / "correct.jsonl"
    _write(inp, [
        _cand("q1", 0, "Maka jawaban adalah panjang sekali dengan banyak langkah \\boxed{4}"),
        _cand("q1", 1, "Jadi adalah \\boxed{4}"),          # shorter Indonesian -> best
        _cand("q1", 2, "First then the answer is \\boxed{4}"),  # English
    ])
    stats = tc.run(inp, tmp_path / "sft", best_per_problem=True)
    cot = [json.loads(l) for l in (tmp_path / "sft" / "cot.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stats["cot_examples"] == 1
    assert cot[0]["messages"][1]["content"] == "Jadi adalah \\boxed{4}"


def test_id_only_drops_english(tmp_path):
    inp = tmp_path / "correct.jsonl"
    _write(inp, [
        _cand("q1", 0, "First then the answer value is \\boxed{4} since we find"),  # English
        _cand("q2", 0, "Maka jawaban adalah \\boxed{6}", pred="6"),                 # Indonesian
    ])
    stats = tc.run(inp, tmp_path / "sft", id_only=True)
    assert stats["cot_examples"] == 1          # english dropped
    assert stats["skipped_lang"] == 1


def test_max_per_problem_caps(tmp_path):
    inp = tmp_path / "correct.jsonl"
    _write(inp, [_cand("q1", i, f"Jadi adalah \\boxed{{4}} versi {i}") for i in range(4)])
    stats = tc.run(inp, tmp_path / "sft", max_per_problem=2)
    assert stats["cot_examples"] == 2
