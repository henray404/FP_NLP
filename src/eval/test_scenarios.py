"""CPU tests untuk S1 (compare_teachers) + S2/S3 (scenario_eval) bagian murni-Python.

Run: python -m pytest src/eval/test_scenarios.py -q
"""
import json

from src.cot_synthesis import compare_teachers as ct
from src.eval import scenario_eval as se


def _write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


# -------------------------------
# S1: retensi + pemenang
# -------------------------------

def test_score_teacher_retention(tmp_path):
    cand = tmp_path / "cot.jsonl"
    corr = tmp_path / "correct.jsonl"
    _write(cand, [{"id": "q1"}, {"id": "q1"}, {"id": "q2"}, {"id": "q2"}])  # 4 baris, 2 soal
    _write(corr, [{"id": "q1"}, {"id": "q2"}])                              # 2 baris, 2 soal
    m = ct.score_teacher([str(cand)], [str(corr)])
    assert m["retention_pct"] == 50.0 and m["reduction_pct"] == 50.0
    assert m["coverage_pct"] == 100.0


def test_compare_picks_highest_retention(tmp_path):
    def mk(name, n_corr):
        c = tmp_path / f"cot_{name}.jsonl"; r = tmp_path / f"correct_{name}.jsonl"
        _write(c, [{"id": f"q{i}"} for i in range(10)])
        _write(r, [{"id": f"q{i}"} for i in range(n_corr)])
        return {"candidates": [str(c)], "correct": [str(r)]}
    res = ct.compare({"gemma": mk("gemma", 3), "deepseek": mk("deepseek", 8),
                      "ernie": mk("ernie", 5)})
    assert res["winner"] == "deepseek"            # 80% retensi tertinggi
    assert "deepseek **(WINNER)**" in ct.render_table(res)


# -------------------------------
# S2/S3: tabel perbandingan
# -------------------------------

def test_render_table_models_x_sets():
    results = {
        "CoT":    {"numglue": {"accuracy": 0.42}, "un": {"accuracy": 0.31}},
        "nonCoT": {"numglue": {"accuracy": 0.20}, "un": {"accuracy": 0.18}},
    }
    table = se.render_table(results)
    assert "| model | numglue | un |" in table
    assert "| CoT | 0.420 | 0.310 |" in table
    assert "| nonCoT | 0.200 | 0.180 |" in table


def test_render_table_missing_cell_dash():
    results = {"CoT": {"numglue": {"accuracy": 0.5}}}     # tak ada 'un'
    table = se.render_table({"CoT": results["CoT"], "x": {"un": {"accuracy": 0.1}}})
    assert "-" in table     # sel kosong -> '-'
