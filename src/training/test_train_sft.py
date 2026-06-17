"""CPU unit tests for the training pipeline plumbing (no GPU, no model download).

Run: python -m pytest src/training/test_train_sft.py -q
"""
import json

import pytest

from src.training.train_sft import (TrainConfig, load_config, read_chatml,
                                     to_training_text, train_val_split,
                                     validate_chatml_row)

CONFIG_DIR = "src/training/configs"


def _row(user="2+2?", assistant="\\boxed{4}"):
    return {"messages": [{"role": "user", "content": user},
                         {"role": "assistant", "content": assistant}]}


# -------------------------------
# TrainConfig / load_config
# -------------------------------

def test_default_config_is_valid():
    cfg = TrainConfig()
    assert cfg.mode == "cot"
    assert "q_proj" in cfg.target_modules
    assert cfg.train_on_responses_only is True


def test_config_rejects_bad_mode():
    with pytest.raises(ValueError):
        TrainConfig(mode="reasoning")


def test_config_rejects_bad_val_ratio():
    with pytest.raises(ValueError):
        TrainConfig(val_ratio=1.0)


@pytest.mark.parametrize("name", ["cot_1.5b", "nocot_1.5b", "cot_0.5b", "nocot_0.5b"])
def test_shipped_configs_load(name):
    cfg = load_config(f"{CONFIG_DIR}/{name}.yaml")
    expected_mode = "nocot" if name.startswith("nocot") else "cot"
    assert cfg.mode == expected_mode
    assert cfg.dataset.endswith(f"{expected_mode}.jsonl")


def test_load_config_override_wins_and_ignores_none():
    cfg = load_config(f"{CONFIG_DIR}/cot_1.5b.yaml",
                      output_dir="outputs/smoke", base_model=None, max_examples=10)
    assert cfg.output_dir == "outputs/smoke"      # override applied
    assert cfg.base_model == "Qwen/Qwen2.5-1.5B"  # None override ignored
    assert cfg.max_examples == 10


def test_load_config_rejects_unknown_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("mode: cot\nnonsense: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(p)


# -------------------------------
# ChatML validation / reading
# -------------------------------

def test_validate_chatml_row():
    assert validate_chatml_row(_row()) is True
    assert validate_chatml_row({"messages": []}) is False
    assert validate_chatml_row({"messages": [{"role": "user", "content": "x"}]}) is False
    # empty assistant content -> unusable
    assert validate_chatml_row(_row(assistant="   ")) is False
    assert validate_chatml_row("not a dict") is False


def test_read_chatml_skips_blank_and_malformed(tmp_path):
    p = tmp_path / "d.jsonl"
    lines = [
        json.dumps(_row("a", "\\boxed{1}")),
        "",                               # blank
        "{not json}",                     # malformed
        json.dumps({"messages": [{"role": "user", "content": "x"}]}),  # invalid (no assistant)
        json.dumps(_row("b", "\\boxed{2}")),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    rows = read_chatml(p)
    assert len(rows) == 2


def test_read_chatml_respects_max_examples(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(json.dumps(_row(str(i))) for i in range(10)), encoding="utf-8")
    assert len(read_chatml(p, max_examples=3)) == 3


# -------------------------------
# split
# -------------------------------

def test_split_no_val_when_ratio_zero():
    rows = [_row(str(i)) for i in range(10)]
    train, val = train_val_split(rows, val_ratio=0.0)
    assert len(train) == 10 and val == []


def test_split_partitions_without_overlap_and_is_deterministic():
    rows = [_row(str(i)) for i in range(20)]
    t1, v1 = train_val_split(rows, val_ratio=0.25, seed=42)
    t2, v2 = train_val_split(rows, val_ratio=0.25, seed=42)
    assert len(v1) == 5 and len(t1) == 15
    assert t1 == t2 and v1 == v2                      # deterministic
    users = lambda rs: {r["messages"][0]["content"] for r in rs}
    assert users(t1).isdisjoint(users(v1))            # no overlap
    assert users(t1) | users(v1) == users(rows)       # covers everything


# -------------------------------
# formatting
# -------------------------------

class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False):
        assert tokenize is False
        return "\n".join(f"<{m['role']}>{m['content']}" for m in messages)


def test_to_training_text_uses_chat_template():
    text = to_training_text(_row("hi", "yo"), _FakeTokenizer())
    assert text == "<user>hi\n<assistant>yo"
