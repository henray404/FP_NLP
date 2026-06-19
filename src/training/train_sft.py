"""
Config-driven SFT (QLoRA) for the CoT vs non-CoT experiment.

Same base model + same hyperparameters; ONLY the dataset differs (data/sft/cot.jsonl vs
data/sft/nocot.jsonl) -> the accuracy gap isolates the CoT effect. Each run produces one
LoRA adapter; pair up cot/nocot adapters per base size and eval them with src.eval.

Stack: **Unsloth** (FastLanguageModel, ~2x faster + lower VRAM on T4) + trl SFTTrainer for
the loop. Training needs a GPU and is lazy-imported in build_and_train(); everything above it
(config, data, split, formatting) is pure Python and unit-tested on CPU
(src/training/test_train_sft.py). Note: Unsloth uses a single GPU, so on Kaggle 2xT4 only one
T4 is used (still faster than the 2-GPU transformers path for a 1.5B QLoRA run).

Usage:
    python -m src.training.train_sft --config src/training/configs/cot_1.5b.yaml
    python -m src.training.train_sft --config src/training/configs/cot_1.5b.yaml \
        --dataset data/sft/cot.jsonl --output-dir outputs/cot_1.5b --max-examples 200
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Qwen ChatML turn markers -> used for response-only loss masking (train on the assistant
# answer, not on the prompt). Override per-base in the yaml if you switch model families.
DEFAULT_INSTRUCTION_TEMPLATE = "<|im_start|>user\n"
DEFAULT_RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


@dataclass
class TrainConfig:
    # what to train
    base_model: str = "Qwen/Qwen2.5-1.5B"
    dataset: str = "data/sft/cot.jsonl"
    output_dir: str = "outputs/cot_1.5b"
    mode: str = "cot"                 # "cot" | "nocot" (bookkeeping only)
    # sequence / quantization
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # optimization
    epochs: float = 2.0
    batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.0
    logging_steps: int = 20
    save_strategy: str = "epoch"
    seed: int = 42
    # data handling
    val_ratio: float = 0.0           # 0 -> no validation split
    max_examples: int | None = None  # cap for smoke tests
    train_on_responses_only: bool = True
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE
    response_template: str = DEFAULT_RESPONSE_TEMPLATE

    def __post_init__(self) -> None:
        if self.mode not in ("cot", "nocot"):
            raise ValueError(f"mode must be 'cot' or 'nocot', got {self.mode!r}")
        if not 0.0 <= self.val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in [0, 1), got {self.val_ratio}")


_FIELD_NAMES = {f.name for f in fields(TrainConfig)}


def load_config(path: str | Path, **overrides) -> TrainConfig:
    """Load a yaml/json config into TrainConfig. Unknown keys raise; CLI overrides win
    (None overrides are ignored so absent CLI flags don't clobber the file)."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    unknown = set(data) - _FIELD_NAMES
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**data)


# -------------------------------
# Data (pure Python, CPU-testable)
# -------------------------------

def validate_chatml_row(obj: object) -> bool:
    """A usable SFT row is {"messages": [...]} with a user turn followed by a non-empty
    assistant turn."""
    if not isinstance(obj, dict):
        return False
    msgs = obj.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False
    roles = [m.get("role") for m in msgs if isinstance(m, dict)]
    if "user" not in roles or "assistant" not in roles:
        return False
    for m in msgs:
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return True
    return False


def read_chatml(path: str | Path, max_examples: int | None = None) -> list[dict]:
    """Read ChatML JSONL, skipping blank/malformed lines and invalid rows."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if validate_chatml_row(obj):
                rows.append(obj)
                if max_examples is not None and len(rows) >= max_examples:
                    break
    return rows


def train_val_split(rows: list[dict], val_ratio: float = 0.0,
                    seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic shuffle + split. val_ratio<=0 -> all rows go to train, val empty."""
    if val_ratio <= 0 or len(rows) < 2:
        return list(rows), []
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(rows) * val_ratio))
    val_idx = set(idx[:n_val])
    train = [rows[i] for i in range(len(rows)) if i not in val_idx]
    val = [rows[i] for i in range(len(rows)) if i in val_idx]
    return train, val


def to_training_text(row: dict, tokenizer) -> str:
    """Render one ChatML row to the model's chat-formatted text (the SFT 'text' field)."""
    return tokenizer.apply_chat_template(row["messages"], tokenize=False)


# -------------------------------
# Training (GPU; lazy imports)
# -------------------------------

def build_and_train(cfg: TrainConfig) -> str:
    """Run SFT for one config and save the LoRA adapter. Returns the output dir.

    Uses **Unsloth** (FastLanguageModel) for ~2x faster training and lower VRAM on the
    free-tier T4. `import unsloth` must come first so it can patch transformers/trl before
    they load. The rest of the SFT loop is still trl's SFTTrainer (Unsloth-patched).
    Everything above this function is framework-agnostic and unit-tested on CPU."""
    import gc

    # Unsloth first: it monkeypatches transformers/peft/trl on import for the fast path.
    from unsloth import FastLanguageModel, is_bfloat16_supported
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    rows = read_chatml(cfg.dataset, max_examples=cfg.max_examples)
    if not rows:
        raise RuntimeError(f"no usable rows in {cfg.dataset}")
    train_rows, val_rows = train_val_split(rows, cfg.val_ratio, cfg.seed)
    print(f"[{cfg.mode}] {cfg.base_model}: train={len(train_rows)} val={len(val_rows)}")

    # Load 4-bit base + tokenizer via Unsloth (handles quantization + fast kernels itself).
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model, max_seq_length=cfg.max_seq_length,
        dtype=None, load_in_4bit=cfg.load_in_4bit)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Attach LoRA adapters through Unsloth (replaces peft.LoraConfig + get_peft_model).
    model = FastLanguageModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", target_modules=cfg.target_modules,
        use_gradient_checkpointing="unsloth", random_state=cfg.seed,
        max_seq_length=cfg.max_seq_length)

    def _to_text(ex):
        return {"text": to_training_text(ex, tok)}

    train_ds = Dataset.from_list(train_rows).map(_to_text, remove_columns=["messages"])
    eval_ds = (Dataset.from_list(val_rows).map(_to_text, remove_columns=["messages"])
               if val_rows else None)

    sft_args = SFTConfig(
        output_dir=cfg.output_dir, num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum, learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio, lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
        logging_steps=cfg.logging_steps, optim="adamw_8bit",
        save_strategy=cfg.save_strategy, seed=cfg.seed, max_seq_length=cfg.max_seq_length,
        report_to="none", dataset_text_field="text",
        eval_strategy="epoch" if eval_ds is not None else "no")

    trainer = SFTTrainer(model=model, tokenizer=tok, args=sft_args,
                         train_dataset=train_ds, eval_dataset=eval_ds)

    if cfg.train_on_responses_only:
        # Mask the prompt so loss is computed on the assistant answer only (Unsloth helper).
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer, instruction_part=cfg.instruction_template,
            response_part=cfg.response_template)

    trainer.train()
    model.save_pretrained(cfg.output_dir)   # LoRA adapter
    tok.save_pretrained(cfg.output_dir)
    Path(cfg.output_dir, "train_config.json").write_text(
        json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved adapter -> {cfg.output_dir}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cfg.output_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Config-driven QLoRA SFT (CoT / non-CoT)")
    ap.add_argument("--config", required=True, help="path to a yaml config (configs/*.yaml)")
    ap.add_argument("--dataset", default=None, help="override config.dataset")
    ap.add_argument("--output-dir", default=None, help="override config.output_dir")
    ap.add_argument("--base-model", default=None, help="override config.base_model")
    ap.add_argument("--max-examples", type=int, default=None,
                    help="override config.max_examples (smoke test)")
    args = ap.parse_args()

    cfg = load_config(args.config, dataset=args.dataset, output_dir=args.output_dir,
                      base_model=args.base_model, max_examples=args.max_examples)
    build_and_train(cfg)


if __name__ == "__main__":
    main()
