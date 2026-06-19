"""
Shared helpers for CoT synthesis: JSONL IO, \\boxed{} extraction, OpenAI/Groq client.

Data schema produced by the data pipeline is {soal, cara, jawaban}:
- soal    : the problem statement
- cara    : reference worked solution (steps) -- NOT used by teacher distillation, kept for later
- jawaban : final answer as a short natural-language sentence (NO \\boxed, e.g. "Hasilnya $y=-x-6$.")

Correctness here is decided by an LLM judge (see filter_solutions.py), so the heuristic
answer-equivalence helpers below are optional shortcuts, off by default.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

T = TypeVar("T")

# -------------------------------
# JSONL IO
# -------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_soal(item: dict) -> str:
    return (item.get("soal") or item.get("question") or "").strip()


def get_cara(item: dict) -> str:
    return str(item.get("cara") or item.get("solution") or "").strip()


def get_jawaban(item: dict) -> str:
    return str(item.get("jawaban") or item.get("answer") or "").strip()


def problem_id(item: dict, idx: int) -> str:
    return str(item.get("question_id") or item.get("id") or f"q{idx:05d}")


# -------------------------------
# OpenAI-compatible client (Groq / DeepSeek / local vLLM server)
# -------------------------------

def openai_client():
    """OpenAI-compatible client. Auto-detects Groq / DeepSeek from whichever API key is set.

    env:
      GROQ_API_KEY     -> base https://api.groq.com/openai/v1   (default if set)
      DEEPSEEK_API_KEY -> base https://api.deepseek.com
      OPENAI_API_KEY   + OPENAI_BASE_URL  -> anything OpenAI-compatible
    OPENAI_BASE_URL always wins if set explicitly.
    """
    from openai import OpenAI

    try:  # best-effort: pick up keys from a .env in the repo root
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    key = (os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("DEEPSEEK_API_KEY"))
    if not key:
        raise RuntimeError(
            "No API key found. Set GROQ_API_KEY (recommended) or OPENAI_API_KEY / DEEPSEEK_API_KEY."
        )
    base = os.environ.get("OPENAI_BASE_URL")
    if not base:
        if os.environ.get("GROQ_API_KEY"):
            base = "https://api.groq.com/openai/v1"
        elif os.environ.get("DEEPSEEK_API_KEY"):
            base = "https://api.deepseek.com"
        else:
            base = "https://api.groq.com/openai/v1"
    return OpenAI(base_url=base, api_key=key)


def with_retry(fn: Callable[[], T], *, tries: int = 10, base_wait: float = 5.0,
               max_wait: float = 60.0) -> T:
    """Call fn() with exponential backoff. Handles Groq free-tier 429 rate limits and
    transient network errors. Re-raises the last error after `tries` attempts."""
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - provider SDKs raise varied types
            last = e
            msg = str(e).lower()
            transient = ("rate" in msg or "429" in msg or "timeout" in msg
                         or "overloaded" in msg or "503" in msg or "502" in msg)
            if attempt == tries - 1 or not transient:
                if not transient:
                    raise
                break
            time.sleep(min(base_wait * (2 ** attempt), max_wait))
    assert last is not None
    raise last


# -------------------------------
# \boxed{} extraction (brace-balanced)
# -------------------------------

def extract_boxed(text: str) -> str | None:
    """Return content of the LAST \\boxed{...}, handling nested braces. None if absent."""
    marker = r"\boxed"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    while i < len(text) and text[i] != "{":
        i += 1
    if i >= len(text):
        return None
    depth = 0
    out = []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
        out.append(ch)
    return None  # unbalanced -> treat as incomplete


def has_boxed(text: str) -> bool:
    return extract_boxed(text) is not None


# -------------------------------
# Optional heuristic answer equivalence (off by default; judge is primary)
# -------------------------------

_WS_RE = re.compile(r"\s+")


def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    s = str(ans).strip()
    inner = extract_boxed(s)
    if inner is not None:
        s = inner
    s = s.strip().strip("$").strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", " ")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("{,}", ".")
    s = s.rstrip(".")
    s = _WS_RE.sub("", s)
    if re.fullmatch(r"-?\d+,\d+", s):
        s = s.replace(",", ".")
    return s.lower()


def _try_math_verify(pred: str, gold: str) -> bool | None:
    try:
        from math_verify import parse, verify  # type: ignore
    except Exception:
        return None
    try:
        g = parse(gold if "\\boxed" in gold or "$" in gold else f"${gold}$")
        p = parse(pred if "\\boxed" in pred or "$" in pred else f"${pred}$")
        return bool(verify(g, p))
    except Exception:
        return None


def answers_equivalent(pred: str, gold: str) -> bool:
    """Cheap string/math_verify equivalence. True only when confidently equal.
    Used only as an optional pre-filter before the LLM judge."""
    if not gold:
        return False
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ and np_ == ng:
        return True
    return _try_math_verify(pred, gold) is True
