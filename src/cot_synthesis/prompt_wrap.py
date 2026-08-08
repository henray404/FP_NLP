"""
Wrap each problem (`soal`) into the Indonesian teacher prompt.

Two prompt variants -- same problem, different instruction -- drive the CoT vs non-CoT experiment:
- "cot":   ask for a detailed step-by-step solution, final answer in \\boxed{}
- "nocot": ask for the final answer only, in \\boxed{}, no steps

The teacher is generated with the "cot" prompt; to_chatml.py reuses both to build the two training sets.
The data has no \\boxed in `jawaban`, so we explicitly force the teacher to emit one (we parse it back out).
"""
from __future__ import annotations

from .utils import get_soal

PROMPT_COT = (
    "Selesaikan soal matematika berikut. Tunjukkan langkah-langkah penyelesaian "
    "secara rinci dan sistematis dalam Bahasa Indonesia. Pastikan jawaban akhir "
    "(dan hanya jawaban) berada di dalam \\boxed{{}}.\n\n{soal}"
)

PROMPT_NOCOT = (
    "Selesaikan soal matematika berikut. Berikan HANYA jawaban akhir di dalam "
    "\\boxed{{}} tanpa langkah-langkah.\n\n{soal}"
)

_TEMPLATES = {"cot": PROMPT_COT, "nocot": PROMPT_NOCOT}


_SENTINEL = "\x00SOAL\x00"


def wrap(soal: str, mode: str = "cot") -> str:
    if mode not in _TEMPLATES:
        raise ValueError(f"mode must be one of {list(_TEMPLATES)}, got {mode!r}")
    return _TEMPLATES[mode].format(soal=soal.strip())


def unwrap(prompt: str, mode: str | None = None) -> str:
    """Kebalikan `wrap`: kupas instruksinya, kembalikan `soal` mentah.

    Dipakai verify_pairing.py untuk membandingkan himpunan soal antara cot.jsonl dan
    nocot.jsonl -- kedua file hanya menyimpan prompt yang sudah dibungkus, tanpa kolom id.
    `mode=None` mencoba semua template. Kalau tidak ada yang cocok, prompt dikembalikan apa
    adanya (sudah di-strip) supaya perbandingan tetap jalan pada data lama.
    """
    templates = _TEMPLATES if mode is None else {mode: _TEMPLATES[mode]}
    for tpl in templates.values():
        head, _, tail = tpl.format(soal=_SENTINEL).partition(_SENTINEL)
        if prompt.startswith(head) and prompt.endswith(tail):
            end = len(prompt) - len(tail) if tail else len(prompt)
            return prompt[len(head):end].strip()
    return prompt.strip()


def wrap_item(item: dict, mode: str = "cot") -> str:
    return wrap(get_soal(item), mode)


def wrap_all(items: list[dict], mode: str = "cot") -> list[str]:
    return [wrap_item(it, mode) for it in items]
