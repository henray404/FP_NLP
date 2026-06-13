"""
Extract math questions from PDFs with page-level traceability and image handling.

Output JSONL fields:
- question, answer, steps
- source_file, source_page, question_id
- source_bbox (if available; None otherwise)
- has_image, image_files, image_bboxes
- extraction_method (text or ocr)
- raw_text (original question text chunk)
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber
import pytesseract

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional
    pipeline = None


# -------------------------------
# Text parsing helpers
# -------------------------------

SOAL_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*SOAL\s+(\d+)\s*[:.)]?\s+(.*?)(?=\n\s*SOAL\s+\d+|\Z)",
    re.DOTALL | re.IGNORECASE,
)

NUMBERED_BLOCK_RE = re.compile(
    r"(?:^|\n)\s*(\d{1,3})[.)]\s+(.*?)(?=\n\s*\d{1,3}[.)]\s+|\Z)",
    re.DOTALL,
)


KUNCI_MARKER_RE = re.compile(
    r"\n\s*(KUNCI|JAWABAN|PEMBAHASAN|PENYELESAIAN)\b",
    re.IGNORECASE,
)

IMAGE_HINT_RE = re.compile(
    r"\b(gambar|diagram|grafik|tabel|foto|ilustrasi|fig|figure)\b",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_answer_sections(text: str) -> str:
    if not text:
        return ""
    m = KUNCI_MARKER_RE.search(text)
    if m:
        return text[: m.start()].strip()
    return text.strip()


def split_questions(text: str) -> Tuple[List[str], bool]:
    """
    Return (blocks, has_numbering).
    - blocks: list of question text blocks
    - has_numbering: True if SOAL/numbered pattern detected
    """
    text = text.strip()
    if not text:
        return [], False

    soal_blocks = [m.group(2).strip() for m in SOAL_BLOCK_RE.finditer(text)]
    if soal_blocks:
        return soal_blocks, True

    numbered_blocks = [m.group(2).strip() for m in NUMBERED_BLOCK_RE.finditer(text)]
    if numbered_blocks:
        return numbered_blocks, True

    return [text], False


def question_mentions_image(text: str) -> bool:
    return bool(IMAGE_HINT_RE.search(text))


def looks_complete(text: str) -> bool:
    text = text.strip()
    return text.endswith((".", "?", "!", ":", ";"))


# -------------------------------
# OCR and VLM captioning
# -------------------------------

@dataclass
class VLMConfig:
    model_id: str
    max_tokens: int = 64


class VLMCaptioner:
    def __init__(self, config: VLMConfig):
        if pipeline is None:
            raise RuntimeError("transformers is not available for VLM captioning")
        self.pipe = pipeline("image-to-text", model=config.model_id)
        self.max_tokens = config.max_tokens

    def caption(self, image) -> str:
        try:
            outputs = self.pipe(image, max_new_tokens=self.max_tokens)
            if outputs and isinstance(outputs, list):
                text = outputs[0].get("generated_text", "")
                return text.strip()
        except Exception:
            return ""
        return ""


def ocr_image(image, ocr_lang: str, ocr_psm: int) -> str:
    config = f"--psm {ocr_psm}"
    return pytesseract.image_to_string(image, lang=ocr_lang, config=config)


def extract_image_text(image, ocr_lang: str, ocr_psm: int, captioner: VLMCaptioner | None) -> str:
    parts: List[str] = []
    ocr_text = clean_text(ocr_image(image, ocr_lang, ocr_psm))
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    if captioner is not None:
        caption = clean_text(captioner.caption(image))
        if caption:
            parts.append(f"VLM: {caption}")
    return "\n".join(parts).strip()


# -------------------------------
# PDF extraction
# -------------------------------


def extract_page_text(page, ocr_lang: str, ocr_psm: int) -> Tuple[str, str]:
    text = page.extract_text() or ""
    method = "text"
    if not text.strip():
        method = "ocr"
        page_image = page.to_image(resolution=300).original
        text = ocr_image(page_image, ocr_lang, ocr_psm)
    return text, method


def extract_page_images(
    page,
    image_dir: Path,
    page_num: int,
    min_area: int,
    ocr_lang: str,
    ocr_psm: int,
    captioner: VLMCaptioner | None,
) -> Tuple[List[Path], List[List[float]], List[str]]:
    image_dir.mkdir(parents=True, exist_ok=True)

    image_files: List[Path] = []
    image_bboxes: List[List[float]] = []
    image_texts: List[str] = []

    for idx, img in enumerate(page.images):
        x0, x1 = float(img.get("x0", 0)), float(img.get("x1", 0))
        top, bottom = float(img.get("top", 0)), float(img.get("bottom", 0))
        area = max(0.0, x1 - x0) * max(0.0, bottom - top)
        if area < float(min_area):
            continue

        bbox = (x0, top, x1, bottom)
        try:
            cropped = page.crop(bbox).to_image(resolution=300).original
        except Exception:
            continue

        filename = f"page_{page_num:03d}_img_{idx + 1:02d}.png"
        out_path = image_dir / filename
        cropped.save(out_path)

        image_files.append(out_path)
        image_bboxes.append([x0, top, x1, bottom])

        image_text = extract_image_text(cropped, ocr_lang, ocr_psm, captioner)
        if image_text:
            image_texts.append(image_text)

    return image_files, image_bboxes, image_texts


def to_rel_path(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def merge_image_text(question_text: str, image_texts: List[str]) -> str:
    if not image_texts:
        return question_text
    return (
        question_text
        + "\n\n[Deskripsi Gambar]\n"
        + "\n".join(image_texts)
    ).strip()


def build_question_record(
    question_text: str,
    raw_text: str,
    pdf_path: Path,
    page_num: int,
    q_idx: int,
    extraction_method: str,
    image_files: List[Path],
    image_bboxes: List[List[float]],
    image_texts: List[str],
    out_dir: Path,
) -> dict:
    question_text = clean_text(question_text)
    question_text = merge_image_text(question_text, image_texts)

    record = {
        "question": question_text,
        "answer": "",
        "steps": "",
        "source_file": pdf_path.name,
        "source_page": page_num,
        "question_id": f"{pdf_path.stem}_p{page_num}_q{q_idx:02d}",
        "source_bbox": None,
        "has_image": bool(image_files),
        "image_files": [to_rel_path(p, out_dir) for p in image_files],
        "image_bboxes": image_bboxes,
        "extraction_method": extraction_method,
        "raw_text": clean_text(raw_text),
    }

    if image_texts:
        record["image_text"] = "\n".join(image_texts)

    return record


def process_pdf(
    pdf_path: Path,
    out_dir: Path,
    image_root: Path,
    ocr_lang: str,
    ocr_psm: int,
    min_image_area: int,
    captioner: VLMCaptioner | None,
    verbose: bool,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{pdf_path.stem}.jsonl"
    image_dir = image_root / pdf_path.stem

    total_questions = 0
    pending: dict | None = None

    with pdfplumber.open(pdf_path) as pdf, open(out_path, "w", encoding="utf-8") as fout:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text, method = extract_page_text(page, ocr_lang, ocr_psm)
            page_text = strip_answer_sections(page_text)
            page_text = clean_text(page_text)

            image_files, image_bboxes, image_texts = extract_page_images(
                page,
                image_dir=image_dir,
                page_num=page_num,
                min_area=min_image_area,
                ocr_lang=ocr_lang,
                ocr_psm=ocr_psm,
                captioner=captioner,
            )

            blocks, has_nums = split_questions(page_text)

            if pending and not has_nums:
                pending["raw_text"] += "\n" + page_text
                existing_image_text = pending.get("image_text", "").strip()
                merged_image_texts = []
                if existing_image_text:
                    merged_image_texts.append(existing_image_text)
                merged_image_texts.extend(image_texts)
                pending["image_text"] = "\n".join(merged_image_texts).strip()
                pending["question"] = merge_image_text(
                    clean_text(pending["raw_text"]),
                    merged_image_texts,
                )
                pending["source_page_end"] = page_num
                pending["image_files"].extend([to_rel_path(p, out_dir) for p in image_files])
                pending["image_bboxes"].extend(image_bboxes)
                continue

            if pending:
                fout.write(json.dumps(pending, ensure_ascii=False) + "\n")
                total_questions += 1
                pending = None

            if not blocks:
                continue

            multi_on_page = len(blocks) > 1

            for idx, block in enumerate(blocks, start=1):
                attach_images = False
                if image_files:
                    if question_mentions_image(block):
                        attach_images = True
                    elif not multi_on_page:
                        attach_images = True

                use_images = image_files if attach_images else []
                use_bboxes = image_bboxes if attach_images else []
                use_image_texts = image_texts if attach_images else []

                record = build_question_record(
                    question_text=block,
                    raw_text=block,
                    pdf_path=pdf_path,
                    page_num=page_num,
                    q_idx=idx,
                    extraction_method=method,
                    image_files=use_images,
                    image_bboxes=use_bboxes,
                    image_texts=use_image_texts,
                    out_dir=out_dir,
                )

                if not has_nums and idx == len(blocks) and not looks_complete(block):
                    pending = record
                    pending["source_page_end"] = page_num
                else:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_questions += 1

        if pending:
            fout.write(json.dumps(pending, ensure_ascii=False) + "\n")
            total_questions += 1

    if verbose:
        print(f"[{total_questions:3d} soal] {pdf_path.name} -> {out_path.name}")

    return total_questions


def iter_pdfs(input_path: Path) -> Iterable[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.pdf"))
    return [input_path]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract math questions from PDFs with page traceability and image handling",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="data/raw/osn",
        help="PDF file or folder (default: data/raw/osn)",
    )
    parser.add_argument("--out-dir", default="data/extracted")
    parser.add_argument("--image-dir", default="data/extracted/images")
    parser.add_argument("--ocr-lang", default="ind+eng")
    parser.add_argument("--ocr-psm", type=int, default=6)
    parser.add_argument("--min-image-area", type=int, default=1000)
    parser.add_argument("--vlm-model", default="")
    parser.add_argument("--vlm-max-tokens", type=int, default=64)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    image_root = Path(args.image_dir)

    captioner = None
    if args.vlm_model:
        captioner = VLMCaptioner(VLMConfig(model_id=args.vlm_model, max_tokens=args.vlm_max_tokens))

    pdf_files = list(iter_pdfs(input_path))
    if not pdf_files:
        print(f"No PDFs found in {input_path}")
        return

    total = 0
    for pdf_path in pdf_files:
        total += process_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir,
            image_root=image_root,
            ocr_lang=args.ocr_lang,
            ocr_psm=args.ocr_psm,
            min_image_area=args.min_image_area,
            captioner=captioner,
            verbose=args.verbose,
        )

    if args.verbose:
        print(f"Total extracted: {total}")


if __name__ == "__main__":
    main()
