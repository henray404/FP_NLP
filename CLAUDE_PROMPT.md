# Prompt for Copilot Agent: Extract Math Questions from PDFs

```
You Please implement Python programs for this workspace to extract each math question from PDFs in data/raw/osn into JSONL files in data/extracted.

Hard requirements:
- Every question must be traceable to its PDF source and page number.
- Some questions include images (especially geometry). Those images must be captured or referenced in the output.
- Keep outputs JSONL and add minimal, stable metadata so downstream filtering still works.

Target output schema (per question, JSONL):
- question: string
- answer: string (optional, if found)
- steps: string (optional)
- source_file: PDF file name
- source_page: 1-based page number
- question_id: stable id, e.g. <pdf_stem>_p<page>_q<index>
- source_bbox: [x0, y0, x1, y1] in PDF coordinates if available
- has_image: bool
- image_files: list of image file paths (relative, if saved)
- image_bboxes: list of [x0, y0, x1, y1] for each image
- extraction_method: text or ocr
- raw_text: original page text chunk

Functional requirements:
- Process either a single PDF or a folder of PDFs via CLI.
- Split multiple questions on a page using numbering or known patterns; keep original order.
- If a page is scanned or text extraction is empty, run OCR (pytesseract) on the page image.
- For images: extract embedded images from PDF (pdfplumber or PyMuPDF), or crop image regions; save them to data/extracted/images/<pdf_stem>/.
- When a question references a figure or diagram, attach the nearest image or all images on that page.
- Keep text cleaning conservative (do not lose math notation). Preserve LaTeX where present.
- Provide clear logging and a summary count per PDF.

Implementation constraints:
- Use existing dependencies in requirements.txt where possible (pdfplumber, pytesseract, requests, langdetect, etc.).
- If you add a new dependency, update requirements.txt and justify it.
- Keep code in src/data_pipeline/. You may add a new module (e.g., extract_questions.py) and optionally refactor existing extract.py to reuse shared logic.

Deliverables:
- Code changes in src/data_pipeline/ for the new extractor.
- Any updates to requirements.txt (if needed).
- Short run instructions with example commands.
```
