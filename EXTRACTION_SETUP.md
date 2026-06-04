# Extraction Setup (Local)

Base install
- Create and activate a virtual environment (optional but recommended).
- Install Python deps:
  - pip install -r requirements.txt

System packages (OCR)
- Install Tesseract OCR and Indonesian language data:
  - sudo apt-get update
  - sudo apt-get install -y tesseract-ocr tesseract-ocr-ind

Optional: VLM image-to-text
- If you want image captioning beyond OCR, install an image-to-text model.
- Minimal extra deps for common captioning models:
  - pip install timm
- Example model you can pass to the CLI:
  - Salesforce/blip-image-captioning-base

Notes
- OCR uses tesseract with --ocr-lang ind+eng by default.
- VLM captioning is optional and only runs when --vlm-model is provided.

Run examples
- Folder of PDFs:
  - python -m src.data_pipeline.extract_questions data/raw/osn -v
- Single PDF:
  - python -m src.data_pipeline.extract_questions data/raw/osn/your_file.pdf -v

- With VLM captioning:
  - python -m src.data_pipeline.extract_questions data/raw/osn -v --vlm-model Salesforce/blip-image-captioning-base
