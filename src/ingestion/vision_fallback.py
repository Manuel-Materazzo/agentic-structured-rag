"""
vision_fallback.py — Vision-mode fallback parser for PDFs with low parsing confidence.

When DoclingParser returns text that the LLM marks as parsing_confidence="low",
this module re-processes the PDF using DoclingParser with OCR/vision enabled,
or falls back to rendering pages as images for GPT-4o vision.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def parse_with_vision(source_path: str) -> str:
    """
    Re-parse a document using OCR/vision mode.

    Strategy:
    1. Try DoclingParser with EasyOCR enabled (handles scanned PDFs)
    2. If that fails, fall back to reading raw bytes and passing to AI vision model

    Args:
        source_path: Path to the PDF or document

    Returns:
        Extracted text string
    """
    log.info(f"[vision_fallback] Starting vision parse for: {source_path}")

    # Strategy 1: DoclingParser with OCR
    try:
        text = _parse_with_docling_ocr(source_path)
        if text and len(text.strip()) > 50:
            log.info(f"[vision_fallback] OCR parse succeeded for {source_path}")
            return text
    except Exception as exc:
        log.warning(f"[vision_fallback] DoclingParser OCR failed: {exc}")

    # Strategy 2: LLM vision (page images)
    try:
        text = _parse_with_llm_vision(source_path)
        if text:
            log.info(f"[vision_fallback] LLM vision parse succeeded for {source_path}")
            return text
    except Exception as exc:
        log.warning(f"[vision_fallback] LLM vision fallback failed: {exc}")

    # Last resort: try reading as plain text
    log.warning(f"[vision_fallback] All vision strategies failed for {source_path}, returning empty string")
    return ""


def _parse_with_docling_ocr(source_path: str) -> str:
    """Parse document using DoclingParser with EasyOCR enabled."""
    from datapizza.modules.parsers.docling import DoclingParser, OCREngine, OCROptions

    ocr_options = OCROptions(engine=OCREngine.EASY_OCR)
    parser = DoclingParser(ocr_options=ocr_options)

    nodes = parser(source_path)
    return "\n\n".join(n.text for n in nodes if n.text)


def _parse_with_llm_vision(source_path: str) -> str:
    """
    Convert PDF pages to images and pass to vison LLM for text extraction.
    Requires pypdfium2 (already installed via docling deps).
    """
    import base64
    import io

    import pypdfium2 as pdfium
    from openai import OpenAI
    from src.app.config import OPENAI_BASE_URL, OPENAI_API_KEY

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    path = Path(source_path)

    pdf = pdfium.PdfDocument(str(path))
    all_text_parts: list[str] = []

    MAX_PAGES = 10  # cap to avoid excessive API costs

    for page_idx in range(min(len(pdf), MAX_PAGES)):
        page = pdf[page_idx]
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()

        # Encode as base64 JPEG
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buf.getvalue()).decode()

        #TODO: Make model and prompt configurable
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract all text from this menu page faithfully, "
                                "preserving dish names, ingredients and quantities. "
                                "Output plain text only."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            max_tokens=2048,
        )
        page_text = response.choices[0].message.content or ""
        all_text_parts.append(f"--- Page {page_idx + 1} ---\n{page_text}")

    return "\n\n".join(all_text_parts)
