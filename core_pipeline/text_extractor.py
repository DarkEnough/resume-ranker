from __future__ import annotations

import io
import logging
import re
from pathlib import Path
from typing import Final

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)

MAX_PDF_PAGES: Final[int] = 30


def _extract_pdf(path: Path) -> str:
    """Simple extraction"""
    with pdfplumber.open(str(path)) as pdf:
        all_text = []
        for page in pdf.pages[:MAX_PDF_PAGES]:
            text = page.extract_text()
            if text:
                all_text.append(text)
        
        full_text = ' '.join(all_text)
        
        # Basic cleanup
        full_text = re.sub(r'(\w)-\s+(\w)', r'\1\2', full_text)  # Fix hyphenation
        full_text = re.sub(r'\s+', ' ', full_text)  # Normalize spaces
        full_text = re.sub(r'\n+', '\n', full_text)  # Normalize newlines
        
        return full_text


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


def _extract_txt(path: Path) -> str:
    return Path(path).read_text(errors="ignore")


def extract_text(path: Path) -> str:
    """
    Returns text for a supported file type, else empty string.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            text = _extract_pdf(path)
            return text
        if suffix in {".docx", ".doc"}:
            return _extract_docx(path)
        if suffix in {".txt", ".text"}:
            return _extract_txt(path)
    except Exception as exc:
        logger.exception("Extraction failed (%s): %s", path.name, exc)
    return ""