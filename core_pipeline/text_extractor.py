from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Final

from pypdf import PdfReader
from docx import Document

logger = logging.getLogger(__name__)

MAX_PDF_PAGES: Final[int] = 30


def _extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    if len(reader.pages) > MAX_PDF_PAGES:
        logger.warning("PDF truncated to first %s pages: %s", MAX_PDF_PAGES, path.name)
    pages = reader.pages[:MAX_PDF_PAGES]
    return "\n".join(page.extract_text() or "" for page in pages)


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
            return _extract_pdf(path)
        if suffix in {".docx", ".doc"}:
            return _extract_docx(path)
        if suffix in {".txt", ".text"}:
            return _extract_txt(path)
    except Exception as exc:
        logger.exception("Extraction failed (%s): %s", path.name, exc)
    return "" 