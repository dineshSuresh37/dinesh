"""
pdf_parser.py – Parse text-based PDFs into Document objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.ingestion.loader import Document


def parse_pdf(
    file_path: str,
    source_file: str | None = None,
    version_label: str = "v1",
) -> List["Document"]:
    """
    Extract text page by page using pdfplumber.

    Attempts to detect section titles per page using two heuristics:
      1. A line whose alphabetic characters are ALL CAPS (≥ 2 words, ≤ 120 chars).
      2. A short line (no trailing period) immediately followed by a blank line.

    Raises
    ------
    FileNotFoundError  – if *file_path* does not exist.
    ValueError         – if the file extension is not .pdf.
    ImportError        – if pdfplumber is not installed.
    """
    from app.ingestion.loader import Document

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path!r}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"TextPdfParser only handles .pdf files, got: {path.suffix!r}"
        )
    if source_file is None:
        source_file = path.name

    try:
        import pdfplumber  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdfplumber is required. Install with: pip install pdfplumber"
        ) from exc

    documents: List[Document] = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_file": source_file,
                        "version_label": version_label,
                        "page_number": page_num,
                        "section_title": _detect_section_title(text),
                        "file_type": "pdf_text",
                    },
                )
            )

    return documents


def _detect_section_title(page_text: str) -> str:
    """Return the first plausible section title found in *page_text*, or ''."""
    lines = page_text.splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line or len(line) > 120:
            continue

        # Heuristic 1: line is ALL CAPS (at least 2 words to reduce noise)
        alpha_chars = [c for c in line if c.isalpha()]
        if alpha_chars and all(c.isupper() for c in alpha_chars) and len(line.split()) >= 2:
            return line

        # Heuristic 2: short line followed by a blank line, not ending with '.'
        next_blank = i + 1 < len(lines) and not lines[i + 1].strip()
        if next_blank and not line.endswith("."):
            return line

    return ""
