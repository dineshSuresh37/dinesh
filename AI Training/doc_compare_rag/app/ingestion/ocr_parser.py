"""
ocr_parser.py – Parse scanned PDFs via OCR into Document objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.ingestion.loader import Document


def parse_pdf_ocr(
    file_path: str,
    source_file: str | None = None,
    version_label: str = "v1",
) -> List["Document"]:
    """
    Convert each PDF page to an image and run pytesseract OCR on it.

    When OCR fails for a page the Document is still emitted with an empty
    ``text`` and an ``ocr_error`` key in its metadata so callers can detect
    and handle the failure without losing page provenance.

    Raises
    ------
    FileNotFoundError  – if *file_path* does not exist.
    ValueError         – if the file extension is not .pdf.
    ImportError        – if pdf2image or pytesseract is not installed.
    """
    from app.ingestion.loader import Document

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path!r}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"OcrPdfParser only handles .pdf files, got: {path.suffix!r}"
        )
    if source_file is None:
        source_file = path.name

    try:
        from pdf2image import convert_from_path  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pdf2image is required. Install with: pip install pdf2image"
        ) from exc

    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required. Install with: pip install pytesseract"
        ) from exc

    images = convert_from_path(file_path)
    documents: List[Document] = []
    for page_num, image in enumerate(images, start=1):
        base_meta = {
            "source_file": source_file,
            "version_label": version_label,
            "page_number": page_num,
            "section_title": "",
            "file_type": "pdf_ocr",
        }
        try:
            text = pytesseract.image_to_string(image).strip()
        except Exception as exc:  # noqa: BLE001
            documents.append(
                Document(text="", metadata={**base_meta, "ocr_error": str(exc)})
            )
            continue

        documents.append(Document(text=text, metadata=base_meta))

    return documents
