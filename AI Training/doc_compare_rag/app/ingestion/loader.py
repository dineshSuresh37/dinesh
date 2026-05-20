"""
loader.py – Format detection and routing to the appropriate parser.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Normalised unit of text with provenance metadata."""

    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parser classes  (thin wrappers — logic lives in the individual parser files)
# ---------------------------------------------------------------------------


class DocxParser:
    """Route .docx parsing to docx_parser.parse_docx."""

    def parse(
        self, file_path: str, source_file: str, version_label: str
    ) -> List[Document]:
        from app.ingestion.docx_parser import parse_docx  # lazy to avoid circulars

        return parse_docx(file_path, source_file, version_label)


class TextPdfParser:
    """Route text-PDF parsing to pdf_parser.parse_pdf."""

    def parse(
        self, file_path: str, source_file: str, version_label: str
    ) -> List[Document]:
        from app.ingestion.pdf_parser import parse_pdf

        return parse_pdf(file_path, source_file, version_label)


class OcrPdfParser:
    """Route scanned-PDF parsing to ocr_parser.parse_pdf_ocr."""

    def parse(
        self, file_path: str, source_file: str, version_label: str
    ) -> List[Document]:
        from app.ingestion.ocr_parser import parse_pdf_ocr

        return parse_pdf_ocr(file_path, source_file, version_label)


# ---------------------------------------------------------------------------
# HTML → .docx helper  (used only for .doc → .docx conversion via mammoth)
# ---------------------------------------------------------------------------


class _HtmlToDocxConverter(HTMLParser):
    """
    Minimal HTML-to-docx converter that preserves heading levels from
    mammoth's HTML output, producing a python-docx Document saved to disk.
    """

    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        try:
            from docx import Document as DocxDoc  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "python-docx is required. Install with: pip install python-docx"
            ) from exc
        self._doc = DocxDoc()
        self._tag: str | None = None
        self._buf: str = ""

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ARG002
        self._tag = tag
        self._buf = ""

    def handle_endtag(self, tag: str) -> None:
        text = self._buf.strip()
        if text:
            if tag in self._HEADING_TAGS:
                self._doc.add_heading(text, level=int(tag[1]))
            elif tag == "p":
                self._doc.add_paragraph(text)
        self._tag = None
        self._buf = ""

    def handle_data(self, data: str) -> None:
        self._buf += data

    def save(self, path: str) -> None:
        self._doc.save(path)


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

_PDF_TEXT_THRESHOLD = 50  # characters on page 1 to treat a PDF as text-extractable


class DocumentLoader:
    """
    Load a document from *file_path*, auto-detect its format, and return a
    normalised list of Document objects.

    Parameters
    ----------
    file_path:     Path to the source document (.docx, .doc, or .pdf).
    version_label: Caller-supplied label stored in every Document's metadata
                   (e.g. "v1", "v2").
    """

    def __init__(self, file_path: str, version_label: str = "v1") -> None:
        self.file_path = file_path
        self.version_label = version_label
        self._source_file = Path(file_path).name

    def load(self) -> List[Document]:
        """Detect the file format and return a list of Document objects."""
        path = Path(self.file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path!r}")

        suffix = path.suffix.lower()
        if suffix == ".docx":
            return DocxParser().parse(
                self.file_path, self._source_file, self.version_label
            )
        if suffix == ".doc":
            return self._load_doc()
        if suffix == ".pdf":
            return self._load_pdf()

        raise ValueError(f"Unsupported file format: {suffix!r}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_doc(self) -> List[Document]:
        """Convert .doc → temp .docx via mammoth, then parse with DocxParser."""
        try:
            import mammoth  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "mammoth is required for .doc conversion. "
                "Install with: pip install mammoth"
            ) from exc

        with open(self.file_path, "rb") as fh:
            result = mammoth.convert_to_html(fh)

        converter = _HtmlToDocxConverter()
        converter.feed(result.value)

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".docx")
        os.close(tmp_fd)
        try:
            converter.save(tmp_path)
            return DocxParser().parse(tmp_path, self._source_file, self.version_label)
        finally:
            os.unlink(tmp_path)

    def _load_pdf(self) -> List[Document]:
        """Route to TextPdfParser or OcrPdfParser based on page-1 text length."""
        try:
            import pdfplumber  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pdfplumber is required for PDF format detection. "
                "Install with: pip install pdfplumber"
            ) from exc

        with pdfplumber.open(self.file_path) as pdf:
            first_page_text = (
                (pdf.pages[0].extract_text() or "") if pdf.pages else ""
            )

        if len(first_page_text.strip()) > _PDF_TEXT_THRESHOLD:
            return TextPdfParser().parse(
                self.file_path, self._source_file, self.version_label
            )
        return OcrPdfParser().parse(
            self.file_path, self._source_file, self.version_label
        )


# ---------------------------------------------------------------------------
# Legacy functional API (backwards compatibility)
# ---------------------------------------------------------------------------


def detect_format(file_path: str) -> str:
    """Return the detected format string for the given file path."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".docx":
        return "docx"
    if suffix == ".doc":
        return "doc"
    if suffix == ".pdf":
        return "pdf"
    raise ValueError(f"Unsupported file format: {suffix!r}")


def load_document(file_path: str) -> List[str]:
    """
    Detect the file format and route to the correct parser.

    Returns a list of page/section text strings.
    """
    fmt = detect_format(file_path)

    if fmt in ("docx", "doc"):
        from app.ingestion.docx_parser import parse_docx

        return [d.text for d in parse_docx(file_path)]

    if fmt == "pdf":
        from app.ingestion.pdf_parser import parse_pdf

        docs = parse_pdf(file_path)
        if all(d.text.strip() == "" for d in docs):
            from app.ingestion.ocr_parser import parse_pdf_ocr

            docs = parse_pdf_ocr(file_path)
        return [d.text for d in docs]

    raise ValueError(f"No parser registered for format: {fmt!r}")


# ---------------------------------------------------------------------------
# __main__ – quick smoke-test from the command line
#   python -m app.ingestion.loader path/to/file.pdf --version v2
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m app.ingestion.loader",
        description="Load a document and print a summary of each extracted chunk.",
    )
    ap.add_argument("file_path", help="Path to the document (.docx, .doc, or .pdf)")
    ap.add_argument(
        "--version",
        metavar="LABEL",
        default="v1",
        help="Version label to attach to each Document (default: v1)",
    )
    args = ap.parse_args()

    try:
        loader = DocumentLoader(args.file_path, version_label=args.version)
        documents = loader.load()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(documents)} chunk(s) from {args.file_path!r}\n")
    for i, doc in enumerate(documents, start=1):
        m = doc.metadata
        ocr_err = f"  [OCR ERROR: {m['ocr_error']}]" if "ocr_error" in m else ""
        print(
            f"[{i:>3}] page={m['page_number']}  type={m['file_type']}"
            f"  section={m['section_title']!r}{ocr_err}"
        )
        preview = doc.text[:140].replace("\n", " ")
        if len(doc.text) > 140:
            preview += "…"
        print(f"       {preview}\n")
