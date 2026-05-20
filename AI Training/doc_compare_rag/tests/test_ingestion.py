"""
test_ingestion.py – Tests for the ingestion layer.

Covers:
  - DocxParser  : real .docx fixture built with python-docx
  - TextPdfParser : real PDF fixture built with fpdf2
  - OcrPdfParser  : pytesseract mocked out
  - DocumentLoader: format routing and metadata completeness
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.loader import (
    Document,
    DocumentLoader,
    OcrPdfParser,
    TextPdfParser,
)
from app.ingestion.docx_parser import parse_docx
from app.ingestion.pdf_parser import parse_pdf

# Required metadata keys every Document must carry.
_REQUIRED_META = {"source_file", "version_label", "page_number", "section_title", "file_type"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_metadata(docs: list[Document]) -> None:
    """Assert every Document in *docs* has all required metadata keys."""
    assert docs, "Expected at least one Document"
    for doc in docs:
        missing = _REQUIRED_META - doc.metadata.keys()
        assert not missing, f"Missing metadata keys {missing} in {doc.metadata}"


# ---------------------------------------------------------------------------
# DocxParser
# ---------------------------------------------------------------------------


class TestDocxParser:
    def test_returns_documents(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="v1")
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_text_is_non_empty(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="v1")
        non_empty = [d for d in docs if d.text.strip()]
        assert non_empty, "Expected at least one Document with non-empty text"

    def test_section_title_detected(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="v1")
        titles = {d.metadata.get("section_title", "") for d in docs}
        # The .docx fixture has headings "Introduction" and "Background".
        assert any("introduction" in t.lower() for t in titles), (
            f"Expected 'Introduction' heading; got titles: {titles}"
        )

    def test_metadata_fields_populated(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="v1")
        _assert_metadata(docs)

    def test_version_label_propagated(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="release-2")
        assert all(d.metadata["version_label"] == "release-2" for d in docs)

    def test_file_type_is_docx(self, minimal_docx_path):
        docs = parse_docx(minimal_docx_path, version_label="v1")
        assert all(d.metadata["file_type"] == "docx" for d in docs)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_docx(str(tmp_path / "nonexistent.docx"), version_label="v1")


# ---------------------------------------------------------------------------
# TextPdfParser (real fpdf2 PDF)
# ---------------------------------------------------------------------------


class TestPdfParser:
    def test_returns_documents(self, minimal_pdf_path):
        docs = parse_pdf(minimal_pdf_path, version_label="v1")
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)

    def test_text_extracted(self, minimal_pdf_path):
        docs = parse_pdf(minimal_pdf_path, version_label="v1")
        combined = " ".join(d.text for d in docs)
        assert len(combined.strip()) > 0

    def test_metadata_fields_populated(self, minimal_pdf_path):
        docs = parse_pdf(minimal_pdf_path, version_label="v1")
        _assert_metadata(docs)

    def test_file_type_is_pdf_text(self, minimal_pdf_path):
        docs = parse_pdf(minimal_pdf_path, version_label="v1")
        assert all(d.metadata["file_type"] == "pdf_text" for d in docs)

    def test_version_label_propagated(self, minimal_pdf_path):
        docs = parse_pdf(minimal_pdf_path, version_label="draft")
        assert all(d.metadata["version_label"] == "draft" for d in docs)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_pdf(str(tmp_path / "ghost.pdf"), version_label="v1")


# ---------------------------------------------------------------------------
# OcrPdfParser (pytesseract mocked)
# ---------------------------------------------------------------------------


class TestOcrParser:
    @staticmethod
    def _mock_pil_image():
        img = MagicMock()
        img.size = (100, 100)
        return img

    def test_ocr_returns_documents(self, minimal_pdf_path):
        fake_image = self._mock_pil_image()
        with (
            patch("pdf2image.convert_from_path", return_value=[fake_image]),
            patch("pytesseract.image_to_string", return_value="Mocked OCR output text."),
        ):
            from app.ingestion.ocr_parser import parse_pdf_ocr

            docs = parse_pdf_ocr(minimal_pdf_path, version_label="v1")

        assert len(docs) >= 1
        assert docs[0].text == "Mocked OCR output text."

    def test_ocr_metadata_populated(self, minimal_pdf_path):
        fake_image = self._mock_pil_image()
        with (
            patch("pdf2image.convert_from_path", return_value=[fake_image]),
            patch("pytesseract.image_to_string", return_value="Some OCR text here."),
        ):
            from app.ingestion.ocr_parser import parse_pdf_ocr

            docs = parse_pdf_ocr(minimal_pdf_path, version_label="scan-v1")

        _assert_metadata(docs)

    def test_ocr_failure_emits_error_doc(self, minimal_pdf_path):
        """A failed OCR page should emit an empty-text Document, not raise."""
        fake_image = self._mock_pil_image()
        with (
            patch("pdf2image.convert_from_path", return_value=[fake_image]),
            patch(
                "pytesseract.image_to_string",
                side_effect=RuntimeError("Tesseract crashed"),
            ),
        ):
            from app.ingestion.ocr_parser import parse_pdf_ocr

            docs = parse_pdf_ocr(minimal_pdf_path, version_label="v1")

        assert len(docs) == 1
        assert docs[0].text == ""
        assert "ocr_error" in docs[0].metadata

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            from app.ingestion.ocr_parser import parse_pdf_ocr

            parse_pdf_ocr(str(tmp_path / "ghost.pdf"), version_label="v1")


# ---------------------------------------------------------------------------
# DocumentLoader – format routing
# ---------------------------------------------------------------------------


class TestDocumentLoader:
    # ── .docx routing ────────────────────────────────────────────────────────

    def test_docx_routes_to_docx_parser(self, minimal_docx_path):
        docs = DocumentLoader(minimal_docx_path, version_label="v1").load()
        assert docs
        assert all(d.metadata["file_type"] == "docx" for d in docs)

    # ── text PDF routing ─────────────────────────────────────────────────────

    def test_text_pdf_routes_to_text_parser(self, minimal_pdf_path):
        docs = DocumentLoader(minimal_pdf_path, version_label="v1").load()
        assert docs
        # TextPdfParser sets file_type = "pdf_text"
        assert all(d.metadata["file_type"] == "pdf_text" for d in docs)

    # ── scanned PDF routing (empty pdfplumber text → OCR) ───────────────────

    def test_scanned_pdf_routes_to_ocr(self, minimal_pdf_path):
        """Mock pdfplumber to return no text, verify OCR path is taken."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_pdf_cm = MagicMock()
        mock_pdf_cm.__enter__ = MagicMock(return_value=mock_pdf_cm)
        mock_pdf_cm.__exit__ = MagicMock(return_value=False)
        mock_pdf_cm.pages = [mock_page]

        fake_image = MagicMock()
        fake_image.size = (100, 100)

        with (
            patch("pdfplumber.open", return_value=mock_pdf_cm),
            patch("pdf2image.convert_from_path", return_value=[fake_image]),
            patch("pytesseract.image_to_string", return_value="OCR result text."),
        ):
            docs = DocumentLoader(minimal_pdf_path, version_label="v1").load()

        assert docs
        texts = [d.text for d in docs]
        assert any(t == "OCR result text." for t in texts)

    # ── unsupported extension ────────────────────────────────────────────────

    def test_unsupported_extension_raises(self, tmp_path):
        fake = tmp_path / "document.xyz"
        fake.write_text("dummy content")
        with pytest.raises(ValueError, match="Unsupported"):
            DocumentLoader(str(fake)).load()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DocumentLoader(str(tmp_path / "missing.docx")).load()

    # ── metadata completeness ────────────────────────────────────────────────

    def test_all_metadata_fields_present_docx(self, minimal_docx_path):
        docs = DocumentLoader(minimal_docx_path, version_label="v2").load()
        _assert_metadata(docs)

    def test_all_metadata_fields_present_pdf(self, minimal_pdf_path):
        docs = DocumentLoader(minimal_pdf_path, version_label="v2").load()
        _assert_metadata(docs)
        # TextPdfParser uses "pdf_text" as the file_type value.
        assert all(d.metadata["file_type"] == "pdf_text" for d in docs)

    def test_source_file_is_filename(self, minimal_docx_path):
        import os
        docs = DocumentLoader(minimal_docx_path, version_label="v1").load()
        expected_name = os.path.basename(minimal_docx_path)
        assert all(d.metadata["source_file"] == expected_name for d in docs)

    def test_version_label_stored(self, minimal_docx_path):
        docs = DocumentLoader(minimal_docx_path, version_label="my-label").load()
        assert all(d.metadata["version_label"] == "my-label" for d in docs)
