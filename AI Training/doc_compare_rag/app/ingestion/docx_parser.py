"""
docx_parser.py – Parse .docx files into Document objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.ingestion.loader import Document


def parse_docx(
    file_path: str,
    source_file: str | None = None,
    version_label: str = "v1",
) -> List["Document"]:
    """
    Parse a .docx file paragraph by paragraph.

    - Detects Heading styles to record the current section title.
    - Extracts table cell text as plain text, appended in document order
      after the section they belong to.

    Raises
    ------
    FileNotFoundError  – if *file_path* does not exist.
    ValueError         – if the file extension is not .docx or .doc.
    ImportError        – if python-docx is not installed.
    """
    from app.ingestion.loader import Document

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path!r}")
    if path.suffix.lower() not in (".docx", ".doc"):
        raise ValueError(
            f"DocxParser only handles .docx/.doc files, got: {path.suffix!r}"
        )
    if source_file is None:
        source_file = path.name

    try:
        from docx import Document as DocxDoc  # type: ignore
        from docx.oxml.ns import qn  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "python-docx is required. Install with: pip install python-docx"
        ) from exc

    doc = DocxDoc(file_path)
    body = doc.element.body
    documents: List[Document] = []
    current_heading = ""
    page_number = 1

    W_P = qn("w:p")
    W_TBL = qn("w:tbl")
    W_T = qn("w:t")
    W_BR = qn("w:br")
    W_BR_TYPE = qn("w:type")
    W_PSTYLE = qn("w:pStyle")
    W_VAL = qn("w:val")
    W_TC = qn("w:tc")

    def _para_text(p_elem) -> str:
        return "".join(t.text or "" for t in p_elem.iter(W_T)).strip()

    def _style_id(p_elem) -> str:
        pStyle = p_elem.find(".//" + W_PSTYLE)
        return (pStyle.get(W_VAL) or "") if pStyle is not None else ""

    def _has_page_break(p_elem) -> bool:
        return any(br.get(W_BR_TYPE) == "page" for br in p_elem.iter(W_BR))

    def _make(text: str, section_title: str) -> Document:
        return Document(
            text=text,
            metadata={
                "source_file": source_file,
                "version_label": version_label,
                "page_number": page_number,
                "section_title": section_title,
                "file_type": "docx",
            },
        )

    for child in body:
        if child.tag == W_P:
            if _has_page_break(child):
                page_number += 1
            text = _para_text(child)
            if not text:
                continue

            style_id = _style_id(child)
            is_heading = style_id.startswith("Heading")
            if is_heading:
                current_heading = text
                section_title = text
            else:
                section_title = current_heading

            documents.append(_make(text, section_title))

        elif child.tag == W_TBL:
            cell_texts: List[str] = []
            for tc in child.iter(W_TC):
                cell_text = "".join(t.text or "" for t in tc.iter(W_T)).strip()
                if cell_text:
                    cell_texts.append(cell_text)
            if cell_texts:
                documents.append(_make(" | ".join(cell_texts), current_heading))

    return documents
