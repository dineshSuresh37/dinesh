"""
helpers.py – Shared utility functions used across the application.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from app.pipeline.chunker import Chunk


def file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """
    Compute a hex digest for the given file.

    Useful for detecting whether a document version has changed without
    re-processing the full content.
    """
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def safe_filename(name: str) -> str:
    """
    Sanitise a string so it can be used safely as a file/collection name.

    Replaces characters that are invalid in file names with underscores.
    """
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def join_pages(pages: List[str], separator: str = "\n\n") -> str:
    """Concatenate a list of page/section strings into a single document string."""
    return separator.join(pages)


def load_env(env_file: str = ".env") -> None:
    """
    Load environment variables from a .env file into os.environ.

    Skips lines that are comments or blank; does NOT overwrite existing env vars.
    Requires: python-dotenv  (falls back silently if not installed).
    """
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv(dotenv_path=env_file, override=False)
    except ImportError:
        # python-dotenv not installed – variables must be set externally.
        pass


def write_temp_file(content: bytes, suffix: str = "") -> str:
    """
    Write bytes to a temporary file and return its path.

    The caller is responsible for deleting the file when done.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, content)
    finally:
        os.close(fd)
    return path


def format_diff_as_html(diff_lines: list[str]) -> str:
    """Render unified-diff lines as coloured HTML suitable for st.components."""
    rows: list[str] = []
    for line in diff_lines:
        escaped = (
            line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        if line.startswith("+") and not line.startswith("+++"):
            rows.append(
                f'<span style="color:green;display:block">{escaped}</span>'
            )
        elif line.startswith("-") and not line.startswith("---"):
            rows.append(
                f'<span style="color:red;display:block">{escaped}</span>'
            )
        else:
            rows.append(f'<span style="display:block">{escaped}</span>')
    return (
        "<pre style='font-family:monospace;font-size:13px'>"
        + "".join(rows)
        + "</pre>"
    )


def truncate_text(text: str, max_chars: int = 300) -> str:
    """Return *text* truncated to *max_chars*, appending an ellipsis if cut."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\u2026"


def build_rag_prompt(question: str, context_chunks: list[Chunk]) -> str:
    """Build a prompt that embeds numbered context passages before the question."""
    passages: list[str] = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk.metadata
        header = (
            f"[{i}] {meta.get('version_label', '?')} / "
            f"{meta.get('section_title') or 'no section'} / "
            f"page {meta.get('page_number', '?')}"
        )
        passages.append(f"{header}\n{chunk.text}")
    context = "\n\n---\n\n".join(passages)
    return (
        "Use the following context passages to answer the question.\n"
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
