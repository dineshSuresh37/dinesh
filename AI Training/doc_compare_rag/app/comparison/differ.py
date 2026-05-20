"""
differ.py – Structural diff between two document versions using difflib.
"""

from __future__ import annotations

import difflib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

if TYPE_CHECKING:
    from app.ingestion.loader import Document


# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@dataclass
class SectionDiff:
    """Diff result for one aligned section pair.

    Attributes
    ----------
    section_title:
        The heading or identifier of this section (may be empty for untitled
        body text).
    status:
        ``"added"``     – section exists only in version B.
        ``"removed"``   – section exists only in version A.
        ``"modified"``  – section exists in both but the content differs.
        ``"unchanged"`` – section exists in both with identical content.
    diff_lines:
        Raw ``difflib.unified_diff`` output lines (include ``+``/``-`` markers
        and the hunk headers).
    text_a:
        Full section text from version A.  Empty string for added sections.
    text_b:
        Full section text from version B.  Empty string for removed sections.
    """

    section_title: str
    status: Literal["added", "removed", "modified", "unchanged"]
    diff_lines: List[str]
    text_a: str
    text_b: str


@dataclass
class ComparisonResult:
    """Full structured comparison between two document versions.

    Attributes
    ----------
    version_a_label:
        Human-readable label for the original version (e.g. ``"v1"``).
    version_b_label:
        Human-readable label for the revised version (e.g. ``"v2"``).
    sections:
        One :class:`SectionDiff` per aligned section pair, in document order.
    summary_stats:
        Counts of sections by status:
        ``{"added": int, "removed": int, "modified": int, "unchanged": int}``.
    """

    version_a_label: str
    version_b_label: str
    sections: List[SectionDiff]
    summary_stats: Dict[str, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _Section:
    """Internal: a run-length-grouped section of a document."""

    title: str
    text: str


def _group_sections(docs: List["Document"]) -> List[_Section]:
    """Merge consecutive Documents that share the same ``section_title``.

    Documents are assumed to be in document order (as produced by the ingestion
    layer).  Consecutive documents whose ``section_title`` metadata values are
    equal are merged into a single :class:`_Section`; their texts are joined
    with ``"\\n\\n"``.

    This means the heading paragraph itself (whose ``section_title`` equals its
    own text in the DOCX parser) is included in the same group as the body
    paragraphs that follow it.
    """
    if not docs:
        return []

    def _title(doc: "Document") -> str:
        return (doc.metadata.get("section_title") or "").strip()

    sections: List[_Section] = []
    current_title = _title(docs[0])
    current_texts: List[str] = [docs[0].text]

    for doc in docs[1:]:
        t = _title(doc)
        if t == current_title:
            current_texts.append(doc.text)
        else:
            sections.append(_Section(current_title, "\n\n".join(current_texts)))
            current_title = t
            current_texts = [doc.text]

    sections.append(_Section(current_title, "\n\n".join(current_texts)))
    return sections


def _align_sections(
    sections_a: List[_Section],
    sections_b: List[_Section],
) -> List[Tuple[Optional[_Section], Optional[_Section]]]:
    """Return an ordered list of aligned ``(section_a, section_b)`` pairs.

    Alignment strategy
    ------------------
    1. Run ``difflib.SequenceMatcher`` on the **title** sequences of both
       versions.  This matches sections that share an identical title,
       regardless of position (title-based alignment).
    2. Within ``"replace"`` opcode blocks — where a span of A titles has no
       direct title match in B — the sections are aligned **positionally**
       using ``itertools.zip_longest``.

    A ``None`` in the first slot means the section was added (only in B).
    A ``None`` in the second slot means the section was removed (only in A).
    """
    titles_a = [s.title for s in sections_a]
    titles_b = [s.title for s in sections_b]

    matcher = difflib.SequenceMatcher(None, titles_a, titles_b, autojunk=False)
    pairs: List[Tuple[Optional[_Section], Optional[_Section]]] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Identical title sequences → direct section-by-section match.
            for sa, sb in zip(sections_a[i1:i2], sections_b[j1:j2]):
                pairs.append((sa, sb))

        elif tag == "insert":
            # Sections that only appear in B.
            for sb in sections_b[j1:j2]:
                pairs.append((None, sb))

        elif tag == "delete":
            # Sections that only appear in A.
            for sa in sections_a[i1:i2]:
                pairs.append((sa, None))

        elif tag == "replace":
            # Title mismatch block → positional fallback.
            for sa, sb in itertools.zip_longest(
                sections_a[i1:i2], sections_b[j1:j2]
            ):
                pairs.append((sa, sb))

    return pairs


def _make_section_diff(
    sa: Optional[_Section],
    sb: Optional[_Section],
    label_a: str,
    label_b: str,
) -> SectionDiff:
    """Produce a :class:`SectionDiff` for a single aligned pair."""
    # Section title: use A's (original) when available, fall back to B's.
    title = (sa.title if sa else "") or (sb.title if sb else "")
    display = title or "(untitled)"

    text_a = sa.text if sa else ""
    text_b = sb.text if sb else ""

    if sa is None:
        status: Literal["added", "removed", "modified", "unchanged"] = "added"
    elif sb is None:
        status = "removed"
    elif text_a == text_b:
        status = "unchanged"
    else:
        status = "modified"

    diff_lines = list(
        difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile=f"{label_a}/{display}",
            tofile=f"{label_b}/{display}",
        )
    )

    return SectionDiff(
        section_title=title,
        status=status,
        diff_lines=diff_lines,
        text_a=text_a,
        text_b=text_b,
    )


# ---------------------------------------------------------------------------
# DocumentDiffer
# ---------------------------------------------------------------------------


class DocumentDiffer:
    """Compare two lists of Document objects and produce a :class:`ComparisonResult`.

    Alignment workflow
    ------------------
    1. Group consecutive Documents that share the same ``section_title``
       metadata value into logical sections (heading + body text).
    2. Align sections across versions using ``difflib.SequenceMatcher`` on
       title sequences (title-based alignment), falling back to positional
       alignment inside mismatched blocks.
    3. For each aligned pair compute the ``difflib.unified_diff`` and
       determine the section status (added / removed / modified / unchanged).

    Usage
    -----
    ::

        differ = DocumentDiffer()
        result = differ.compare(docs_v1, docs_v2)
    """

    def compare(
        self,
        version_a: List["Document"],
        version_b: List["Document"],
        version_a_label: Optional[str] = None,
        version_b_label: Optional[str] = None,
    ) -> ComparisonResult:
        """Compare *version_a* against *version_b*.

        Parameters
        ----------
        version_a, version_b:
            Lists of :class:`~app.ingestion.loader.Document` objects from the
            ingestion layer (or chunking pipeline).
        version_a_label, version_b_label:
            Human-readable version identifiers.  When omitted, the
            ``version_label`` field from the first document's metadata is used,
            with ``"Version A"`` / ``"Version B"`` as the ultimate fallback.

        Returns
        -------
        ComparisonResult
            Structured diff with per-section status, raw diff lines, and
            aggregate counts.
        """
        label_a = version_a_label or (
            (version_a[0].metadata.get("version_label") or "Version A")
            if version_a
            else "Version A"
        )
        label_b = version_b_label or (
            (version_b[0].metadata.get("version_label") or "Version B")
            if version_b
            else "Version B"
        )

        sections_a = _group_sections(version_a)
        sections_b = _group_sections(version_b)

        aligned = _align_sections(sections_a, sections_b)

        section_diffs: List[SectionDiff] = [
            _make_section_diff(sa, sb, label_a, label_b)
            for sa, sb in aligned
        ]

        summary_stats: Dict[str, int] = {
            "added":     sum(1 for s in section_diffs if s.status == "added"),
            "removed":   sum(1 for s in section_diffs if s.status == "removed"),
            "modified":  sum(1 for s in section_diffs if s.status == "modified"),
            "unchanged": sum(1 for s in section_diffs if s.status == "unchanged"),
        }

        return ComparisonResult(
            version_a_label=label_a,
            version_b_label=label_b,
            sections=section_diffs,
            summary_stats=summary_stats,
        )


# ---------------------------------------------------------------------------
# Legacy functional API (used by streamlit_app.py)
# ---------------------------------------------------------------------------


def diff_texts(text_a: str, text_b: str) -> List[str]:
    """Produce a unified diff between two plain-text strings.

    .. deprecated::
        Prefer :class:`DocumentDiffer` which works with ``Document`` objects
        and returns a typed :class:`ComparisonResult`.

    Returns
    -------
    List[str]
        Unified-diff lines including ``+``/``-`` markers.
    """
    return list(
        difflib.unified_diff(
            text_a.splitlines(keepends=True),
            text_b.splitlines(keepends=True),
            fromfile="version_A",
            tofile="version_B",
        )
    )


def diff_chunks(
    chunks_a: List[str],
    chunks_b: List[str],
) -> Dict[str, List[str]]:
    """Classify text chunks as added, removed, or unchanged.

    .. deprecated::
        Prefer :class:`DocumentDiffer`.

    Returns
    -------
    Dict[str, List[str]]
        Keys ``"added"``, ``"removed"``, ``"unchanged"``.
    """
    set_a = set(chunks_a)
    set_b = set(chunks_b)
    return {
        "added":     [c for c in chunks_b if c not in set_a],
        "removed":   [c for c in chunks_a if c not in set_b],
        "unchanged": [c for c in chunks_a if c in set_b],
    }


def html_diff(text_a: str, text_b: str) -> str:
    """Generate an HTML side-by-side diff (for Streamlit rendering).

    .. deprecated::
        Prefer :class:`DocumentDiffer`.

    Returns
    -------
    str
        HTML produced by :class:`difflib.HtmlDiff`.
    """
    return difflib.HtmlDiff().make_file(
        text_a.splitlines(),
        text_b.splitlines(),
        fromdesc="Version A",
        todesc="Version B",
    )
