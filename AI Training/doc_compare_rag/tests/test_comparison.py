"""
test_comparison.py – Tests for the document comparison engine.

Covers:
  - DocumentDiffer : modified / added / removed detection, summary_stats
  - DiffSummariser : section_summaries, overall_summary (LLM mocked)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.comparison.differ import ComparisonResult, DocumentDiffer, SectionDiff
from app.comparison.summariser import DiffSummariser, EnrichedComparisonResult
from app.ingestion.loader import Document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(text: str, section: str, version: str = "v1", page: int = 1) -> Document:
    return Document(
        text=text,
        metadata={
            "source_file": "test.docx",
            "version_label": version,
            "page_number": page,
            "section_title": section,
            "file_type": "docx",
        },
    )


def _compare(docs_a: list[Document], docs_b: list[Document]) -> ComparisonResult:
    return DocumentDiffer().compare(docs_a, docs_b, "A", "B")


# ---------------------------------------------------------------------------
# DocumentDiffer
# ---------------------------------------------------------------------------


class TestDocumentDiffer:
    # ── single-section cases ────────────────────────────────────────────────

    def test_identical_sections_are_unchanged(self):
        docs = [_doc("Same text here.", "Intro")]
        result = _compare(docs, docs)
        assert all(s.status == "unchanged" for s in result.sections)
        assert result.summary_stats["unchanged"] == 1
        assert result.summary_stats["modified"] == 0

    def test_modified_section_detected(self):
        docs_a = [_doc("Original content.", "Intro")]
        docs_b = [_doc("Modified content.", "Intro")]
        result = _compare(docs_a, docs_b)

        modified = [s for s in result.sections if s.status == "modified"]
        assert len(modified) == 1
        assert modified[0].section_title == "Intro"

    def test_modified_section_carries_both_texts(self):
        docs_a = [_doc("Version A text.", "Intro")]
        docs_b = [_doc("Version B text.", "Intro")]
        result = _compare(docs_a, docs_b)

        sec = next(s for s in result.sections if s.status == "modified")
        assert "Version A text." in sec.text_a
        assert "Version B text." in sec.text_b

    def test_modified_section_has_diff_lines(self):
        docs_a = [_doc("Old line.", "Body")]
        docs_b = [_doc("New line.", "Body")]
        result = _compare(docs_a, docs_b)

        sec = next(s for s in result.sections if s.status == "modified")
        assert any(line.startswith("-") for line in sec.diff_lines)
        assert any(line.startswith("+") for line in sec.diff_lines)

    # ── added section ────────────────────────────────────────────────────────

    def test_added_section_detected(self):
        docs_a = [_doc("Section A content.", "Section A")]
        docs_b = [
            _doc("Section A content.", "Section A"),
            _doc("Newly added text.", "Section B"),
        ]
        result = _compare(docs_a, docs_b)

        added = [s for s in result.sections if s.status == "added"]
        assert len(added) == 1
        assert added[0].section_title == "Section B"

    def test_added_section_text_a_is_empty(self):
        docs_a = [_doc("Existing.", "Alpha")]
        docs_b = [_doc("Existing.", "Alpha"), _doc("New.", "Beta")]
        result = _compare(docs_a, docs_b)

        added = next(s for s in result.sections if s.status == "added")
        assert added.text_a == ""
        assert added.text_b  # non-empty

    # ── removed section ──────────────────────────────────────────────────────

    def test_removed_section_detected(self):
        docs_a = [
            _doc("Section A content.", "Section A"),
            _doc("Old section content.", "Section B"),
        ]
        docs_b = [_doc("Section A content.", "Section A")]
        result = _compare(docs_a, docs_b)

        removed = [s for s in result.sections if s.status == "removed"]
        assert len(removed) == 1
        assert removed[0].section_title == "Section B"

    def test_removed_section_text_b_is_empty(self):
        docs_a = [_doc("Existing.", "Alpha"), _doc("Vanishing.", "Beta")]
        docs_b = [_doc("Existing.", "Alpha")]
        result = _compare(docs_a, docs_b)

        removed = next(s for s in result.sections if s.status == "removed")
        assert removed.text_b == ""
        assert removed.text_a  # non-empty

    # ── summary_stats ────────────────────────────────────────────────────────

    def test_summary_stats_all_unchanged(self):
        docs = [_doc("Same.", "S1"), _doc("Also same.", "S2")]
        result = _compare(docs, docs)
        assert result.summary_stats == {"added": 0, "removed": 0, "modified": 0, "unchanged": 2}

    def test_summary_stats_mixed(self):
        # A: [Keep, Change, Gone]  B: [Keep, New, Change]
        # SequenceMatcher LCS = [Keep, Change]:
        #   equal  Keep  → unchanged (same text)
        #   insert New   → added
        #   equal  Change → modified (different text)
        #   delete Gone  → removed
        docs_a = [
            _doc("Same text.", "Keep"),
            _doc("Old text.", "Change"),
            _doc("Removed text.", "Gone"),
        ]
        docs_b = [
            _doc("Same text.", "Keep"),
            _doc("Brand new section.", "New"),
            _doc("New text.", "Change"),
        ]
        result = _compare(docs_a, docs_b)

        assert result.summary_stats["unchanged"] == 1
        assert result.summary_stats["modified"] == 1
        assert result.summary_stats["removed"] == 1
        assert result.summary_stats["added"] == 1

    def test_summary_stats_keys_always_present(self):
        result = _compare([], [])
        for key in ("added", "removed", "modified", "unchanged"):
            assert key in result.summary_stats

    # ── result structure ─────────────────────────────────────────────────────

    def test_version_labels_propagated(self):
        docs = [_doc("Text.", "S1")]
        result = DocumentDiffer().compare(docs, docs, "draft", "final")
        assert result.version_a_label == "draft"
        assert result.version_b_label == "final"

    def test_sections_list_contains_section_diff_objects(self):
        docs_a = [_doc("A text.", "S1")]
        docs_b = [_doc("B text.", "S1")]
        result = _compare(docs_a, docs_b)
        assert result.sections
        assert all(isinstance(s, SectionDiff) for s in result.sections)

    def test_empty_vs_empty_produces_no_sections(self):
        result = _compare([], [])
        assert result.sections == []

    def test_empty_a_all_sections_added(self):
        docs_b = [_doc("New.", "S1"), _doc("Also new.", "S2")]
        result = _compare([], docs_b)
        assert all(s.status == "added" for s in result.sections)
        assert result.summary_stats["added"] == 2

    def test_empty_b_all_sections_removed(self):
        docs_a = [_doc("Old.", "S1"), _doc("Also old.", "S2")]
        result = _compare(docs_a, [])
        assert all(s.status == "removed" for s in result.sections)
        assert result.summary_stats["removed"] == 2


# ---------------------------------------------------------------------------
# DiffSummariser (LLM mocked via AsyncMock on _llm_call)
# ---------------------------------------------------------------------------


class TestDiffSummariser:
    """All tests mock DiffSummariser._llm_call so no API keys are needed."""

    _SECTION_REPLY = "The section was changed to emphasise new requirements."
    _OVERALL_REPLY = "Overall, one section was modified with minor wording changes."

    @pytest.fixture()
    def comparison_one_modified(self):
        docs_a = [_doc("Original body text.", "Methods")]
        docs_b = [_doc("Revised body text.", "Methods")]
        return _compare(docs_a, docs_b)

    @pytest.fixture()
    def comparison_multi(self):
        docs_a = [
            _doc("Unchanged text.", "Intro"),
            _doc("Old body.", "Body"),
        ]
        docs_b = [
            _doc("Unchanged text.", "Intro"),
            _doc("New body.", "Body"),
        ]
        return _compare(docs_a, docs_b)

    # ── section_summaries ────────────────────────────────────────────────────

    def test_section_summaries_has_entry_per_modified_section(
        self, comparison_one_modified
    ):
        summariser = DiffSummariser(provider="openai")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value=self._SECTION_REPLY),
        ):
            enriched = summariser.summarise(comparison_one_modified)

        assert isinstance(enriched.section_summaries, dict)
        assert len(enriched.section_summaries) == 1
        assert "Methods" in enriched.section_summaries
        assert enriched.section_summaries["Methods"] == self._SECTION_REPLY

    def test_section_summaries_excludes_unchanged_sections(self, comparison_multi):
        summariser = DiffSummariser(provider="openai")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value=self._SECTION_REPLY),
        ):
            enriched = summariser.summarise(comparison_multi)

        # Only "Body" was modified; "Intro" was unchanged and must not appear.
        assert "Intro" not in enriched.section_summaries
        assert "Body" in enriched.section_summaries

    def test_section_summaries_empty_when_no_modified(self):
        docs = [_doc("Same text.", "S1")]
        result = _compare(docs, docs)  # all unchanged

        summariser = DiffSummariser(provider="openai")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value=self._OVERALL_REPLY),
        ):
            enriched = summariser.summarise(result)

        assert enriched.section_summaries == {}

    # ── overall_summary ──────────────────────────────────────────────────────

    def test_overall_summary_is_non_empty_string(self, comparison_one_modified):
        summariser = DiffSummariser(provider="openai")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value=self._OVERALL_REPLY),
        ):
            enriched = summariser.summarise(comparison_one_modified)

        assert isinstance(enriched.overall_summary, str)
        assert len(enriched.overall_summary) > 0

    def test_overall_summary_equals_mocked_reply(self, comparison_one_modified):
        summariser = DiffSummariser(provider="anthropic")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value=self._OVERALL_REPLY),
        ):
            enriched = summariser.summarise(comparison_one_modified)

        assert enriched.overall_summary == self._OVERALL_REPLY

    # ── EnrichedComparisonResult structure ───────────────────────────────────

    def test_enriched_result_inherits_comparison_fields(self, comparison_one_modified):
        summariser = DiffSummariser(provider="openai")
        with patch.object(
            summariser,
            "_llm_call",
            new=AsyncMock(return_value="summary"),
        ):
            enriched = summariser.summarise(comparison_one_modified)

        assert isinstance(enriched, EnrichedComparisonResult)
        assert enriched.version_a_label == "A"
        assert enriched.version_b_label == "B"
        assert enriched.sections == comparison_one_modified.sections
        assert enriched.summary_stats == comparison_one_modified.summary_stats

    # ── LLM failure isolation ────────────────────────────────────────────────

    def test_llm_error_in_section_produces_placeholder(self):
        """A failed LLM call for one section should not abort others."""
        docs_a = [_doc("Old A.", "S1"), _doc("Old B.", "S2")]
        docs_b = [_doc("New A.", "S1"), _doc("New B.", "S2")]
        result = _compare(docs_a, docs_b)

        call_count = 0

        async def _flaky(system, user):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("API timeout")
            return "OK"

        summariser = DiffSummariser(provider="openai", max_retries=0)
        with patch.object(summariser, "_llm_call", new=_flaky):
            enriched = summariser.summarise(result)

        # One of the two modified sections should have the error placeholder.
        placeholder_values = [
            v for v in enriched.section_summaries.values()
            if v.startswith("[Summary unavailable")
        ]
        assert len(placeholder_values) >= 1

    # ── provider validation ──────────────────────────────────────────────────

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
            DiffSummariser(provider="grok")
