"""
summariser.py – LLM-powered plain-language summary of document comparison results.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

from app.comparison.differ import ComparisonResult, SectionDiff

# ---------------------------------------------------------------------------
# Model constants (override via environment variables)
# ---------------------------------------------------------------------------

# Anthropic: claude-sonnet-4-6 is the current Sonnet 4 model ID.
# Override with ANTHROPIC_MODEL env var if you need a specific pinned version
# (e.g. "claude-sonnet-4-20250514").
_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# Clip section text sent to the LLM to avoid exceeding context limits.
_MAX_SECTION_CHARS = 3_000

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SECTION_SYSTEM = (
    "You are a document analyst. Compare these two versions of a document section "
    "and explain the key changes in 2-3 plain sentences. Be specific about what was "
    "added, removed, or changed. Do not repeat the full text."
)

_SECTION_USER = (
    "Section: {section_title}\n\n"
    "Version A:\n{text_a}\n\n"
    "Version B:\n{text_b}\n\n"
    "Changes summary:"
)

_OVERALL_SYSTEM = (
    "You are a document analyst. Given a structured comparison between two versions "
    "of a document, write a concise overall summary (3-5 sentences) that highlights "
    "the most significant changes. Focus on what matters to a non-technical reader."
)

_OVERALL_USER = (
    "Document comparison: {version_a_label} → {version_b_label}\n\n"
    "Statistics:\n"
    "  Added sections:     {added}\n"
    "  Removed sections:   {removed}\n"
    "  Modified sections:  {modified}\n"
    "  Unchanged sections: {unchanged}\n\n"
    "Section-level changes:\n{section_changes}\n\n"
    "Section summaries (for modified sections):\n{section_summaries}\n\n"
    "Overall document summary:"
)


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class EnrichedComparisonResult(ComparisonResult):
    """A :class:`ComparisonResult` enriched with LLM-generated summaries.

    Attributes
    ----------
    section_summaries:
        Mapping from ``section_title`` → plain-English summary string for each
        **modified** section.  Added and removed sections are not summarised
        per-section (their status is self-evident).
    overall_summary:
        A 3-5 sentence plain-English summary of the full comparison.
    """

    section_summaries: Dict[str, str] = field(default_factory=dict)
    overall_summary: str = ""


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return ``True`` when *exc* is a rate-limit error from either provider."""
    # Check by type name to avoid hard import dependencies at module load.
    try:
        import openai  # type: ignore

        if isinstance(exc, openai.RateLimitError):
            return True
    except (ImportError, AttributeError):
        pass
    try:
        import anthropic  # type: ignore

        if isinstance(exc, anthropic.RateLimitError):
            return True
    except (ImportError, AttributeError):
        pass
    return False


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------


async def _retry_with_backoff(
    coro_fn: Callable[[], Coroutine[Any, Any, str]],
    max_retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> str:
    """Call *coro_fn* and retry on rate-limit errors with exponential back-off.

    Parameters
    ----------
    coro_fn:
        A zero-argument callable that returns a new coroutine on each call.
        It is re-invoked on every retry attempt.
    max_retries:
        Maximum number of additional attempts after the first failure.
    base_delay:
        Initial sleep duration in seconds (doubled on each subsequent attempt).
    max_delay:
        Upper bound on sleep duration before jitter is added.

    Raises
    ------
    Exception
        Re-raises the last exception when all retries are exhausted, or
        immediately for non-rate-limit errors.
    """
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn()
        except Exception as exc:
            if not _is_rate_limit_error(exc) or attempt == max_retries:
                raise
            # Exponential back-off with full jitter to avoid thundering-herd.
            delay = min(base_delay * (2.0 ** attempt) + random.uniform(0.0, 1.0), max_delay)
            await asyncio.sleep(delay)

    raise RuntimeError("_retry_with_backoff: unreachable")  # keeps type-checker happy


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, max_chars: int = _MAX_SECTION_CHARS) -> str:
    """Clip *text* to *max_chars*, appending an omission note if truncated."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return text[:max_chars] + f"\n… [{omitted} characters omitted]"


# ---------------------------------------------------------------------------
# DiffSummariser
# ---------------------------------------------------------------------------


class DiffSummariser:
    """Generate plain-English summaries for a :class:`ComparisonResult`.

    For each **modified** section an LLM is called with the exact prompt
    template specified in the module docstring.  An overall document summary
    is also produced, incorporating the per-section summaries.

    LLM provider is selected by the ``LLM_PROVIDER`` environment variable
    (``"anthropic"`` or ``"openai"``; default ``"openai"``).

    Concurrent LLM calls are throttled by an ``asyncio.Semaphore`` to avoid
    hitting provider rate limits.

    Both a synchronous entry point (:meth:`summarise`) and an async one
    (:meth:`asummarise`) are provided.  The sync version is safe to call from
    within a running event loop (e.g. Streamlit) because it offloads the async
    work to a dedicated thread pool.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"openai"``.  Reads ``LLM_PROVIDER`` env var;
        defaults to ``"openai"``.
    max_retries:
        Retry budget for rate-limit errors per LLM call (default 4).
    max_concurrent:
        Maximum number of simultaneous LLM calls (default 5).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        max_retries: int = 4,
        max_concurrent: int = 5,
    ) -> None:
        self.provider = (
            provider or os.environ.get("LLM_PROVIDER", "openai")
        ).lower()
        if self.provider not in {"openai", "anthropic"}:
            raise ValueError(
                f"Unknown LLM_PROVIDER: {self.provider!r}. "
                "Set LLM_PROVIDER to 'openai' or 'anthropic'."
            )
        self.max_retries = max_retries
        self._max_concurrent = max_concurrent
        # Lazy-initialised async clients (created inside the event loop).
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def summarise(self, result: ComparisonResult) -> EnrichedComparisonResult:
        """Synchronous entry point — safe to call from any context.

        Internally runs the async implementation.  When called from inside a
        running event loop (e.g. Streamlit, Jupyter), the async work is
        delegated to a background thread to avoid ``"cannot run nested event
        loops"`` errors.

        Parameters
        ----------
        result:
            A :class:`ComparisonResult` produced by :class:`DocumentDiffer`.

        Returns
        -------
        EnrichedComparisonResult
            The same result enriched with ``section_summaries`` and
            ``overall_summary``.
        """
        try:
            asyncio.get_running_loop()
            # We are inside an existing event loop — delegate to a new thread
            # that can safely call asyncio.run().
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.asummarise(result)).result()
        except RuntimeError:
            # No running loop — create one directly.
            return asyncio.run(self.asummarise(result))

    async def asummarise(self, result: ComparisonResult) -> EnrichedComparisonResult:
        """Async entry point — call with ``await`` from async code.

        Parameters
        ----------
        result:
            A :class:`ComparisonResult` produced by :class:`DocumentDiffer`.

        Returns
        -------
        EnrichedComparisonResult
        """
        # Create a fresh semaphore bound to the current event loop.
        sem = asyncio.Semaphore(self._max_concurrent)
        modified = [s for s in result.sections if s.status == "modified"]

        async def _guarded(section: SectionDiff) -> str:
            async with sem:
                return await self._summarise_section(section)

        raw = await asyncio.gather(
            *[_guarded(s) for s in modified],
            return_exceptions=True,
        )

        section_summaries: Dict[str, str] = {}
        for section, outcome in zip(modified, raw):
            if isinstance(outcome, BaseException):
                section_summaries[section.section_title] = (
                    f"[Summary unavailable: {outcome}]"
                )
            else:
                section_summaries[section.section_title] = outcome  # type: ignore[assignment]

        overall_summary = await self._summarise_overall(result, section_summaries)

        return EnrichedComparisonResult(
            version_a_label=result.version_a_label,
            version_b_label=result.version_b_label,
            sections=result.sections,
            summary_stats=result.summary_stats,
            section_summaries=section_summaries,
            overall_summary=overall_summary,
        )

    # ------------------------------------------------------------------
    # Per-section summary
    # ------------------------------------------------------------------

    async def _summarise_section(self, section: SectionDiff) -> str:
        """Generate a 2-3 sentence plain-English summary for one section diff."""
        user_prompt = _SECTION_USER.format(
            section_title=section.section_title or "(untitled)",
            text_a=_truncate(section.text_a),
            text_b=_truncate(section.text_b),
        )
        return await _retry_with_backoff(
            lambda: self._llm_call(_SECTION_SYSTEM, user_prompt),
            max_retries=self.max_retries,
        )

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------

    async def _summarise_overall(
        self,
        result: ComparisonResult,
        section_summaries: Dict[str, str],
    ) -> str:
        """Generate a 3-5 sentence overall document comparison summary."""
        stats = result.summary_stats

        # Build a bullet list of non-unchanged sections.
        change_bullets: List[str] = []
        for sec in result.sections:
            if sec.status != "unchanged":
                label = f"[{sec.status.upper()}]"
                title = sec.section_title or "(untitled)"
                change_bullets.append(f"  {label} {title}")

        section_change_text = (
            "\n".join(change_bullets) if change_bullets else "  (no structural changes)"
        )

        # Summarise per-section LLM outputs.
        section_summary_text = (
            "\n\n".join(
                f"  '{title}':\n  {summary}"
                for title, summary in section_summaries.items()
            )
            if section_summaries
            else "  (no modified sections)"
        )

        user_prompt = _OVERALL_USER.format(
            version_a_label=result.version_a_label,
            version_b_label=result.version_b_label,
            added=stats.get("added", 0),
            removed=stats.get("removed", 0),
            modified=stats.get("modified", 0),
            unchanged=stats.get("unchanged", 0),
            section_changes=section_change_text,
            section_summaries=section_summary_text,
        )

        return await _retry_with_backoff(
            lambda: self._llm_call(_OVERALL_SYSTEM, user_prompt),
            max_retries=self.max_retries,
        )

    # ------------------------------------------------------------------
    # LLM dispatch
    # ------------------------------------------------------------------

    async def _llm_call(self, system: str, user: str) -> str:
        """Route to the active provider backend."""
        if self.provider == "anthropic":
            return await self._call_anthropic(system, user)
        return await self._call_openai(system, user)

    async def _call_openai(self, system: str, user: str) -> str:
        """Call the OpenAI chat completions API (async)."""
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAI summarisation. "
                "Install with: pip install openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Export it or add it to your .env file."
            )

        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=api_key)

        response = await self._openai_client.chat.completions.create(
            model=_OPENAI_MODEL,
            max_tokens=512,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""

    async def _call_anthropic(self, system: str, user: str) -> str:
        """Call the Anthropic messages API (async)."""
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic is required for Anthropic summarisation. "
                "Install with: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Export it or add it to your .env file."
            )

        if self._anthropic_client is None:
            self._anthropic_client = AsyncAnthropic(api_key=api_key)

        response = await self._anthropic_client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


# ---------------------------------------------------------------------------
# Legacy functional API (used by streamlit_app.py)
# ---------------------------------------------------------------------------


_LEGACY_SYSTEM = (
    "You are an expert document analyst. "
    "You will receive a list of changes between two versions of a document. "
    "Summarise the changes in plain, concise language that a non-technical "
    "reader can understand. Group related changes together and highlight the "
    "most important modifications first."
)


def summarise_diff(diff_lines: List[str], model: Optional[str] = None) -> str:
    """Use an LLM to produce a human-readable summary of a unified diff.

    .. deprecated::
        Prefer :class:`DiffSummariser` which works with typed
        :class:`ComparisonResult` objects, supports both OpenAI and Anthropic,
        and generates per-section summaries with retry logic.

    Parameters
    ----------
    diff_lines:
        Output from :func:`~app.comparison.differ.diff_texts`.
    model:
        Optional OpenAI model override (defaults to ``SUMMARISER_MODEL`` env
        var, then ``"gpt-4o-mini"``).

    Returns
    -------
    str
        Plain-language summary string.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai is required for diff summarisation. "
            "Install with: pip install openai"
        ) from exc

    if not diff_lines:
        return "No differences were detected between the two document versions."

    diff_text = "".join(diff_lines)
    resolved_model = model or os.environ.get("SUMMARISER_MODEL", "gpt-4o-mini")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _LEGACY_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Here is the diff between the two document versions:\n\n"
                    f"```\n{diff_text}\n```\n\n"
                    "Please provide a clear, plain-language summary of the changes."
                ),
            },
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""
