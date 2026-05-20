"""
main.py – CLI entry point for the document comparison pipeline.

Usage:
    python -m app.main --file1 v1.pdf  --label1 "Version 1" \
                       --file2 v2.docx --label2 "Version 2"

Optional flags:
    --no-llm        Skip the LLM summarisation step.
    --provider      Override LLM_PROVIDER env var (openai | anthropic).
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

STATUS_SYMBOL: dict[str, str] = {
    "added": "+ ",
    "removed": "- ",
    "modified": "~ ",
    "unchanged": "  ",
}


def _print_report(result, file1: str, file2: str) -> None:
    width = 72
    sep = "=" * width

    print(sep)
    print("DOCUMENT COMPARISON REPORT")
    print(f"  A : {file1}")
    print(f"  B : {file2}")
    print(f"  A label : {result.version_a_label}")
    print(f"  B label : {result.version_b_label}")
    print(sep)

    stats = result.summary_stats
    print(
        f"  Sections — unchanged: {stats['unchanged']}  "
        f"modified: {stats['modified']}  "
        f"added: {stats['added']}  "
        f"removed: {stats['removed']}"
    )
    print()

    overall = getattr(result, "overall_summary", "")
    if overall:
        print("OVERALL SUMMARY")
        print("-" * width)
        for line in textwrap.wrap(overall, width=width):
            print(line)
        print()

    section_summaries: dict[str, str] = getattr(result, "section_summaries", {})

    print("SECTIONS")
    print("-" * width)
    for sec in result.sections:
        sym = STATUS_SYMBOL.get(sec.status, "? ")
        print(f"{sym}[{sec.status.upper()}] {sec.section_title}")

        llm_text = section_summaries.get(sec.section_title, "")
        if llm_text:
            for line in textwrap.wrap(f"  LLM: {llm_text}", width=width):
                print(line)

        if sec.status == "modified" and sec.diff_lines:
            shown = sec.diff_lines[:10]
            for dl in shown:
                print(f"    {dl}")
            if len(sec.diff_lines) > 10:
                print(f"    … ({len(sec.diff_lines) - 10} more diff lines)")

        print()

    print(sep)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two document versions through the RAG pipeline."
    )
    parser.add_argument("--file1", required=True, help="Path to first document")
    parser.add_argument("--label1", default="Version A", help="Label for first document")
    parser.add_argument("--file2", required=True, help="Path to second document")
    parser.add_argument("--label2", default="Version B", help="Label for second document")
    parser.add_argument(
        "--no-llm", action="store_true", help="Skip LLM summarisation"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="Override LLM_PROVIDER env var",
    )
    args = parser.parse_args(argv)

    for attr, path_str in (("file1", args.file1), ("file2", args.file2)):
        if not Path(path_str).exists():
            print(f"Error: file not found: {path_str}", file=sys.stderr)
            return 1

    from config import get_config  # type: ignore

    cfg = get_config()

    # ── Ingestion ────────────────────────────────────────────────────────────
    print("Ingesting documents …", flush=True)
    from app.ingestion.loader import DocumentLoader

    docs_a = DocumentLoader(args.file1, version_label=args.label1).load()
    docs_b = DocumentLoader(args.file2, version_label=args.label2).load()
    print(
        f"  Loaded {len(docs_a)} section(s) from A, "
        f"{len(docs_b)} section(s) from B.",
        flush=True,
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    print("Chunking …", flush=True)
    from app.pipeline.chunker import TextChunker

    chunker = TextChunker(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    chunks_a = chunker.chunk(docs_a)
    chunks_b = chunker.chunk(docs_b)
    print(
        f"  {len(chunks_a)} chunk(s) from A, {len(chunks_b)} chunk(s) from B.",
        flush=True,
    )

    # ── Embedding & indexing ─────────────────────────────────────────────────
    print("Embedding and indexing …", flush=True)
    from app.pipeline.embedder import Embedder
    from app.pipeline.vector_store import VectorStoreManager

    embedder = Embedder(
        model_name=cfg.embedding_model, cache_dir=cfg.embedding_cache_dir
    )
    vsm = VectorStoreManager(embedder, data_dir=cfg.data_dir)

    emb_a = embedder.embed(chunks_a)
    vsm.index(args.label1, chunks_a, emb_a)

    emb_b = embedder.embed(chunks_b)
    vsm.index(args.label2, chunks_b, emb_b)
    print("  Indexing complete.", flush=True)

    # ── Comparison ───────────────────────────────────────────────────────────
    print("Comparing …", flush=True)
    from app.comparison.differ import DocumentDiffer

    comparison = DocumentDiffer().compare(docs_a, docs_b, args.label1, args.label2)

    # ── Summarisation ────────────────────────────────────────────────────────
    if args.no_llm:
        result = comparison
    else:
        provider = args.provider or cfg.llm_provider
        try:
            cfg.validate()
        except EnvironmentError as exc:
            print(f"Warning: {exc}  (skipping LLM summarisation)", file=sys.stderr)
            result = comparison
        else:
            print(f"Summarising with {provider} …", flush=True)
            from app.comparison.summariser import DiffSummariser

            result = DiffSummariser(provider=provider).summarise(comparison)

    # ── Report ───────────────────────────────────────────────────────────────
    _print_report(result, args.file1, args.file2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
