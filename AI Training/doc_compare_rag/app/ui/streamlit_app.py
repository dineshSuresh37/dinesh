"""
streamlit_app.py – Streamlit UI for document comparison with RAG.

Run with:
    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

import streamlit as st

from app.utils.helpers import load_env

load_env()

st.set_page_config(
    page_title="Document Compare + RAG",
    page_icon="📄",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: dict = {
    "documents": {},        # version_label -> List[Document]
    "chunks": {},           # version_label -> List[Chunk]
    "embeddings": {},       # version_label -> List[np.ndarray]
    "embedder": None,
    "vsm": None,            # VectorStoreManager
    "comparison_result": None,
    "rag_results": None,
    "rag_answer": None,
}

for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_upload(uploaded_file) -> str:
    """Write an UploadedFile to a temp path, return the path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as fh:
        fh.write(uploaded_file.read())
        return fh.name


def _status_badge(status: str) -> str:
    colors = {
        "unchanged": "🟢",
        "modified":  "🟡",
        "removed":   "🔴",
        "added":     "🔵",
    }
    return f"{colors.get(status, '⚪')} **{status.upper()}**"


def _diff_html(diff_lines: List[str]) -> str:
    """Render unified diff lines with inline HTML colouring."""
    rows = []
    for line in diff_lines:
        escaped = (
            line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        if line.startswith("+") and not line.startswith("+++"):
            rows.append(f'<span style="background:#d4edda;display:block">{escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            rows.append(f'<span style="background:#f8d7da;display:block">{escaped}</span>')
        else:
            rows.append(f'<span style="display:block">{escaped}</span>')
    return "<pre style='font-size:12px;overflow:auto'>" + "".join(rows) + "</pre>"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("Document Compare + RAG")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents (.docx, .doc, .pdf)",
        type=["docx", "doc", "pdf"],
        accept_multiple_files=True,
    )

    # Per-file version labels
    version_labels: List[str] = []
    if uploaded_files:
        st.markdown("**Assign version labels:**")
        for i, uf in enumerate(uploaded_files):
            default = f"v{i + 1}"
            label = st.text_input(
                uf.name,
                value=default,
                key=f"vlabel_{i}_{uf.name}",
            )
            version_labels.append(label)

    process_clicked = st.button("Process Documents", type="primary", disabled=not uploaded_files)

    st.divider()

    # Compare dropdowns (only meaningful once docs are processed)
    processed_versions = list(st.session_state["documents"].keys())
    compare_disabled = len(processed_versions) < 2

    if compare_disabled:
        st.caption("Process at least 2 documents to enable comparison.")

    version_a_choice = st.selectbox(
        "Compare (A – original)",
        options=processed_versions if processed_versions else ["—"],
        disabled=compare_disabled,
        key="sel_a",
    )
    version_b_choice = st.selectbox(
        "vs (B – revised)",
        options=processed_versions if processed_versions else ["—"],
        index=min(1, len(processed_versions) - 1) if len(processed_versions) >= 2 else 0,
        disabled=compare_disabled,
        key="sel_b",
    )

    llm_provider = st.selectbox(
        "LLM provider",
        options=["openai", "anthropic"],
        index=0,
        key="llm_provider",
    )

    run_comparison = st.button(
        "Run Comparison",
        type="primary",
        disabled=compare_disabled,
    )

# ---------------------------------------------------------------------------
# Process Documents
# ---------------------------------------------------------------------------

if process_clicked and uploaded_files:
    from app.ingestion.loader import DocumentLoader
    from app.pipeline.chunker import TextChunker
    from app.pipeline.embedder import Embedder
    from app.pipeline.vector_store import VectorStoreManager

    # Initialise shared objects once (persist across reruns via session state)
    if st.session_state["embedder"] is None:
        st.session_state["embedder"] = Embedder()

    embedder: Embedder = st.session_state["embedder"]

    if st.session_state["vsm"] is None:
        st.session_state["vsm"] = VectorStoreManager(embedder)

    vsm: VectorStoreManager = st.session_state["vsm"]
    chunker = TextChunker()

    progress = st.sidebar.progress(0, text="Starting…")
    n = len(uploaded_files)

    for idx, (uf, label) in enumerate(zip(uploaded_files, version_labels)):
        step_pct = int((idx / n) * 100)
        progress.progress(step_pct, text=f"Ingesting {uf.name}…")

        try:
            tmp_path = _save_upload(uf)
            loader = DocumentLoader(tmp_path, version_label=label)
            docs = loader.load()
        except Exception as exc:
            st.sidebar.error(f"Failed to load {uf.name}: {exc}")
            os.unlink(tmp_path)
            continue
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        st.session_state["documents"][label] = docs

        progress.progress(step_pct + int(0.3 * 100 / n), text=f"Chunking {uf.name}…")
        chunks = chunker.chunk(docs)
        st.session_state["chunks"][label] = chunks

        progress.progress(step_pct + int(0.6 * 100 / n), text=f"Embedding {uf.name}…")
        try:
            embeddings = embedder.embed(chunks)
            st.session_state["embeddings"][label] = embeddings
        except Exception as exc:
            st.sidebar.error(f"Embedding failed for {uf.name}: {exc}")
            continue

        progress.progress(step_pct + int(0.85 * 100 / n), text=f"Indexing {uf.name}…")
        try:
            vsm.index(label, chunks, embeddings)
        except Exception as exc:
            st.sidebar.error(f"Indexing failed for {uf.name}: {exc}")

    progress.progress(100, text="Done.")
    st.sidebar.success(f"Processed {n} document(s).")

# ---------------------------------------------------------------------------
# Run Comparison
# ---------------------------------------------------------------------------

if run_comparison and not compare_disabled:
    if version_a_choice == version_b_choice:
        st.sidebar.error("Choose two different versions to compare.")
    elif version_a_choice not in st.session_state["documents"]:
        st.sidebar.error(f"Version '{version_a_choice}' not found. Process it first.")
    elif version_b_choice not in st.session_state["documents"]:
        st.sidebar.error(f"Version '{version_b_choice}' not found. Process it first.")
    else:
        from app.comparison.differ import DocumentDiffer
        from app.comparison.summariser import DiffSummariser

        docs_a = st.session_state["documents"][version_a_choice]
        docs_b = st.session_state["documents"][version_b_choice]

        with st.spinner("Comparing documents…"):
            try:
                differ = DocumentDiffer()
                raw_result = differ.compare(
                    docs_a,
                    docs_b,
                    version_a_label=version_a_choice,
                    version_b_label=version_b_choice,
                )
            except Exception as exc:
                st.error(f"Comparison failed: {exc}")
                raw_result = None

        if raw_result is not None:
            with st.spinner("Generating LLM summaries…"):
                try:
                    summariser = DiffSummariser(provider=st.session_state["llm_provider"])
                    enriched = summariser.summarise(raw_result)
                    st.session_state["comparison_result"] = enriched
                except Exception as exc:
                    st.warning(
                        f"LLM summarisation failed ({exc}). Showing structural diff only."
                    )
                    st.session_state["comparison_result"] = raw_result

# ---------------------------------------------------------------------------
# Main area – tabs
# ---------------------------------------------------------------------------

tab_compare, tab_rag, tab_overview = st.tabs([
    "Comparison Results",
    "Ask a Question (RAG)",
    "Document Overview",
])

# ── Tab 1: Comparison Results ────────────────────────────────────────────────

with tab_compare:
    result = st.session_state["comparison_result"]

    if result is None:
        st.info("Process documents and run a comparison to see results here.")
    else:
        # Overall summary
        overall_summary = getattr(result, "overall_summary", "")
        if overall_summary:
            st.subheader("Overall Summary")
            st.write(overall_summary)
            st.divider()

        # Stats row
        stats = result.summary_stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Added",     stats.get("added",     0))
        c2.metric("Removed",   stats.get("removed",   0))
        c3.metric("Modified",  stats.get("modified",  0))
        c4.metric("Unchanged", stats.get("unchanged", 0))
        st.divider()

        # Per-section cards
        section_summaries: Dict[str, str] = getattr(result, "section_summaries", {})

        for sec in result.sections:
            title_display = sec.section_title or "(untitled section)"
            badge = _status_badge(sec.status)

            with st.container():
                header_col, badge_col = st.columns([4, 1])
                header_col.markdown(f"**{title_display}**")
                badge_col.markdown(badge)

                if sec.status == "unchanged":
                    st.caption("No changes in this section.")

                elif sec.status == "added":
                    st.caption("This section was added in the revised version.")
                    with st.expander("Show added text"):
                        st.text(sec.text_b[:2000] + ("…" if len(sec.text_b) > 2000 else ""))

                elif sec.status == "removed":
                    st.caption("This section was removed in the revised version.")
                    with st.expander("Show removed text"):
                        st.text(sec.text_a[:2000] + ("…" if len(sec.text_a) > 2000 else ""))

                elif sec.status == "modified":
                    llm_explanation = section_summaries.get(sec.section_title, "")
                    if llm_explanation:
                        st.write(llm_explanation)
                    with st.expander("Show raw diff"):
                        if sec.diff_lines:
                            st.components.v1.html(
                                _diff_html(sec.diff_lines),
                                height=300,
                                scrolling=True,
                            )
                        else:
                            st.caption("(no diff lines)")

                st.divider()

# ── Tab 2: Ask a Question (RAG) ──────────────────────────────────────────────

with tab_rag:
    vsm: Optional[object] = st.session_state["vsm"]
    processed_versions = list(st.session_state["documents"].keys())

    if not processed_versions:
        st.info("Process documents first to enable RAG search.")
    else:
        query = st.text_input("Enter your question:", key="rag_query")

        version_options = ["All versions"] + processed_versions
        rag_version = st.selectbox(
            "Search in:",
            options=version_options,
            key="rag_version_sel",
        )

        search_clicked = st.button("Search", key="rag_search_btn")

        if search_clicked and query.strip():
            with st.spinner("Searching…"):
                try:
                    if rag_version == "All versions":
                        raw_chunks_by_version = vsm.search_all_versions(
                            query,
                            top_k_per_version=5,
                        )
                    else:
                        raw_chunks_by_version = {
                            rag_version: vsm.search(query, rag_version, top_k=5)
                        }
                    st.session_state["rag_results"] = raw_chunks_by_version
                except Exception as exc:
                    st.error(f"Search failed: {exc}")
                    st.session_state["rag_results"] = None

            # LLM answer
            if st.session_state["rag_results"]:
                all_chunks = [
                    chunk
                    for chunks in st.session_state["rag_results"].values()
                    for chunk in chunks
                ]
                context = "\n\n---\n\n".join(
                    f"[{c.metadata.get('version_label', '?')} / "
                    f"{c.metadata.get('section_title', '?')} / "
                    f"p{c.metadata.get('page_number', '?')}]\n{c.text}"
                    for c in all_chunks
                )
                with st.spinner("Generating answer…"):
                    try:
                        provider = st.session_state.get("llm_provider", "openai")
                        if provider == "anthropic":
                            import anthropic as _anth
                            _client = _anth.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                            _resp = _client.messages.create(
                                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                                max_tokens=512,
                                system=(
                                    "You are a document analyst. Answer the question using only "
                                    "the provided context. Be concise and accurate."
                                ),
                                messages=[
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                                        ),
                                    }
                                ],
                            )
                            answer = _resp.content[0].text
                        else:
                            from openai import OpenAI as _OAI
                            _client = _OAI(api_key=os.environ["OPENAI_API_KEY"])
                            _resp = _client.chat.completions.create(
                                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
                                max_tokens=512,
                                temperature=0.3,
                                messages=[
                                    {
                                        "role": "system",
                                        "content": (
                                            "You are a document analyst. Answer the question "
                                            "using only the provided context. Be concise and accurate."
                                        ),
                                    },
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                                        ),
                                    },
                                ],
                            )
                            answer = _resp.choices[0].message.content or ""
                        st.session_state["rag_answer"] = answer
                    except Exception as exc:
                        st.session_state["rag_answer"] = None
                        st.warning(f"LLM answer unavailable: {exc}")

        # Display results
        if st.session_state["rag_results"]:
            answer = st.session_state.get("rag_answer")
            if answer:
                st.subheader("Answer")
                st.write(answer)
                st.divider()

            st.subheader("Retrieved Passages")
            for ver_label, chunks in st.session_state["rag_results"].items():
                st.markdown(f"**Version: {ver_label}**")
                for chunk in chunks:
                    meta = chunk.metadata
                    score_pct = f"{chunk.score * 100:.1f}%"
                    info_line = (
                        f"📄 `{meta.get('source_file', '?')}` · "
                        f"version `{meta.get('version_label', '?')}` · "
                        f"section *{meta.get('section_title') or '—'}* · "
                        f"page {meta.get('page_number', '?')} · "
                        f"relevance **{score_pct}**"
                    )
                    with st.container():
                        st.caption(info_line)
                        st.info(chunk.text)

# ── Tab 3: Document Overview ─────────────────────────────────────────────────

with tab_overview:
    documents_map = st.session_state["documents"]
    chunks_map = st.session_state["chunks"]

    if not documents_map:
        st.info("No documents processed yet. Upload and process files using the sidebar.")
    else:
        # Summary table
        rows = []
        for ver, docs in documents_map.items():
            source_file = docs[0].metadata.get("source_file", "—") if docs else "—"
            file_type = docs[0].metadata.get("file_type", "—") if docs else "—"
            pages = max(
                (d.metadata.get("page_number", 1) for d in docs),
                default=0,
            )
            sections = len({d.metadata.get("section_title", "") for d in docs if d.metadata.get("section_title")})
            chunk_count = len(chunks_map.get(ver, []))
            rows.append({
                "Version":    ver,
                "File":       os.path.basename(source_file),
                "Type":       file_type,
                "Pages":      pages,
                "Sections":   sections,
                "Chunks":     chunk_count,
            })

        st.subheader("Indexed Documents")
        st.table(rows)
        st.divider()

        # Per-version raw text preview
        for ver, docs in documents_map.items():
            with st.expander(f"Raw text preview – {ver}"):
                preview_text = "\n\n".join(d.text for d in docs)
                st.text(preview_text[:5000] + ("…" if len(preview_text) > 5000 else ""))
