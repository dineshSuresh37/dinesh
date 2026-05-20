# doc_compare_rag

A document comparison and Q&A system powered by a Retrieval-Augmented Generation (RAG) pipeline. Compare two versions of a document (PDF or DOCX), receive an AI-generated summary of what changed, and ask natural-language questions against the indexed content.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1 – Ingestion                                         │
│  DocxParser · TextPdfParser · OcrPdfParser · DocumentLoader  │
│  (section detection, metadata tagging, OCR fallback)         │
├─────────────────────────────────────────────────────────────┤
│  Layer 2 – RAG Pipeline                                      │
│  TextChunker · Embedder (disk cache) · VectorStoreManager    │
│  (FAISS or Chroma, per-version indices)                      │
├─────────────────────────────────────────────────────────────┤
│  Layer 3 – Comparison Engine                                 │
│  DocumentDiffer (SequenceMatcher diff) · DiffSummariser      │
│  (async LLM calls, per-section + overall summaries)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 4 – UI                                                │
│  Streamlit app – sidebar upload, 3 tabs:                     │
│    Comparison Results · Ask a Question (RAG) · Doc Overview  │
├─────────────────────────────────────────────────────────────┤
│  Layer 5 – QA / Testing                                      │
│  pytest suite – real fixtures, mocked LLM + binary deps      │
│  72 tests, zero external API calls required                  │
└─────────────────────────────────────────────────────────────┘
```

## Requirements

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) binary (only needed for scanned PDFs)
- An OpenAI **or** Anthropic API key

## Setup

```bash
# 1. Clone and enter the project directory
cd doc_compare_rag

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
make install
# or: pip install -r requirements.txt

# 4. Copy the example env file and fill in your keys
cp .env.example .env
```

### `.env` reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `OPENAI_MODEL` | `gpt-4o` | Chat model for OpenAI |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | Chat model for Anthropic |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_CACHE_DIR` | `.cache/embeddings` | Disk cache for computed embeddings |
| `VECTOR_STORE` | `faiss` | `faiss` (in-memory) or `chroma` (persistent) |
| `DATA_DIR` | `data` | Directory for FAISS index files |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Chroma persistence directory |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |

## CLI Usage

```bash
python -m app.main \
  --file1 docs/contract_v1.pdf  --label1 "Version 1" \
  --file2 docs/contract_v2.docx --label2 "Version 2"
```

**Flags:**

| Flag | Description |
|---|---|
| `--file1 PATH` | Path to the first document (required) |
| `--label1 LABEL` | Version label for the first document |
| `--file2 PATH` | Path to the second document (required) |
| `--label2 LABEL` | Version label for the second document |
| `--no-llm` | Skip LLM summarisation (comparison only) |
| `--provider openai\|anthropic` | Override `LLM_PROVIDER` env var |

**Example output:**

```
Ingesting documents …
  Loaded 4 section(s) from A, 5 section(s) from B.
Chunking …
  12 chunk(s) from A, 15 chunk(s) from B.
Embedding and indexing …
  Indexing complete.
Comparing …
Summarising with openai …
========================================================================
DOCUMENT COMPARISON REPORT
  A : docs/contract_v1.pdf
  B : docs/contract_v2.docx
  A label : Version 1
  B label : Version 2
========================================================================
  Sections — unchanged: 2  modified: 2  added: 1  removed: 0

OVERALL SUMMARY
------------------------------------------------------------------------
Clause 3 was expanded to include new data-retention obligations …
```

## Streamlit UI

```bash
make run
# or: streamlit run app/ui/streamlit_app.py
```

1. Upload one or more documents in the sidebar and assign a version label to each.
2. Click **Process Documents** to ingest, chunk, embed, and index.
3. Use the **Comparison Results** tab to select two versions and run a diff (with optional LLM summary).
4. Use the **Ask a Question** tab to query across all indexed documents.
5. Use the **Document Overview** tab to inspect per-version metadata.

## Running Tests

```bash
make test
# or: pytest tests/ -v --tb=short
```

The suite runs without any external API keys or binary dependencies (LLM calls and pytesseract/pdf2image are mocked).

## Linting

```bash
make lint
# or: ruff check app/ tests/ && mypy app/ --ignore-missing-imports
```
