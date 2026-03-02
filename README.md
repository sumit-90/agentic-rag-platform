# Agentic RAG Platform

A production-grade Retrieval-Augmented Generation system with agentic orchestration,
hybrid retrieval, tiered document parsing, reranking, guardrails, and RAGAS-based evaluation.

---

## Architecture Overview

```
User Query
    │
    ▼
[FastAPI Layer]              ← thin routes, dependency injection only
    │
    ▼
[Agent Service]              ← orchestration + LangChain tool-calling loop
    │
    ├──▶ [Guardrail Service]      ← input sanitization + output validation
    │
    ├──▶ [Retrieval Service]      ← hybrid dense + sparse search
    │         │
    │         ├──▶ [Embedding Cache]       (Redis / in-memory)
    │         ├──▶ [Dense Retriever]       (Pinecone / Qdrant)
    │         └──▶ [Sparse Retriever]      (BM25)
    │
    ├──▶ [Reranking Service]      ← Cohere reranker / cross-encoder
    │
    └──▶ [LLM Response]           ← OpenAI GPT (via LangChain)
```

---

## Ingestion Pipeline

```
Raw File (PDF / DOCX / HTML)
    │
    ▼
[ParserFactory]              ← selects parser by file extension
    │
    ▼
[ParsedDocument]             ← typed Pydantic model (content + metadata)
    │
    ▼
[ChunkerFactory]             ← selects fixed or semantic chunker
    │
    ▼
[list[Chunk]]                ← typed Pydantic model (id + content + metadata)
    │
    ▼
[Embedder]                   ← OpenAI or HuggingFace, batched
    │
    ▼
[Cache Check]                ← Redis (prod) / in-memory (dev)
    │
    ▼
[VectorStore]                ← Pinecone or Qdrant
```

---

## Project Structure

```
agentic-rag-platform/
│
├── app/                    # Entry points only (FastAPI app factory + routes)
│   ├── main.py
│   └── api/
│       ├── dependencies.py
│       └── routes/         # health, ingest, query, agent
│
├── services/               # Business logic — one file per domain capability
│   ├── ingestion_service.py
│   ├── indexing_service.py
│   ├── retrieval_service.py
│   ├── reranking_service.py
│   ├── agent_service.py
│   ├── tool_service.py
│   ├── guardrail_service.py
│   └── evaluation_service.py
│
├── core/                   # Reusable low-level components
│   ├── parsing/            # PDF, DOCX, HTML parsers + factory
│   ├── chunking/           # Fixed, semantic chunkers + factory
│   ├── embeddings/         # OpenAI + HuggingFace embedders (batched)
│   ├── vectorstore/        # Pinecone + Qdrant store wrappers
│   ├── hybrid_search/      # Dense, sparse, hybrid retrievers
│   ├── cache/              # Redis (prod) + in-memory (dev) cache
│   └── prompts/            # RAG, agent, guardrail prompt templates
│
├── agents/                 # Agentic orchestration layer
│   ├── base_agent.py
│   ├── rag_agent.py
│   ├── tools/              # retrieval, rerank, summarize tools
│   └── memory/             # conversation memory
│
├── guardrails/             # Input + output validation
├── evaluation/             # RAGAS evaluator + dataset builder + metrics
├── config/                 # Pydantic settings + structlog config
├── ui/                     # Streamlit frontend
├── data/                   # raw / processed / evaluation datasets
├── tests/                  # unit + integration tests
├── scripts/                # CLI: batch ingestion + evaluation runner
└── deployment/             # Dockerfile + docker-compose
```

---

## Tech Stack

| Layer              | Technology                                          |
|--------------------|-----------------------------------------------------|
| API                | FastAPI + Uvicorn                                   |
| Orchestration      | LangChain Agents                                    |
| LLM                | OpenAI GPT-4o (via langchain-openai)                |
| Vector Store       | Pinecone / Qdrant                                   |
| Embeddings         | OpenAI text-embedding-3-small / HuggingFace ST      |
| Sparse Search      | BM25 (rank-bm25)                                    |
| Reranking          | Cohere rerank-english-v3.0                          |
| Parsing            | PyMuPDF, python-docx, BeautifulSoup4, Unstructured  |
| Chunking           | LangChain RecursiveCharacterTextSplitter / Semantic |
| Caching            | Redis (prod) / cachetools in-memory (dev)           |
| Evaluation         | RAGAS + HuggingFace datasets                        |
| UI                 | Streamlit                                           |
| Config             | Pydantic Settings (nested, env-driven)              |
| Logging            | structlog (JSON in prod, console in dev)            |
| Retry              | tenacity (exponential backoff)                      |
| Testing            | pytest + pytest-asyncio + pytest-cov                |

---

## Core Components (Built)

### Parsing Layer — `core/parsing/`
Tiered document parsing with a factory pattern. All parsers return a typed
`ParsedDocument` Pydantic model — consistent contract regardless of file format.

| Parser | Library | Notes |
|---|---|---|
| `PDFParser` | PyMuPDF (fitz) | Page-by-page extraction, encrypted PDF guard |
| `DOCXParser` | python-docx | Filters empty paragraphs |
| `HTMLParser` | BeautifulSoup4 + lxml | Strips script/style tags, extracts title |
| `ParserFactory` | — | Registry-based dispatch by file extension |

---

### Chunking Layer — `core/chunking/`
Strategy-based chunking with factory dispatch. All chunkers consume `ParsedDocument`
and return `list[Chunk]` with deterministic IDs and full metadata inheritance.

| Chunker | Strategy | Notes |
|---|---|---|
| `FixedChunker` | `RecursiveCharacterTextSplitter` | Size + overlap from settings |
| `SemanticChunker` | LangChain `SemanticChunker` | OpenAI embeddings, slow-threshold warning |
| `ChunkerFactory` | — | Registry dispatch by strategy name |

**Chunk ID generation:** `md5(source + chunk_index)` — deterministic, deduplication-safe.

---

### Embeddings Layer — `core/embeddings/`
Batch-optimized embedding with provider abstraction. Ingestion pipeline always
calls `embed_batch()` — never `embed()` in a loop.

| Embedder | Provider | Batch Size | Notes |
|---|---|---|---|
| `OpenAIEmbedder` | OpenAI API | 256 (configurable) | Retry with exponential backoff |
| `HFEmbedder` | sentence-transformers | 64 (configurable) | Local inference, no API calls |

**Retry policy:** 3 attempts, exponential backoff (1s → 10s), `reraise=True`.

---

### Cache Layer — `core/cache/`
Drop-in replaceable cache backends behind a common `BaseCache` interface.
Cache failures never crash the main pipeline — always degrade gracefully.

| Backend | Use Case | TTL Support |
|---|---|---|
| `InMemoryCache` | Development / testing | ✅ via cachetools |
| `RedisCache` | Production | ✅ native Redis TTL |

**Key convention:** `BaseCache.make_key("embedding", "doc_id", "chunk_0")` → `"embedding:doc_id:chunk_0"`

---

## Quickstart

```bash
# 1. Clone repository
git clone <repo-url>
cd agentic-rag-platform

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env
# Open .env and fill in your API keys

# 5. Start API server
uvicorn app.main:app --reload --port 8000

# 6. Start UI (separate terminal)
streamlit run ui/streamlit_app.py
```

---

## Environment Variables

> All nested settings use `__` as delimiter (e.g. `OPENAI__API_KEY` maps to `settings.openai.api_key`)

| Variable                        | Required | Default                     | Description                        |
|---------------------------------|----------|-----------------------------|------------------------------------|
| `OPENAI__API_KEY`               | ✅        | —                           | OpenAI API key                     |
| `OPENAI__MODEL_NAME`            | ❌        | `gpt-4o`                    | LLM model identifier               |
| `OPENAI__TEMPERATURE`           | ❌        | `0.0`                       | Sampling temperature (0.0–2.0)     |
| `OPENAI__MAX_TOKENS`            | ❌        | `2048`                      | Max tokens per LLM response        |
| `PINECONE__API_KEY`             | ✅        | —                           | Pinecone API key                   |
| `PINECONE__INDEX_NAME`          | ❌        | `rag-index`                 | Target Pinecone index              |
| `PINECONE__ENVIRONMENT`         | ❌        | `us-east-1-aws`             | Pinecone cloud region              |
| `PINECONE__NAMESPACE`           | ❌        | `default`                   | Index namespace                    |
| `COHERE__API_KEY`               | ✅        | —                           | Cohere reranking API key           |
| `COHERE__RERANK_MODEL`          | ❌        | `rerank-english-v3.0`       | Reranking model                    |
| `COHERE__TOP_N`                 | ❌        | `5`                         | Results to keep after reranking    |
| `QDRANT__URL`                   | ❌        | `http://localhost`          | Qdrant server URL                  |
| `QDRANT__COLLECTION_NAME`       | ❌        | `rag-collection`            | Qdrant collection name             |
| `QDRANT__PORT`                  | ❌        | `6333`                      | Qdrant server port                 |
| `REDIS__URL`                    | ❌        | `redis://localhost:6379`    | Redis connection URL               |
| `REDIS__TTL_SECONDS`            | ❌        | `3600`                      | Cache entry TTL (seconds)          |
| `EMBEDDING__PROVIDER`           | ❌        | `openai`                    | openai / huggingface               |
| `EMBEDDING__MODEL_NAME`         | ❌        | `text-embedding-3-small`    | Embedding model identifier         |
| `EMBEDDING__BATCH_SIZE`         | ❌        | `256`                       | Texts per API batch call           |
| `CHUNKING__CHUNK_SIZE`          | ❌        | `512`                       | Target tokens per chunk            |
| `CHUNKING__CHUNK_OVERLAP`       | ❌        | `90`                        | Token overlap between chunks       |
| `CHUNKING__STRATEGY`            | ❌        | `fixed`                     | fixed / semantic                   |
| `RETRIEVAL__TOP_K`              | ❌        | `10`                        | Candidates before reranking        |
| `RETRIEVAL__MODE`               | ❌        | `hybrid`                    | dense / sparse / hybrid            |
| `RETRIEVAL__SCORE_THRESHOLD`    | ❌        | `0.7`                       | Minimum relevance score (0.0–1.0)  |
| `APP__ENV`                      | ❌        | `dev`                       | dev / staging / prod               |
| `APP__LOG_LEVEL`                | ❌        | `INFO`                      | DEBUG / INFO / WARNING / ERROR     |

---

## Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v
```

---

## Batch Ingestion (CLI)

```bash
python scripts/ingest_batch.py --source data/raw/
```

---

## Running RAGAS Evaluation

```bash
python scripts/evaluate.py --dataset data/evaluation/testset.json
```

---

## API Endpoints

| Method | Endpoint       | Description                        |
|--------|----------------|------------------------------------|
| GET    | `/health`      | Health check                       |
| POST   | `/ingest`      | Upload + ingest documents          |
| POST   | `/query`       | Direct RAG query (no agent)        |
| POST   | `/agent/run`   | Agentic query with tool calling    |

---

## Development Status

| Component              | Status          | Details                                          |
|------------------------|-----------------|--------------------------------------------------|
| Project Scaffold       | ✅ Done         | Full folder structure, packages, gitignore       |
| Config + Logging       | ✅ Done         | Pydantic BaseSettings, structlog, .env.example   |
| Parsing Layer          | ✅ Done         | PDF, DOCX, HTML parsers + factory pattern        |
| Chunking Layer         | ✅ Done         | Fixed + semantic chunkers + factory pattern      |
| Embeddings Layer       | ✅ Done         | OpenAI + HuggingFace, batch optimized, retries   |
| Cache Layer            | ✅ Done         | Redis + in-memory backends                       |
| Vector Store Layer     | ✅ Done         | Pinecone + Qdrant wrappers                       |
| Ingestion Service      | ✅ Done         | Orchestrates parse → chunk → embed → index       |
| Hybrid Retrieval       | ✅ Done         | Dense + sparse + hybrid retriever                |
| Reranking              | 🔲 In Progress  | Cohere reranker integration                      |
| Agent Layer            | 🔲 Pending      | RAG agent + tool calling + memory                |
| Guardrails             | 🔲 Pending      | Input/output validation                          |
| Evaluation (RAGAS)     | 🔲 Pending      | Faithfulness, relevancy, precision metrics       |
| FastAPI Routes         | 🔲 Pending      | ingest, query, agent, health endpoints           |
| Streamlit UI           | 🔲 Pending      | Chat interface + document upload                 |
| Deployment             | 🔲 Pending      | Dockerfile + docker-compose                      |
