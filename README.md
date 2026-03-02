# Agentic RAG Platform

A production-grade Retrieval-Augmented Generation system with agentic orchestration,
hybrid retrieval, tiered document parsing, reranking, guardrails, and RAGAS-based evaluation.

---

## Architecture Overview

```
User Query
    |
    v
[FastAPI Layer]              <- thin routes, dependency injection only
    |
    v
[Agent Service]              <- guardrail checks + RAGAgent orchestration
    |
    |---> [Guardrail Service]      <- input sanitization + output validation
    |
    |---> [RAG Agent]              <- retrieve -> rerank -> generate
    |         |
    |         |---> [Retrieval Service]   <- hybrid dense + sparse search
    |         |         |---> [Embedding Cache]     (Redis / in-memory)
    |         |         |---> [Dense Retriever]     (Pinecone / Qdrant)
    |         |         '---> [Sparse Retriever]    (BM25)
    |         |
    |         |---> [Reranking Service]  <- Cohere rerank-english-v3.0
    |         |
    |         '---> [OpenAI GPT-4o]     <- final answer generation
    |
    '---> [Conversation Memory]    <- sliding-window multi-turn history
```

---

## Ingestion Pipeline

```
Raw File (PDF / DOCX / HTML)
    |
    v
[ParserFactory]              <- selects parser by file extension
    |
    v
[ParsedDocument]             <- typed Pydantic model (content + metadata)
    |
    v
[ChunkerFactory]             <- selects fixed or semantic chunker
    |
    v
[list[Chunk]]                <- deterministic ID via md5(source + index)
    |
    v
[Cache Check]                <- skip already-indexed chunks (Redis / in-memory)
    |
    v
[Embedder]                   <- OpenAI or HuggingFace, batch-optimized
    |
    v
[VectorStore]                <- Pinecone or Qdrant
```

---

## Project Structure

```
agentic-rag-platform/
|
|-- app/                    # FastAPI application layer (thin routes only)
|   |-- main.py             # App factory + uvicorn entry point
|   '-- api/
|       |-- dependencies.py # lru_cache singleton factories for all services
|       '-- routes/         # health, ingest, query, agent
|
|-- services/               # Business logic - one file per domain capability
|   |-- ingestion_service.py
|   |-- retrieval_service.py
|   |-- reranking_service.py
|   |-- agent_service.py    # GuardrailError + RAGAgent orchestration
|   |-- guardrail_service.py
|   '-- evaluation_service.py
|
|-- core/                   # Reusable low-level components
|   |-- parsing/            # PDF, DOCX, HTML parsers + factory
|   |-- chunking/           # Fixed, semantic chunkers + factory
|   |-- embeddings/         # OpenAI + HuggingFace embedders (batched)
|   |-- vectorstore/        # Pinecone + Qdrant store wrappers + factory
|   |-- hybrid_search/      # Dense, sparse, hybrid retrievers (RRF)
|   |-- cache/              # Redis (prod) + in-memory (dev) cache
|   '-- prompts/            # RAG prompt templates
|
|-- agents/                 # Agentic orchestration layer
|   |-- base_agent.py       # ABC contract
|   |-- rag_agent.py        # Main RAG agent (retrieve -> rerank -> generate)
|   '-- memory/
|       '-- conversation_memory.py  # Sliding-window multi-turn memory
|
|-- guardrails/             # Input + output validation (rule-based)
|   |-- input_guard.py      # GuardResult + InputGuard (length + injection)
|   '-- output_guard.py     # OutputGuard (empty + truncation)
|
|-- evaluation/             # RAGAS evaluation pipeline
|   |-- metrics.py          # Default metric suite
|   |-- dataset_builder.py  # EvalSample -> RAGAS EvaluationDataset
|   '-- ragas_evaluator.py  # Runs evaluate() and returns score dict
|
|-- config/                 # Settings + logging
|   |-- settings.py         # Pydantic BaseSettings (nested, env-driven)
|   '-- logging_config.py   # structlog (JSON prod / console dev)
|
|-- ui/
|   '-- streamlit_app.py    # Chat UI + document upload sidebar
|
|-- data/                   # raw / processed / evaluation datasets
|-- tests/                  # unit + integration tests
|-- scripts/                # CLI: batch ingestion + evaluation runner
'-- deployment/
    |-- Dockerfile
    '-- docker-compose.yml  # API + UI + Redis + Qdrant
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| LLM | OpenAI GPT-4o (direct SDK) |
| Vector Store | Pinecone / Qdrant |
| Embeddings | OpenAI text-embedding-3-small / HuggingFace ST |
| Sparse Search | BM25 (rank-bm25) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranking | Cohere rerank-english-v3.0 |
| Parsing | PyMuPDF, python-docx, BeautifulSoup4 + lxml |
| Chunking | LangChain RecursiveCharacterTextSplitter / Semantic |
| Caching | Redis (prod) / in-memory (dev) |
| Guardrails | Rule-based (length + injection detection) |
| Evaluation | RAGAS + HuggingFace datasets |
| UI | Streamlit |
| Config | Pydantic Settings (nested, env-driven) |
| Logging | structlog (JSON in prod, console in dev) |
| Retry | tenacity (exponential backoff) |
| Testing | pytest + pytest-asyncio + pytest-cov |
| Deployment | Docker + docker-compose |

---

## Core Components

### Parsing Layer -- `core/parsing/`

| Parser | Library | Notes |
|---|---|---|
| `PDFParser` | PyMuPDF (fitz) | Page-by-page extraction, encrypted PDF guard |
| `DOCXParser` | python-docx | Filters empty paragraphs |
| `HTMLParser` | BeautifulSoup4 + lxml | Strips script/style tags before extraction |
| `ParserFactory` | -- | Registry-based dispatch by file extension |

---

### Chunking Layer -- `core/chunking/`

| Chunker | Strategy | Notes |
|---|---|---|
| `FixedChunker` | `RecursiveCharacterTextSplitter` | Size + overlap from settings |
| `SemanticChunker` | LangChain `SemanticChunker` | OpenAI embeddings, slow-path warning |

**Chunk ID:** `md5(source + chunk_index)` -- deterministic, deduplication-safe.

---

### Embeddings Layer -- `core/embeddings/`

| Embedder | Batch Size | Notes |
|---|---|---|
| `OpenAIEmbedder` | 256 (configurable) | tenacity retry (3 attempts, exp backoff) |
| `HFEmbedder` | 64 (configurable) | Local SentenceTransformer inference |

---

### Cache Layer -- `core/cache/`

| Backend | Use Case | TTL |
|---|---|---|
| `InMemoryCache` | Development / testing | Per-item via monotonic clock |
| `RedisCache` | Production | Native Redis TTL |

**Key convention:** `make_key("embedding", "doc_id")` -> `"embedding:doc_id"`

---

### Hybrid Retrieval -- `core/hybrid_search/`

| Component | Role |
|---|---|
| `DenseRetriever` | Thin wrapper over `BaseVectorStore.search()` |
| `SparseRetriever` | BM25Okapi -- zero-score results filtered out |
| `HybridRetriever` | RRF fusion (k=60), `fetch_k = top_k x 2` for richer candidates |

Score threshold applied to dense mode only -- incompatible with RRF scores.

---

### Guardrails -- `guardrails/`

| Guard | Checks |
|---|---|
| `InputGuard` | Min/max length + 5 prompt-injection pattern strings (case-insensitive) |
| `OutputGuard` | Empty response + max length (truncates with `[truncated]` suffix) |

`AgentService` raises `GuardrailError(stage="input"|"output")` -- routes convert to HTTP 400/500.

---

### Agent Layer -- `agents/`

| Component | Role |
|---|---|
| `BaseAgent` | ABC -- defines `run(query, **kwargs) -> Any` contract |
| `RAGAgent` | Retrieve -> Rerank -> Build context -> Call GPT-4o -> Return `AgentResponse` |
| `ConversationMemory` | Sliding-window history (`max_turns=20`), injected between system + user messages |

---

### Evaluation -- `evaluation/`

| Metric | What it measures |
|---|---|
| `Faithfulness` | Is the answer grounded in retrieved context? |
| `ResponseRelevancy` | Does the answer address the question? |
| `LLMContextPrecisionWithReference` | Are retrieved chunks relevant to the question? |
| `LLMContextRecall` | Does retrieved context cover the ground-truth answer? |

---

## Quickstart

```bash
# 1. Clone and enter directory
git clone <repo-url>
cd agentic-rag-platform

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env -- fill in OPENAI__API_KEY, PINECONE__API_KEY, COHERE__API_KEY

# 5. Start API server
uvicorn app.main:app --reload --port 8000

# 6. Start Streamlit UI (separate terminal)
streamlit run ui/streamlit_app.py
```

---

## Docker Deployment

```bash
cd deployment

# Build and start all services (API + UI + Redis + Qdrant)
docker-compose up --build

# API:      http://localhost:8000
# UI:       http://localhost:8501
# API docs: http://localhost:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check -- returns status, version, env |
| POST | `/ingest` | Upload + ingest documents (multipart/form-data) |
| POST | `/query` | Stateless RAG query (no memory) |
| POST | `/agent/run` | Stateful agentic query with session memory |
| GET | `/docs` | Interactive Swagger UI |

---

## Environment Variables

> All nested settings use `__` as delimiter (e.g. `OPENAI__API_KEY` maps to `settings.openai.api_key`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI__API_KEY` | Yes | -- | OpenAI API key |
| `OPENAI__MODEL_NAME` | No | `gpt-4o` | LLM model |
| `OPENAI__TEMPERATURE` | No | `0.0` | Sampling temperature |
| `OPENAI__MAX_TOKENS` | No | `2048` | Max tokens per response |
| `PINECONE__API_KEY` | Yes | -- | Pinecone API key |
| `PINECONE__INDEX_NAME` | No | `rag-index` | Target index |
| `PINECONE__ENVIRONMENT` | No | `us-east-1-aws` | Cloud region |
| `PINECONE__NAMESPACE` | No | `default` | Index namespace |
| `COHERE__API_KEY` | Yes | -- | Cohere reranking API key |
| `COHERE__RERANK_MODEL` | No | `rerank-english-v3.0` | Reranking model |
| `COHERE__TOP_N` | No | `5` | Results after reranking |
| `QDRANT__URL` | No | `http://localhost` | Qdrant server URL |
| `QDRANT__COLLECTION_NAME` | No | `rag-collection` | Collection name |
| `QDRANT__PORT` | No | `6333` | Qdrant port |
| `REDIS__URL` | No | `redis://localhost:6379` | Redis connection URL |
| `REDIS__TTL_SECONDS` | No | `3600` | Cache TTL (seconds) |
| `EMBEDDING__PROVIDER` | No | `openai` | `openai` / `huggingface` |
| `EMBEDDING__MODEL_NAME` | No | `text-embedding-3-small` | Embedding model |
| `EMBEDDING__BATCH_SIZE` | No | `256` | Texts per API batch |
| `CHUNKING__CHUNK_SIZE` | No | `512` | Target tokens per chunk |
| `CHUNKING__CHUNK_OVERLAP` | No | `90` | Token overlap |
| `CHUNKING__STRATEGY` | No | `fixed` | `fixed` / `semantic` |
| `RETRIEVAL__TOP_K` | No | `10` | Candidates before reranking |
| `RETRIEVAL__MODE` | No | `hybrid` | `dense` / `sparse` / `hybrid` |
| `RETRIEVAL__SCORE_THRESHOLD` | No | `0.7` | Min score (dense mode only) |
| `APP__ENV` | No | `dev` | `dev` / `staging` / `prod` |
| `APP__LOG_LEVEL` | No | `INFO` | Log verbosity level |

---

## Running Tests

```bash
pytest tests/ -v --cov=. --cov-report=term-missing   # all tests + coverage
pytest tests/unit/ -v                                  # unit only
pytest tests/integration/ -v                           # integration only
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

## Development Status

| Component | Status | Details |
|---|---|---|
| Project Scaffold | Done | Full folder structure, packages, gitignore |
| Config + Logging | Done | Pydantic BaseSettings, structlog, .env.example |
| Parsing Layer | Done | PDF, DOCX, HTML parsers + factory pattern |
| Chunking Layer | Done | Fixed + semantic chunkers + factory pattern |
| Embeddings Layer | Done | OpenAI + HuggingFace, batch-optimized, retries |
| Cache Layer | Done | Redis + in-memory backends, per-item TTL |
| Vector Store Layer | Done | Pinecone + Qdrant wrappers + factory |
| Ingestion Service | Done | parse -> chunk -> cache-filter -> embed -> index |
| Hybrid Retrieval | Done | Dense + BM25 sparse + RRF hybrid |
| Reranking Service | Done | Cohere rerank-english-v3.0 integration |
| Agent Layer | Done | RAGAgent + BaseAgent ABC + ConversationMemory |
| Guardrails | Done | InputGuard + OutputGuard + GuardrailService |
| Agent Service | Done | GuardrailError orchestration around RAGAgent |
| Evaluation (RAGAS) | Done | Faithfulness, Relevancy, Precision, Recall |
| FastAPI Routes | Done | /health, /ingest, /query, /agent/run |
| Streamlit UI | Done | Chat interface + document upload sidebar |
| Deployment | Done | Dockerfile + docker-compose (API + UI + Redis + Qdrant) |
