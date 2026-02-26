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
│   ├── embeddings/         # OpenAI + HuggingFace embedders
│   ├── vectorstore/        # Pinecone + Qdrant store wrappers
│   ├── hybrid_search/      # Dense, sparse, hybrid retrievers
│   └── cache/              # Redis (prod) + in-memory (dev) cache
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

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| API                | FastAPI + Uvicorn                       |
| Orchestration      | LangChain Agents                        |
| LLM                | OpenAI GPT (via langchain-openai)       |
| Vector Store       | Pinecone / Qdrant                       |
| Embeddings         | OpenAI / HuggingFace sentence-transformers |
| Sparse Search      | BM25 (rank-bm25)                        |
| Reranking          | Cohere API                              |
| Parsing            | PyMuPDF, python-docx, BeautifulSoup4, Unstructured |
| Caching            | Redis (prod) / cachetools in-memory (dev) |
| Evaluation         | RAGAS + HuggingFace datasets            |
| UI                 | Streamlit                               |
| Config             | Pydantic Settings                       |
| Logging            | structlog (structured JSON)             |
| Testing            | pytest + pytest-asyncio + pytest-cov    |

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

| Variable                | Required | Description                          |
|-------------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`        | ✅        | OpenAI API key                       |
| `PINECONE_API_KEY`      | ✅        | Pinecone API key                     |
| `PINECONE_INDEX_NAME`   | ✅        | Target Pinecone index name           |
| `COHERE_API_KEY`        | ✅        | Cohere reranking API key             |
| `QDRANT_URL`            | ❌        | Qdrant instance URL (if using Qdrant)|
| `REDIS_URL`             | ❌        | Redis URL (default: in-memory cache) |
| `ENV`                   | ❌        | dev / staging / prod (default: dev)  |
| `LOG_LEVEL`             | ❌        | DEBUG / INFO / WARNING (default: INFO)|

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

| Method | Endpoint          | Description                        |
|--------|-------------------|------------------------------------|
| GET    | `/health`         | Health check                       |
| POST   | `/ingest`         | Upload + ingest documents          |
| POST   | `/query`          | Direct RAG query (no agent)        |
| POST   | `/agent/run`      | Agentic query with tool calling    |

---

## Development Status

| Component            | Status        |
|----------------------|---------------|
| Project Scaffold     | ✅ Done        |
| Config + Logging     | 🔲 In Progress |
| Ingestion Pipeline   | 🔲 Pending     |
| Parsing Layer        | 🔲 Pending     |
| Chunking Layer       | 🔲 Pending     |
| Indexing Pipeline    | 🔲 Pending     |
| Hybrid Retrieval     | 🔲 Pending     |
| Reranking            | 🔲 Pending     |
| Cache Layer          | 🔲 Pending     |
| Agent Layer          | 🔲 Pending     |
| Guardrails           | 🔲 Pending     |
| Evaluation (RAGAS)   | 🔲 Pending     |
| FastAPI Routes       | 🔲 Pending     |
| Streamlit UI         | 🔲 Pending     |
| Deployment           | 🔲 Pending     |
