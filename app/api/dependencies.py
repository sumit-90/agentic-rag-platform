from functools import lru_cache

from agents.rag_agent import RAGAgent
from config.settings import settings
from core.cache.in_memory_cache import InMemoryCache
from core.cache.redis_cache import RedisCache
from core.embeddings.openai_embedder import OpenAIEmbedder
from core.hybrid_search.hybrid_retriever import HybridRetriever
from core.vectorstore.vectorstore_factory import VectorStoreFactory
from services.agent_service import AgentService
from services.guardrail_service import GuardrailService
from services.ingestion_service import IngestionService
from services.reranking_service import RerankingService
from services.retrieval_service import RetrievalService


@lru_cache(maxsize=1)
def get_embedder() -> OpenAIEmbedder:
    return OpenAIEmbedder()


@lru_cache(maxsize=1)
def get_vector_store():
    return VectorStoreFactory.get_store()


@lru_cache(maxsize=1)
def get_cache():
    if settings.app.env == "prod":
        return RedisCache()
    return InMemoryCache()


@lru_cache(maxsize=1)
def get_ingestion_service() -> IngestionService:
    return IngestionService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        cache=get_cache(),
    )


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    retriever = HybridRetriever(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
    )
    return RetrievalService(hybrid_retriever=retriever, cache=get_cache())


@lru_cache(maxsize=1)
def get_reranking_service() -> RerankingService:
    return RerankingService()


@lru_cache(maxsize=1)
def get_guardrail_service() -> GuardrailService:
    return GuardrailService()


@lru_cache(maxsize=1)
def get_rag_agent() -> RAGAgent:
    return RAGAgent(
        retrieval_service=get_retrieval_service(),
        reranking_service=get_reranking_service(),
    )


@lru_cache(maxsize=1)
def get_agent_service() -> AgentService:
    return AgentService(
        agent=get_rag_agent(),
        guardrail_service=get_guardrail_service(),
    )
