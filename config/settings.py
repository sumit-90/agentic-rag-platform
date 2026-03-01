from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseModel):
    env: Literal["dev", "staging", "prod"] = "dev"
    debug: bool = False
    app_name: str = "Agentic RAG Platform"
    version: str = "0.1.0"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


class OpenAISettings(BaseModel):
    api_key: SecretStr
    model_name: str = "gpt-4o"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)


class EmbeddingSettings(BaseModel):
    provider: Literal["openai", "huggingface"] = "openai"
    model_name: str = "text-embedding-3-small"
    batch_size: int = Field(default=256, gt=0)


class PineconeSettings(BaseModel):
    api_key: SecretStr
    index_name: str = "rag-index"
    environment: str = "us-east-1-aws"
    namespace: str = "default"


class CohereSettings(BaseModel):
    api_key: SecretStr
    rerank_model: str = "rerank-english-v3.0"
    top_n: int = Field(default=5, gt=0)


class QdrantSettings(BaseModel):
    url: str = "http://localhost"
    collection_name: str = "rag-collection"
    port: int = Field(default=6333, gt=0, lt=65536)


class RedisSettings(BaseModel):
    url: str = "redis://localhost:6379"
    ttl_seconds: int = Field(default=3600, gt=0)


class ChunkingSettings(BaseModel):
    chunk_size: int = Field(default=512, gt=0)
    chunk_overlap: int = Field(default=90, ge=0)
    strategy: Literal["fixed", "semantic"] = "fixed"


class RetrievalSettings(BaseModel):
    top_k: int = Field(default=10, gt=0)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class VectorStoreSettings(BaseModel):
    provider: Literal["pinecone", "qdrant"] = "pinecone"
    upsert_batch_size: int = Field(default=100, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    app: AppSettings = AppSettings()
    openai: OpenAISettings
    embedding: EmbeddingSettings = EmbeddingSettings()
    pinecone: PineconeSettings
    cohere: CohereSettings
    qdrant: QdrantSettings = QdrantSettings()
    redis: RedisSettings = RedisSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    retrieval: RetrievalSettings = RetrievalSettings()
    vectorstore: VectorStoreSettings = VectorStoreSettings()


settings = Settings()
