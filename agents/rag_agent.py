import time
from typing import Any

import openai
import structlog
from pydantic import BaseModel

from agents.base_agent import BaseAgent
from agents.memory.conversation_memory import ConversationMemory
from config.settings import settings
from core.prompts.rag_prompts import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from services.reranking_service import RerankingService
from services.retrieval_service import RetrievalService

logger = structlog.get_logger(__name__)


class AgentResponse(BaseModel):
    answer: str
    sources: list[str]      # unique source paths, order-preserved
    retrieval_count: int    # results before reranking
    reranked_count: int     # results after reranking
    model: str
    duration_s: float


class RAGAgent(BaseAgent):
    def __init__(
        self,
        retrieval_service: RetrievalService,
        reranking_service: RerankingService,
        model: str | None = None,
        temperature: float | None = None,
    ) -> None:
        self._retrieval = retrieval_service
        self._reranking = reranking_service
        self._model = model or settings.openai.model_name
        self._temperature = temperature if temperature is not None else settings.openai.temperature
        self._client = openai.OpenAI(
            api_key=settings.openai.api_key.get_secret_value()
        )


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        filters: dict | None = None,
        memory: ConversationMemory | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        log = logger.bind(query=query, model=self._model)
        t0 = time.perf_counter()

        # Step 1 — Retrieve
        retrieved = self._retrieval.retrieve(
            query=query,
            top_k=top_k,
            mode=mode,
            filters=filters,
        )

        # Step 2 — Rerank
        reranked = self._reranking.rerank(query=query, results=retrieved)

        # Step 3 — Build context block
        context = "\n\n".join(
            f"[{i + 1}] (source: {r.source})\n{r.content}"
            for i, r in enumerate(reranked)
        )

        # Step 4 — Build messages: [system] [history…] [current user turn]
        history = memory.get_messages() if memory is not None else []
        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            *history,
            {
                "role": "user",
                "content": RAG_USER_TEMPLATE.format(
                    context=context,
                    question=query,
                ),
            },
        ]

        # Step 5 — Call OpenAI
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        answer = response.choices[0].message.content or ""

        # Step 6 — Persist turn to memory
        if memory is not None:
            memory.add("user", query)
            memory.add("assistant", answer)

        # Step 7 — Return structured response
        sources = list(dict.fromkeys(r.source for r in reranked))
        duration = round(time.perf_counter() - t0, 4)

        log.info(
            "rag_agent_complete",
            retrieval_count=len(retrieved),
            reranked_count=len(reranked),
            duration_s=duration,
        )

        return AgentResponse(
            answer=answer,
            sources=sources,
            retrieval_count=len(retrieved),
            reranked_count=len(reranked),
            model=self._model,
            duration_s=duration,
        )
