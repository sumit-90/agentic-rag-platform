import structlog

from agents.memory.conversation_memory import ConversationMemory
from agents.rag_agent import AgentResponse, RAGAgent
from services.guardrail_service import GuardrailService

logger = structlog.get_logger(__name__)


class GuardrailError(Exception):
    """Raised when an input or output guardrail check fails."""

    def __init__(self, message: str, stage: str = "input") -> None:
        super().__init__(message)
        self.stage = stage   # "input" | "output"


class AgentService:
    """Orchestrates guardrail checks around RAGAgent.run().

    Responsibility boundary
    -----------------------
    - AgentService owns the guardrail + agent wiring.
    - FastAPI routes stay thin: they only catch GuardrailError and convert
      it to the appropriate HTTP status code.
    """

    def __init__(
        self,
        agent: RAGAgent,
        guardrail_service: GuardrailService,
    ) -> None:
        self._agent = agent
        self._guardrail = guardrail_service

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
    ) -> AgentResponse:
        # Step 1 — Input guardrail
        input_check = self._guardrail.check_input(query)
        if not input_check.passed:
            raise GuardrailError(
                input_check.reason or "Input rejected by guardrail.",
                stage="input",
            )

        # Step 2 — Run the RAG agent
        response = self._agent.run(
            query=query,
            top_k=top_k,
            mode=mode,
            filters=filters,
            memory=memory,
        )

        # Step 3 — Output guardrail
        output_check = self._guardrail.check_output(response.answer)
        if not output_check.passed:
            raise GuardrailError(
                output_check.reason or "Output rejected by guardrail.",
                stage="output",
            )

        # Use sanitized content (may be truncated by OutputGuard)
        return response.model_copy(
            update={"answer": output_check.content or response.answer}
        )
