from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agents.memory.conversation_memory import ConversationMemory
from agents.rag_agent import AgentResponse
from app.api.dependencies import get_agent_service
from services.agent_service import AgentService, GuardrailError

router = APIRouter()

# In-process session store — replace with Redis for multi-replica deployments.
_sessions: dict[str, ConversationMemory] = {}


class AgentRequest(BaseModel):
    query: str
    session_id: str = "default"
    top_k: int | None = None
    mode: str | None = None
    filters: dict | None = None


@router.post("/agent/run", response_model=AgentResponse, tags=["agent"])
def agent_run(
    request: AgentRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    """Stateful agentic query with per-session conversation memory."""
    memory = _sessions.setdefault(request.session_id, ConversationMemory())

    try:
        return service.run(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            filters=request.filters,
            memory=memory,
        )
    except GuardrailError as exc:
        status = 400 if exc.stage == "input" else 500
        raise HTTPException(status_code=status, detail=str(exc))
