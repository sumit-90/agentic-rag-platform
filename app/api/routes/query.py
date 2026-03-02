from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from agents.rag_agent import AgentResponse
from app.api.dependencies import get_agent_service
from services.agent_service import AgentService, GuardrailError

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None
    mode: str | None = None
    filters: dict | None = None


@router.post("/query", response_model=AgentResponse, tags=["retrieval"])
def query(
    request: QueryRequest,
    service: AgentService = Depends(get_agent_service),
) -> AgentResponse:
    """Stateless RAG query — no conversation memory."""
    try:
        return service.run(
            query=request.query,
            top_k=request.top_k,
            mode=request.mode,
            filters=request.filters,
        )
    except GuardrailError as exc:
        status = 400 if exc.stage == "input" else 500
        raise HTTPException(status_code=status, detail=str(exc))
