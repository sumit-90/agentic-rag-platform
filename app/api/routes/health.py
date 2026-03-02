from fastapi import APIRouter
from pydantic import BaseModel

from config.settings import settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    env: str


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app.version,
        env=settings.app.env,
    )
