import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.api.dependencies import get_ingestion_service
from services.ingestion_service import IngestionResult, IngestionService

router = APIRouter()

_ALLOWED_EXTENSIONS = {"pdf", "docx", "html", "htm"}


class IngestResponse(BaseModel):
    results: list[IngestionResult]


@router.post("/ingest", response_model=IngestResponse, tags=["ingestion"])
async def ingest(
    files: list[UploadFile] = File(...),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    results: list[IngestionResult] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for upload in files:
            filename = upload.filename or "unknown"
            ext = filename.rsplit(".", 1)[-1].lower()

            if ext not in _ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unsupported file type: .{ext}. Allowed: {sorted(_ALLOWED_EXTENSIONS)}",
                )

            tmp_path = Path(tmp_dir) / filename
            tmp_path.write_bytes(await upload.read())
            results.append(service.ingest(tmp_path))

    return IngestResponse(results=results)
