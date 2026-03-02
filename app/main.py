from fastapi import FastAPI

from app.api.routes import api_router
from config.logging_config import setup_logging
from config.settings import settings


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app.app_name,
        version=settings.app.version,
        debug=settings.app.debug,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(api_router)
    return app


# ASGI entry point — referenced by uvicorn app.main:app
app = create_app()
