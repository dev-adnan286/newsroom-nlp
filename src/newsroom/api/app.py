"""Application factory with lifespan-managed artifact loading and global error handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from newsroom import __version__
from newsroom.api.routes import router
from newsroom.api.state import AppState, load_state
from newsroom.config import Settings, get_settings
from newsroom.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    state: AppState | None = None,
    classifier_path: Path | None = None,
    index_dir: Path | None = None,
) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Configuration; defaults to environment-derived settings.
        state: Pre-built state (used by tests to inject in-memory artifacts).
        classifier_path / index_dir: Override artifact locations.
    """
    cfg = settings or get_settings()
    setup_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.newsroom = state or load_state(
            cfg, classifier_path=classifier_path, index_dir=index_dir
        )
        logger.info(
            "Newsroom API ready (classifier=%s, index=%s, auth=%s)",
            app.state.newsroom.predictor is not None,
            app.state.newsroom.engine is not None,
            cfg.auth_enabled,
        )
        yield
        logger.info("Newsroom API shutting down")

    app = FastAPI(
        title="Newsroom NLP API",
        version=__version__,
        description="Classification, embeddings, hybrid search, keywords and summarisation for news text.",
        lifespan=lifespan,
    )
    app.include_router(router)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": _reason(exc.status_code), "detail": exc.detail},
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "validation_error", "detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(ValueError)
    async def _value_error(_: Request, exc: ValueError) -> JSONResponse:
        logger.warning("Bad request: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "bad_request", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal_error", "detail": "An unexpected error occurred"},
        )

    return app


def _reason(code: int) -> str:
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        422: "unprocessable",
        503: "service_unavailable",
    }.get(code, "http_error")


__all__ = ["create_app"]
