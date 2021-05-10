"""FastAPI dependencies: state access and optional API-key authentication."""

from __future__ import annotations

import hmac

from fastapi import HTTPException, Request, status

from newsroom.api.state import AppState


def get_state(request: Request) -> AppState:
    """Fetch the :class:`AppState` attached to the application."""
    state = getattr(request.app.state, "newsroom", None)
    if state is None:  # pragma: no cover - only if lifespan never ran
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Application state not initialised"
        )
    return state


async def require_api_key(request: Request) -> None:
    """Reject the request unless the configured API key is present (no-op when disabled)."""
    state = get_state(request)
    settings = state.settings
    if not settings.auth_enabled:
        return
    provided = request.headers.get(settings.api_key_header)
    expected = settings.api_key or ""
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing or invalid API key (header {settings.api_key_header})",
            headers={"WWW-Authenticate": "ApiKey"},
        )


def require_predictor(request: Request) -> AppState:
    state = get_state(request)
    if state.predictor is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No classifier is loaded; train one with `newsroom train`",
        )
    return state


def require_engine(request: Request) -> AppState:
    state = get_state(request)
    if state.engine is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "No search index is loaded; build one with `newsroom build-index` or POST /index/documents",
        )
    return state


__all__ = ["get_state", "require_api_key", "require_engine", "require_predictor"]
