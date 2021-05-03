"""FastAPI service exposing classification, embeddings, search, keywords and summaries."""

from newsroom.api.app import create_app
from newsroom.api.state import AppState

__all__ = ["AppState", "create_app"]
