"""Mutable application state: loaded classifier and search engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from newsroom.classification.predictor import Predictor
from newsroom.config import Settings
from newsroom.embeddings.base import get_provider
from newsroom.search.query import SearchEngine
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AppState:
    """Artifacts shared by request handlers. Mutations are guarded by a lock."""

    settings: Settings
    predictor: Predictor | None = None
    engine: SearchEngine | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    @property
    def documents_indexed(self) -> int:
        return len(self.engine) if self.engine is not None else 0

    def ensure_engine(self, df: pd.DataFrame) -> tuple[int, bool]:
        """Add documents to the engine, creating it on first use. Returns ``(added, created)``."""
        with self.lock:
            if self.engine is None:
                provider = get_provider(
                    self.settings.embedding_provider,
                    **(
                        {"dimension": self.settings.embedding_dim}
                        if self.settings.embedding_provider == "tfidf_svd"
                        else {"model_name": self.settings.sentence_transformer_model}
                    ),
                )
                self.engine = SearchEngine(provider).build(df)
                return len(self.engine), True
            return self.engine.add_documents(df), False


def load_state(
    settings: Settings, *, classifier_path: Path | None = None, index_dir: Path | None = None
) -> AppState:
    """Load whatever artifacts exist on disk; missing artifacts leave endpoints degraded."""
    state = AppState(settings=settings)
    clf_path = classifier_path or settings.classifier_path
    idx_dir = index_dir or settings.index_dir
    if clf_path.exists():
        try:
            state.predictor = Predictor.from_path(clf_path)
            logger.info("Loaded classifier from %s", clf_path)
        except Exception as exc:  # noqa: BLE001 - keep API booting even with a bad artifact
            logger.error("Failed to load classifier %s: %s", clf_path, exc)
    else:
        logger.warning("No classifier found at %s; /classify will return 503", clf_path)
    if idx_dir.exists():
        try:
            state.engine = SearchEngine.load(idx_dir)
            logger.info("Loaded search index from %s", idx_dir)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load search index %s: %s", idx_dir, exc)
    else:
        logger.warning(
            "No search index found at %s; /search will return 503 until documents are indexed",
            idx_dir,
        )
    return state


__all__ = ["AppState", "load_state"]
