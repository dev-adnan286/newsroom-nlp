"""The :class:`EmbeddingProvider` protocol and a small factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Anything that maps texts to dense vectors.

    Implementations must expose a stable ``name``, the output ``dimension``, and an
    ``embed_texts`` method returning a 2-D ``float32`` array of shape ``(n_texts, dimension)``.
    Providers that need fitting (e.g. TF-IDF + SVD) expose ``fit``/``is_fitted``.
    """

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    @property
    def is_fitted(self) -> bool: ...

    def fit(self, texts: Sequence[str]) -> EmbeddingProvider: ...

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray: ...


def get_provider(name: str = "tfidf_svd", **kwargs: Any) -> EmbeddingProvider:
    """Instantiate a provider by name.

    Args:
        name: ``"tfidf_svd"`` (offline) or ``"sentence_transformers"`` (optional extra).
        **kwargs: Forwarded to the provider constructor.
    """
    key = name.strip().lower().replace("-", "_")
    if key in {"tfidf_svd", "tfidf", "lsa"}:
        from newsroom.embeddings.tfidf_svd import TfidfSvdProvider

        return TfidfSvdProvider(**kwargs)
    if key in {"sentence_transformers", "sbert", "st"}:
        from newsroom.embeddings.sentence_transformers_provider import SentenceTransformerProvider

        return SentenceTransformerProvider(**kwargs)
    raise ValueError(f"Unknown embedding provider: {name!r}")


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalise a 2-D array to unit L2 length (zero rows stay zero)."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return (arr / norms).astype(np.float32)


__all__ = ["EmbeddingProvider", "get_provider", "l2_normalize"]
