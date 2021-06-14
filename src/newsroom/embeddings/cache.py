"""On-disk embedding cache keyed by provider name and text content hash."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from newsroom.embeddings.base import EmbeddingProvider
from newsroom.utils.hashing import content_hash
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Cache vectors as ``.npy`` files so repeated embedding calls are free.

    Files are stored under ``<root>/<provider-name>/<hash>.npy``. The cache is transparent:
    :meth:`embed` returns vectors in the same order as the input texts, computing only the
    misses through the wrapped provider.
    """

    def __init__(self, root: str | Path, provider: EmbeddingProvider) -> None:
        self.root = Path(root)
        self.provider = provider
        self._dir = self.root / _safe_name(provider.name)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> Path:
        return self._dir / f"{content_hash(text)}.npy"

    def has(self, text: str) -> bool:
        return self._path(text).exists()

    def get(self, text: str) -> np.ndarray | None:
        p = self._path(text)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except (OSError, ValueError) as exc:  # corrupted file -> treat as miss
            logger.warning("Discarding unreadable cache entry %s: %s", p.name, exc)
            p.unlink(missing_ok=True)
            return None

    def put(self, text: str, vector: np.ndarray) -> None:
        np.save(self._path(text), np.asarray(vector, dtype=np.float32))

    def embed(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Return embeddings for ``texts``, computing and storing only cache misses."""
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.provider.dimension), dtype=np.float32)
        results: list[np.ndarray | None] = [None] * n
        miss_idx: list[int] = []
        for i, text in enumerate(texts):
            vec = self.get(text)
            if vec is None:
                miss_idx.append(i)
            else:
                results[i] = vec
        self.hits += n - len(miss_idx)
        self.misses += len(miss_idx)
        if miss_idx:
            computed = self.provider.embed_texts(
                [texts[i] for i in miss_idx], batch_size=batch_size
            )
            for j, i in enumerate(miss_idx):
                results[i] = computed[j]
                self.put(texts[i], computed[j])
        return np.vstack([r for r in results if r is not None]).astype(np.float32)

    def clear(self) -> int:
        """Delete all cached vectors for this provider; returns the number removed."""
        removed = 0
        for p in self._dir.glob("*.npy"):
            p.unlink()
            removed += 1
        return removed

    def __len__(self) -> int:
        return sum(1 for _ in self._dir.glob("*.npy"))


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)


__all__ = ["EmbeddingCache"]
