"""In-memory dense vector index with cosine top-k, incremental adds and persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from newsroom.embeddings.base import l2_normalize
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SearchHit:
    """A retrieved document with its score and rank."""

    doc_id: str
    score: float
    rank: int
    metadata: dict[str, Any]


class VectorIndex:
    """Brute-force cosine similarity index over L2-normalised float32 vectors.

    Suitable for corpora up to a few hundred thousand documents; exact results, no ANN.
    Vectors are stored in a pre-allocated buffer that doubles when full so repeated
    :meth:`add` calls are amortised O(1).
    """

    def __init__(self, dimension: int, *, initial_capacity: int = 1024) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors = np.zeros((max(1, initial_capacity), dimension), dtype=np.float32)
        self._size = 0
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._metadata: list[dict[str, Any]] = []

    # -- properties --------------------------------------------------------------------
    def __len__(self) -> int:
        return self._size

    @property
    def ids(self) -> list[str]:
        return list(self._ids)

    @property
    def vectors(self) -> np.ndarray:
        """View of the stored vectors (``(n, dimension)``)."""
        return self._vectors[: self._size]

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._id_to_row

    def get_metadata(self, doc_id: str) -> dict[str, Any]:
        return self._metadata[self._id_to_row[doc_id]]

    # -- mutation ----------------------------------------------------------------------
    def _ensure_capacity(self, extra: int) -> None:
        needed = self._size + extra
        if needed <= self._vectors.shape[0]:
            return
        new_cap = max(needed, self._vectors.shape[0] * 2)
        grown = np.zeros((new_cap, self.dimension), dtype=np.float32)
        grown[: self._size] = self._vectors[: self._size]
        self._vectors = grown

    def add(
        self,
        ids: Sequence[str],
        vectors: np.ndarray,
        metadata: Sequence[dict[str, Any]] | None = None,
        *,
        overwrite: bool = False,
    ) -> int:
        """Add documents; returns the number of vectors inserted or updated.

        Raises:
            ValueError: on dimension mismatch, duplicate ids (unless ``overwrite``) or
                length mismatches between ``ids``, ``vectors`` and ``metadata``.
        """
        vecs = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if vecs.shape[0] != len(ids):
            raise ValueError("ids and vectors must have the same length")
        if vecs.shape[1] != self.dimension:
            raise ValueError(f"Expected vectors of dimension {self.dimension}, got {vecs.shape[1]}")
        meta = list(metadata) if metadata is not None else [{} for _ in ids]
        if len(meta) != len(ids):
            raise ValueError("metadata length must match ids")
        if len(set(ids)) != len(ids):
            raise ValueError("ids must be unique within a single add() call")

        new_rows: list[int] = []
        for i, doc_id in enumerate(ids):
            if doc_id in self._id_to_row:
                if not overwrite:
                    raise ValueError(f"Document id already indexed: {doc_id!r}")
                row = self._id_to_row[doc_id]
                self._vectors[row] = vecs[i]
                self._metadata[row] = dict(meta[i])
            else:
                new_rows.append(i)
        if new_rows:
            self._ensure_capacity(len(new_rows))
            for i in new_rows:
                row = self._size
                self._vectors[row] = vecs[i]
                self._ids.append(str(ids[i]))
                self._id_to_row[str(ids[i])] = row
                self._metadata.append(dict(meta[i]))
                self._size += 1
        return len(ids)

    def remove(self, doc_id: str) -> bool:
        """Remove a document by id (O(n) compaction). Returns ``True`` if it existed."""
        row = self._id_to_row.pop(doc_id, None)
        if row is None:
            return False
        last = self._size - 1
        if row != last:
            self._vectors[row] = self._vectors[last]
            moved_id = self._ids[last]
            self._ids[row] = moved_id
            self._metadata[row] = self._metadata[last]
            self._id_to_row[moved_id] = row
        self._ids.pop()
        self._metadata.pop()
        self._vectors[last] = 0
        self._size -= 1
        return True

    # -- querying ----------------------------------------------------------------------
    def search(
        self, query_vector: np.ndarray, top_k: int = 10, *, min_score: float = -1.0
    ) -> list[SearchHit]:
        """Return the ``top_k`` most similar documents by cosine similarity."""
        if self._size == 0 or top_k <= 0:
            return []
        q = l2_normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        if q.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {q.shape[0]} != index dimension {self.dimension}")
        scores = self._vectors[: self._size] @ q
        k = min(top_k, self._size)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        hits: list[SearchHit] = []
        for rank, row in enumerate(order, start=1):
            s = float(scores[row])
            if s < min_score:
                continue
            hits.append(SearchHit(self._ids[row], s, rank, self._metadata[row]))
        return hits

    def search_batch(self, query_vectors: np.ndarray, top_k: int = 10) -> list[list[SearchHit]]:
        """Vectorised multi-query search."""
        return [self.search(q, top_k) for q in np.asarray(query_vectors, dtype=np.float32)]

    # -- persistence -------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Save vectors (``.npy``) and ids/metadata (``.json``) to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self._vectors[: self._size])
        with (d / "index_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {"dimension": self.dimension, "ids": self._ids, "metadata": self._metadata},
                fh,
                ensure_ascii=False,
                default=str,
            )
        logger.info("Saved vector index with %d documents to %s", self._size, d)
        return d

    @classmethod
    def load(cls, directory: str | Path) -> VectorIndex:
        """Load an index saved by :meth:`save`."""
        d = Path(directory)
        meta_path = d / "index_meta.json"
        vec_path = d / "vectors.npy"
        if not meta_path.exists() or not vec_path.exists():
            raise FileNotFoundError(f"No vector index found in {d}")
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        vectors = np.load(vec_path)
        index = cls(int(meta["dimension"]), initial_capacity=max(1, len(meta["ids"])))
        if len(meta["ids"]):
            index.add(meta["ids"], vectors, meta["metadata"])
        return index


__all__ = ["SearchHit", "VectorIndex"]
