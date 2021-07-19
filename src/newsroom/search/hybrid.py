"""Hybrid lexical + dense retrieval via reciprocal rank fusion (RRF)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from newsroom.embeddings.base import EmbeddingProvider
from newsroom.search.bm25 import BM25
from newsroom.search.index import VectorIndex


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one.

    Each document receives ``sum_i w_i / (k + rank_i)`` over the rankings in which it
    appears (rank is 1-based). Ties are broken by first appearance for determinism.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    w = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(w) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    fused: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking, weight in zip(rankings, w, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)
            if doc_id not in first_seen:
                first_seen[doc_id] = order
                order += 1
    return sorted(fused.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


@dataclass
class HybridConfig:
    """Tunable knobs for :class:`HybridSearcher`."""

    rrf_k: int = 60
    bm25_weight: float = 1.0
    vector_weight: float = 1.0
    candidate_multiplier: int = 4

    @classmethod
    def from_dict(cls, data: Mapping[str, object] | None) -> HybridConfig:
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)  # type: ignore[arg-type]


class HybridSearcher:
    """Combine a :class:`BM25` index and a :class:`VectorIndex` sharing document ids."""

    def __init__(
        self,
        bm25: BM25,
        index: VectorIndex,
        provider: EmbeddingProvider,
        config: HybridConfig | None = None,
    ) -> None:
        self.bm25 = bm25
        self.index = index
        self.provider = provider
        self.config = config or HybridConfig()

    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        mode: str = "hybrid",
        bm25_weight: float | None = None,
        vector_weight: float | None = None,
    ) -> list[tuple[str, float, dict[str, float]]]:
        """Return ``(doc_id, fused_score, {"bm25": s, "vector": s})`` triples.

        Args:
            query: Free-text query.
            top_k: Number of results.
            mode: ``"hybrid"``, ``"bm25"`` or ``"vector"``.
            bm25_weight / vector_weight: Optional per-request overrides.
        """
        if mode not in {"hybrid", "bm25", "vector"}:
            raise ValueError(f"Unknown search mode: {mode!r}")
        if top_k <= 0 or not query.strip():
            return []
        n_candidates = max(top_k, top_k * self.config.candidate_multiplier)
        bm25_hits = self.bm25.search(query, n_candidates) if mode != "vector" else []
        vector_hits: list[tuple[str, float]] = []
        if mode != "bm25":
            q_vec = self.provider.embed_texts([query])
            vector_hits = [(h.doc_id, h.score) for h in self.index.search(q_vec[0], n_candidates)]

        bm25_scores = dict(bm25_hits)
        vector_scores = dict(vector_hits)
        if mode == "bm25":
            fused = [(d, s) for d, s in bm25_hits]
        elif mode == "vector":
            fused = [(d, s) for d, s in vector_hits]
        else:
            fused = reciprocal_rank_fusion(
                [[d for d, _ in bm25_hits], [d for d, _ in vector_hits]],
                k=self.config.rrf_k,
                weights=[
                    self.config.bm25_weight if bm25_weight is None else bm25_weight,
                    self.config.vector_weight if vector_weight is None else vector_weight,
                ],
            )
        results = []
        for doc_id, score in fused[:top_k]:
            results.append(
                (
                    doc_id,
                    float(score),
                    {
                        "bm25": float(bm25_scores.get(doc_id, 0.0)),
                        "vector": float(vector_scores.get(doc_id, 0.0)),
                    },
                )
            )
        return results

    def score_matrix(
        self, queries: Sequence[str], mode: str = "hybrid", top_k: int = 10
    ) -> list[list[str]]:
        """Convenience for evaluation: ranked id lists for many queries."""
        return [[d for d, _, _ in self.search(q, top_k, mode=mode)] for q in queries]


def min_max(scores: np.ndarray) -> np.ndarray:
    """Scale a score vector to ``[0, 1]`` (constant vectors map to zeros)."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


__all__ = ["HybridConfig", "HybridSearcher", "min_max", "reciprocal_rank_fusion"]
