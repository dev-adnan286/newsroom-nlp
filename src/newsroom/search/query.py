"""High-level search engine tying together documents, embeddings, BM25 and the vector index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from newsroom.embeddings.base import EmbeddingProvider, get_provider
from newsroom.embeddings.tfidf_svd import TfidfSvdProvider
from newsroom.search.bm25 import BM25
from newsroom.search.hybrid import HybridConfig, HybridSearcher
from newsroom.search.index import VectorIndex
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)

_PROVIDER_FILE = "provider.joblib"
_DOCS_FILE = "documents.json"
_ENGINE_META = "engine_meta.json"


@dataclass
class SearchResult:
    """One search hit enriched with document text and metadata."""

    doc_id: str
    score: float
    rank: int
    title: str
    text: str
    label: str | None = None
    component_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def snippet(self, max_chars: int = 200) -> str:
        return (
            self.text if len(self.text) <= max_chars else self.text[: max_chars - 1].rstrip() + "…"
        )


class SearchEngine:
    """Owns the document store plus BM25/vector indexes and exposes ``search``/``add``."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        config: HybridConfig | None = None,
        *,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        self.provider = provider
        self.config = config or HybridConfig()
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.documents: dict[str, dict[str, Any]] = {}
        self.bm25: BM25 | None = None
        self.index: VectorIndex | None = None
        self._hybrid: HybridSearcher | None = None

    # -- building ----------------------------------------------------------------------
    @staticmethod
    def _rows(df: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, rec in enumerate(df.to_dict(orient="records")):
            doc_id = str(rec.get("doc_id") or f"doc-{i:05d}")
            text = str(rec.get("text") or "")
            title = str(rec.get("title") or text[:60])
            label = rec.get("label")
            published = rec.get("published_at")
            rows.append(
                {
                    "doc_id": doc_id,
                    "text": text,
                    "title": title,
                    "label": (
                        None
                        if label is None or (isinstance(label, float) and pd.isna(label))
                        else str(label)
                    ),
                    "published_at": (
                        None if published is None or pd.isna(published) else str(published)
                    ),
                }
            )
        return rows

    def build(self, df: pd.DataFrame) -> SearchEngine:
        """Fit the provider (if needed) and index every row of ``df``."""
        rows = self._rows(df)
        if not rows:
            raise ValueError("Cannot build a search engine from an empty DataFrame")
        texts = [r["text"] for r in rows]
        ids = [r["doc_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("doc_id values must be unique")
        if not self.provider.is_fitted:
            self.provider.fit(texts)
        vectors = self.provider.embed_texts(texts)
        self.index = VectorIndex(self.provider.dimension, initial_capacity=len(ids))
        self.index.add(ids, vectors, [{"title": r["title"], "label": r["label"]} for r in rows])
        self.bm25 = BM25(k1=self.bm25_k1, b=self.bm25_b).fit(ids, texts)
        self.documents = {r["doc_id"]: r for r in rows}
        self._hybrid = HybridSearcher(self.bm25, self.index, self.provider, self.config)
        logger.info(
            "Built search engine over %d documents (dim=%d)", len(ids), self.provider.dimension
        )
        return self

    def add_documents(self, df: pd.DataFrame) -> int:
        """Incrementally add documents. BM25 is refitted over the full corpus (cheap)."""
        if self.index is None or self.bm25 is None:
            return len(self.build(df).documents)
        rows = self._rows(df)
        if not rows:
            return 0
        new_rows = [r for r in rows if r["doc_id"] not in self.documents]
        if not new_rows:
            return 0
        vectors = self.provider.embed_texts([r["text"] for r in new_rows])
        self.index.add(
            [r["doc_id"] for r in new_rows],
            vectors,
            [{"title": r["title"], "label": r["label"]} for r in new_rows],
        )
        for r in new_rows:
            self.documents[r["doc_id"]] = r
        all_ids = list(self.documents.keys())
        self.bm25 = BM25(k1=self.bm25_k1, b=self.bm25_b).fit(
            all_ids, [self.documents[i]["text"] for i in all_ids]
        )
        self._hybrid = HybridSearcher(self.bm25, self.index, self.provider, self.config)
        return len(new_rows)

    # -- querying ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.documents)

    def search(
        self, query: str, top_k: int = 10, *, mode: str = "hybrid", **weights: float
    ) -> list[SearchResult]:
        """Run a query and return enriched results."""
        if self._hybrid is None:
            raise RuntimeError("SearchEngine has no index; call build() or load() first")
        results: list[SearchResult] = []
        for rank, (doc_id, score, components) in enumerate(
            self._hybrid.search(query, top_k, mode=mode, **weights), start=1
        ):
            doc = self.documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    rank=rank,
                    title=doc["title"],
                    text=doc["text"],
                    label=doc.get("label"),
                    component_scores=components,
                    metadata={"published_at": doc.get("published_at")},
                )
            )
        return results

    # -- persistence -------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist provider, indexes and documents into ``directory``."""
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Nothing to save; build the engine first")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if isinstance(self.provider, TfidfSvdProvider):
            self.provider.save(d / _PROVIDER_FILE)
            provider_kind = "tfidf_svd"
        else:
            provider_kind = self.provider.name
        self.index.save(d)
        self.bm25.save(d)
        with (d / _DOCS_FILE).open("w", encoding="utf-8") as fh:
            json.dump(list(self.documents.values()), fh, ensure_ascii=False)
        with (d / _ENGINE_META).open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "provider": provider_kind,
                    "dimension": self.provider.dimension,
                    "n_documents": len(self.documents),
                    "hybrid": self.config.__dict__,
                    "bm25": {"k1": self.bm25_k1, "b": self.bm25_b},
                },
                fh,
            )
        logger.info("Saved search engine to %s", d)
        return d

    @classmethod
    def load(
        cls, directory: str | Path, *, provider: EmbeddingProvider | None = None
    ) -> SearchEngine:
        """Load an engine saved with :meth:`save`."""
        d = Path(directory)
        meta_path = d / _ENGINE_META
        if not meta_path.exists():
            raise FileNotFoundError(f"No search engine found in {d}")
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        if provider is None:
            if meta["provider"] == "tfidf_svd":
                provider = TfidfSvdProvider.load(d / _PROVIDER_FILE)
            else:
                name, _, model = str(meta["provider"]).partition(":")
                provider = get_provider(name, model_name=model) if model else get_provider(name)
        bm25_params = meta.get("bm25", {})
        engine = cls(
            provider,
            HybridConfig.from_dict(meta.get("hybrid")),
            bm25_k1=float(bm25_params.get("k1", 1.5)),
            bm25_b=float(bm25_params.get("b", 0.75)),
        )
        engine.index = VectorIndex.load(d)
        engine.bm25 = BM25.load(d)
        with (d / _DOCS_FILE).open("r", encoding="utf-8") as fh:
            docs = json.load(fh)
        engine.documents = {doc["doc_id"]: doc for doc in docs}
        engine._hybrid = HybridSearcher(engine.bm25, engine.index, provider, engine.config)
        logger.info("Loaded search engine with %d documents from %s", len(engine.documents), d)
        return engine


def build_search_engine(
    df: pd.DataFrame,
    *,
    provider_name: str = "tfidf_svd",
    dimension: int = 256,
    config: HybridConfig | None = None,
    provider_kwargs: dict[str, Any] | None = None,
) -> SearchEngine:
    """One-liner: choose a provider by name and index ``df``."""
    kwargs = dict(provider_kwargs or {})
    if provider_name == "tfidf_svd":
        kwargs.setdefault("dimension", dimension)
    provider = get_provider(provider_name, **kwargs)
    return SearchEngine(provider, config).build(df)


def ids_from_results(results: Sequence[SearchResult]) -> list[str]:
    """Extract ordered document ids from results."""
    return [r.doc_id for r in results]


__all__ = ["SearchEngine", "SearchResult", "build_search_engine", "ids_from_results"]
