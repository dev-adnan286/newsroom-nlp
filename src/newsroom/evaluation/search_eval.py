"""Retrieval evaluation: Precision@k, Recall@k, MRR and nDCG on generated relevance sets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from newsroom.keywords.textrank import TextRankExtractor

if TYPE_CHECKING:
    from newsroom.search.query import SearchEngine


@dataclass
class RelevanceSet:
    """Queries with graded relevance judgements (``doc_id -> gain``)."""

    queries: dict[str, str]
    relevance: dict[str, dict[str, float]]

    def __len__(self) -> int:
        return len(self.queries)

    def relevant_ids(self, query_id: str) -> set[str]:
        return {d for d, g in self.relevance.get(query_id, {}).items() if g > 0}


@dataclass
class RetrievalMetrics:
    """Averaged retrieval metrics over a query set."""

    k: int
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    n_queries: int
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "n_queries": self.n_queries,
        }


# --------------------------------------------------------------------------------------
# Single-query metrics
# --------------------------------------------------------------------------------------
def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-``k`` results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(ranked[:k])
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / k


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-``k``."""
    if not relevant:
        return 0.0
    return sum(1 for d in ranked[:k] if d in relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant result, or 0."""
    for i, d in enumerate(ranked, start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rankings: Sequence[Sequence[str]], relevants: Sequence[set[str]]) -> float:
    """Average reciprocal rank across queries."""
    if len(rankings) != len(relevants):
        raise ValueError("rankings and relevants must be the same length")
    if not rankings:
        return 0.0
    return float(
        np.mean([reciprocal_rank(r, rel) for r, rel in zip(rankings, relevants, strict=True)])
    )


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with ``log2(i + 1)`` discounts."""
    g = np.asarray(gains[:k], dtype=np.float64)
    if g.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, g.size + 2))
    return float(np.sum(g / discounts))


def ndcg_at_k(ranked: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    """Normalised DCG using graded relevance."""
    gains = [float(relevance.get(d, 0.0)) for d in ranked[:k]]
    ideal = sorted((float(g) for g in relevance.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


# --------------------------------------------------------------------------------------
# Batch evaluation
# --------------------------------------------------------------------------------------
def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]], relevance_set: RelevanceSet, k: int = 10
) -> RetrievalMetrics:
    """Average metrics over every query in ``relevance_set`` present in ``rankings``."""
    per_query: dict[str, dict[str, float]] = {}
    for qid in relevance_set.queries:
        ranked = list(rankings.get(qid, []))
        rel_ids = relevance_set.relevant_ids(qid)
        per_query[qid] = {
            "precision_at_k": precision_at_k(ranked, rel_ids, k),
            "recall_at_k": recall_at_k(ranked, rel_ids, k),
            "reciprocal_rank": reciprocal_rank(ranked, rel_ids),
            "ndcg_at_k": ndcg_at_k(ranked, relevance_set.relevance.get(qid, {}), k),
        }
    n = len(per_query)
    if n == 0:
        raise ValueError("relevance_set contains no queries")

    def _mean(key: str) -> float:
        return float(np.mean([v[key] for v in per_query.values()]))

    return RetrievalMetrics(
        k=k,
        precision_at_k=_mean("precision_at_k"),
        recall_at_k=_mean("recall_at_k"),
        mrr=_mean("reciprocal_rank"),
        ndcg_at_k=_mean("ndcg_at_k"),
        n_queries=n,
        per_query=per_query,
    )


def build_relevance_set(
    df: pd.DataFrame,
    *,
    n_queries: int = 30,
    seed: int = 42,
    keywords_per_query: int = 4,
    label_gain: float = 1.0,
    source_gain: float = 3.0,
) -> RelevanceSet:
    """Generate queries from documents and judge relevance by shared label.

    For each sampled document a query is built from its TextRank keywords. The source
    document receives ``source_gain``; every other document with the same label receives
    ``label_gain``. This yields a cheap but meaningful proxy for topical relevance.
    """
    required = {"doc_id", "text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame needs columns {sorted(required)}")
    rng = np.random.RandomState(seed)
    n = min(n_queries, len(df))
    idx = rng.choice(len(df), n, replace=False)
    extractor = TextRankExtractor(max_phrase_words=1)
    queries: dict[str, str] = {}
    relevance: dict[str, dict[str, float]] = {}
    by_label: dict[str, list[str]] = {}
    for doc_id, label in zip(df["doc_id"].astype(str), df["label"].astype(str), strict=True):
        by_label.setdefault(label, []).append(doc_id)
    for i in idx:
        row = df.iloc[int(i)]
        kws = extractor.extract(str(row["text"]), top_k=keywords_per_query, phrases=False)
        query_text = " ".join(k.text for k in kws) or str(row["text"])[:50]
        qid = f"q-{row['doc_id']}"
        queries[qid] = query_text
        judgements = {d: label_gain for d in by_label[str(row["label"])]}
        judgements[str(row["doc_id"])] = source_gain
        relevance[qid] = judgements
    return RelevanceSet(queries, relevance)


def evaluate_search_engine(
    engine: SearchEngine,
    relevance_set: RelevanceSet,
    *,
    k: int = 10,
    modes: Sequence[str] = ("bm25", "vector", "hybrid"),
) -> dict[str, RetrievalMetrics]:
    """Evaluate each retrieval mode of a :class:`SearchEngine` on ``relevance_set``."""
    results: dict[str, RetrievalMetrics] = {}
    for mode in modes:
        rankings = {
            qid: [r.doc_id for r in engine.search(q, k, mode=mode)]
            for qid, q in relevance_set.queries.items()
        }
        results[mode] = evaluate_rankings(rankings, relevance_set, k)
    return results


__all__ = [
    "RelevanceSet",
    "RetrievalMetrics",
    "build_relevance_set",
    "dcg_at_k",
    "evaluate_rankings",
    "evaluate_search_engine",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
