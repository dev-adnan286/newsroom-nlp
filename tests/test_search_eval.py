from __future__ import annotations

import pandas as pd
import pytest

from newsroom.evaluation.search_eval import (
    RelevanceSet,
    build_relevance_set,
    dcg_at_k,
    evaluate_rankings,
    evaluate_search_engine,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from newsroom.search.query import SearchEngine


def test_precision_recall_rr() -> None:
    ranked = ["a", "b", "c", "d"]
    rel = {"b", "d", "z"}
    assert precision_at_k(ranked, rel, 2) == 0.5
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert recall_at_k(ranked, rel, 4) == pytest.approx(2 / 3)
    assert recall_at_k(ranked, set(), 4) == 0.0
    assert reciprocal_rank(ranked, rel) == 0.5
    assert reciprocal_rank(ranked, {"q"}) == 0.0
    assert precision_at_k([], rel, 3) == 0.0
    with pytest.raises(ValueError):
        precision_at_k(ranked, rel, 0)


def test_mrr_and_ndcg() -> None:
    assert mean_reciprocal_rank([["a", "b"], ["b", "a"]], [{"b"}, {"b"}]) == pytest.approx(0.75)
    assert mean_reciprocal_rank([], []) == 0.0
    with pytest.raises(ValueError):
        mean_reciprocal_rank([["a"]], [])
    assert dcg_at_k([3, 2, 3], 3) == pytest.approx(3 + 2 / 1.5849625 + 3 / 2, rel=1e-4)
    assert dcg_at_k([], 3) == 0.0
    relevance = {"a": 3.0, "b": 1.0}
    assert ndcg_at_k(["a", "b"], relevance, 2) == pytest.approx(1.0)
    assert 0 < ndcg_at_k(["b", "a"], relevance, 2) < 1.0
    assert ndcg_at_k(["x"], {}, 2) == 0.0


def test_evaluate_rankings() -> None:
    rel = RelevanceSet(
        queries={"q1": "one", "q2": "two"}, relevance={"q1": {"a": 1.0}, "q2": {"b": 2.0, "c": 1.0}}
    )
    assert len(rel) == 2 and rel.relevant_ids("q2") == {"b", "c"}
    metrics = evaluate_rankings({"q1": ["a", "x"], "q2": ["c", "b"]}, rel, k=2)
    assert metrics.n_queries == 2 and metrics.k == 2
    assert metrics.mrr == pytest.approx(1.0)
    assert metrics.recall_at_k == pytest.approx(1.0)
    assert metrics.precision_at_k == pytest.approx(0.75)
    assert set(metrics.per_query) == {"q1", "q2"}
    assert metrics.to_dict()["ndcg_at_k"] > 0.5
    with pytest.raises(ValueError):
        evaluate_rankings({}, RelevanceSet({}, {}), 2)


def test_build_relevance_set(corpus_df: pd.DataFrame) -> None:
    rel = build_relevance_set(corpus_df, n_queries=5, seed=1)
    assert len(rel) == 5
    qid, query = next(iter(rel.queries.items()))
    assert query.strip()
    source = qid.removeprefix("q-")
    assert rel.relevance[qid][source] == 3.0
    assert len(rel.relevant_ids(qid)) == 30  # every doc with the same label
    with pytest.raises(ValueError):
        build_relevance_set(pd.DataFrame({"text": ["x"]}))


def test_evaluate_search_engine(search_engine: SearchEngine, corpus_df: pd.DataFrame) -> None:
    rel = build_relevance_set(corpus_df, n_queries=8, seed=3)
    results = evaluate_search_engine(search_engine, rel, k=5)
    assert set(results) == {"bm25", "vector", "hybrid"}
    for mode, m in results.items():
        assert 0.0 <= m.precision_at_k <= 1.0 and 0.0 <= m.ndcg_at_k <= 1.0, mode
    assert results["hybrid"].precision_at_k > 0.5
    assert results["bm25"].mrr > 0.5
