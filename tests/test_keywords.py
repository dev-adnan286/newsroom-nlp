from __future__ import annotations

import numpy as np
import pytest

from newsroom.keywords.textrank import (
    Keyword,
    TextRankExtractor,
    extract_keywords,
    extract_keywords_batch,
    pagerank,
)
from newsroom.keywords.tfidf_keywords import TfidfKeywordExtractor


def test_pagerank_uniform_on_symmetric_ring() -> None:
    n = 5
    adj = np.zeros((n, n))
    for i in range(n):
        adj[i, (i + 1) % n] = adj[(i + 1) % n, i] = 1.0
    ranks = pagerank(adj)
    assert ranks.shape == (n,)
    assert np.allclose(ranks, 1 / n, atol=1e-6)
    assert pagerank(np.zeros((0, 0))).size == 0


def test_pagerank_hub_scores_highest_and_handles_dangling() -> None:
    adj = np.zeros((4, 4))
    for j in range(1, 4):
        adj[0, j] = adj[j, 0] = 1.0
    adj[3, :] = 0  # node 3 dangles (no outgoing weight)
    ranks = pagerank(adj)
    assert ranks.argmax() == 0
    assert ranks.sum() == pytest.approx(1.0, abs=1e-6)


def test_textrank_extracts_relevant_keywords(sample_article: str) -> None:
    keywords = TextRankExtractor().extract(sample_article, top_k=5)
    assert 0 < len(keywords) <= 5
    assert all(isinstance(k, Keyword) and k.score > 0 for k in keywords)
    joined = " ".join(k.text for k in keywords)
    assert any(
        term in joined
        for term in (
            "interest rates",
            "central bank",
            "inflation",
            "rates",
            "bank",
            "market",
            "conference",
        )
    )
    assert not any(w in {"the", "and", "by", "on"} for k in keywords for w in k.text.split())
    # Stemmed duplicates (market/markets) collapse into one keyword.
    assert not ({"market", "markets"} <= {k.text for k in keywords})
    scores = [k.score for k in keywords]
    assert scores == sorted(scores, reverse=True)


def test_textrank_single_words_mode(sample_article: str) -> None:
    words = TextRankExtractor(stem=False).extract(sample_article, top_k=4, phrases=False)
    assert all(" " not in k.text for k in words)
    assert all(k.text not in {"the", "and"} for k in words)


def test_textrank_edge_cases() -> None:
    assert TextRankExtractor().extract("") == []
    assert TextRankExtractor().extract("the and of") == []
    single = TextRankExtractor().extract("Inflation", top_k=3)
    assert len(single) == 1 and single[0].text == "inflation"
    with pytest.raises(ValueError):
        TextRankExtractor(window=1)


def test_functional_helpers(sample_article: str) -> None:
    assert extract_keywords(sample_article, top_k=2)
    batch = extract_keywords_batch([sample_article, "Striker scores twice."], top_k=2)
    assert len(batch) == 2 and batch[1]


def test_tfidf_keyword_extractor(small_texts: list[str]) -> None:
    extractor = TfidfKeywordExtractor(min_df=1)
    with pytest.raises(RuntimeError):
        extractor.extract("x")
    extractor.fit(small_texts)
    assert extractor.is_fitted
    kws = extractor.extract(
        "The striker scored a penalty as fans celebrated the derby win", top_k=4
    )
    assert 0 < len(kws) <= 4
    assert kws == sorted(kws, key=lambda k: -k.score)
    assert extractor.extract("") == []
    assert len(extractor.extract_batch(small_texts[:2], top_k=3)) == 2
    with pytest.raises(ValueError):
        TfidfKeywordExtractor().fit([])
