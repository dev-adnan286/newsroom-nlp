from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newsroom.embeddings.tfidf_svd import TfidfSvdProvider
from newsroom.topics.clustering import ClusterResult, TopicClusterer, select_k_by_silhouette
from newsroom.topics.lda import LdaTopicModel


@pytest.fixture(scope="module")
def embeddings(corpus_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    texts = corpus_df["text"].tolist()
    provider = TfidfSvdProvider(dimension=24).fit(texts)
    return provider.embed_texts(texts), texts


def test_select_k_by_silhouette_on_separable_blobs() -> None:
    rng = np.random.RandomState(0)
    centers = np.array([[0, 0], [10, 10], [-10, 10]])
    x = np.vstack([c + rng.randn(30, 2) for c in centers])
    best, scores = select_k_by_silhouette(x, [2, 3, 4, 5])
    assert best == 3 and set(scores) == {2, 3, 4, 5}
    with pytest.raises(ValueError):
        select_k_by_silhouette(x, [200])


def test_clusterer_with_fixed_k(embeddings: tuple[np.ndarray, list[str]]) -> None:
    x, texts = embeddings
    result = TopicClusterer(k=6, n_top_terms=5).fit(x, texts)
    assert isinstance(result, ClusterResult)
    assert result.k == 6 and len(result.labels) == len(texts)
    assert sum(result.sizes.values()) == len(texts)
    assert all(len(v) <= 5 for v in result.top_terms.values())
    summary = result.summary()
    assert len(summary) == 6 and {"cluster", "size", "top_terms"} <= set(summary[0])


def test_clusterer_auto_k_and_predict(embeddings: tuple[np.ndarray, list[str]]) -> None:
    x, texts = embeddings
    clusterer = TopicClusterer(k_range=(3, 4, 6), minibatch=True)
    result = clusterer.fit(x)
    assert result.k in {3, 4, 6} and result.k_scores and result.top_terms == {}
    preds = clusterer.predict(x[:5])
    assert preds.shape == (5,)


def test_clusterer_validation() -> None:
    with pytest.raises(RuntimeError):
        TopicClusterer(k=2).predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        TopicClusterer(k=2).fit(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        TopicClusterer(k=10).fit(np.random.rand(5, 3))
    with pytest.raises(ValueError):
        TopicClusterer(k=2).top_terms(np.array([0, 1]), ["only one"])


def test_lda_topic_model(corpus_df: pd.DataFrame) -> None:
    texts = corpus_df["text"].tolist()[:90]
    model = LdaTopicModel(n_topics=3, max_iter=5, min_df=1)
    with pytest.raises(RuntimeError):
        model.transform(texts[:1])
    result = model.fit(texts)
    assert result.n_topics == 3 and result.doc_topics.shape == (90, 3)
    assert np.allclose(result.doc_topics.sum(axis=1), 1.0, atol=1e-3)
    assert result.perplexity > 0
    assert len(result.summary(4)) == 3 and len(result.summary(4)[0]["terms"]) == 4  # type: ignore[arg-type]
    assert result.dominant_topics().shape == (90,)
    dist = model.transform(texts[:2])
    assert dist.shape == (2, 3)
    with pytest.raises(ValueError):
        LdaTopicModel(n_topics=1)
    with pytest.raises(ValueError):
        LdaTopicModel(n_topics=5).fit(texts[:3])
