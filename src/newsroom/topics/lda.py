"""Latent Dirichlet Allocation wrapper with readable topic summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import clean_text


@dataclass
class LdaResult:
    """Fitted LDA outputs."""

    n_topics: int
    topic_terms: dict[int, list[tuple[str, float]]]
    doc_topics: np.ndarray
    perplexity: float

    def dominant_topics(self) -> np.ndarray:
        return np.argmax(self.doc_topics, axis=1)

    def summary(self, n_terms: int = 8) -> list[dict[str, object]]:
        return [
            {"topic": t, "terms": [w for w, _ in terms[:n_terms]]}
            for t, terms in sorted(self.topic_terms.items())
        ]


class LdaTopicModel:
    """Bag-of-words LDA with sensible defaults for short news articles."""

    def __init__(
        self,
        n_topics: int = 6,
        *,
        n_top_terms: int = 10,
        max_iter: int = 15,
        random_state: int = 42,
        min_df: int | float = 2,
        max_df: float = 0.9,
        max_features: int | None = 20000,
    ) -> None:
        if n_topics < 2:
            raise ValueError("n_topics must be >= 2")
        self.n_topics = n_topics
        self.n_top_terms = n_top_terms
        self._vectorizer = CountVectorizer(
            stop_words=list(ENGLISH_STOPWORDS),
            preprocessor=lambda t: clean_text(t, numbers=True),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self._lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            learning_method="batch",
            random_state=random_state,
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: Sequence[str]) -> LdaResult:
        """Fit the model and return topic-term / document-topic summaries."""
        docs = [str(t) for t in texts]
        if len(docs) < self.n_topics:
            raise ValueError("Need at least as many documents as topics")
        counts = self._vectorizer.fit_transform(docs)
        if counts.shape[1] == 0:
            raise ValueError("Vocabulary is empty after preprocessing")
        doc_topics = self._lda.fit_transform(counts)
        self._fitted = True
        perplexity = float(self._lda.perplexity(counts))
        return LdaResult(self.n_topics, self.topic_terms(), doc_topics, perplexity)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("LdaTopicModel is not fitted")
        return self._lda.transform(self._vectorizer.transform([str(t) for t in texts]))

    def topic_terms(self, n_terms: int | None = None) -> dict[int, list[tuple[str, float]]]:
        """Return the top weighted terms for each topic as ``(term, probability)`` pairs."""
        if not self._fitted:
            raise RuntimeError("LdaTopicModel is not fitted")
        n = n_terms or self.n_top_terms
        names = self._vectorizer.get_feature_names_out()
        components = self._lda.components_ / self._lda.components_.sum(axis=1, keepdims=True)
        out: dict[int, list[tuple[str, float]]] = {}
        for t, row in enumerate(components):
            idx = np.argsort(-row)[:n]
            out[t] = [(str(names[i]), float(row[i])) for i in idx]
        return out


__all__ = ["LdaResult", "LdaTopicModel"]
