"""Corpus-aware keyword extraction using TF-IDF weights."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from newsroom.keywords.textrank import Keyword
from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import clean_text


class TfidfKeywordExtractor:
    """Fit on a corpus, then rank a document's terms by their TF-IDF weight.

    Unlike TextRank this captures *distinctiveness* across the corpus: terms common in
    every document are down-weighted even if frequent locally.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        *,
        min_df: int | float = 1,
        max_df: float = 0.9,
        max_features: int | None = 50000,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
            stop_words=list(ENGLISH_STOPWORDS),
            sublinear_tf=True,
            preprocessor=lambda t: clean_text(t, numbers=True),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, corpus: Sequence[str]) -> TfidfKeywordExtractor:
        if len(corpus) == 0:
            raise ValueError("corpus must not be empty")
        self._vectorizer.fit([str(t) for t in corpus])
        self._fitted = True
        return self

    def extract(self, text: str, top_k: int = 10) -> list[Keyword]:
        """Top ``top_k`` terms of ``text`` by TF-IDF weight."""
        if not self._fitted:
            raise RuntimeError("Call fit() on a corpus before extracting keywords")
        row = self._vectorizer.transform([text])
        if row.nnz == 0:
            return []
        names = self._vectorizer.get_feature_names_out()
        coo = row.tocoo()
        order = np.argsort(-coo.data)[:top_k]
        return [Keyword(str(names[coo.col[i]]), float(coo.data[i])) for i in order]

    def extract_batch(self, texts: Sequence[str], top_k: int = 10) -> list[list[Keyword]]:
        return [self.extract(t, top_k) for t in texts]


__all__ = ["TfidfKeywordExtractor"]
