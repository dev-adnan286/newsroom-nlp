"""Extractive summarisation by TF-IDF centroid similarity or TextRank over sentences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from newsroom.keywords.textrank import pagerank
from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import clean_text, sentence_split


@dataclass
class Summary:
    """Selected sentences (in original order) with their scores."""

    sentences: list[str]
    scores: list[float]
    indices: list[int]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


class ExtractiveSummarizer:
    """Pick the most central sentences of a document.

    Methods:
        * ``centroid`` - cosine similarity between each sentence and the document's TF-IDF
          centroid, with a position prior favouring early sentences (news lead bias).
        * ``textrank`` - PageRank over the sentence similarity graph.
    """

    def __init__(
        self,
        method: str = "centroid",
        *,
        position_bias: float = 0.15,
        redundancy_penalty: float = 0.6,
    ) -> None:
        if method not in {"centroid", "textrank"}:
            raise ValueError("method must be 'centroid' or 'textrank'")
        self.method = method
        self.position_bias = position_bias
        self.redundancy_penalty = redundancy_penalty

    def _sentence_matrix(self, sentences: list[str]) -> np.ndarray:
        vectorizer = TfidfVectorizer(
            stop_words=list(ENGLISH_STOPWORDS),
            preprocessor=lambda t: clean_text(t, numbers=True),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
            sublinear_tf=True,
        )
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:  # empty vocabulary (e.g. all stopwords)
            return np.zeros((len(sentences), 1))
        dense = matrix.toarray().astype(np.float64)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return dense / np.where(norms == 0, 1.0, norms)

    def _scores(self, sentences: list[str]) -> np.ndarray:
        x = self._sentence_matrix(sentences)
        n = len(sentences)
        if self.method == "centroid":
            centroid = x.mean(axis=0)
            c_norm = np.linalg.norm(centroid)
            centroid = centroid / c_norm if c_norm > 0 else centroid
            base = x @ centroid
        else:
            sims = x @ x.T
            np.fill_diagonal(sims, 0.0)
            sims[sims < 0.05] = 0.0
            base = pagerank(sims)
        position = 1.0 + self.position_bias * (1.0 - np.arange(n) / max(1, n - 1))
        return np.asarray(base) * position

    def summarize(self, text: str, n_sentences: int = 3, *, ratio: float | None = None) -> Summary:
        """Return the ``n_sentences`` best sentences (or a ``ratio`` of the document)."""
        sentences = sentence_split(text)
        if not sentences:
            return Summary([], [], [])
        if ratio is not None:
            if not 0.0 < ratio <= 1.0:
                raise ValueError("ratio must be in (0, 1]")
            n_sentences = max(1, int(round(len(sentences) * ratio)))
        n_sentences = max(1, min(n_sentences, len(sentences)))
        if len(sentences) == 1:
            return Summary(sentences, [1.0], [0])
        scores = self._scores(sentences)
        matrix = self._sentence_matrix(sentences)
        chosen: list[int] = []
        remaining = list(range(len(sentences)))
        adjusted = scores.copy()
        # Greedy selection with a redundancy penalty (maximal marginal relevance style).
        while remaining and len(chosen) < n_sentences:
            best = max(remaining, key=lambda i: (adjusted[i], -i))
            chosen.append(best)
            remaining.remove(best)
            for i in remaining:
                adjusted[i] -= self.redundancy_penalty * float(matrix[i] @ matrix[best]) * scores[i]
        chosen.sort()
        return Summary([sentences[i] for i in chosen], [float(scores[i]) for i in chosen], chosen)


def summarize(text: str, n_sentences: int = 3, method: str = "centroid") -> str:
    """Functional shortcut returning the summary text."""
    return ExtractiveSummarizer(method).summarize(text, n_sentences).text


__all__ = ["ExtractiveSummarizer", "Summary", "summarize"]
