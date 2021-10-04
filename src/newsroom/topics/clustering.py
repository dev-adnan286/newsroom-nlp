"""K-means topic clustering over document embeddings with silhouette-based ``k`` selection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import clean_text
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClusterResult:
    """Cluster assignments plus per-cluster descriptive terms."""

    labels: np.ndarray
    k: int
    silhouette: float
    top_terms: dict[int, list[str]]
    sizes: dict[int, int]
    k_scores: dict[int, float] = field(default_factory=dict)

    def summary(self) -> list[dict[str, object]]:
        return [
            {"cluster": c, "size": self.sizes[c], "top_terms": self.top_terms.get(c, [])}
            for c in sorted(self.sizes)
        ]


def select_k_by_silhouette(
    embeddings: np.ndarray,
    k_values: Sequence[int],
    *,
    random_state: int = 42,
    sample_size: int | None = 2000,
    minibatch: bool = False,
) -> tuple[int, dict[int, float]]:
    """Fit K-means for every ``k`` and return the best by silhouette score."""
    n = embeddings.shape[0]
    candidates = [k for k in k_values if 2 <= k < n]
    if not candidates:
        raise ValueError(f"No valid k in {list(k_values)} for {n} samples")
    scores: dict[int, float] = {}
    rng = np.random.RandomState(random_state)
    sample_idx = None
    if sample_size is not None and n > sample_size:
        sample_idx = rng.choice(n, sample_size, replace=False)
    for k in candidates:
        model = _make_model(k, random_state, minibatch)
        labels = model.fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            continue
        if sample_idx is not None:
            scores[k] = float(silhouette_score(embeddings[sample_idx], labels[sample_idx]))
        else:
            scores[k] = float(silhouette_score(embeddings, labels))
    best = max(scores, key=lambda k: (scores[k], -k))
    logger.info(
        "Silhouette selection picked k=%d (%s)", best, {k: round(v, 3) for k, v in scores.items()}
    )
    return best, scores


def _make_model(k: int, random_state: int, minibatch: bool) -> KMeans | MiniBatchKMeans:
    if minibatch:
        return MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=3, batch_size=1024)
    return KMeans(n_clusters=k, random_state=random_state, n_init=5)


class TopicClusterer:
    """Cluster embeddings and label each cluster with its most characteristic terms."""

    def __init__(
        self,
        k: int | None = None,
        *,
        k_range: Sequence[int] = (3, 4, 5, 6, 8, 10),
        random_state: int = 42,
        minibatch: bool = False,
        n_top_terms: int = 10,
    ) -> None:
        self.k = k
        self.k_range = list(k_range)
        self.random_state = random_state
        self.minibatch = minibatch
        self.n_top_terms = n_top_terms
        self.model: KMeans | MiniBatchKMeans | None = None

    def fit(self, embeddings: np.ndarray, texts: Sequence[str] | None = None) -> ClusterResult:
        """Cluster ``embeddings``; if ``texts`` is given compute top terms per cluster."""
        x = np.asarray(embeddings, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] < 3:
            raise ValueError("embeddings must be a 2-D array with at least 3 rows")
        k_scores: dict[int, float] = {}
        k = self.k
        if k is None:
            k, k_scores = select_k_by_silhouette(
                x, self.k_range, random_state=self.random_state, minibatch=self.minibatch
            )
        if k >= x.shape[0]:
            raise ValueError(f"k={k} must be smaller than the number of samples ({x.shape[0]})")
        self.model = _make_model(k, self.random_state, self.minibatch)
        labels = self.model.fit_predict(x)
        sil = float(silhouette_score(x, labels)) if len(np.unique(labels)) > 1 else -1.0
        sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}
        top_terms = self.top_terms(labels, texts) if texts is not None else {}
        return ClusterResult(
            labels=labels,
            k=int(k),
            silhouette=sil,
            top_terms=top_terms,
            sizes=sizes,
            k_scores=k_scores,
        )

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("TopicClusterer is not fitted")
        return self.model.predict(np.asarray(embeddings, dtype=np.float32))

    def top_terms(self, labels: np.ndarray, texts: Sequence[str]) -> dict[int, list[str]]:
        """Terms with the highest mean TF-IDF inside each cluster relative to the corpus."""
        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        vectorizer = TfidfVectorizer(
            stop_words=list(ENGLISH_STOPWORDS),
            preprocessor=lambda t: clean_text(t, numbers=True),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z'-]+\b",
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform([str(t) for t in texts])
        names = vectorizer.get_feature_names_out()
        global_mean = np.asarray(matrix.mean(axis=0)).ravel()
        out: dict[int, list[str]] = {}
        for c in np.unique(labels):
            rows = np.where(labels == c)[0]
            cluster_mean = np.asarray(matrix[rows].mean(axis=0)).ravel()
            distinct = cluster_mean - 0.5 * global_mean
            idx = np.argsort(-distinct)[: self.n_top_terms]
            out[int(c)] = [str(names[i]) for i in idx if distinct[i] > 0]
        return out


__all__ = ["ClusterResult", "TopicClusterer", "select_k_by_silhouette"]
