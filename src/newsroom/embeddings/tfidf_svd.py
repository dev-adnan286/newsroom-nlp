"""Offline embedding provider: TF-IDF followed by TruncatedSVD (LSA)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np

from newsroom.embeddings.base import l2_normalize
from newsroom.features.vectorizers import VectorizerConfig, build_lsa_pipeline
from newsroom.preprocessing.text import TextCleaner
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)


class TfidfSvdProvider:
    """Dense document embeddings from an LSA pipeline.

    The provider must be fitted on a reference corpus before ``embed_texts`` is called.
    The number of SVD components is automatically capped by the vocabulary size so small
    corpora (as in tests) still work.
    """

    def __init__(
        self,
        dimension: int = 256,
        *,
        vectorizer_config: VectorizerConfig | None = None,
        random_state: int = 42,
        clean: bool = True,
    ) -> None:
        if dimension < 2:
            raise ValueError("dimension must be >= 2")
        self._requested_dim = dimension
        self._config = vectorizer_config or VectorizerConfig(min_df=1, use_char_ngrams=False)
        self._random_state = random_state
        self._clean = clean
        self._cleaner = TextCleaner(stem=True, remove_stopwords=True)
        self._pipeline = build_lsa_pipeline(dimension, self._config, random_state=random_state)
        self._fitted = False
        self._dimension = dimension

    # -- protocol ----------------------------------------------------------------------
    @property
    def name(self) -> str:
        return f"tfidf_svd_{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _prepare(self, texts: Sequence[str]) -> list[str]:
        docs = [str(t) if t is not None else "" for t in texts]
        return self._cleaner.transform(docs) if self._clean else docs

    def fit(self, texts: Sequence[str]) -> TfidfSvdProvider:
        """Fit TF-IDF vocabulary and the SVD projection on ``texts``."""
        if len(texts) < 2:
            raise ValueError("At least two documents are required to fit the LSA provider")
        docs = self._prepare(texts)
        tfidf = self._pipeline.named_steps["tfidf"]
        matrix = tfidf.fit_transform(docs)
        n_features = matrix.shape[1]
        max_components = max(1, min(n_features - 1, len(docs) - 1))
        n_components = min(self._requested_dim, max_components)
        if n_components != self._requested_dim:
            logger.warning(
                "Capping SVD components from %d to %d (features=%d, docs=%d)",
                self._requested_dim,
                n_components,
                n_features,
                len(docs),
            )
        svd = self._pipeline.named_steps["svd"]
        svd.set_params(n_components=n_components)
        reduced = svd.fit_transform(matrix)
        self._pipeline.named_steps["norm"].fit(reduced)
        self._dimension = n_components
        self._fitted = True
        logger.info("Fitted TF-IDF/SVD provider: %d features -> %d dims", n_features, n_components)
        return self

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Embed texts into L2-normalised vectors of shape ``(n, dimension)``."""
        if not self._fitted:
            raise RuntimeError("TfidfSvdProvider must be fitted before embedding texts")
        if len(texts) == 0:
            return np.zeros((0, self._dimension), dtype=np.float32)
        tfidf = self._pipeline.named_steps["tfidf"]
        svd = self._pipeline.named_steps["svd"]
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = self._prepare(texts[start : start + batch_size])
            reduced = svd.transform(tfidf.transform(chunk))
            outputs.append(np.asarray(reduced, dtype=np.float32))
        return l2_normalize(np.vstack(outputs))

    # -- persistence --------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        """Serialise the fitted provider with joblib."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> TfidfSvdProvider:
        """Load a provider previously saved with :meth:`save`."""
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} does not contain a {cls.__name__}")
        return obj


__all__ = ["TfidfSvdProvider"]
