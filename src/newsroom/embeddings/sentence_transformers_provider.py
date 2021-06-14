"""Optional provider backed by ``sentence-transformers`` (install the ``transformers`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from newsroom.embeddings.base import l2_normalize
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)

_INSTALL_HINT = (
    "sentence-transformers is not installed. Install the optional extra with "
    '`pip install "newsroom-nlp[transformers]"` or switch to the offline '
    "`tfidf_svd` provider (NEWSROOM_EMBEDDING_PROVIDER=tfidf_svd)."
)


class SentenceTransformerProvider:
    """Wraps a pretrained sentence-transformers model behind :class:`EmbeddingProvider`.

    The model is loaded lazily on first use so importing this module never requires torch.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._normalize = normalize
        self._model = model
        self._dimension: int | None = None
        if model is not None:
            self._dimension = int(model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return f"sentence_transformers:{self._model_name}"

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._load()
        assert self._dimension is not None
        return self._dimension

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, texts: Sequence[str]) -> SentenceTransformerProvider:
        """Pretrained models need no fitting; returns ``self``."""
        return self

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        logger.info("Loading sentence-transformers model %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device=self._device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
        """Encode texts with the underlying model."""
        model = self._load()
        if len(texts) == 0:
            return np.zeros((0, self.dimension), dtype=np.float32)
        vectors = model.encode(
            [str(t) for t in texts],
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        return l2_normalize(arr) if self._normalize else arr


__all__ = ["SentenceTransformerProvider"]
