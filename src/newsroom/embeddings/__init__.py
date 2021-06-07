"""Embedding providers behind a common protocol, plus a disk cache."""

from newsroom.embeddings.base import EmbeddingProvider, get_provider
from newsroom.embeddings.cache import EmbeddingCache
from newsroom.embeddings.sentence_transformers_provider import SentenceTransformerProvider
from newsroom.embeddings.tfidf_svd import TfidfSvdProvider

__all__ = [
    "EmbeddingCache",
    "EmbeddingProvider",
    "SentenceTransformerProvider",
    "TfidfSvdProvider",
    "get_provider",
]
