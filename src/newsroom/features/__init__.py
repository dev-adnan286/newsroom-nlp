"""Feature extraction: TF-IDF unions, hashing and LSA."""

from newsroom.features.vectorizers import (
    VectorizerConfig,
    build_hashing_vectorizer,
    build_lsa_pipeline,
    build_tfidf_vectorizer,
    build_word_char_union,
)

__all__ = [
    "VectorizerConfig",
    "build_hashing_vectorizer",
    "build_lsa_pipeline",
    "build_tfidf_vectorizer",
    "build_word_char_union",
]
