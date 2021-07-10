"""Factories for scikit-learn text vectorisers used across the toolkit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer


@dataclass
class VectorizerConfig:
    """Hyper-parameters for the word/char TF-IDF feature union."""

    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    min_df: int | float = 2
    max_df: float = 0.95
    max_features: int | None = 60000
    sublinear_tf: bool = True
    use_char_ngrams: bool = True
    char_weight: float = 0.5

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VectorizerConfig:
        """Build a config from a mapping (e.g. parsed YAML); unknown keys are ignored."""
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        for key in ("word_ngram_range", "char_ngram_range"):
            if key in known and isinstance(known[key], list | tuple):
                known[key] = (int(known[key][0]), int(known[key][1]))
        return cls(**known)


def build_tfidf_vectorizer(config: VectorizerConfig | None = None) -> TfidfVectorizer:
    """Word-level TF-IDF vectoriser."""
    cfg = config or VectorizerConfig()
    return TfidfVectorizer(
        ngram_range=cfg.word_ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        sublinear_tf=cfg.sublinear_tf,
        token_pattern=r"(?u)(?:<\w+>|\b\w[\w']*\b)",
        dtype=np.float32,
    )


def build_char_vectorizer(config: VectorizerConfig | None = None) -> TfidfVectorizer:
    """Character n-gram TF-IDF vectoriser (``char_wb`` analyser)."""
    cfg = config or VectorizerConfig()
    max_feats = cfg.max_features if cfg.max_features is None else max(1000, cfg.max_features // 2)
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=max_feats,
        sublinear_tf=cfg.sublinear_tf,
        dtype=np.float32,
    )


def build_word_char_union(config: VectorizerConfig | None = None) -> FeatureUnion | TfidfVectorizer:
    """Return a ``FeatureUnion`` of word + char TF-IDF, or just the word vectoriser."""
    cfg = config or VectorizerConfig()
    if not cfg.use_char_ngrams:
        return build_tfidf_vectorizer(cfg)
    return FeatureUnion(
        [("word", build_tfidf_vectorizer(cfg)), ("char", build_char_vectorizer(cfg))],
        transformer_weights={"word": 1.0, "char": cfg.char_weight},
    )


def build_hashing_vectorizer(
    n_features: int = 2**18,
    ngram_range: tuple[int, int] = (1, 2),
    *,
    tfidf: bool = True,
) -> Pipeline | HashingVectorizer:
    """Stateless hashing vectoriser, optionally followed by a TF-IDF re-weighting."""
    hasher = HashingVectorizer(
        n_features=n_features,
        ngram_range=ngram_range,
        alternate_sign=False,
        norm=None if tfidf else "l2",
    )
    if not tfidf:
        return hasher
    return Pipeline([("hash", hasher), ("tfidf", TfidfTransformer(sublinear_tf=True))])


def build_lsa_pipeline(
    n_components: int = 256,
    config: VectorizerConfig | None = None,
    *,
    random_state: int = 42,
    n_iter: int = 7,
) -> Pipeline:
    """TF-IDF -> TruncatedSVD -> L2 normalisation (classic Latent Semantic Analysis)."""
    return Pipeline(
        [
            ("tfidf", build_word_char_union(config)),
            (
                "svd",
                TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=n_iter),
            ),
            ("norm", Normalizer(copy=False)),
        ]
    )


__all__ = [
    "VectorizerConfig",
    "build_char_vectorizer",
    "build_hashing_vectorizer",
    "build_lsa_pipeline",
    "build_tfidf_vectorizer",
    "build_word_char_union",
]
