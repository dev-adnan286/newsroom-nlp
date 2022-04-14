from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.pipeline import FeatureUnion

from newsroom.features.vectorizers import (
    VectorizerConfig,
    build_hashing_vectorizer,
    build_lsa_pipeline,
    build_tfidf_vectorizer,
    build_word_char_union,
)


def test_config_from_dict_coerces_ranges() -> None:
    cfg = VectorizerConfig.from_dict({"word_ngram_range": [1, 3], "min_df": 1, "unknown": 5})
    assert cfg.word_ngram_range == (1, 3)
    assert cfg.min_df == 1
    assert VectorizerConfig.from_dict(None) == VectorizerConfig()


def test_word_char_union(small_texts: list[str]) -> None:
    union = build_word_char_union(VectorizerConfig(min_df=1))
    assert isinstance(union, FeatureUnion)
    matrix = union.fit_transform(small_texts)
    assert sparse.issparse(matrix)
    assert matrix.shape[0] == len(small_texts)
    word_dim = union.transformer_list[0][1].transform(small_texts).shape[1]
    assert matrix.shape[1] > word_dim


def test_word_only_when_char_disabled(small_texts: list[str]) -> None:
    vec = build_word_char_union(VectorizerConfig(min_df=1, use_char_ngrams=False))
    assert not isinstance(vec, FeatureUnion)
    matrix = vec.fit_transform(small_texts)
    assert matrix.dtype == np.float32


def test_tfidf_vectorizer_handles_placeholder_tokens() -> None:
    vec = build_tfidf_vectorizer(VectorizerConfig(min_df=1))
    vec.fit(["visit <url> now", "<url> again", "no link here"])
    assert "<url>" in vec.vocabulary_


def test_hashing_vectorizer_is_stateless(small_texts: list[str]) -> None:
    hv = build_hashing_vectorizer(n_features=2**12)
    matrix = hv.fit_transform(small_texts)
    assert matrix.shape == (len(small_texts), 2**12)
    raw = build_hashing_vectorizer(n_features=2**10, tfidf=False)
    assert raw.transform(small_texts).shape[1] == 2**10


def test_lsa_pipeline_outputs_unit_rows(small_texts: list[str]) -> None:
    lsa = build_lsa_pipeline(n_components=8, config=VectorizerConfig(min_df=1))
    dense = lsa.fit_transform(small_texts)
    assert dense.shape == (len(small_texts), 8)
    norms = np.linalg.norm(dense, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
