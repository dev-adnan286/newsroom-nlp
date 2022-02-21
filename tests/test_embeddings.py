from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from newsroom.embeddings.base import EmbeddingProvider, get_provider, l2_normalize
from newsroom.embeddings.cache import EmbeddingCache
from newsroom.embeddings.sentence_transformers_provider import SentenceTransformerProvider
from newsroom.embeddings.tfidf_svd import TfidfSvdProvider


def test_l2_normalize() -> None:
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_tfidf_svd_provider_fit_and_embed(small_texts: list[str]) -> None:
    provider = TfidfSvdProvider(dimension=16)
    assert not provider.is_fitted
    with pytest.raises(RuntimeError):
        provider.embed_texts(["x"])
    provider.fit(small_texts)
    assert provider.is_fitted
    assert isinstance(provider, EmbeddingProvider)
    vectors = provider.embed_texts(small_texts[:5], batch_size=2)
    assert vectors.shape == (5, provider.dimension)
    assert vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert provider.embed_texts([]).shape == (0, provider.dimension)


def test_tfidf_svd_caps_dimension_for_tiny_corpus() -> None:
    provider = TfidfSvdProvider(dimension=500).fit(
        ["alpha beta gamma", "beta gamma delta", "delta epsilon zeta"]
    )
    assert provider.dimension < 500
    assert provider.name.startswith("tfidf_svd_")


def test_tfidf_svd_similarity_is_meaningful(small_texts: list[str]) -> None:
    provider = TfidfSvdProvider(dimension=16).fit(small_texts)
    a, b, c = provider.embed_texts(
        [
            "The striker scored a late penalty in the cup final",
            "The goalkeeper saved a penalty in the derby final",
            "The central bank raised interest rates amid inflation",
        ]
    )
    assert float(a @ b) > float(a @ c)


def test_provider_save_load(tmp_path: Path, small_texts: list[str]) -> None:
    provider = TfidfSvdProvider(dimension=8).fit(small_texts)
    path = provider.save(tmp_path / "p.joblib")
    loaded = TfidfSvdProvider.load(path)
    assert np.allclose(loaded.embed_texts(small_texts[:2]), provider.embed_texts(small_texts[:2]))
    with pytest.raises(ValueError):
        TfidfSvdProvider(dimension=1)


def test_get_provider_factory() -> None:
    assert isinstance(get_provider("tfidf_svd", dimension=8), TfidfSvdProvider)
    assert isinstance(get_provider("sbert"), SentenceTransformerProvider)
    with pytest.raises(ValueError):
        get_provider("unknown")


def test_sentence_transformer_provider_missing_dependency_message() -> None:
    provider = SentenceTransformerProvider("some-model")
    try:
        import sentence_transformers  # noqa: F401

        pytest.skip("sentence-transformers is installed; cannot test the missing-dependency path")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="transformers"):
        provider.embed_texts(["hello"])


def test_sentence_transformer_provider_with_fake_model() -> None:
    class FakeModel:
        def get_sentence_embedding_dimension(self) -> int:
            return 4

        def encode(self, texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False):  # type: ignore[no-untyped-def]
            return np.tile(np.arange(1, 5, dtype=np.float32), (len(texts), 1))

    provider = SentenceTransformerProvider("fake", model=FakeModel())
    assert provider.dimension == 4 and provider.is_fitted
    vecs = provider.embed_texts(["a", "b"])
    assert vecs.shape == (2, 4)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


def test_embedding_cache_hits_and_misses(tmp_path: Path, small_texts: list[str]) -> None:
    provider = TfidfSvdProvider(dimension=8).fit(small_texts)
    cache = EmbeddingCache(tmp_path / "cache", provider)
    first = cache.embed(small_texts[:3])
    assert cache.misses == 3 and cache.hits == 0 and len(cache) == 3
    second = cache.embed(small_texts[:3] + small_texts[3:4])
    assert cache.hits == 3 and cache.misses == 4
    assert np.allclose(first, second[:3])
    assert cache.has(small_texts[0]) and not cache.has("never seen")
    assert cache.embed([]).shape == (0, 8)
    assert cache.clear() == 4 and len(cache) == 0


def test_embedding_cache_discards_corrupt_entries(tmp_path: Path, small_texts: list[str]) -> None:
    provider = TfidfSvdProvider(dimension=8).fit(small_texts)
    cache = EmbeddingCache(tmp_path / "cache", provider)
    cache.embed(small_texts[:1])
    corrupt = next((tmp_path / "cache").rglob("*.npy"))
    corrupt.write_bytes(b"not a numpy file")
    assert cache.get(small_texts[0]) is None
    assert not corrupt.exists()
