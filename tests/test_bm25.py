from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from newsroom.search.bm25 import BM25

DOCS = {
    "d1": "The central bank raised interest rates to fight inflation.",
    "d2": "Interest rates stayed flat as the bank paused its hiking cycle.",
    "d3": "The striker scored a hat-trick in the championship final.",
    "d4": "A new vaccine trial reported strong results against influenza.",
    "d5": "Inflation inflation inflation: prices keep rising says the central bank.",
}


@pytest.fixture(scope="module")
def bm25() -> BM25:
    return BM25().fit(list(DOCS), list(DOCS.values()))


def test_fit_and_basic_ranking(bm25: BM25) -> None:
    assert bm25.is_fitted and len(bm25) == 5 and bm25.vocabulary_size > 10
    hits = bm25.search("central bank inflation", top_k=3)
    ids = [d for d, _ in hits]
    assert ids[0] in {"d1", "d5"}
    assert "d3" not in ids
    assert all(s > 0 for _, s in hits)
    assert hits == sorted(hits, key=lambda h: -h[1])


def test_term_frequency_saturation(bm25: BM25) -> None:
    scores = bm25.score("inflation")
    assert scores.shape == (5,)
    # d5 repeats the term but BM25 saturates, so it should not be 3x the single-mention score.
    assert scores[4] > scores[0] > 0
    assert scores[4] < 3 * scores[0]


def test_unknown_query_terms_return_empty(bm25: BM25) -> None:
    assert bm25.search("zebra quantum", top_k=5) == []
    assert np.all(bm25.score("zebra") == 0)
    assert bm25.search("bank", top_k=0) == []


def test_validation() -> None:
    with pytest.raises(ValueError):
        BM25(k1=-1)
    with pytest.raises(ValueError):
        BM25(b=1.5)
    with pytest.raises(ValueError):
        BM25().fit([], [])
    with pytest.raises(ValueError):
        BM25().fit(["a"], ["x", "y"])
    with pytest.raises(RuntimeError):
        BM25().score("x")


def test_length_normalisation_prefers_shorter_docs() -> None:
    short = "bank rates"
    long = "bank rates " + " ".join(["filler"] * 40)
    model = BM25(b=0.75).fit(["short", "long"], [short, long])
    scores = model.score("bank rates")
    assert scores[0] > scores[1]
    no_norm = BM25(b=0.0).fit(["short", "long"], [short, long])
    s2 = no_norm.score("bank rates")
    assert abs(s2[0] - s2[1]) < 1e-5


def test_save_and_load(tmp_path: Path, bm25: BM25) -> None:
    bm25.save(tmp_path)
    loaded = BM25.load(tmp_path)
    assert loaded.search("central bank inflation", 3) == bm25.search("central bank inflation", 3)
    with pytest.raises(FileNotFoundError):
        BM25.load(tmp_path / "missing")


def test_custom_tokenizer() -> None:
    model = BM25(tokenizer=lambda t: t.lower().split()).fit(
        ["a", "b"], ["Alpha beta", "beta gamma"]
    )
    assert [d for d, _ in model.search("alpha", 2)] == ["a"]
