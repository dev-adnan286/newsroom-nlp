from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from newsroom.search.index import SearchHit, VectorIndex


def _unit(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_add_and_search_returns_cosine_ranking() -> None:
    index = VectorIndex(dimension=3, initial_capacity=2)
    index.add(
        ["a", "b", "c"],
        np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]),
        [{"t": 1}, {"t": 2}, {"t": 3}],
    )
    assert len(index) == 3 and "a" in index and "zz" not in index
    hits = index.search(np.array([1.0, 0.0, 0.0]), top_k=2)
    assert [h.doc_id for h in hits] == ["a", "c"]
    assert hits[0].rank == 1 and hits[0].score == pytest.approx(1.0)
    assert isinstance(hits[0], SearchHit) and hits[0].metadata == {"t": 1}
    assert index.get_metadata("b") == {"t": 2}


def test_capacity_growth_and_vectors_view() -> None:
    index = VectorIndex(dimension=4, initial_capacity=1)
    rng = np.random.RandomState(0)
    ids = [f"d{i}" for i in range(50)]
    index.add(ids, rng.rand(50, 4))
    assert len(index) == 50 and index.vectors.shape == (50, 4)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0, atol=1e-5)
    assert index.ids == ids


def test_duplicate_and_overwrite() -> None:
    index = VectorIndex(dimension=2)
    index.add(["a"], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="already indexed"):
        index.add(["a"], np.array([[0.0, 1.0]]))
    index.add(["a"], np.array([[0.0, 1.0]]), overwrite=True)
    assert len(index) == 1
    assert index.search(np.array([0.0, 1.0]), 1)[0].score == pytest.approx(1.0)
    with pytest.raises(ValueError):
        index.add(["x", "x"], np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_validation_errors() -> None:
    with pytest.raises(ValueError):
        VectorIndex(dimension=0)
    index = VectorIndex(dimension=2)
    with pytest.raises(ValueError):
        index.add(["a"], np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        index.add(["a", "b"], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        index.add(["a"], np.array([[1.0, 0.0]]), metadata=[{}, {}])
    index.add(["a"], np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError):
        index.search(np.array([1.0, 0.0, 0.0]))
    assert VectorIndex(dimension=2).search(np.array([1.0, 0.0])) == []


def test_remove_keeps_index_consistent() -> None:
    index = VectorIndex(dimension=2)
    index.add(["a", "b", "c"], np.array([[1, 0], [0, 1], [1, 1]]))
    assert index.remove("a") and not index.remove("a")
    assert len(index) == 2 and index.ids == ["c", "b"]
    assert index.search(np.array([0.0, 1.0]), 1)[0].doc_id == "b"
    assert index.get_metadata("c") == {}


def test_min_score_and_batch() -> None:
    index = VectorIndex(dimension=2)
    index.add(["a", "b"], np.array([[1, 0], [0, 1]]))
    assert [h.doc_id for h in index.search(np.array([1.0, 0.0]), 2, min_score=0.5)] == ["a"]
    batch = index.search_batch(np.array([[1.0, 0.0], [0.0, 1.0]]), top_k=1)
    assert [b[0].doc_id for b in batch] == ["a", "b"]


def test_save_and_load(tmp_path: Path) -> None:
    index = VectorIndex(dimension=3)
    index.add(["a", "b"], np.array([[1, 0, 0], [0, 1, 0]]), [{"title": "A"}, {"title": "B"}])
    index.save(tmp_path)
    loaded = VectorIndex.load(tmp_path)
    assert loaded.ids == ["a", "b"] and loaded.dimension == 3
    assert loaded.get_metadata("b") == {"title": "B"}
    assert loaded.search(_unit([0, 1, 0]), 1)[0].doc_id == "b"
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "missing")
