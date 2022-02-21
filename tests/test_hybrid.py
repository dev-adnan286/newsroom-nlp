from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from newsroom.search.hybrid import HybridConfig, HybridSearcher, min_max, reciprocal_rank_fusion
from newsroom.search.query import SearchEngine, SearchResult


def test_rrf_basic_fusion() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[0] == "a"  # tie broken by first appearance
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert ids[-2:] == ["c", "d"] or ids[-2:] == ["d", "c"]


def test_rrf_weights_and_validation() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=10, weights=[0.1, 1.0])
    assert fused[0][0] == "b"
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=0)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
    assert reciprocal_rank_fusion([]) == []


def test_min_max() -> None:
    assert np.allclose(min_max(np.array([1.0, 3.0, 2.0])), [0.0, 1.0, 0.5])
    assert np.all(min_max(np.array([2.0, 2.0])) == 0)
    assert min_max(np.array([])).size == 0


def test_hybrid_config_from_dict() -> None:
    cfg = HybridConfig.from_dict({"rrf_k": 10, "bm25_weight": 2.0, "junk": 1})
    assert cfg.rrf_k == 10 and cfg.bm25_weight == 2.0 and cfg.vector_weight == 1.0


def test_engine_modes_and_results(search_engine: SearchEngine) -> None:
    query = "central bank interest rates inflation earnings"
    hybrid = search_engine.search(query, top_k=5)
    assert len(hybrid) == 5 and all(isinstance(r, SearchResult) for r in hybrid)
    assert [r.rank for r in hybrid] == [1, 2, 3, 4, 5]
    assert hybrid[0].label == "business"
    assert {"bm25", "vector"} <= set(hybrid[0].component_scores)
    bm25_only = search_engine.search(query, top_k=5, mode="bm25")
    vec_only = search_engine.search(query, top_k=5, mode="vector")
    assert bm25_only and vec_only
    assert bm25_only[0].component_scores["vector"] == 0.0
    assert hybrid[0].snippet(30).endswith("…")
    with pytest.raises(ValueError):
        search_engine.search(query, mode="fuzzy")
    assert search_engine.search("   ", top_k=3) == []


def test_hybrid_weights_shift_ranking(search_engine: SearchEngine) -> None:
    assert search_engine._hybrid is not None
    searcher: HybridSearcher = search_engine._hybrid
    query = "goalkeeper penalty derby"
    bm25_heavy = searcher.search(query, 5, bm25_weight=5.0, vector_weight=0.01)
    bm25_only = searcher.search(query, 5, mode="bm25")
    assert bm25_heavy[0][0] == bm25_only[0][0]
    assert searcher.search(query, 0) == []


def test_engine_add_documents_and_persistence(
    search_engine: SearchEngine, index_dir: Path, tmp_path: Path
) -> None:
    loaded = SearchEngine.load(index_dir)
    assert len(loaded) == len(search_engine)
    before = loaded.search("volcanic eruption lava", 3)
    new_docs = pd.DataFrame(
        {
            "doc_id": ["new-1", "new-2"],
            "text": [
                "A volcanic eruption sent lava flowing toward villages as scientists monitored seismic activity.",
                "Lava from the volcano cooled overnight, the observatory said.",
            ],
            "title": ["Volcano erupts", "Lava cools"],
            "label": ["science", "science"],
        }
    )
    added = loaded.add_documents(new_docs)
    assert added == 2 and len(loaded) == len(search_engine) + 2
    assert loaded.add_documents(new_docs) == 0  # idempotent
    after = loaded.search("volcanic eruption lava", 3)
    assert after[0].doc_id == "new-1"
    assert before[0].doc_id != "new-1"
    out = loaded.save(tmp_path / "idx2")
    again = SearchEngine.load(out)
    assert "new-2" in again.documents


def test_engine_errors(tmp_path: Path) -> None:
    from newsroom.embeddings.tfidf_svd import TfidfSvdProvider

    engine = SearchEngine(TfidfSvdProvider(dimension=8))
    with pytest.raises(RuntimeError):
        engine.search("x")
    with pytest.raises(RuntimeError):
        engine.save(tmp_path)
    with pytest.raises(ValueError):
        engine.build(pd.DataFrame({"text": []}))
    with pytest.raises(ValueError, match="unique"):
        engine.build(pd.DataFrame({"doc_id": ["a", "a"], "text": ["one two", "three four"]}))
    with pytest.raises(FileNotFoundError):
        SearchEngine.load(tmp_path / "missing")
