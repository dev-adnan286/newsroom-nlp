from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from newsroom.data.corpus import (
    CATEGORIES,
    CorpusGenerator,
    CorpusSpec,
    generate_corpus,
    load_corpus_csv,
    save_corpus_csv,
)


def test_generate_corpus_shape_and_columns(corpus_df: pd.DataFrame) -> None:
    assert list(corpus_df.columns) == ["doc_id", "text", "label", "title", "published_at"]
    assert len(corpus_df) == 180
    assert set(corpus_df["label"]) == set(CATEGORIES)
    assert corpus_df["label"].value_counts().min() == 30
    assert corpus_df["doc_id"].is_unique
    assert pd.api.types.is_datetime64_any_dtype(corpus_df["published_at"])
    assert corpus_df["text"].str.len().min() > 40


def test_generation_is_deterministic() -> None:
    a = generate_corpus(30, seed=7)
    b = generate_corpus(30, seed=7)
    c = generate_corpus(30, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a["text"].equals(c["text"])


def test_spec_validation() -> None:
    with pytest.raises(ValueError):
        CorpusSpec(n_docs=0)
    with pytest.raises(ValueError):
        CorpusSpec(min_sentences=4, max_sentences=2)
    with pytest.raises(ValueError):
        CorpusSpec(categories=("technology", "cooking"))


def test_subset_of_categories() -> None:
    df = CorpusGenerator(CorpusSpec(n_docs=20, categories=("sports", "health"))).generate()
    assert set(df["label"]) == {"sports", "health"}


def test_csv_roundtrip(tmp_path: Path, corpus_df: pd.DataFrame) -> None:
    path = save_corpus_csv(corpus_df.head(10), tmp_path / "c.csv")
    loaded = load_corpus_csv(path)
    assert len(loaded) == 10
    assert loaded["doc_id"].tolist() == corpus_df["doc_id"].head(10).tolist()
    assert pd.api.types.is_datetime64_any_dtype(loaded["published_at"])


def test_load_corpus_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus_csv(tmp_path / "missing.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["x"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_corpus_csv(bad)
    unlabeled = tmp_path / "unlabeled.csv"
    pd.DataFrame({"text": ["hello world", "second doc"]}).to_csv(unlabeled, index=False)
    df = load_corpus_csv(unlabeled, require_label=False)
    assert {"doc_id", "title", "published_at"} <= set(df.columns)
