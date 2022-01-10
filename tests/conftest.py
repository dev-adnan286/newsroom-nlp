"""Shared fixtures: a small deterministic corpus and models trained once per session."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

from newsroom.classification.trainer import (  # noqa: E402
    TrainConfig,
    TrainResult,
    save_model,
    train_classifier,
)
from newsroom.data.corpus import generate_corpus  # noqa: E402
from newsroom.features.vectorizers import VectorizerConfig  # noqa: E402
from newsroom.search.query import SearchEngine, build_search_engine  # noqa: E402


@pytest.fixture(scope="session")
def corpus_df() -> pd.DataFrame:
    """180 synthetic documents (30 per category)."""
    return generate_corpus(n_docs=180, seed=123)


@pytest.fixture(scope="session")
def small_texts(corpus_df: pd.DataFrame) -> list[str]:
    return corpus_df["text"].tolist()[:40]


@pytest.fixture(scope="session")
def train_result(corpus_df: pd.DataFrame) -> TrainResult:
    cfg = TrainConfig(
        model="linear_svc",
        calibrate=True,
        grid_search=False,
        vectorizer=VectorizerConfig(min_df=1, max_features=20000, use_char_ngrams=True),
    )
    return train_classifier(corpus_df, cfg)


@pytest.fixture(scope="session")
def model_path(train_result: TrainResult, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("model") / "classifier.joblib"
    save_model(train_result, path)
    return path


@pytest.fixture(scope="session")
def search_engine(corpus_df: pd.DataFrame) -> SearchEngine:
    return build_search_engine(corpus_df, dimension=48)


@pytest.fixture(scope="session")
def index_dir(search_engine: SearchEngine, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("index") / "index"
    search_engine.save(path)
    return path


@pytest.fixture(scope="session")
def corpus_csv(corpus_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    from newsroom.data.corpus import save_corpus_csv

    path = tmp_path_factory.mktemp("data") / "corpus.csv"
    save_corpus_csv(corpus_df, path)
    return path


SAMPLE_ARTICLE = (
    "The central bank raised interest rates by a quarter point on Wednesday, citing persistent "
    "inflation and a tight labour market. Officials signalled that further increases remain possible "
    "if price pressures do not ease. Markets reacted calmly, with bond yields edging higher and the "
    "currency strengthening against the dollar. Analysts said the decision was widely expected. "
    "The bank's governor will hold a press conference later today to explain the outlook."
)


@pytest.fixture
def sample_article() -> str:
    return SAMPLE_ARTICLE
