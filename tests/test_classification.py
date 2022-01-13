from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

from newsroom.classification.explain import explain_document, top_terms_per_class
from newsroom.classification.predictor import Predictor
from newsroom.classification.trainer import (
    SUPPORTED_MODELS,
    TrainConfig,
    TrainResult,
    build_pipeline,
    load_model,
    train_classifier,
)
from newsroom.features.vectorizers import VectorizerConfig

FAST_VEC = VectorizerConfig(min_df=1, use_char_ngrams=False, max_features=5000)


def test_train_config_validation() -> None:
    with pytest.raises(ValueError):
        TrainConfig(model="random_forest")
    with pytest.raises(ValueError):
        TrainConfig(test_size=1.5)
    cfg = TrainConfig.from_dict(
        {"model": "sgd", "vectorizer": {"min_df": 1}, "cv_folds": 2, "bogus": 1}
    )
    assert cfg.model == "sgd" and cfg.vectorizer.min_df == 1 and cfg.cv_folds == 2
    assert cfg.to_dict()["model"] == "sgd"


def test_trained_model_is_accurate(train_result: TrainResult) -> None:
    assert train_result.metrics.accuracy > 0.8
    assert train_result.metrics.macro_f1 > 0.8
    assert set(train_result.labels) == set(train_result.metrics.labels)
    assert train_result.train_size + train_result.test_size == 180


@pytest.mark.parametrize("model", [m for m in SUPPORTED_MODELS if m != "linear_svc"])
def test_all_models_train_and_expose_probabilities(model: str, corpus_df: pd.DataFrame) -> None:
    cfg = TrainConfig(model=model, vectorizer=FAST_VEC, grid_search=False)
    result = train_classifier(corpus_df, cfg)
    predictor = Predictor(result.pipeline)
    pred = predictor.predict("Doctors at the hospital approved a new vaccine trial for patients")
    assert pred.label in result.labels
    assert 0.0 < pred.confidence <= 1.0
    assert abs(sum(pred.probabilities.values()) - 1.0) < 1e-4


def test_grid_search_path(corpus_df: pd.DataFrame) -> None:
    cfg = TrainConfig(
        model="logistic_regression", vectorizer=FAST_VEC, grid_search=True, cv_folds=2
    )
    result = train_classifier(corpus_df.sample(90, random_state=0), cfg)
    assert result.best_params and "clf__C" in result.best_params


def test_uncalibrated_svc_uses_softmax_fallback(corpus_df: pd.DataFrame) -> None:
    cfg = TrainConfig(model="linear_svc", calibrate=False, vectorizer=FAST_VEC)
    result = train_classifier(corpus_df, cfg)
    predictor = Predictor(result.pipeline)
    preds = predictor.predict_batch(
        ["Striker scores twice in the derby", "Shares fell after weak earnings"]
    )
    assert len(preds) == 2
    assert all(abs(sum(p.probabilities.values()) - 1.0) < 1e-6 for p in preds)
    assert predictor.predict_batch([]) == []


def test_training_requires_columns_and_labels() -> None:
    with pytest.raises(ValueError):
        train_classifier(pd.DataFrame({"text": ["a", "b"]}))
    with pytest.raises(ValueError, match="two distinct"):
        train_classifier(
            pd.DataFrame({"text": ["a b c", "b c d", "c d e", "d e f"], "label": ["x"] * 4})
        )


def test_save_load_predictor(model_path: Path, train_result: TrainResult) -> None:
    payload = load_model(model_path)
    assert payload["labels"] == train_result.labels
    assert payload["metrics"]["accuracy"] == train_result.metrics.accuracy
    predictor = Predictor.from_path(model_path)
    assert predictor.metadata["config"]["model"] == "linear_svc"
    pred = predictor.predict(
        "Parliament passed the budget bill after a heated debate on voter turnout"
    )
    assert pred.label == "politics"
    assert [lab for lab, _ in pred.top(2)][0] == "politics"
    assert predictor.predict_labels(["The rover mission observed dark matter signals"]) == [
        "science"
    ]


def test_load_model_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "nope.joblib")
    bad = tmp_path / "bad.joblib"
    joblib.dump({"something": 1}, bad)
    with pytest.raises(ValueError):
        load_model(bad)


def test_predictor_rejects_unfitted_pipeline() -> None:
    with pytest.raises(ValueError):
        Predictor(build_pipeline(TrainConfig(vectorizer=FAST_VEC)))


def test_top_terms_per_class(train_result: TrainResult) -> None:
    terms = top_terms_per_class(train_result.pipeline, n_terms=5)
    assert set(terms) == set(train_result.labels)
    for label, pairs in terms.items():
        assert len(pairs) == 5
        assert all(isinstance(t, str) and isinstance(w, float) for t, w in pairs)
        assert pairs == sorted(pairs, key=lambda p: -p[1]), label
    sports_terms = " ".join(t for t, _ in terms["sports"])
    assert any(
        w in sports_terms
        for w in ("fan", "coach", "rival", "match", "stadium", "striker", "penalti")
    )


def test_top_terms_for_naive_bayes(corpus_df: pd.DataFrame) -> None:
    result = train_classifier(corpus_df, TrainConfig(model="complement_nb", vectorizer=FAST_VEC))
    terms = top_terms_per_class(result.pipeline, n_terms=3)
    assert all(len(v) == 3 for v in terms.values())


def test_explain_document(train_result: TrainResult) -> None:
    text = "The goalkeeper saved a penalty and the striker scored in the cup final"
    explanation = explain_document(train_result.pipeline, text, n_terms=5)
    assert explanation["predicted_label"] == "sports"
    assert explanation["explained_label"] == "sports"
    assert 0 < len(explanation["top_features"]) <= 5
    assert all(f["contribution"] > 0 for f in explanation["top_features"])
    other = explain_document(train_result.pipeline, text, label="business")
    assert other["explained_label"] == "business"
    with pytest.raises(ValueError):
        explain_document(train_result.pipeline, text, label="cooking")
