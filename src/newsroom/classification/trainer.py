"""Train linear text classifiers with optional calibration and grid search."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from newsroom import __version__
from newsroom.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from newsroom.features.vectorizers import VectorizerConfig, build_word_char_union
from newsroom.preprocessing.text import TextCleaner
from newsroom.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_MODELS: tuple[str, ...] = ("linear_svc", "logistic_regression", "complement_nb", "sgd")

MODEL_ARTIFACT_KEY = "pipeline"


@dataclass
class TrainConfig:
    """Everything needed to reproduce a training run."""

    model: str = "linear_svc"
    calibrate: bool = True
    grid_search: bool = False
    cv_folds: int = 3
    test_size: float = 0.2
    random_seed: int = 42
    vectorizer: VectorizerConfig = field(default_factory=VectorizerConfig)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {SUPPORTED_MODELS}, got {self.model!r}")
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> TrainConfig:
        """Create a config from a parsed YAML/JSON mapping."""
        data = dict(data or {})
        vec = VectorizerConfig.from_dict(data.pop("vectorizer", None))
        known = {
            k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "vectorizer"
        }
        return cls(vectorizer=vec, **known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    """Outcome of :func:`train_classifier`."""

    pipeline: Pipeline
    metrics: ClassificationMetrics
    config: TrainConfig
    labels: list[str]
    train_size: int
    test_size: int
    best_params: dict[str, Any]
    train_seconds: float


def _base_estimator(model: str, seed: int) -> BaseEstimator:
    if model == "linear_svc":
        return LinearSVC(C=1.0, random_state=seed, dual="auto")
    if model == "logistic_regression":
        return LogisticRegression(C=4.0, max_iter=2000, random_state=seed)
    if model == "complement_nb":
        return ComplementNB(alpha=0.3)
    if model == "sgd":
        return SGDClassifier(
            loss="modified_huber", alpha=1e-4, max_iter=50, tol=1e-3, random_state=seed
        )
    raise ValueError(f"Unsupported model: {model}")


def _needs_calibration(model: str) -> bool:
    # ComplementNB / LogisticRegression already expose predict_proba; SGD modified_huber does too.
    return model == "linear_svc"


def build_pipeline(config: TrainConfig | None = None) -> Pipeline:
    """Assemble ``TextCleaner -> TF-IDF union -> classifier`` (optionally calibrated)."""
    cfg = config or TrainConfig()
    estimator = _base_estimator(cfg.model, cfg.random_seed)
    if cfg.calibrate and _needs_calibration(cfg.model):
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=min(cfg.cv_folds, 3))
    cleaner_kwargs = {k: v for k, v in cfg.preprocessing.items() if k in TextCleaner().get_params()}
    return Pipeline(
        [
            ("clean", TextCleaner(**cleaner_kwargs)),
            ("features", build_word_char_union(cfg.vectorizer)),
            ("clf", estimator),
        ]
    )


def _param_grid(config: TrainConfig) -> dict[str, list[Any]]:
    """Small, fast grid so demos finish quickly."""
    grid: dict[str, list[Any]] = {}
    if config.vectorizer.use_char_ngrams:
        grid["features__word__ngram_range"] = [(1, 1), (1, 2)]
    else:
        grid["features__ngram_range"] = [(1, 1), (1, 2)]
    clf_prefix = (
        "clf__estimator__" if (config.calibrate and _needs_calibration(config.model)) else "clf__"
    )
    if config.model in {"linear_svc", "logistic_regression"}:
        grid[f"{clf_prefix}C"] = [0.5, 1.0, 2.0]
    elif config.model == "complement_nb":
        grid[f"{clf_prefix}alpha"] = [0.1, 0.3, 1.0]
    elif config.model == "sgd":
        grid[f"{clf_prefix}alpha"] = [1e-5, 1e-4, 1e-3]
    return grid


def train_classifier(
    df: pd.DataFrame,
    config: TrainConfig | None = None,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> TrainResult:
    """Train, evaluate on a stratified hold-out split and return the fitted pipeline.

    Args:
        df: DataFrame with at least ``text_column`` and ``label_column``.
        config: Training configuration.
        text_column: Name of the text column.
        label_column: Name of the label column.
    """
    cfg = config or TrainConfig()
    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(f"DataFrame must contain {text_column!r} and {label_column!r}")
    texts = df[text_column].fillna("").astype(str).to_numpy()
    labels = df[label_column].astype(str).to_numpy()
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("At least two distinct labels are required for training")
    counts = pd.Series(labels).value_counts()
    stratify = labels if counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=stratify
    )
    pipeline = build_pipeline(cfg)
    started = time.perf_counter()
    best_params: dict[str, Any] = {}

    if cfg.grid_search:
        folds = min(cfg.cv_folds, int(pd.Series(y_train).value_counts().min()))
        if folds < 2:
            logger.warning("Too few samples per class for grid search; fitting directly")
            pipeline.fit(x_train, y_train)
        else:
            search = GridSearchCV(
                pipeline,
                _param_grid(cfg),
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.random_seed),
                scoring="f1_macro",
                n_jobs=cfg.n_jobs,
                refit=True,
            )
            search.fit(x_train, y_train)
            pipeline = search.best_estimator_
            best_params = dict(search.best_params_)
            logger.info("Grid search best macro-F1=%.4f params=%s", search.best_score_, best_params)
    else:
        pipeline.fit(x_train, y_train)

    elapsed = time.perf_counter() - started
    y_pred = pipeline.predict(x_test)
    metrics = compute_classification_metrics(y_test, y_pred, labels=classes)
    logger.info(
        "Trained %s in %.2fs | accuracy=%.4f macro-F1=%.4f (test n=%d)",
        cfg.model,
        elapsed,
        metrics.accuracy,
        metrics.macro_f1,
        len(y_test),
    )
    return TrainResult(
        pipeline=pipeline,
        metrics=metrics,
        config=cfg,
        labels=classes,
        train_size=int(len(y_train)),
        test_size=int(len(y_test)),
        best_params=best_params,
        train_seconds=elapsed,
    )


def save_model(result: TrainResult, path: str | Path) -> Path:
    """Persist a trained pipeline and its metadata with joblib."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        MODEL_ARTIFACT_KEY: result.pipeline,
        "labels": list(result.labels),
        "config": result.config.to_dict(),
        "metrics": result.metrics.to_dict(),
        "best_params": result.best_params,
        "package_version": __version__,
        "train_size": result.train_size,
        "test_size": result.test_size,
    }
    joblib.dump(payload, p, compress=3)
    logger.info("Saved classifier to %s", p)
    return p


def load_model(path: str | Path) -> dict[str, Any]:
    """Load a payload saved by :func:`save_model`; raises a clear error on bad files."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model artifact not found: {p}")
    payload = joblib.load(p)
    if not isinstance(payload, dict) or MODEL_ARTIFACT_KEY not in payload:
        raise ValueError(f"{p} is not a newsroom classifier artifact")
    return payload


def class_prior(labels: np.ndarray) -> dict[str, float]:
    """Return relative class frequencies."""
    values, counts = np.unique(labels, return_counts=True)
    total = float(counts.sum())
    return {str(v): float(c) / total for v, c in zip(values, counts, strict=True)}


__all__ = [
    "MODEL_ARTIFACT_KEY",
    "SUPPORTED_MODELS",
    "TrainConfig",
    "TrainResult",
    "build_pipeline",
    "class_prior",
    "load_model",
    "save_model",
    "train_classifier",
]
