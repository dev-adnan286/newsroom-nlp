"""Supervised text classification: training, prediction and explanation."""

from newsroom.classification.explain import explain_document, top_terms_per_class
from newsroom.classification.predictor import Prediction, Predictor
from newsroom.classification.trainer import (
    SUPPORTED_MODELS,
    TrainConfig,
    TrainResult,
    build_pipeline,
    load_model,
    save_model,
    train_classifier,
)

__all__ = [
    "SUPPORTED_MODELS",
    "Prediction",
    "Predictor",
    "TrainConfig",
    "TrainResult",
    "build_pipeline",
    "explain_document",
    "load_model",
    "save_model",
    "top_terms_per_class",
    "train_classifier",
]
