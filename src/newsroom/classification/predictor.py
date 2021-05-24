"""Inference wrapper around a trained classification pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from newsroom.classification.trainer import MODEL_ARTIFACT_KEY, load_model


@dataclass
class Prediction:
    """A single classification outcome."""

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)

    def top(self, k: int = 3) -> list[tuple[str, float]]:
        """Top-``k`` ``(label, probability)`` pairs sorted by probability."""
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]


class Predictor:
    """Thin, typed facade over a fitted pipeline with probability handling.

    If the underlying estimator lacks ``predict_proba`` (e.g. an uncalibrated ``LinearSVC``),
    decision-function scores are converted with a softmax so confidences stay comparable.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        labels: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metadata = metadata or {}
        clf = pipeline.named_steps.get("clf", pipeline.steps[-1][1])
        classes = getattr(clf, "classes_", None)
        if classes is None:
            raise ValueError("Pipeline's final estimator is not fitted (no classes_)")
        self.labels: list[str] = [str(c) for c in classes]
        if labels and sorted(labels) != sorted(self.labels):
            raise ValueError("Provided labels do not match the fitted estimator's classes")

    @classmethod
    def from_path(cls, path: str | Path) -> Predictor:
        """Load a :class:`Predictor` from a joblib artifact created by ``save_model``."""
        payload = load_model(path)
        meta = {k: v for k, v in payload.items() if k != MODEL_ARTIFACT_KEY}
        return cls(payload[MODEL_ARTIFACT_KEY], payload.get("labels"), meta)

    # -- scoring -----------------------------------------------------------------------
    def _scores(self, texts: Sequence[str]) -> np.ndarray:
        clf = self.pipeline.named_steps.get("clf", self.pipeline.steps[-1][1])
        if hasattr(clf, "predict_proba"):
            return np.asarray(self.pipeline.predict_proba(list(texts)), dtype=np.float64)
        raw = np.asarray(self.pipeline.decision_function(list(texts)), dtype=np.float64)
        if raw.ndim == 1:  # binary: convert margin to two-column scores
            raw = np.stack([-raw, raw], axis=1)
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, text: str) -> Prediction:
        """Classify one document."""
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify many documents at once."""
        if len(texts) == 0:
            return []
        cleaned = [str(t) if t is not None else "" for t in texts]
        probs = self._scores(cleaned)
        out: list[Prediction] = []
        for row in probs:
            idx = int(np.argmax(row))
            out.append(
                Prediction(
                    label=self.labels[idx],
                    confidence=float(row[idx]),
                    probabilities={lab: float(p) for lab, p in zip(self.labels, row, strict=True)},
                )
            )
        return out

    def predict_labels(self, texts: Sequence[str]) -> list[str]:
        """Return only the predicted labels (fast path without probabilities)."""
        if len(texts) == 0:
            return []
        return [str(x) for x in self.pipeline.predict([str(t) for t in texts])]


__all__ = ["Prediction", "Predictor"]
