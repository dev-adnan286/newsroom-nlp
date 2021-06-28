"""Classification metrics with a serialisable container."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassificationMetrics:
    """Aggregate and per-class scores for a multi-class classifier."""

    accuracy: float
    macro_f1: float
    micro_f1: float
    weighted_f1: float
    labels: list[str]
    per_class: dict[str, dict[str, float]]
    confusion: list[list[int]]
    n_samples: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "labels": list(self.labels),
            "per_class": self.per_class,
            "confusion": self.confusion,
            "n_samples": self.n_samples,
            **({"extra": self.extra} if self.extra else {}),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationMetrics:
        return cls(
            accuracy=float(data["accuracy"]),
            macro_f1=float(data["macro_f1"]),
            micro_f1=float(data["micro_f1"]),
            weighted_f1=float(data["weighted_f1"]),
            labels=list(data["labels"]),
            per_class=dict(data["per_class"]),
            confusion=[list(map(int, row)) for row in data["confusion"]],
            n_samples=int(data["n_samples"]),
            extra=dict(data.get("extra", {})),
        )

    def per_class_table(self) -> list[dict[str, Any]]:
        """Rows suitable for tabular rendering."""
        return [
            {
                "label": lab,
                **{
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in self.per_class[lab].items()
                },
            }
            for lab in self.labels
        ]

    def worst_classes(self, n: int = 3) -> list[tuple[str, float]]:
        """Classes with the lowest F1."""
        pairs = [(lab, self.per_class[lab]["f1"]) for lab in self.labels]
        return sorted(pairs, key=lambda kv: kv[1])[:n]


def compute_classification_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    *,
    labels: Sequence[str] | None = None,
) -> ClassificationMetrics:
    """Compute accuracy, macro/micro/weighted F1, per-class P/R/F1/support and confusion matrix."""
    yt = np.asarray([str(v) for v in y_true])
    yp = np.asarray([str(v) for v in y_pred])
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if yt.size == 0:
        raise ValueError("Cannot compute metrics on empty inputs")
    label_list = list(labels) if labels is not None else sorted(set(yt.tolist()) | set(yp.tolist()))
    precision, recall, f1, support = precision_recall_fscore_support(
        yt, yp, labels=label_list, zero_division=0
    )
    per_class = {
        lab: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, lab in enumerate(label_list)
    }
    return ClassificationMetrics(
        accuracy=float(accuracy_score(yt, yp)),
        macro_f1=float(f1_score(yt, yp, labels=label_list, average="macro", zero_division=0)),
        micro_f1=float(f1_score(yt, yp, labels=label_list, average="micro", zero_division=0)),
        weighted_f1=float(f1_score(yt, yp, labels=label_list, average="weighted", zero_division=0)),
        labels=label_list,
        per_class=per_class,
        confusion=confusion_matrix(yt, yp, labels=label_list).astype(int).tolist(),
        n_samples=int(yt.size),
    )


def classification_report_text(metrics: ClassificationMetrics) -> str:
    """Plain-text report resembling scikit-learn's ``classification_report``."""
    width = max(len(lab) for lab in metrics.labels + ["weighted avg"])
    lines = [f"{'':<{width}}  precision  recall  f1-score  support"]
    for lab in metrics.labels:
        pc = metrics.per_class[lab]
        lines.append(
            f"{lab:<{width}}  {pc['precision']:9.3f}  {pc['recall']:6.3f}  {pc['f1']:8.3f}  {pc['support']:7d}"
        )
    lines.append("")
    lines.append(
        f"{'accuracy':<{width}}  {'':9}  {'':6}  {metrics.accuracy:8.3f}  {metrics.n_samples:7d}"
    )
    lines.append(
        f"{'macro avg':<{width}}  {'':9}  {'':6}  {metrics.macro_f1:8.3f}  {metrics.n_samples:7d}"
    )
    lines.append(
        f"{'weighted avg':<{width}}  {'':9}  {'':6}  {metrics.weighted_f1:8.3f}  {metrics.n_samples:7d}"
    )
    return "\n".join(lines)


__all__ = ["ClassificationMetrics", "classification_report_text", "compute_classification_metrics"]
