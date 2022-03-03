from __future__ import annotations

from pathlib import Path

import pytest

from newsroom.evaluation.metrics import (
    ClassificationMetrics,
    classification_report_text,
    compute_classification_metrics,
)
from newsroom.evaluation.report import write_json_report, write_markdown_report
from newsroom.evaluation.search_eval import RetrievalMetrics
from newsroom.utils.io import read_json


def test_compute_metrics_perfect_and_imperfect() -> None:
    y = ["a", "b", "a", "c"]
    perfect = compute_classification_metrics(y, y)
    assert perfect.accuracy == 1.0 and perfect.macro_f1 == 1.0 and perfect.micro_f1 == 1.0
    assert perfect.labels == ["a", "b", "c"] and perfect.n_samples == 4
    assert perfect.confusion == [[2, 0, 0], [0, 1, 0], [0, 0, 1]]

    pred = ["a", "a", "a", "c"]
    m = compute_classification_metrics(y, pred, labels=["a", "b", "c"])
    assert m.accuracy == pytest.approx(0.75)
    assert m.per_class["b"]["recall"] == 0.0 and m.per_class["b"]["support"] == 1
    assert m.per_class["a"]["precision"] == pytest.approx(2 / 3)
    assert m.worst_classes(1)[0][0] == "b"
    assert m.per_class_table()[0]["label"] == "a"


def test_metrics_roundtrip_and_report_text() -> None:
    m = compute_classification_metrics(["x", "y"], ["x", "x"])
    restored = ClassificationMetrics.from_dict(m.to_dict())
    assert restored.accuracy == m.accuracy and restored.confusion == m.confusion
    text = classification_report_text(m)
    assert "accuracy" in text and "macro avg" in text and "x" in text


def test_metrics_validation() -> None:
    with pytest.raises(ValueError):
        compute_classification_metrics(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        compute_classification_metrics([], [])


def test_report_writers(tmp_path: Path) -> None:
    m = compute_classification_metrics(["a", "b", "b"], ["a", "b", "a"])
    retrieval = {
        "hybrid": RetrievalMetrics(
            k=5, precision_at_k=0.5, recall_at_k=0.4, mrr=0.9, ndcg_at_k=0.7, n_queries=3
        )
    }
    md = write_markdown_report(
        tmp_path / "r.md", classification=m, retrieval=retrieval, metadata={"model": "svc"}
    )
    content = md.read_text(encoding="utf-8")
    assert "## Classification report" in content and "## Retrieval evaluation" in content
    assert "| hybrid | 5 |" in content and "| model | svc |" in content
    js = write_json_report(
        tmp_path / "r.json", classification=m, retrieval=retrieval, metadata={"model": "svc"}
    )
    data = read_json(js)
    assert data["classification"]["accuracy"] == pytest.approx(2 / 3)
    assert data["retrieval"]["hybrid"]["mrr"] == 0.9 and data["metadata"]["model"] == "svc"
