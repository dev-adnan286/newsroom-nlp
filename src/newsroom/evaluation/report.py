"""Write evaluation reports as Markdown and JSON."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from newsroom.evaluation.metrics import ClassificationMetrics
from newsroom.evaluation.search_eval import RetrievalMetrics
from newsroom.utils.io import write_json


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def classification_markdown(
    metrics: ClassificationMetrics, *, title: str = "Classification report"
) -> str:
    """Render classification metrics (summary, per-class table, confusion matrix)."""
    lines = [f"## {title}", ""]
    lines.append(
        _md_table(
            ["Metric", "Value"],
            [
                ["Accuracy", f"{metrics.accuracy:.4f}"],
                ["Macro F1", f"{metrics.macro_f1:.4f}"],
                ["Micro F1", f"{metrics.micro_f1:.4f}"],
                ["Weighted F1", f"{metrics.weighted_f1:.4f}"],
                ["Samples", metrics.n_samples],
            ],
        )
    )
    lines += ["", "### Per-class", ""]
    lines.append(
        _md_table(
            ["Label", "Precision", "Recall", "F1", "Support"],
            [
                [
                    lab,
                    f"{pc['precision']:.3f}",
                    f"{pc['recall']:.3f}",
                    f"{pc['f1']:.3f}",
                    pc["support"],
                ]
                for lab, pc in ((lab, metrics.per_class[lab]) for lab in metrics.labels)
            ],
        )
    )
    lines += ["", "### Confusion matrix (rows = true, cols = predicted)", ""]
    lines.append(
        _md_table(
            ["", *metrics.labels],
            [[lab, *row] for lab, row in zip(metrics.labels, metrics.confusion, strict=True)],
        )
    )
    return "\n".join(lines)


def retrieval_markdown(
    results: Mapping[str, RetrievalMetrics], *, title: str = "Retrieval evaluation"
) -> str:
    """Render one row per retrieval mode."""
    rows = [
        [
            mode,
            m.k,
            f"{m.precision_at_k:.3f}",
            f"{m.recall_at_k:.3f}",
            f"{m.mrr:.3f}",
            f"{m.ndcg_at_k:.3f}",
            m.n_queries,
        ]
        for mode, m in results.items()
    ]
    return "\n".join(
        [
            f"## {title}",
            "",
            _md_table(["Mode", "k", "P@k", "R@k", "MRR", "nDCG@k", "Queries"], rows),
        ]
    )


def write_markdown_report(
    path: str | Path,
    *,
    classification: ClassificationMetrics | None = None,
    retrieval: Mapping[str, RetrievalMetrics] | None = None,
    metadata: Mapping[str, Any] | None = None,
    title: str = "Newsroom NLP evaluation report",
) -> Path:
    """Write a Markdown report combining the provided sections."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        f"# {title}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
    ]
    if metadata:
        sections.append(_md_table(["Key", "Value"], [[k, v] for k, v in metadata.items()]))
        sections.append("")
    if classification is not None:
        sections.append(classification_markdown(classification))
        sections.append("")
    if retrieval:
        sections.append(retrieval_markdown(retrieval))
        sections.append("")
    p.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return p


def write_json_report(
    path: str | Path,
    *,
    classification: ClassificationMetrics | None = None,
    retrieval: Mapping[str, RetrievalMetrics] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write the same content as JSON for programmatic consumption."""
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    if classification is not None:
        payload["classification"] = classification.to_dict()
    if retrieval:
        payload["retrieval"] = {mode: m.to_dict() for mode, m in retrieval.items()}
    return write_json(payload, path)


__all__ = [
    "classification_markdown",
    "retrieval_markdown",
    "write_json_report",
    "write_markdown_report",
]
