"""Evaluation: classification metrics, retrieval metrics and report writers."""

from newsroom.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from newsroom.evaluation.report import write_json_report, write_markdown_report
from newsroom.evaluation.search_eval import (
    RelevanceSet,
    RetrievalMetrics,
    build_relevance_set,
    evaluate_rankings,
    evaluate_search_engine,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "ClassificationMetrics",
    "RelevanceSet",
    "RetrievalMetrics",
    "build_relevance_set",
    "compute_classification_metrics",
    "evaluate_rankings",
    "evaluate_search_engine",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "write_json_report",
    "write_markdown_report",
]
