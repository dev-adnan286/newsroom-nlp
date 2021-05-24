"""Model explanations derived from linear coefficients."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion, Pipeline


def _feature_names(features: Any) -> np.ndarray:
    """Feature names for a TfidfVectorizer or a word/char FeatureUnion."""
    if isinstance(features, FeatureUnion):
        names: list[str] = []
        for name, transformer in features.transformer_list:
            names.extend(f"{name}:{f}" for f in transformer.get_feature_names_out())
        return np.asarray(names, dtype=object)
    return np.asarray(features.get_feature_names_out(), dtype=object)


def _coefficients(clf: Any) -> tuple[np.ndarray, list[str]]:
    """Extract a ``(n_classes, n_features)`` coefficient matrix from a linear estimator."""
    if isinstance(clf, CalibratedClassifierCV):
        members = [cc.estimator for cc in clf.calibrated_classifiers_]
        coefs = np.mean([np.asarray(m.coef_) for m in members], axis=0)
        classes = [str(c) for c in clf.classes_]
    elif hasattr(clf, "coef_"):
        coefs = np.asarray(clf.coef_)
        classes = [str(c) for c in clf.classes_]
    elif hasattr(clf, "feature_log_prob_"):
        coefs = np.asarray(clf.feature_log_prob_)
        classes = [str(c) for c in clf.classes_]
    else:
        raise TypeError(f"Estimator {type(clf).__name__} does not expose linear coefficients")
    if coefs.ndim == 1:
        coefs = np.stack([-coefs, coefs], axis=0)
    if coefs.shape[0] == 1 and len(classes) == 2:
        coefs = np.vstack([-coefs, coefs])
    return coefs, classes


def top_terms_per_class(
    pipeline: Pipeline, n_terms: int = 15, *, strip_prefix: bool = True
) -> dict[str, list[tuple[str, float]]]:
    """Return the most positively weighted features for each class.

    Args:
        pipeline: A fitted ``clean -> features -> clf`` pipeline.
        n_terms: Number of terms to return per class.
        strip_prefix: Drop the ``word:``/``char:`` prefix from feature names.
    """
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    names = _feature_names(features)
    coefs, classes = _coefficients(clf)
    if coefs.shape[1] != len(names):
        raise ValueError("Coefficient/feature length mismatch; was the pipeline fitted?")
    out: dict[str, list[tuple[str, float]]] = {}
    for row, label in zip(coefs, classes, strict=True):
        idx = np.argsort(row)[::-1][:n_terms]
        terms = []
        for i in idx:
            name = str(names[i])
            if strip_prefix and ":" in name:
                name = name.split(":", 1)[1]
            terms.append((name, float(row[i])))
        out[label] = terms
    return out


def explain_document(
    pipeline: Pipeline, text: str, n_terms: int = 10, label: str | None = None
) -> dict[str, Any]:
    """Explain a prediction as the features that contributed most to the chosen class.

    Contribution of feature ``j`` for class ``c`` is ``x_j * w_cj``; the top ``n_terms`` by
    contribution are returned alongside the predicted label.
    """
    cleaned = pipeline.named_steps["clean"].transform([text])
    features = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    x = features.transform(cleaned)
    if not sparse.issparse(x):
        x = sparse.csr_matrix(x)
    x = x.tocsr()
    coefs, classes = _coefficients(clf)
    predicted = str(pipeline.predict([text])[0])
    target = label or predicted
    if target not in classes:
        raise ValueError(f"Unknown label {target!r}; expected one of {classes}")
    row = coefs[classes.index(target)]
    names = _feature_names(features)
    contributions = x.multiply(row).tocsr()
    idx = contributions.indices
    vals = contributions.data
    order = np.argsort(vals)[::-1][:n_terms]
    top = []
    for k in order:
        name = str(names[idx[k]])
        if ":" in name:
            name = name.split(":", 1)[1]
        top.append({"term": name, "contribution": float(vals[k])})
    return {"predicted_label": predicted, "explained_label": target, "top_features": top}


__all__ = ["explain_document", "top_terms_per_class"]
