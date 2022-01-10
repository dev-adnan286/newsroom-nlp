"""Matplotlib figures saved to disk (headless ``Agg`` backend, no display required)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA, TruncatedSVD  # noqa: E402

_PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
]


def _finish(fig: plt.Figure, path: str | Path, dpi: int) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: str | Path,
    *,
    normalize: bool = True,
    title: str = "Confusion matrix",
    dpi: int = 120,
) -> Path:
    """Heatmap of a confusion matrix with annotated cells."""
    cm = np.asarray(confusion, dtype=np.float64)
    if cm.shape[0] != cm.shape[1] or cm.shape[0] != len(labels):
        raise ValueError("confusion must be square and match labels")
    display = cm
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display = cm / np.where(row_sums == 0, 1.0, row_sums)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(labels), 0.9 + 0.8 * len(labels)))
    im = ax.imshow(display, cmap="Blues", vmin=0, vmax=display.max() if display.size else 1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = display.max() / 2 if display.size else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = f"{display[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if display[i, j] > threshold else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, path, dpi)


def plot_class_distribution(
    labels: Sequence[str], path: str | Path, *, title: str = "Class distribution", dpi: int = 120
) -> Path:
    """Bar chart of label frequencies."""
    values, counts = np.unique(np.asarray([str(v) for v in labels]), return_counts=True)
    order = np.argsort(-counts)
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(values)), 4))
    ax.bar(
        values[order],
        counts[order],
        color=[_PALETTE[i % len(_PALETTE)] for i in range(len(values))],
    )
    ax.set_ylabel("Documents")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    for i, c in enumerate(counts[order]):
        ax.text(i, c, str(int(c)), ha="center", va="bottom", fontsize=8)
    return _finish(fig, path, dpi)


def project_2d(
    embeddings: np.ndarray, *, method: str = "pca", random_state: int = 42
) -> np.ndarray:
    """Reduce embeddings to two dimensions (``pca``, ``svd`` or optional ``umap``)."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("embeddings must be 2-D with at least two columns")
    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "umap-learn is not installed; use method='pca' or install the `umap` extra"
            ) from exc
        return np.asarray(umap.UMAP(n_components=2, random_state=random_state).fit_transform(x))
    if method == "svd":
        return np.asarray(TruncatedSVD(n_components=2, random_state=random_state).fit_transform(x))
    if method == "pca":
        return np.asarray(PCA(n_components=2, random_state=random_state).fit_transform(x))
    raise ValueError(f"Unknown projection method: {method!r}")


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: Sequence[str],
    path: str | Path,
    *,
    method: str = "pca",
    title: str | None = None,
    max_points: int = 3000,
    random_state: int = 42,
    dpi: int = 120,
) -> Path:
    """Scatter plot of a 2-D projection coloured by label."""
    x = np.asarray(embeddings, dtype=np.float32)
    labs = np.asarray([str(v) for v in labels])
    if x.shape[0] != labs.shape[0]:
        raise ValueError("embeddings and labels must have the same length")
    if x.shape[0] > max_points:
        idx = np.random.RandomState(random_state).choice(x.shape[0], max_points, replace=False)
        x, labs = x[idx], labs[idx]
    coords = project_2d(x, method=method, random_state=random_state)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, lab in enumerate(sorted(set(labs.tolist()))):
        mask = labs == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10,
            alpha=0.6,
            label=lab,
            color=_PALETTE[i % len(_PALETTE)],
        )
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_title(title or f"{method.upper()} projection of document embeddings")
    ax.set_xticks([])
    ax.set_yticks([])
    return _finish(fig, path, dpi)


def plot_top_terms(
    terms_by_class: Mapping[str, Sequence[tuple[str, float]]],
    path: str | Path,
    *,
    n_terms: int = 10,
    title: str = "Top weighted terms per class",
    dpi: int = 120,
) -> Path:
    """Grid of horizontal bar charts, one panel per class."""
    classes = list(terms_by_class)
    if not classes:
        raise ValueError("terms_by_class is empty")
    n_cols = min(3, len(classes))
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
    for i, cls in enumerate(classes):
        ax = axes[i // n_cols][i % n_cols]
        pairs = list(terms_by_class[cls])[:n_terms][::-1]
        names = [p[0] for p in pairs]
        weights = [float(p[1]) for p in pairs]
        ax.barh(names, weights, color=_PALETTE[i % len(_PALETTE)])
        ax.set_title(cls, fontsize=10)
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", labelsize=7)
    for j in range(len(classes), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    fig.suptitle(title)
    return _finish(fig, path, dpi)


__all__ = [
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_embedding_projection",
    "plot_top_terms",
    "project_2d",
]
