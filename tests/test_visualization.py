from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from newsroom.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_embedding_projection,
    plot_top_terms,
    project_2d,
)


def test_confusion_matrix_plot(tmp_path: Path) -> None:
    out = plot_confusion_matrix([[5, 1], [2, 7]], ["a", "b"], tmp_path / "cm.png")
    assert out.exists() and out.stat().st_size > 1000
    out2 = plot_confusion_matrix(
        [[5, 1], [2, 7]], ["a", "b"], tmp_path / "cm_raw.png", normalize=False
    )
    assert out2.exists()
    with pytest.raises(ValueError):
        plot_confusion_matrix([[1, 2, 3]], ["a"], tmp_path / "bad.png")


def test_class_distribution_and_top_terms(tmp_path: Path) -> None:
    assert plot_class_distribution(["a", "b", "a", "c"], tmp_path / "dist.png").exists()
    terms = {"a": [("x", 1.0), ("y", 0.5)], "b": [("z", 0.9)], "c": [("w", 0.1)], "d": [("v", 0.2)]}
    assert plot_top_terms(terms, tmp_path / "terms.png", n_terms=1).exists()
    with pytest.raises(ValueError):
        plot_top_terms({}, tmp_path / "empty.png")


def test_projection_and_scatter(tmp_path: Path) -> None:
    rng = np.random.RandomState(0)
    x = rng.rand(40, 6)
    labels = ["a"] * 20 + ["b"] * 20
    assert project_2d(x, method="pca").shape == (40, 2)
    assert project_2d(x, method="svd").shape == (40, 2)
    with pytest.raises(ValueError):
        project_2d(x, method="tsne")
    out = plot_embedding_projection(x, labels, tmp_path / "proj.png", max_points=25)
    assert out.exists()
    with pytest.raises(ValueError):
        plot_embedding_projection(x, labels[:10], tmp_path / "bad.png")
