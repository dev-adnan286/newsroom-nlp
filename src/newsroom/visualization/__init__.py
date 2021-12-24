"""Static PNG plots rendered with the Agg backend."""

from newsroom.visualization.plots import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_embedding_projection,
    plot_top_terms,
)

__all__ = [
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_embedding_projection",
    "plot_top_terms",
]
