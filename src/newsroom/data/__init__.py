"""Datasets: synthetic newsroom corpus generation and external loaders."""

from newsroom.data.corpus import (
    CATEGORIES,
    CorpusGenerator,
    generate_corpus,
    load_corpus_csv,
    load_twenty_newsgroups,
    save_corpus_csv,
)

__all__ = [
    "CATEGORIES",
    "CorpusGenerator",
    "generate_corpus",
    "load_corpus_csv",
    "load_twenty_newsgroups",
    "save_corpus_csv",
]
