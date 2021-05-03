"""Newsroom NLP: a text-intelligence toolkit and API for news and article text.

The package is organised around small, composable modules:

* :mod:`newsroom.data` - synthetic corpus generation and dataset loading
* :mod:`newsroom.preprocessing` - normalisation, tokenisation, stemming
* :mod:`newsroom.features` / :mod:`newsroom.embeddings` - vectorisers and embedding providers
* :mod:`newsroom.classification` - training, prediction and explanation of linear classifiers
* :mod:`newsroom.search` - BM25, vector and hybrid retrieval
* :mod:`newsroom.keywords`, :mod:`newsroom.topics`, :mod:`newsroom.summarization`,
  :mod:`newsroom.similarity` - analysis utilities
* :mod:`newsroom.evaluation` - classification and retrieval metrics plus report writers
* :mod:`newsroom.api` - FastAPI service exposing the toolkit over HTTP
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
