"""Okapi BM25 ranking implemented with sparse matrices."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
from scipy import sparse

from newsroom.preprocessing.text import PreprocessConfig, preprocess

Tokenizer = Callable[[str], list[str]]


def default_tokenizer(text: str) -> list[str]:
    """Clean, drop stopwords and stem (shared with the rest of the toolkit)."""
    return preprocess(
        text, PreprocessConfig(stem=True, remove_stopwords=True, replace_numbers=False)
    )


class BM25:
    """Okapi BM25 with the standard ``k1``/``b`` parameters.

    Scores are precomputed into a sparse ``(n_docs, vocab)`` matrix at index time so a
    query reduces to summing a handful of columns.
    """

    def __init__(
        self, k1: float = 1.5, b: float = 0.75, *, tokenizer: Tokenizer | None = None
    ) -> None:
        if k1 < 0 or not 0.0 <= b <= 1.0:
            raise ValueError("k1 must be >= 0 and b in [0, 1]")
        self.k1 = k1
        self.b = b
        self._tokenizer: Tokenizer = tokenizer or default_tokenizer
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self._scores: sparse.csc_matrix | None = None
        self._ids: list[str] = []
        self._avgdl: float = 0.0
        self._doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)

    # -- indexing ---------------------------------------------------------------------
    def fit(self, ids: Sequence[str], documents: Sequence[str]) -> BM25:
        """Tokenise ``documents`` and precompute BM25 term weights."""
        if len(ids) != len(documents):
            raise ValueError("ids and documents must be the same length")
        if len(documents) == 0:
            raise ValueError("Cannot fit BM25 on an empty corpus")
        tokenised = [self._tokenizer(doc) for doc in documents]
        vocab: dict[str, int] = {}
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        doc_lengths = np.zeros(len(tokenised), dtype=np.float32)
        for row, tokens in enumerate(tokenised):
            doc_lengths[row] = len(tokens)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok, tf in counts.items():
                col = vocab.setdefault(tok, len(vocab))
                rows.append(row)
                cols.append(col)
                vals.append(float(tf))
        n_docs = len(tokenised)
        n_terms = max(1, len(vocab))
        tf_matrix = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n_docs, n_terms), dtype=np.float32
        )
        df = np.asarray((tf_matrix > 0).sum(axis=0)).ravel()
        idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = float(doc_lengths.mean()) if n_docs else 0.0
        avgdl = avgdl if avgdl > 0 else 1.0

        tf_coo = tf_matrix.tocoo()
        length_norm = self.k1 * (1.0 - self.b + self.b * doc_lengths[tf_coo.row] / avgdl)
        weights = idf[tf_coo.col] * (tf_coo.data * (self.k1 + 1.0)) / (tf_coo.data + length_norm)
        score_matrix = sparse.csr_matrix(
            (weights, (tf_coo.row, tf_coo.col)), shape=tf_matrix.shape, dtype=np.float32
        )

        self._vocab = vocab
        self._idf = idf
        self._scores = score_matrix.tocsc()
        self._ids = [str(i) for i in ids]
        self._avgdl = avgdl
        self._doc_lengths = doc_lengths
        return self

    @property
    def is_fitted(self) -> bool:
        return self._scores is not None

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocab)

    def __len__(self) -> int:
        return len(self._ids)

    # -- querying ----------------------------------------------------------------------
    def score(self, query: str) -> np.ndarray:
        """Return BM25 scores for every indexed document (dense vector)."""
        if self._scores is None:
            raise RuntimeError("BM25 must be fitted before scoring")
        cols = [self._vocab[t] for t in self._tokenizer(query) if t in self._vocab]
        if not cols:
            return np.zeros(len(self._ids), dtype=np.float32)
        return np.asarray(self._scores[:, cols].sum(axis=1)).ravel().astype(np.float32)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` for the best ``top_k`` documents with positive scores."""
        scores = self.score(query)
        if top_k <= 0 or scores.size == 0:
            return []
        k = min(top_k, scores.size)
        part = np.argpartition(-scores, k - 1)[:k]
        order = part[np.argsort(-scores[part], kind="stable")]
        return [(self._ids[i], float(scores[i])) for i in order if scores[i] > 0]

    # -- persistence -------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Persist the index as ``bm25_scores.npz`` + ``bm25_meta.json``."""
        if self._scores is None:
            raise RuntimeError("Nothing to save: BM25 is not fitted")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(d / "bm25_scores.npz", self._scores.tocsr())
        with (d / "bm25_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "ids": self._ids,
                    "vocab": self._vocab,
                    "idf": self._idf.tolist(),
                    "avgdl": self._avgdl,
                    "doc_lengths": self._doc_lengths.tolist(),
                },
                fh,
                ensure_ascii=False,
            )
        return d

    @classmethod
    def load(cls, directory: str | Path, *, tokenizer: Tokenizer | None = None) -> BM25:
        """Load an index created by :meth:`save`."""
        d = Path(directory)
        meta_path = d / "bm25_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No BM25 index found in {d}")
        with meta_path.open("r", encoding="utf-8") as fh:
            meta = json.load(fh)
        obj = cls(k1=float(meta["k1"]), b=float(meta["b"]), tokenizer=tokenizer)
        obj._scores = sparse.load_npz(d / "bm25_scores.npz").tocsc()
        obj._ids = list(meta["ids"])
        obj._vocab = dict(meta["vocab"])
        obj._idf = np.asarray(meta["idf"], dtype=np.float32)
        obj._avgdl = float(meta["avgdl"])
        obj._doc_lengths = np.asarray(meta["doc_lengths"], dtype=np.float32)
        return obj


__all__ = ["BM25", "default_tokenizer"]
