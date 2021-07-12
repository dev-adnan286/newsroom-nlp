"""TextRank keyword extraction using a co-occurrence graph and numpy power iteration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from newsroom.preprocessing.stemmer import PorterStemmer
from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import clean_text, tokenize


@dataclass(frozen=True)
class Keyword:
    """A keyword (or key-phrase) with its score."""

    text: str
    score: float


def pagerank(
    adjacency: np.ndarray, *, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Weighted PageRank via power iteration.

    Args:
        adjacency: Symmetric non-negative weight matrix ``(n, n)``.
        damping: Random-jump damping factor.
        max_iter: Maximum iterations.
        tol: L1 convergence threshold.
    """
    n = adjacency.shape[0]
    if n == 0:
        return np.zeros(0)
    weights = np.asarray(adjacency, dtype=np.float64)
    out_strength = weights.sum(axis=1)
    dangling = out_strength == 0
    out_strength = np.where(dangling, 1.0, out_strength)
    transition = weights / out_strength[:, None]  # row-stochastic (rows with no edges are zero)
    ranks = np.full(n, 1.0 / n)
    teleport = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        dangling_mass = ranks[dangling].sum() if dangling.any() else 0.0
        new_ranks = (1 - damping) * teleport + damping * (
            transition.T @ ranks + dangling_mass * teleport
        )
        if np.abs(new_ranks - ranks).sum() < tol:
            ranks = new_ranks
            break
        ranks = new_ranks
    return ranks


class TextRankExtractor:
    """Extract keywords/key-phrases from a single document.

    Candidate tokens (non-stopword alphabetic words with length >= ``min_word_length``) are
    connected when they co-occur within a sliding window; PageRank scores the graph, then
    adjacent high-scoring words are merged back into phrases from the original text.
    """

    def __init__(
        self,
        window: int = 4,
        *,
        damping: float = 0.85,
        min_word_length: int = 3,
        max_phrase_words: int = 3,
        stem: bool = True,
        stopwords: frozenset[str] | None = None,
    ) -> None:
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.damping = damping
        self.min_word_length = min_word_length
        self.max_phrase_words = max_phrase_words
        self.stem = stem
        self.stopwords = stopwords or ENGLISH_STOPWORDS
        self._stemmer = PorterStemmer()

    def _candidates(self, text: str) -> tuple[list[str], list[str | None]]:
        """Return (all tokens, candidate key per position or None)."""
        tokens = tokenize(clean_text(text, numbers=False))
        keys: list[str | None] = []
        for tok in tokens:
            if (
                tok.startswith("<")
                or len(tok) < self.min_word_length
                or tok in self.stopwords
                or not tok.isalpha()
            ):
                keys.append(None)
            else:
                keys.append(self._stemmer.stem(tok) if self.stem else tok)
        return tokens, keys

    def extract(self, text: str, top_k: int = 10, *, phrases: bool = True) -> list[Keyword]:
        """Return up to ``top_k`` keywords ordered by score."""
        tokens, keys = self._candidates(text)
        vocab: dict[str, int] = {}
        surface: dict[str, dict[str, int]] = {}
        for tok, key in zip(tokens, keys, strict=True):
            if key is None:
                continue
            vocab.setdefault(key, len(vocab))
            surface.setdefault(key, {})
            surface[key][tok] = surface[key].get(tok, 0) + 1
        if not vocab:
            return []
        n = len(vocab)
        adjacency = np.zeros((n, n), dtype=np.float64)
        positions = [(i, keys[i]) for i in range(len(keys)) if keys[i] is not None]
        for a in range(len(positions)):
            pos_a, key_a = positions[a]
            for b in range(a + 1, len(positions)):
                pos_b, key_b = positions[b]
                if pos_b - pos_a >= self.window:
                    break
                if key_a == key_b:
                    continue
                i, j = vocab[key_a], vocab[key_b]  # type: ignore[index]
                adjacency[i, j] += 1.0
                adjacency[j, i] += 1.0
        scores = pagerank(adjacency, damping=self.damping)
        key_scores = {key: float(scores[idx]) for key, idx in vocab.items()}

        def _best_surface(key: str) -> str:
            forms = surface[key]
            return max(forms.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]

        if not phrases or self.max_phrase_words <= 1:
            ranked = sorted(key_scores.items(), key=lambda kv: -kv[1])[:top_k]
            return [Keyword(_best_surface(k), s) for k, s in ranked]

        # Canonical TextRank post-processing: keep the top third of ranked words, then merge
        # runs of adjacent selected words in the original token stream into phrases.
        n_selected = max(top_k, int(np.ceil(len(key_scores) / 3)))
        selected = {k for k, _ in sorted(key_scores.items(), key=lambda kv: -kv[1])[:n_selected]}
        phrase_scores: dict[tuple[str, ...], float] = {}
        phrase_surface: dict[tuple[str, ...], dict[str, int]] = {}
        i = 0
        while i < len(tokens):
            if keys[i] is None or keys[i] not in selected:
                i += 1
                continue
            j = i
            while (
                j < len(tokens)
                and keys[j] is not None
                and keys[j] in selected
                and (j - i) < self.max_phrase_words
            ):
                j += 1
            phrase_keys = tuple(k for k in keys[i:j] if k is not None)
            phrase_text = " ".join(tokens[i:j])
            phrase_scores[phrase_keys] = sum(key_scores[k] for k in phrase_keys)
            forms = phrase_surface.setdefault(phrase_keys, {})
            forms[phrase_text] = forms.get(phrase_text, 0) + 1
            i = j
        ranked_phrases = sorted(phrase_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        # Skip phrases whose stems are fully contained in an already chosen phrase.
        chosen: list[Keyword] = []
        chosen_keys: list[tuple[str, ...]] = []
        for phrase_keys, score in ranked_phrases:
            if any(set(phrase_keys) <= set(ck) for ck in chosen_keys):
                continue
            forms = phrase_surface[phrase_keys]
            surface_text = max(forms.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]
            chosen.append(Keyword(surface_text, float(score)))
            chosen_keys.append(phrase_keys)
            if len(chosen) >= top_k:
                break
        return chosen


def extract_keywords(text: str, top_k: int = 10, **kwargs: object) -> list[Keyword]:
    """Functional shortcut for :class:`TextRankExtractor`."""
    return TextRankExtractor(**kwargs).extract(text, top_k=top_k)  # type: ignore[arg-type]


def extract_keywords_batch(
    texts: Sequence[str], top_k: int = 10, **kwargs: object
) -> list[list[Keyword]]:
    """Extract keywords for many documents with one extractor instance."""
    extractor = TextRankExtractor(**kwargs)  # type: ignore[arg-type]
    return [extractor.extract(t, top_k=top_k) for t in texts]


__all__ = ["Keyword", "TextRankExtractor", "extract_keywords", "extract_keywords_batch", "pagerank"]
