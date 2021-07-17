"""Text cleaning primitives and a scikit-learn compatible :class:`TextCleaner` transformer."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from newsroom.preprocessing.stemmer import PorterStemmer
from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS

_URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_NUMBER_RE = re.compile(r"(?<![\w-])[+-]?\d[\d,]*(?:\.\d+)?%?(?![\w-])")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|<[a-z]+>", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")

_CONTRACTIONS: dict[str, str] = {
    "can't": "cannot",
    "won't": "will not",
    "shan't": "shall not",
    "n't": " not",
    "'re": " are",
    "'ve": " have",
    "'ll": " will",
    "'d": " would",
    "'m": " am",
    "let's": "let us",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is",
    "there's": "there is",
    "here's": "here is",
    "he's": "he is",
    "she's": "she is",
}
_CONTRACTION_RE = re.compile(
    "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)),
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalisation, strip accents and collapse fancy quotes/dashes."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    return (
        text.replace("‘", "'")
        .replace("’", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace(" ", " ")
    )


def expand_contractions(text: str) -> str:
    """Expand common English contractions (``can't`` -> ``cannot``)."""

    def _replace(match: re.Match[str]) -> str:
        key = match.group(0).lower()
        return _CONTRACTIONS.get(key, match.group(0))

    return _CONTRACTION_RE.sub(_replace, text)


def strip_html(text: str) -> str:
    """Remove simple HTML tags."""
    return _HTML_TAG_RE.sub(" ", text)


def replace_urls(text: str, token: str = "<url>") -> str:
    """Replace URLs with a placeholder token."""
    return _URL_RE.sub(token, text)


def replace_emails(text: str, token: str = "<email>") -> str:
    """Replace e-mail addresses with a placeholder token."""
    return _EMAIL_RE.sub(token, text)


def replace_numbers(text: str, token: str = "<num>") -> str:
    """Replace numeric literals (including thousands separators/percentages) with a token."""
    return _NUMBER_RE.sub(token, text)


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    urls: bool = True,
    emails: bool = True,
    numbers: bool = False,
    contractions: bool = True,
    html: bool = True,
) -> str:
    """Normalise raw text into a compact, whitespace-collapsed string.

    Args:
        text: Raw input.
        lowercase: Lowercase the output.
        urls: Replace URLs with ``<url>``.
        emails: Replace e-mails with ``<email>``.
        numbers: Replace numbers with ``<num>``.
        contractions: Expand contractions.
        html: Strip HTML tags.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = normalize_unicode(text)
    if html:
        text = strip_html(text)
    if urls:
        text = replace_urls(text)
    if emails:
        text = replace_emails(text)
    if contractions:
        text = expand_contractions(text)
    if numbers:
        text = replace_numbers(text)
    if lowercase:
        text = text.lower()
    return _WHITESPACE_RE.sub(" ", text).strip()


def tokenize(text: str, *, min_length: int = 1) -> list[str]:
    """Regex tokenizer keeping alphanumerics, inner apostrophes and ``<placeholder>`` tokens."""
    return [t for t in _TOKEN_RE.findall(text) if len(t) >= min_length]


def remove_stopwords(
    tokens: Iterable[str], stopwords: frozenset[str] | set[str] | None = None
) -> list[str]:
    """Drop tokens that are in ``stopwords`` (default: built-in English list)."""
    sw = ENGLISH_STOPWORDS if stopwords is None else stopwords
    return [t for t in tokens if t not in sw]


def ngrams(tokens: Sequence[str], n: int, *, sep: str = " ") -> list[str]:
    """Return contiguous ``n``-grams joined by ``sep``."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return []
    return [sep.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_range(tokens: Sequence[str], low: int, high: int) -> list[str]:
    """Return all n-grams for ``low <= n <= high``."""
    out: list[str] = []
    for n in range(low, high + 1):
        out.extend(ngrams(tokens, n))
    return out


def sentence_split(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    text = _WHITESPACE_RE.sub(" ", normalize_unicode(text)).strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class PreprocessConfig:
    """Options for :func:`preprocess` and :class:`TextCleaner`."""

    lowercase: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    replace_numbers: bool = False
    expand_contractions: bool = True
    remove_stopwords: bool = True
    stem: bool = True
    min_token_length: int = 2
    extra_stopwords: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PreprocessConfig:
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "extra_stopwords" in known:
            known["extra_stopwords"] = frozenset(known["extra_stopwords"])
        return cls(**known)


_STEMMER = PorterStemmer()


def preprocess(text: str, config: PreprocessConfig | None = None) -> list[str]:
    """Full pipeline: clean -> tokenize -> stopwords -> stem. Returns tokens."""
    cfg = config or PreprocessConfig()
    cleaned = clean_text(
        text,
        lowercase=cfg.lowercase,
        urls=cfg.replace_urls,
        emails=cfg.replace_emails,
        numbers=cfg.replace_numbers,
        contractions=cfg.expand_contractions,
    )
    tokens = tokenize(cleaned, min_length=cfg.min_token_length)
    if cfg.remove_stopwords:
        stop = ENGLISH_STOPWORDS | cfg.extra_stopwords if cfg.extra_stopwords else ENGLISH_STOPWORDS
        tokens = remove_stopwords(tokens, stop)
    if cfg.stem:
        tokens = [t if t.startswith("<") else _STEMMER.stem(t) for t in tokens]
    return tokens


class TextCleaner(BaseEstimator, TransformerMixin):
    """scikit-learn transformer that turns raw documents into cleaned strings.

    It can be placed directly in front of a ``TfidfVectorizer`` inside a ``Pipeline``.
    Output is a list of space-joined preprocessed tokens so downstream vectorisers can
    use their default tokenisation.
    """

    def __init__(
        self,
        lowercase: bool = True,
        replace_urls: bool = True,
        replace_emails: bool = True,
        replace_numbers: bool = False,
        expand_contractions: bool = True,
        remove_stopwords: bool = True,
        stem: bool = True,
        min_token_length: int = 2,
    ) -> None:
        self.lowercase = lowercase
        self.replace_urls = replace_urls
        self.replace_emails = replace_emails
        self.replace_numbers = replace_numbers
        self.expand_contractions = expand_contractions
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.min_token_length = min_token_length

    def _config(self) -> PreprocessConfig:
        return PreprocessConfig(
            lowercase=self.lowercase,
            replace_urls=self.replace_urls,
            replace_emails=self.replace_emails,
            replace_numbers=self.replace_numbers,
            expand_contractions=self.expand_contractions,
            remove_stopwords=self.remove_stopwords,
            stem=self.stem,
            min_token_length=self.min_token_length,
        )

    def fit(self, X: Iterable[str], y: Any = None) -> TextCleaner:  # noqa: N803
        """No-op fit (stateless transformer)."""
        return self

    def transform(self, X: Iterable[str]) -> list[str]:  # noqa: N803
        """Return one cleaned string per input document."""
        cfg = self._config()
        if isinstance(X, np.ndarray):
            X = X.tolist()
        return [" ".join(preprocess(doc, cfg)) for doc in X]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(["cleaned_text"], dtype=object)


__all__ = [
    "PreprocessConfig",
    "TextCleaner",
    "clean_text",
    "expand_contractions",
    "ngram_range",
    "ngrams",
    "normalize_unicode",
    "preprocess",
    "remove_stopwords",
    "replace_emails",
    "replace_numbers",
    "replace_urls",
    "sentence_split",
    "strip_html",
    "tokenize",
]
