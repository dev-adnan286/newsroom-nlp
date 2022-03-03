from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from newsroom.preprocessing.stemmer import PorterStemmer, get_stemmer, stem_tokens
from newsroom.preprocessing.stopwords import ENGLISH_STOPWORDS
from newsroom.preprocessing.text import (
    PreprocessConfig,
    TextCleaner,
    clean_text,
    expand_contractions,
    ngram_range,
    ngrams,
    normalize_unicode,
    preprocess,
    remove_stopwords,
    replace_numbers,
    sentence_split,
    tokenize,
)


def test_normalize_unicode_strips_accents_and_quotes() -> None:
    assert normalize_unicode("Café “quoted” – dash") == 'Cafe "quoted" - dash'


def test_expand_contractions() -> None:
    assert (
        expand_contractions("I can't believe they're here, it's fine")
        == "I cannot believe they are here, it is fine"
    )
    assert expand_contractions("She won't go") == "She will not go"


def test_clean_text_replaces_urls_emails_numbers() -> None:
    raw = "Visit https://example.com/x?y=1 or mail me@site.org, sales rose 12.5% to 1,200 units <b>now</b>"
    cleaned = clean_text(raw, numbers=True)
    assert "<url>" in cleaned and "<email>" in cleaned and "<num>" in cleaned
    assert "example.com" not in cleaned and "<b>" not in cleaned
    assert cleaned == cleaned.lower()


def test_clean_text_handles_non_string() -> None:
    assert clean_text(None) == ""  # type: ignore[arg-type]
    assert clean_text(42) == "42"  # type: ignore[arg-type]


def test_tokenize_keeps_placeholders_and_apostrophes() -> None:
    tokens = tokenize("the <url> isn't broken, o'neil scored 3")
    assert tokens == ["the", "<url>", "isn't", "broken", "o'neil", "scored", "3"]
    assert tokenize("a bb ccc", min_length=2) == ["bb", "ccc"]


def test_remove_stopwords_and_ngrams() -> None:
    assert remove_stopwords(["the", "quick", "fox", "is"]) == ["quick", "fox"]
    assert "the" in ENGLISH_STOPWORDS
    assert ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
    assert ngrams(["a"], 2) == []
    assert ngram_range(["a", "b", "c"], 1, 2) == ["a", "b", "c", "a b", "b c"]
    with pytest.raises(ValueError):
        ngrams(["a"], 0)


def test_replace_numbers_keeps_alphanumeric_tokens() -> None:
    assert replace_numbers("5G networks and 3 devices") == "5G networks and <num> devices"


def test_sentence_split() -> None:
    text = "First sentence here. Second one follows! Is this third? Yes."
    assert sentence_split(text) == [
        "First sentence here.",
        "Second one follows!",
        "Is this third?",
        "Yes.",
    ]
    assert sentence_split("   ") == []


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("running", "run"),
        ("hopping", "hop"),
        ("relational", "relat"),
        ("conditional", "condit"),
        ("happiness", "happi"),
        ("generalization", "gener"),
        ("agreed", "agre"),
        ("cats", "cat"),
        ("is", "is"),
    ],
)
def test_porter_stemmer(word: str, expected: str) -> None:
    assert PorterStemmer().stem(word) == expected


def test_stem_helpers() -> None:
    assert stem_tokens(["cats", "running"]) == ["cat", "run"]
    stemmer = get_stemmer(prefer_nltk=False)
    assert stemmer("walked") == "walk"


def test_preprocess_pipeline() -> None:
    tokens = preprocess("The runners were running quickly towards https://x.io in 2024!")
    assert "<url>" in tokens
    assert "the" not in tokens and "were" not in tokens
    assert "run" in tokens or "runner" in tokens
    cfg = PreprocessConfig(stem=False, remove_stopwords=False, replace_numbers=True)
    assert "the" in preprocess("The 5 dogs", cfg) and "<num>" in preprocess("The 5 dogs", cfg)


def test_text_cleaner_in_sklearn_pipeline() -> None:
    docs = ["Stocks rallied sharply on strong earnings.", "The striker scored twice in the derby."]
    pipe = Pipeline([("clean", TextCleaner()), ("tfidf", TfidfVectorizer())])
    matrix = pipe.fit_transform(docs)
    assert matrix.shape[0] == 2
    cleaned = TextCleaner().transform(np.asarray(docs))
    assert isinstance(cleaned, list) and all(isinstance(c, str) for c in cleaned)
    assert TextCleaner().get_params()["stem"] is True
    assert list(TextCleaner().get_feature_names_out()) == ["cleaned_text"]
