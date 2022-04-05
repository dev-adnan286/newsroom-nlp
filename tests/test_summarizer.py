from __future__ import annotations

import pytest

from newsroom.summarization.extractive import ExtractiveSummarizer, Summary, summarize


def test_centroid_summary_keeps_order_and_length(sample_article: str) -> None:
    summary = ExtractiveSummarizer("centroid").summarize(sample_article, n_sentences=2)
    assert isinstance(summary, Summary)
    assert len(summary.sentences) == 2 and summary.indices == sorted(summary.indices)
    assert all(s in sample_article for s in summary.sentences)
    assert summary.text == " ".join(summary.sentences)
    assert len(summary.scores) == 2


def test_textrank_summary(sample_article: str) -> None:
    summary = ExtractiveSummarizer("textrank").summarize(sample_article, n_sentences=3)
    assert len(summary.sentences) == 3
    assert "interest rates" in summary.text or "inflation" in summary.text


def test_ratio_and_bounds(sample_article: str) -> None:
    summarizer = ExtractiveSummarizer()
    assert len(summarizer.summarize(sample_article, ratio=0.4).sentences) == 2
    assert len(summarizer.summarize(sample_article, n_sentences=50).sentences) == 5
    with pytest.raises(ValueError):
        summarizer.summarize(sample_article, ratio=1.5)
    with pytest.raises(ValueError):
        ExtractiveSummarizer("abstractive")


def test_edge_cases() -> None:
    assert ExtractiveSummarizer().summarize("").sentences == []
    single = ExtractiveSummarizer().summarize("Only one sentence here.", 3)
    assert single.sentences == ["Only one sentence here."] and single.scores == [1.0]
    stop_only = ExtractiveSummarizer().summarize("The and of. It is the. Of and the.", 1)
    assert len(stop_only.sentences) == 1


def test_functional_helper(sample_article: str) -> None:
    out = summarize(sample_article, 1)
    assert out and out in sample_article
