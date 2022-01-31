from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from newsroom import __version__
from newsroom.cli import app

runner = CliRunner()


def test_version_and_help() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and __version__ in result.output
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0 and "generate-corpus" in result.output and "serve" in result.output


def test_generate_train_evaluate_flow(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.csv"
    result = runner.invoke(
        app, ["generate-corpus", "--n-docs", "120", "--output", str(corpus), "--seed", "3"]
    )
    assert result.exit_code == 0, result.output
    assert corpus.exists() and "120 documents" in result.output

    model = tmp_path / "clf.joblib"
    result = runner.invoke(
        app,
        [
            "train",
            "--input",
            str(corpus),
            "--output",
            str(model),
            "--model",
            "complement_nb",
            "--no-grid-search",
        ],
    )
    assert result.exit_code == 0, result.output
    assert model.exists() and "accuracy" in result.output

    reports = tmp_path / "reports"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input",
            str(corpus),
            "--model",
            str(model),
            "--report-dir",
            str(reports),
            "--no-plots",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (reports / "evaluation.md").exists() and (reports / "evaluation.json").exists()
    data = json.loads((reports / "evaluation.json").read_text(encoding="utf-8"))
    assert data["classification"]["accuracy"] > 0.5


def test_train_with_yaml_config(tmp_path: Path, corpus_csv: Path) -> None:
    cfg = tmp_path / "clf.yaml"
    cfg.write_text(
        "model: sgd\ncalibrate: false\ngrid_search: false\nvectorizer:\n  min_df: 1\n  use_char_ngrams: false\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "train",
            "--input",
            str(corpus_csv),
            "--output",
            str(tmp_path / "m.joblib"),
            "--config",
            str(cfg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(sgd)" in result.output


def test_build_index_and_search(tmp_path: Path, corpus_csv: Path) -> None:
    index = tmp_path / "index"
    result = runner.invoke(
        app,
        ["build-index", "--input", str(corpus_csv), "--output", str(index), "--dimension", "32"],
    )
    assert result.exit_code == 0, result.output
    assert (index / "engine_meta.json").exists()

    result = runner.invoke(
        app,
        ["search", "quarterly earnings beat expectations", "--index", str(index), "--top-k", "3"],
    )
    assert result.exit_code == 0, result.output
    assert "business" in result.output

    result = runner.invoke(
        app,
        [
            "search",
            "vaccine trial",
            "--index",
            str(index),
            "--top-k",
            "2",
            "--mode",
            "bm25",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"doc_id"' in result.output

    result = runner.invoke(
        app,
        [
            "evaluate",
            "--input",
            str(corpus_csv),
            "--model",
            str(tmp_path / "none.joblib"),
            "--index",
            str(index),
        ],
    )
    assert result.exit_code == 1 and "error" in result.output


def test_keywords_and_summarize(tmp_path: Path, sample_article: str) -> None:
    result = runner.invoke(app, ["keywords", sample_article, "--top-k", "3"])
    assert result.exit_code == 0, result.output
    assert len([line for line in result.output.strip().splitlines() if line.strip()]) == 3

    article = tmp_path / "a.txt"
    article.write_text(sample_article, encoding="utf-8")
    result = runner.invoke(
        app, ["summarize", "--file", str(article), "--sentences", "2", "--method", "textrank"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.strip()

    result = runner.invoke(app, ["summarize"])
    assert result.exit_code == 1
    result = runner.invoke(app, ["keywords", "--file", str(tmp_path / "missing.txt")])
    assert result.exit_code == 1


def test_cluster_and_dedup(tmp_path: Path, corpus_csv: Path) -> None:
    out = tmp_path / "clusters.json"
    result = runner.invoke(
        app,
        [
            "cluster",
            "--input",
            str(corpus_csv),
            "--k",
            "4",
            "--dimension",
            "16",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["k"] == 4 and len(data["clusters"]) == 4

    result = runner.invoke(
        app, ["dedup", "--input", str(corpus_csv), "--threshold", "0.3", "--limit", "3"]
    )
    assert result.exit_code == 0, result.output
    assert "near-duplicate pairs" in result.output


def test_missing_input_reports_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["train", "--input", str(tmp_path / "missing.csv"), "--output", str(tmp_path / "m.joblib")],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_run(app_obj, **kwargs):  # type: ignore[no-untyped-def]
        calls["app"] = app_obj
        calls.update(kwargs)

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = runner.invoke(app, ["serve", "--port", "9999", "--artifacts", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls["port"] == 9999 and calls["app"] is not None
