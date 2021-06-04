"""Typer command-line interface: ``newsroom <command> [options]``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from newsroom import __version__
from newsroom.config import get_settings
from newsroom.utils.io import read_yaml
from newsroom.utils.logging import setup_logging

app = typer.Typer(
    name="newsroom",
    help="Newsroom NLP - text intelligence toolkit for news and article text.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)
console = Console()
err_console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"newsroom-nlp {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option(
            "--log-level", envvar="NEWSROOM_LOG_LEVEL", help="DEBUG, INFO, WARNING, ERROR"
        ),
    ] = "INFO",
) -> None:
    setup_logging(log_level)


def _fail(message: str, code: int = 1) -> None:
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code)


def _load_config(path: Path | None, key: str | None = None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        _fail(f"config file not found: {path}")
    data = read_yaml(path)
    return data if key is None else data.get(key, data)


# --------------------------------------------------------------------------------------
@app.command("generate-corpus")
def generate_corpus_cmd(
    output: Annotated[Path, typer.Option("--output", "-o", help="CSV path to write")] = Path(
        "artifacts/corpus.csv"
    ),
    n_docs: Annotated[
        int, typer.Option("--n-docs", "-n", min=6, help="Number of documents")
    ] = 2000,
    seed: Annotated[int, typer.Option("--seed", help="Random seed")] = 42,
) -> None:
    """Generate the synthetic labelled news corpus as CSV."""
    from newsroom.data.corpus import generate_corpus, save_corpus_csv

    df = generate_corpus(n_docs=n_docs, seed=seed)
    save_corpus_csv(df, output)
    counts = df["label"].value_counts().to_dict()
    console.print(f"[green]Wrote[/] {len(df)} documents to {output}  {counts}")


@app.command("train")
def train_cmd(
    input: Annotated[Path, typer.Option("--input", "-i", help="Corpus CSV with text,label")] = Path(
        "artifacts/corpus.csv"
    ),
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Model artifact path (.joblib)")
    ] = Path("artifacts/classifier.joblib"),
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config (configs/classifier.yaml)")
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="linear_svc | logistic_regression | complement_nb | sgd"
        ),
    ] = None,
    grid_search: Annotated[
        bool | None,
        typer.Option(
            "--grid-search/--no-grid-search",
            help="Run a small GridSearchCV (default: from config, else off)",
        ),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
) -> None:
    """Train a text classifier and print hold-out metrics."""
    from newsroom.classification.trainer import TrainConfig, save_model, train_classifier
    from newsroom.data.corpus import load_corpus_csv

    cfg_dict = _load_config(config)
    if model:
        cfg_dict["model"] = model
    if grid_search is not None:
        cfg_dict["grid_search"] = grid_search
    if seed is not None:
        cfg_dict["random_seed"] = seed
    try:
        train_cfg = TrainConfig.from_dict(cfg_dict)
        df = load_corpus_csv(input)
        result = train_classifier(df, train_cfg)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
        return
    save_model(result, output)
    m = result.metrics
    table = Table(title=f"Hold-out metrics ({train_cfg.model})")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("accuracy", f"{m.accuracy:.4f}")
    table.add_row("macro F1", f"{m.macro_f1:.4f}")
    table.add_row("weighted F1", f"{m.weighted_f1:.4f}")
    table.add_row("train / test", f"{result.train_size} / {result.test_size}")
    table.add_row("seconds", f"{result.train_seconds:.1f}")
    console.print(table)
    if result.best_params:
        console.print(f"best params: {result.best_params}")
    console.print(f"[green]Saved[/] model to {output}")


@app.command("evaluate")
def evaluate_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")] = Path("artifacts/corpus.csv"),
    model: Annotated[Path, typer.Option("--model", "-m")] = Path("artifacts/classifier.joblib"),
    report_dir: Annotated[Path, typer.Option("--report-dir", "-r")] = Path("artifacts/reports"),
    plots: Annotated[bool, typer.Option("--plots/--no-plots", help="Also save PNG figures")] = True,
    search_index: Annotated[
        Path | None, typer.Option("--index", help="Also evaluate a saved search index")
    ] = None,
    n_queries: Annotated[int, typer.Option("--n-queries", min=1)] = 30,
) -> None:
    """Evaluate a trained classifier (and optionally a search index); write Markdown + JSON reports."""
    from newsroom.classification.explain import top_terms_per_class
    from newsroom.classification.predictor import Predictor
    from newsroom.data.corpus import load_corpus_csv
    from newsroom.evaluation.metrics import (
        classification_report_text,
        compute_classification_metrics,
    )
    from newsroom.evaluation.report import write_json_report, write_markdown_report
    from newsroom.evaluation.search_eval import build_relevance_set, evaluate_search_engine

    try:
        df = load_corpus_csv(input)
        predictor = Predictor.from_path(model)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
        return
    y_pred = predictor.predict_labels(df["text"].tolist())
    metrics = compute_classification_metrics(
        df["label"].astype(str).tolist(), y_pred, labels=predictor.labels
    )
    console.print(classification_report_text(metrics))

    retrieval = None
    if search_index is not None:
        from newsroom.search.query import SearchEngine

        engine = SearchEngine.load(search_index)
        rel = build_relevance_set(df, n_queries=n_queries)
        retrieval = evaluate_search_engine(engine, rel, k=10)
        for mode, rm in retrieval.items():
            console.print(
                f"{mode:>7}: P@10={rm.precision_at_k:.3f} R@10={rm.recall_at_k:.3f} MRR={rm.mrr:.3f} nDCG@10={rm.ndcg_at_k:.3f}"
            )

    meta = {"model": str(model), "corpus": str(input), "documents": len(df)}
    md = write_markdown_report(
        report_dir / "evaluation.md", classification=metrics, retrieval=retrieval, metadata=meta
    )
    write_json_report(
        report_dir / "evaluation.json", classification=metrics, retrieval=retrieval, metadata=meta
    )
    if plots:
        from newsroom.visualization.plots import (
            plot_class_distribution,
            plot_confusion_matrix,
            plot_top_terms,
        )

        plot_confusion_matrix(
            metrics.confusion, metrics.labels, report_dir / "confusion_matrix.png"
        )
        plot_class_distribution(df["label"].tolist(), report_dir / "class_distribution.png")
        try:
            plot_top_terms(
                top_terms_per_class(predictor.pipeline, 10), report_dir / "top_terms.png"
            )
        except (TypeError, ValueError) as exc:
            err_console.print(f"[yellow]skipping top-terms plot:[/] {exc}")
    console.print(f"[green]Reports written[/] to {md.parent}")


@app.command("build-index")
def build_index_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")] = Path("artifacts/corpus.csv"),
    output: Annotated[Path, typer.Option("--output", "-o", help="Index directory")] = Path(
        "artifacts/index"
    ),
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="YAML config (configs/search.yaml)")
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", help="tfidf_svd | sentence_transformers")
    ] = None,
    dimension: Annotated[int | None, typer.Option("--dimension", min=2)] = None,
) -> None:
    """Build BM25 + vector indexes over a corpus and save them."""
    from newsroom.data.corpus import load_corpus_csv
    from newsroom.search.hybrid import HybridConfig
    from newsroom.search.query import build_search_engine

    cfg = _load_config(config)
    emb = cfg.get("embedding", {})
    provider_name = provider or emb.get("provider") or get_settings().embedding_provider
    dim = dimension or int(emb.get("dimension", get_settings().embedding_dim))
    kwargs: dict[str, Any] = {}
    if provider_name == "sentence_transformers":
        kwargs["model_name"] = emb.get("model_name", get_settings().sentence_transformer_model)
    try:
        df = load_corpus_csv(input, require_label=False)
        engine = build_search_engine(
            df,
            provider_name=provider_name,
            dimension=dim,
            config=HybridConfig.from_dict(cfg.get("hybrid")),
            provider_kwargs=kwargs,
        )
    except (ValueError, FileNotFoundError, ImportError) as exc:
        _fail(str(exc))
        return
    engine.save(output)
    console.print(
        f"[green]Indexed[/] {len(engine)} documents (provider={engine.provider.name}, dim={engine.provider.dimension}) -> {output}"
    )


@app.command("search")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Free-text query")],
    index: Annotated[Path, typer.Option("--index", help="Index directory")] = Path(
        "artifacts/index"
    ),
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1)] = 5,
    mode: Annotated[str, typer.Option("--mode", help="hybrid | bm25 | vector")] = "hybrid",
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of a table")] = False,
) -> None:
    """Query a saved index."""
    from newsroom.search.query import SearchEngine

    try:
        engine = SearchEngine.load(index)
        results = engine.search(query, top_k, mode=mode)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    if as_json:
        console.print_json(json.dumps([r.__dict__ for r in results], default=str))
        return
    table = Table(title=f"{mode} results for: {query!r}")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("doc_id")
    table.add_column("label")
    table.add_column("title")
    for r in results:
        table.add_row(str(r.rank), f"{r.score:.4f}", r.doc_id, r.label or "-", r.title[:70])
    console.print(table)


@app.command("keywords")
def keywords_cmd(
    text: Annotated[str | None, typer.Argument(help="Text (or use --file)")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f", help="Read text from a file")] = None,
    top_k: Annotated[int, typer.Option("--top-k", "-k", min=1)] = 10,
    phrases: Annotated[bool, typer.Option("--phrases/--no-phrases")] = True,
) -> None:
    """Extract TextRank keywords from a document."""
    from newsroom.keywords.textrank import TextRankExtractor

    content = _read_text(text, file)
    for kw in TextRankExtractor().extract(content, top_k=top_k, phrases=phrases):
        console.print(f"{kw.score:8.4f}  {kw.text}")


@app.command("summarize")
def summarize_cmd(
    text: Annotated[str | None, typer.Argument(help="Text (or use --file)")] = None,
    file: Annotated[Path | None, typer.Option("--file", "-f")] = None,
    n_sentences: Annotated[int, typer.Option("--sentences", "-n", min=1)] = 3,
    method: Annotated[str, typer.Option("--method", help="centroid | textrank")] = "centroid",
) -> None:
    """Produce an extractive summary."""
    from newsroom.summarization.extractive import ExtractiveSummarizer

    content = _read_text(text, file)
    try:
        summary = ExtractiveSummarizer(method).summarize(content, n_sentences)
    except ValueError as exc:
        _fail(str(exc))
        return
    console.print(summary.text)


@app.command("cluster")
def cluster_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")] = Path("artifacts/corpus.csv"),
    k: Annotated[
        int | None,
        typer.Option("--k", help="Fixed number of clusters (default: silhouette selection)"),
    ] = None,
    dimension: Annotated[int, typer.Option("--dimension", min=2)] = 128,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write cluster summary JSON")
    ] = None,
    lda: Annotated[bool, typer.Option("--lda", help="Also fit an LDA topic model")] = False,
) -> None:
    """Cluster documents by embedding and show descriptive terms per cluster."""
    from newsroom.data.corpus import load_corpus_csv
    from newsroom.embeddings.tfidf_svd import TfidfSvdProvider
    from newsroom.topics.clustering import TopicClusterer

    try:
        df = load_corpus_csv(input, require_label=False)
        texts = df["text"].tolist()
        provider = TfidfSvdProvider(dimension=dimension).fit(texts)
        result = TopicClusterer(k=k).fit(provider.embed_texts(texts), texts)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
        return
    table = Table(title=f"{result.k} clusters (silhouette={result.silhouette:.3f})")
    table.add_column("cluster", justify="right")
    table.add_column("size", justify="right")
    table.add_column("top terms")
    for row in result.summary():
        table.add_row(str(row["cluster"]), str(row["size"]), ", ".join(row["top_terms"]))  # type: ignore[arg-type]
    console.print(table)
    payload: dict[str, Any] = {
        "k": result.k,
        "silhouette": result.silhouette,
        "clusters": result.summary(),
    }
    if lda:
        from newsroom.topics.lda import LdaTopicModel

        lda_result = LdaTopicModel(n_topics=result.k).fit(texts)
        for topic in lda_result.summary():
            console.print(f"LDA topic {topic['topic']}: {', '.join(topic['terms'])}")  # type: ignore[arg-type]
        payload["lda"] = lda_result.summary()
    if output:
        from newsroom.utils.io import write_json

        write_json(payload, output)
        console.print(f"[green]Wrote[/] {output}")


@app.command("dedup")
def dedup_cmd(
    input: Annotated[Path, typer.Option("--input", "-i")] = Path("artifacts/corpus.csv"),
    threshold: Annotated[float, typer.Option("--threshold", min=0.05, max=1.0)] = 0.8,
    limit: Annotated[int, typer.Option("--limit", min=1)] = 20,
) -> None:
    """Report near-duplicate document pairs using MinHash."""
    from newsroom.data.corpus import load_corpus_csv
    from newsroom.similarity.dedup import NearDuplicateDetector

    try:
        df = load_corpus_csv(input, require_label=False)
    except (ValueError, FileNotFoundError) as exc:
        _fail(str(exc))
        return
    detector = NearDuplicateDetector(threshold=threshold)
    detector.add_many(df["doc_id"].astype(str).tolist(), df["text"].tolist())
    pairs = detector.find_duplicates()
    console.print(f"{len(pairs)} near-duplicate pairs (threshold={threshold})")
    for pair in pairs[:limit]:
        console.print(f"{pair.similarity:.3f}  {pair.left}  <->  {pair.right}")


@app.command("serve")
def serve_cmd(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port", min=1, max=65535)] = None,
    artifacts: Annotated[
        Path | None,
        typer.Option("--artifacts", help="Directory holding classifier.joblib and index/"),
    ] = None,
    reload: Annotated[
        bool, typer.Option("--reload", help="Auto-reload (development only)")
    ] = False,
) -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn

    from newsroom.api.app import create_app
    from newsroom.config import Settings

    base = get_settings()
    settings = base.model_copy(update={"artifacts_dir": artifacts}) if artifacts else base
    if not isinstance(settings, Settings):  # pragma: no cover - defensive
        _fail("invalid settings")
    application = create_app(settings)
    uvicorn.run(
        application,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


def _read_text(text: str | None, file: Path | None) -> str:
    if file is not None:
        if not file.exists():
            _fail(f"file not found: {file}")
        return file.read_text(encoding="utf-8")
    if text is None or not text.strip():
        _fail("provide TEXT or --file")
    return str(text)


if __name__ == "__main__":  # pragma: no cover
    app()
