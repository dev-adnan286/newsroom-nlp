"""Route handlers."""

from __future__ import annotations

import uuid

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from newsroom import __version__
from newsroom.api.deps import get_state, require_api_key, require_engine, require_predictor
from newsroom.api.schemas import (
    ClassifyItem,
    ClassifyRequest,
    ClassifyResponse,
    EmbedRequest,
    EmbedResponse,
    HealthResponse,
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    KeywordOut,
    KeywordsRequest,
    KeywordsResponse,
    LabelScore,
    ModelInfo,
    ModelsResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from newsroom.api.state import AppState
from newsroom.classification.explain import explain_document
from newsroom.keywords.textrank import TextRankExtractor
from newsroom.summarization.extractive import ExtractiveSummarizer

router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


def _check_batch(n: int, state: AppState) -> None:
    if n > state.settings.api_max_batch:
        raise HTTPException(
            413,
            f"Batch of {n} exceeds the maximum of {state.settings.api_max_batch}",
        )


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    """Liveness/readiness probe. ``degraded`` means at least one artifact is missing."""
    ok = state.predictor is not None and state.engine is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        version=__version__,
        classifier_loaded=state.predictor is not None,
        index_loaded=state.engine is not None,
        documents_indexed=state.documents_indexed,
    )


@router.get("/models", response_model=ModelsResponse, tags=["system"])
def models(state: AppState = Depends(get_state)) -> ModelsResponse:
    """Describe the loaded artifacts."""
    items: list[ModelInfo] = []
    if state.predictor is not None:
        meta = state.predictor.metadata
        items.append(
            ModelInfo(
                name=str(meta.get("config", {}).get("model", "classifier")),
                kind="classifier",
                loaded=True,
                details={
                    "labels": state.predictor.labels,
                    "metrics": {
                        k: meta.get("metrics", {}).get(k) for k in ("accuracy", "macro_f1")
                    },
                    "package_version": meta.get("package_version"),
                },
            )
        )
    else:
        items.append(ModelInfo(name="classifier", kind="classifier", loaded=False))
    if state.engine is not None:
        items.append(
            ModelInfo(
                name=state.engine.provider.name,
                kind="embedding",
                loaded=True,
                details={"dimension": state.engine.provider.dimension},
            )
        )
        items.append(
            ModelInfo(
                name="hybrid_index",
                kind="index",
                loaded=True,
                details={"documents": len(state.engine), "rrf_k": state.engine.config.rrf_k},
            )
        )
    else:
        items.append(
            ModelInfo(name=state.settings.embedding_provider, kind="embedding", loaded=False)
        )
        items.append(ModelInfo(name="hybrid_index", kind="index", loaded=False))
    return ModelsResponse(models=items)


@protected.post("/classify", response_model=ClassifyResponse, tags=["nlp"])
def classify(
    payload: ClassifyRequest, state: AppState = Depends(require_predictor)
) -> ClassifyResponse:
    """Classify one or more documents, returning calibrated label probabilities."""
    _check_batch(len(payload.texts), state)
    assert state.predictor is not None
    predictions = state.predictor.predict_batch(payload.texts)
    items: list[ClassifyItem] = []
    for text, pred in zip(payload.texts, predictions, strict=True):
        explanation = None
        if payload.explain:
            try:
                explanation = explain_document(state.predictor.pipeline, text, n_terms=8)[
                    "top_features"
                ]
            except (TypeError, ValueError, KeyError):
                explanation = []
        items.append(
            ClassifyItem(
                label=pred.label,
                confidence=pred.confidence,
                top=[LabelScore(label=lab, probability=p) for lab, p in pred.top(payload.top_k)],
                explanation=explanation,
            )
        )
    model_name = str(state.predictor.metadata.get("config", {}).get("model", "classifier"))
    return ClassifyResponse(model=model_name, results=items)


@protected.post("/embed", response_model=EmbedResponse, tags=["nlp"])
def embed(payload: EmbedRequest, state: AppState = Depends(require_engine)) -> EmbedResponse:
    """Return dense embeddings from the active provider."""
    _check_batch(len(payload.texts), state)
    assert state.engine is not None
    provider = state.engine.provider
    vectors = provider.embed_texts(payload.texts)
    if not payload.normalize:
        # Providers return unit vectors; scale by text length as a crude magnitude proxy.
        lengths = [max(1, len(t.split())) for t in payload.texts]
        vectors = vectors * (pd.Series(lengths).to_numpy(dtype="float32")[:, None] ** 0.5)
    return EmbedResponse(
        provider=provider.name, dimension=provider.dimension, vectors=vectors.tolist()
    )


@protected.post("/search", response_model=SearchResponse, tags=["search"])
def search(payload: SearchRequest, state: AppState = Depends(require_engine)) -> SearchResponse:
    """Hybrid (BM25 + vector) search with optional per-request weights."""
    assert state.engine is not None
    weights: dict[str, float] = {}
    if payload.bm25_weight is not None:
        weights["bm25_weight"] = payload.bm25_weight
    if payload.vector_weight is not None:
        weights["vector_weight"] = payload.vector_weight
    with state.lock:
        results = state.engine.search(payload.query, payload.top_k, mode=payload.mode, **weights)
        total = len(state.engine)
    hits = [
        SearchHitOut(
            doc_id=r.doc_id,
            rank=r.rank,
            score=r.score,
            title=r.title,
            snippet=r.snippet(payload.snippet_chars),
            label=r.label,
            component_scores=r.component_scores,
            published_at=r.metadata.get("published_at"),
        )
        for r in results
    ]
    return SearchResponse(query=payload.query, mode=payload.mode, total_indexed=total, hits=hits)


@protected.post("/keywords", response_model=KeywordsResponse, tags=["nlp"])
def keywords(payload: KeywordsRequest) -> KeywordsResponse:
    """TextRank keywords/key-phrases for a single document."""
    extractor = TextRankExtractor(window=payload.window)
    kws = extractor.extract(payload.text, top_k=payload.top_k, phrases=payload.phrases)
    return KeywordsResponse(keywords=[KeywordOut(text=k.text, score=k.score) for k in kws])


@protected.post("/summarize", response_model=SummarizeResponse, tags=["nlp"])
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    """Extractive summary of a document."""
    summary = ExtractiveSummarizer(payload.method).summarize(payload.text, payload.n_sentences)
    return SummarizeResponse(
        summary=summary.text,
        sentences=summary.sentences,
        indices=summary.indices,
        scores=summary.scores,
    )


@protected.post(
    "/index/documents",
    response_model=IndexDocumentsResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["search"],
)
def index_documents(
    payload: IndexDocumentsRequest, state: AppState = Depends(get_state)
) -> IndexDocumentsResponse:
    """Add documents to the search index (creating it if none is loaded)."""
    _check_batch(len(payload.documents), state)
    rows = []
    for doc in payload.documents:
        rows.append(
            {
                "doc_id": doc.doc_id or f"api-{uuid.uuid4().hex[:12]}",
                "text": doc.text,
                "title": doc.title or doc.text[:60],
                "label": doc.label,
                "published_at": doc.published_at,
            }
        )
    df = pd.DataFrame(rows)
    if df["doc_id"].duplicated().any():
        raise HTTPException(422, "doc_id values must be unique within a request")
    if state.engine is None and len(df) < 2:
        raise HTTPException(422, "At least two documents are required to create a new index")
    added, created = state.ensure_engine(df)
    return IndexDocumentsResponse(
        added=added, total_indexed=state.documents_indexed, created_index=created
    )


router.include_router(protected)

__all__ = ["router"]
