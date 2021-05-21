"""Pydantic v2 request/response models for the HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SearchMode = Literal["hybrid", "bm25", "vector"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# -- classification ----------------------------------------------------------------------
class ClassifyRequest(_Strict):
    texts: list[str] = Field(..., min_length=1, description="Documents to classify")
    top_k: int = Field(3, ge=1, le=50, description="Number of label probabilities to return")
    explain: bool = Field(False, description="Include top contributing terms per document")

    @field_validator("texts")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if any(not isinstance(t, str) or not t.strip() for t in value):
            raise ValueError("every text must be a non-empty string")
        return value


class LabelScore(BaseModel):
    label: str
    probability: float


class ClassifyItem(BaseModel):
    label: str
    confidence: float
    top: list[LabelScore]
    explanation: list[dict[str, Any]] | None = None


class ClassifyResponse(BaseModel):
    model: str
    results: list[ClassifyItem]


# -- embeddings ---------------------------------------------------------------------------
class EmbedRequest(_Strict):
    texts: list[str] = Field(..., min_length=1)
    normalize: bool = True


class EmbedResponse(BaseModel):
    provider: str
    dimension: int
    vectors: list[list[float]]


# -- search -------------------------------------------------------------------------------
class SearchRequest(_Strict):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    mode: SearchMode = "hybrid"
    bm25_weight: float | None = Field(None, ge=0.0, le=10.0)
    vector_weight: float | None = Field(None, ge=0.0, le=10.0)
    snippet_chars: int = Field(200, ge=20, le=2000)


class SearchHitOut(BaseModel):
    doc_id: str
    rank: int
    score: float
    title: str
    snippet: str
    label: str | None = None
    component_scores: dict[str, float]
    published_at: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    total_indexed: int
    hits: list[SearchHitOut]


# -- keywords -----------------------------------------------------------------------------
class KeywordsRequest(_Strict):
    text: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=50)
    phrases: bool = True
    window: int = Field(4, ge=2, le=20)


class KeywordOut(BaseModel):
    text: str
    score: float


class KeywordsResponse(BaseModel):
    keywords: list[KeywordOut]


# -- summarisation ------------------------------------------------------------------------
class SummarizeRequest(_Strict):
    text: str = Field(..., min_length=1)
    n_sentences: int = Field(3, ge=1, le=20)
    method: Literal["centroid", "textrank"] = "centroid"


class SummarizeResponse(BaseModel):
    summary: str
    sentences: list[str]
    indices: list[int]
    scores: list[float]


# -- indexing -----------------------------------------------------------------------------
class DocumentIn(_Strict):
    doc_id: str | None = Field(None, max_length=200)
    text: str = Field(..., min_length=1)
    title: str | None = Field(None, max_length=500)
    label: str | None = None
    published_at: str | None = None


class IndexDocumentsRequest(_Strict):
    documents: list[DocumentIn] = Field(..., min_length=1)


class IndexDocumentsResponse(BaseModel):
    added: int
    total_indexed: int
    created_index: bool


# -- misc ---------------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    classifier_loaded: bool
    index_loaded: bool
    documents_indexed: int


class ModelInfo(BaseModel):
    name: str
    kind: Literal["classifier", "embedding", "index"]
    loaded: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None


__all__ = [
    "ClassifyItem",
    "ClassifyRequest",
    "ClassifyResponse",
    "DocumentIn",
    "EmbedRequest",
    "EmbedResponse",
    "ErrorResponse",
    "HealthResponse",
    "IndexDocumentsRequest",
    "IndexDocumentsResponse",
    "KeywordOut",
    "KeywordsRequest",
    "KeywordsResponse",
    "LabelScore",
    "ModelInfo",
    "ModelsResponse",
    "SearchHitOut",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "SummarizeRequest",
    "SummarizeResponse",
]
