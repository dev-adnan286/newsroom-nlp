from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from newsroom.api.app import create_app
from newsroom.api.state import AppState
from newsroom.classification.predictor import Predictor
from newsroom.config import Settings
from newsroom.search.query import SearchEngine


@pytest.fixture(scope="module")
def client(model_path: Path, search_engine: SearchEngine) -> Iterator[TestClient]:
    settings = Settings(api_key=None, log_level="WARNING", api_max_batch=8)
    state = AppState(
        settings=settings, predictor=Predictor.from_path(model_path), engine=search_engine
    )
    with TestClient(create_app(settings, state=state)) as c:
        yield c


def test_health_and_models(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok" and body["classifier_loaded"] and body["index_loaded"]
    assert body["documents_indexed"] >= 180

    models = client.get("/models").json()["models"]
    kinds = {m["kind"]: m for m in models}
    assert kinds["classifier"]["loaded"] and "labels" in kinds["classifier"]["details"]
    assert kinds["embedding"]["details"]["dimension"] > 0
    assert kinds["index"]["details"]["documents"] >= 180


def test_classify(client: TestClient) -> None:
    resp = client.post(
        "/classify",
        json={
            "texts": [
                "The striker scored twice as fans celebrated the derby win",
                "Shares fell after weak quarterly earnings",
            ],
            "top_k": 2,
            "explain": True,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "linear_svc"
    labels = [r["label"] for r in body["results"]]
    assert labels == ["sports", "business"]
    assert len(body["results"][0]["top"]) == 2
    assert body["results"][0]["explanation"] and "term" in body["results"][0]["explanation"][0]


def test_classify_validation(client: TestClient) -> None:
    assert client.post("/classify", json={"texts": []}).status_code == 422
    assert client.post("/classify", json={"texts": ["  "]}).status_code == 422
    assert client.post("/classify", json={"texts": ["x"], "extra": 1}).status_code == 422
    resp = client.post("/classify", json={"texts": ["x"] * 9})
    assert resp.status_code == 413 and resp.json()["error"] == "payload_too_large"


def test_embed(client: TestClient) -> None:
    resp = client.post("/embed", json={"texts": ["central bank rates", "cup final penalty"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dimension"] == len(body["vectors"][0]) and len(body["vectors"]) == 2
    assert body["provider"].startswith("tfidf_svd")
    raw = client.post("/embed", json={"texts": ["central bank rates"], "normalize": False}).json()
    assert len(raw["vectors"][0]) == body["dimension"]


def test_search_modes(client: TestClient) -> None:
    resp = client.post(
        "/search", json={"query": "central bank interest rates inflation", "top_k": 3}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "hybrid" and len(body["hits"]) == 3
    assert body["hits"][0]["rank"] == 1 and body["hits"][0]["label"] == "business"
    assert "bm25" in body["hits"][0]["component_scores"]
    for mode in ("bm25", "vector"):
        r = client.post(
            "/search",
            json={"query": "vaccine trial patients", "top_k": 2, "mode": mode, "bm25_weight": 2.0},
        )
        assert r.status_code == 200 and r.json()["mode"] == mode
    assert client.post("/search", json={"query": "", "top_k": 3}).status_code == 422
    assert client.post("/search", json={"query": "x", "mode": "fuzzy"}).status_code == 422


def test_keywords_and_summarize(client: TestClient, sample_article: str) -> None:
    resp = client.post("/keywords", json={"text": sample_article, "top_k": 4})
    assert resp.status_code == 200
    kws = resp.json()["keywords"]
    assert 0 < len(kws) <= 4 and all(k["score"] > 0 for k in kws)

    resp = client.post(
        "/summarize", json={"text": sample_article, "n_sentences": 2, "method": "textrank"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sentences"]) == 2 and body["summary"]
    assert (
        client.post("/summarize", json={"text": sample_article, "method": "magic"}).status_code
        == 422
    )


def test_index_documents(client: TestClient) -> None:
    before = client.get("/health").json()["documents_indexed"]
    resp = client.post(
        "/index/documents",
        json={
            "documents": [
                {
                    "doc_id": "api-volcano",
                    "text": "A volcanic eruption sent lava toward coastal villages overnight.",
                    "title": "Volcano",
                    "label": "science",
                },
                {
                    "text": "Lava flows from the volcano slowed as the observatory monitored tremors."
                },
            ]
        },
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert (
        body["added"] == 2
        and body["total_indexed"] == before + 2
        and body["created_index"] is False
    )
    hits = client.post("/search", json={"query": "volcanic eruption lava", "top_k": 2}).json()[
        "hits"
    ]
    assert hits[0]["doc_id"] == "api-volcano"
    dup = client.post(
        "/index/documents",
        json={"documents": [{"doc_id": "d", "text": "a b"}, {"doc_id": "d", "text": "c d"}]},
    )
    assert dup.status_code == 422


def test_unknown_route_uses_error_schema(client: TestClient) -> None:
    resp = client.get("/nope")
    assert resp.status_code == 404 and resp.json()["error"] == "not_found"


def test_degraded_state_returns_503(tmp_path: Path) -> None:
    settings = Settings(api_key=None, log_level="WARNING", artifacts_dir=tmp_path)
    with TestClient(create_app(settings)) as c:
        health = c.get("/health").json()
        assert health["status"] == "degraded" and not health["classifier_loaded"]
        assert c.post("/classify", json={"texts": ["x"]}).status_code == 503
        assert c.post("/search", json={"query": "x"}).status_code == 503
        assert c.post("/embed", json={"texts": ["x"]}).status_code == 503
        # A single document cannot bootstrap an index, two can.
        assert (
            c.post("/index/documents", json={"documents": [{"text": "only one"}]}).status_code
            == 422
        )
        created = c.post(
            "/index/documents",
            json={
                "documents": [
                    {"text": "central bank raises rates on inflation"},
                    {"text": "striker scores in cup final"},
                    {"text": "vaccine trial shows promise"},
                ]
            },
        )
        assert created.status_code == 201 and created.json()["created_index"] is True
        assert c.get("/health").json()["index_loaded"]
        assert c.post("/search", json={"query": "cup final", "top_k": 1}).status_code == 200


def test_api_key_auth(model_path: Path, search_engine: SearchEngine) -> None:
    settings = Settings(api_key="secret-key", log_level="WARNING")
    state = AppState(
        settings=settings, predictor=Predictor.from_path(model_path), engine=search_engine
    )
    with TestClient(create_app(settings, state=state)) as c:
        assert c.get("/health").status_code == 200  # public
        denied = c.post("/keywords", json={"text": "hello world"})
        assert denied.status_code == 401 and denied.json()["error"] == "unauthorized"
        assert (
            c.post(
                "/keywords", json={"text": "hello world"}, headers={"X-API-Key": "wrong"}
            ).status_code
            == 401
        )
        ok = c.post(
            "/keywords",
            json={"text": "central bank raises rates"},
            headers={"X-API-Key": "secret-key"},
        )
        assert ok.status_code == 200


def test_blank_api_key_disables_auth() -> None:
    assert Settings(api_key="   ").auth_enabled is False
    assert Settings(api_key="abc").auth_enabled is True
    with pytest.raises(ValueError):
        Settings(log_level="loud")
