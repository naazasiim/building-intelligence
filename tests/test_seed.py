from fastapi.testclient import TestClient

from building_with_rag.app import app

client = TestClient(app)


def test_healthz():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_models():
    ids = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert len(ids) == 6 and "rag-hyde" in ids


def test_query_placeholder():
    r = client.post("/v1/query", json={"question": "q", "pattern": "semantic"}).json()
    assert r["status"] == "not_implemented" and r["pattern"] == "semantic"


def test_chat_sse():
    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "rag-semantic",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
            "rag_options": {"pattern": "semantic"},
        },
    )
    assert r.text.strip().endswith("data: [DONE]")
