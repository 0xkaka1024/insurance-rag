from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_index_served():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Playground" in resp.text
    assert "保险条款问答" in resp.text


def test_static_mounted():
    resp = client.get("/static/index.html")
    assert resp.status_code == 200


def test_configs_endpoint_matches_rag_config():
    resp = client.get("/configs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunking"] == ["fixed", "structural"]
    assert body["retrieval"] == ["vector", "hybrid"]
    assert body["rerank"] == [False, True]
