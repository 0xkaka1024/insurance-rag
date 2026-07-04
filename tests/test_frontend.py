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


def test_corpus_view_present():
    resp = client.get("/")
    assert "语料" in resp.text
    assert "view-corpus" in resp.text
    assert "corpus-chunks" in resp.text
    assert "去 Playground 检索" in resp.text  # 语料 → Playground 跳转入口
    assert "在语料中定位" in resp.text  # 检索结果 → 语料反向跳转入口


def test_configs_endpoint_matches_rag_config():
    resp = client.get("/configs")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunking"] == ["fixed", "structural"]
    assert body["retrieval"] == ["vector", "hybrid"]
    assert body["rerank"] == [False, True]
    assert body["production"] == {
        "chunking": "structural",
        "retrieval": "hybrid",
        "rerank": True,
    }  # /ask 锁定的生产配置，前端问答视图展示用
