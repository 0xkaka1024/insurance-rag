import httpx
import pytest

from app.core.config import Settings
from app.rag.reranker import RerankClient, RerankUnavailable


def _client(handler, max_retries: int = 2) -> tuple[RerankClient, list[float]]:
    """基于 httpx.MockTransport 的客户端；sleep 不真睡，只记录退避序列。"""
    sleeps: list[float] = []
    settings = Settings(_env_file=None, max_retries=max_retries)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return RerankClient(settings=settings, client=http, sleep=sleeps.append), sleeps


def test_rerank_returns_sorted_index_score_pairs():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.35},
                ]
            },
        )

    client, _ = _client(handler)
    out = client.rerank("等待期", ["a", "b", "c"], top_n=2)
    assert out == [(2, 0.91), (0, 0.35)]


def test_retries_on_5xx_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    client, sleeps = _client(handler, max_retries=3)
    assert client.rerank("q", ["a"], top_n=1) == [(0, 0.8)]
    assert sleeps == [1, 2]  # 指数退避


def test_gives_up_after_max_retries():
    def handler(request):
        return httpx.Response(500, json={})

    client, sleeps = _client(handler, max_retries=2)
    with pytest.raises(RerankUnavailable):
        client.rerank("q", ["a"], top_n=1)
    assert sleeps == [1, 2]


def test_no_retry_on_auth_error():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(401, json={})

    client, sleeps = _client(handler)
    with pytest.raises(RerankUnavailable):
        client.rerank("q", ["a"], top_n=1)
    assert calls["n"] == 1  # 快速失败，不浪费重试
    assert sleeps == []


def test_retries_on_timeout():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectTimeout("boom")
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.7}]})

    client, _ = _client(handler)
    assert client.rerank("q", ["a"], top_n=1) == [(0, 0.7)]


def test_top_n_clamped_to_documents():
    seen = {}

    def handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    client, _ = _client(handler)
    client.rerank("q", ["a", "b"], top_n=10)
    assert seen["top_n"] == 2
