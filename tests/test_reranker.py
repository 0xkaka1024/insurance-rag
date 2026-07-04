import httpx
import pytest

from app.core.config import Settings
from app.rag.reranker import RerankClient, RerankUnavailable


def _client(handler, max_retries: int = 2) -> tuple[RerankClient, list[float]]:
    """基于 httpx.MockTransport 的客户端；sleep 不真睡，只记录退避序列。"""
    sleeps: list[float] = []
    settings = Settings(_env_file=None, rerank_max_retries=max_retries)
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


def _breaker_client(handler, cooldown: float = 60.0):
    """熔断测试专用：max_retries=0（每次调用一次 HTTP）+ 可拨动的假时钟。"""
    clock = {"t": 0.0}
    settings = Settings(
        _env_file=None, rerank_max_retries=0, rerank_breaker_cooldown_s=cooldown
    )
    http = httpx.Client(transport=httpx.MockTransport(handler))
    client = RerankClient(
        settings=settings, client=http, sleep=lambda _s: None, now=lambda: clock["t"]
    )
    return client, clock


def test_breaker_opens_after_three_failures_and_recovers_after_cooldown():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503, json={})

    client, clock = _breaker_client(handler, cooldown=60)
    for _ in range(3):
        with pytest.raises(RerankUnavailable):
            client.rerank("q", ["a"], top_n=1)
    assert calls["n"] == 3

    with pytest.raises(RerankUnavailable, match="circuit open"):
        client.rerank("q", ["a"], top_n=1)
    assert calls["n"] == 3  # 熔断中不再发 HTTP，立即降级

    clock["t"] = 61.0  # 冷却期过后恢复尝试
    with pytest.raises(RerankUnavailable):
        client.rerank("q", ["a"], top_n=1)
    assert calls["n"] == 4


def test_breaker_streak_resets_on_success():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] <= 2 or calls["n"] >= 4:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.9}]})

    client, _ = _breaker_client(handler)
    for _ in range(2):  # 连败 2 次（未达阈值）
        with pytest.raises(RerankUnavailable):
            client.rerank("q", ["a"], top_n=1)
    assert client.rerank("q", ["a"], top_n=1) == [(0, 0.9)]  # 成功复位计数

    for _ in range(2):  # 再连败 2 次：streak=2 < 3，电路仍闭合
        with pytest.raises(RerankUnavailable):
            client.rerank("q", ["a"], top_n=1)
    assert calls["n"] == 5  # 每次都真实发了 HTTP（未熔断）


def test_top_n_clamped_to_documents():
    seen = {}

    def handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"results": []})

    client, _ = _client(handler)
    client.rerank("q", ["a", "b"], top_n=10)
    assert seen["top_n"] == 2
