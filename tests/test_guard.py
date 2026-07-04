"""防滥用三道闸：鉴权 / 每 IP 限流 / 每日额度熔断（AbuseGuard 单元 + 端点接线）。"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import guard as guard_module
from app.api.guard import AbuseGuard, client_ip
from app.api.routes import get_pipeline
from app.core.config import Settings, get_settings
from app.main import app
from app.rag.pipeline import RagPipeline
from tests.test_ask import FakeLLM, FakeRetriever


class FakeRequest:
    def __init__(self, headers: dict | None = None, host: str = "1.2.3.4"):
        self.headers = headers or {}
        self.client = type("C", (), {"host": host})()


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_all_gates_off_by_default():
    AbuseGuard().check(FakeRequest(), _settings())  # 不抛 = 本地/CI 零负担


def test_client_ip_prefers_forwarded_first_hop():
    req = FakeRequest(headers={"x-forwarded-for": "9.9.9.9, 10.0.0.1"})
    assert client_ip(req) == "9.9.9.9"  # HF 反代后取真实来源
    assert client_ip(FakeRequest()) == "1.2.3.4"


def test_auth_gate():
    s = _settings(api_auth_token="sk-secret")
    g = AbuseGuard()
    with pytest.raises(HTTPException) as e:
        g.check(FakeRequest(), s)
    assert e.value.status_code == 401
    with pytest.raises(HTTPException):
        g.check(FakeRequest(headers={"authorization": "Bearer wrong"}), s)
    g.check(FakeRequest(headers={"authorization": "Bearer sk-secret"}), s)  # 通过


def test_rate_limit_per_ip_with_window_rollover():
    clock = {"t": 60.0}
    g = AbuseGuard(now=lambda: clock["t"])
    s = _settings(rate_limit_per_minute=2)
    g.check(FakeRequest(), s)
    g.check(FakeRequest(), s)
    with pytest.raises(HTTPException) as e:
        g.check(FakeRequest(), s)
    assert e.value.status_code == 429
    g.check(FakeRequest(host="5.6.7.8"), s)  # 其他 IP 不受影响

    clock["t"] = 121.0  # 下一分钟窗口重置
    g.check(FakeRequest(), s)


def test_daily_budget_blocks_then_rolls_over():
    clock = {"t": 0.0}
    g = AbuseGuard(now=lambda: clock["t"])
    s = _settings(daily_request_budget=2)
    g.check(FakeRequest(), s)
    g.check(FakeRequest(), s)
    with pytest.raises(HTTPException) as e:
        g.check(FakeRequest(), s)
    assert e.value.status_code == 429
    assert "额度" in e.value.detail  # 成本熔断话术

    clock["t"] = 86400.0 + 60  # 次日额度恢复
    g.check(FakeRequest(), s)


def test_rate_limited_request_does_not_consume_daily_budget():
    """三道闸顺序：先限流后额度——被限掉的请求不吃每日额度。"""
    clock = {"t": 60.0}
    g = AbuseGuard(now=lambda: clock["t"])
    s = _settings(rate_limit_per_minute=1, daily_request_budget=2)
    g.check(FakeRequest(), s)  # 消耗：限流 1/1，额度 1/2
    with pytest.raises(HTTPException):
        g.check(FakeRequest(), s)  # 被限流拦下，额度仍 1/2
    clock["t"] = 121.0
    g.check(FakeRequest(), s)  # 额度 2/2，仍可通过
    clock["t"] = 181.0
    with pytest.raises(HTTPException) as e:
        g.check(FakeRequest(), s)  # 此时才轮到额度熔断
    assert "额度" in e.value.detail


def test_expensive_endpoints_wired_to_guard(monkeypatch):
    """接线验证：/ask、/playground/ask、/retrieve 都过闸，鉴权开启即 401。"""
    monkeypatch.setenv("API_AUTH_TOKEN", "sk-guard-test")
    get_settings.cache_clear()
    monkeypatch.setattr(guard_module, "_guard", AbuseGuard())
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(FakeRetriever(), FakeLLM())
    client = TestClient(app)
    try:
        for path in ("/ask", "/playground/ask", "/retrieve"):
            assert client.post(path, json={"question": "q"}).status_code == 401, path
        ok = client.post(
            "/ask", json={"question": "等待期多少天？"},
            headers={"authorization": "Bearer sk-guard-test"},
        )
        assert ok.status_code == 200
        # 廉价端点不设闸：健康探针和静态页不能被鉴权挡住
        assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.clear()
