import json
import logging
import threading
import time

from fastapi.testclient import TestClient

from app.api import routes
from app.core.logging import JsonFormatter, request_id_var
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_request_id_generated():
    resp = client.get("/health")
    assert len(resp.headers["x-request-id"]) == 12


def test_request_id_honors_inbound_header():
    resp = client.get("/health", headers={"x-request-id": "abc123"})
    assert resp.headers["x-request-id"] == "abc123"


def test_unhandled_exception_returns_json_with_request_id():
    """非流式 500 不再是纯文本：JSON 错误体带 request_id，前端可解析可报障。"""

    class BoomPipeline:
        def ask(self, q, cfg):
            raise RuntimeError("boom")

    routes_app_client = TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides[routes.get_pipeline] = lambda: BoomPipeline()
    try:
        resp = routes_app_client.post(
            "/ask", json={"question": "q"}, headers={"x-request-id": "rid-err-1"}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 500
    body = resp.json()
    assert body["request_id"] == "rid-err-1"
    assert "detail" in body


def test_get_pipeline_cold_start_builds_serially(monkeypatch):
    """冷启动并发首请求不得并行构建 pipeline（chromadb 共享系统注册表非线程安全）。"""
    active, max_active = 0, 0
    gauge = threading.Lock()

    def slow_build(settings):
        nonlocal active, max_active
        with gauge:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)  # 无串行化保护时，并发线程必然在此重叠
        with gauge:
            active -= 1
        return object()

    monkeypatch.setattr(routes, "build_pipeline", slow_build)
    routes.get_pipeline.cache_clear()
    threads = [threading.Thread(target=routes.get_pipeline) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max_active == 1


def test_json_formatter_emits_valid_json_with_request_id():
    token = request_id_var.set("req-42")
    try:
        record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello 中文", None, None)
        record.extra_fields = {"elapsed_ms": 3.5}
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)
    entry = json.loads(line)
    assert entry["request_id"] == "req-42"
    assert entry["msg"] == "hello 中文"
    assert entry["elapsed_ms"] == 3.5
