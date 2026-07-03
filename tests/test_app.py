import json
import logging

from fastapi.testclient import TestClient

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
