"""就绪自检：probe_readiness 纯函数、/ready 端点、启动 fail-fast。"""

import pytest
from fastapi.testclient import TestClient

from app.api import routes
from app.api.routes import get_indexer, probe_readiness
from app.core.config import Settings, get_settings
from app.main import app

client = TestClient(app)


class FakeCollection:
    def __init__(self, count: int):
        self._count = count

    def count(self) -> int:
        return self._count


class FakeIndexer:
    def __init__(self, count: int = 5):
        self._count = count

    def collection(self, strategy: str) -> FakeCollection:
        return FakeCollection(self._count)


def _settings(tmp_path, *, keys=True, **kw) -> Settings:
    return Settings(
        _env_file=None,
        index_dir=tmp_path,
        deepseek_api_key="x" if keys else "",
        siliconflow_api_key="y" if keys else "",
        **kw,
    )


def _make_bm25(tmp_path):
    (tmp_path / "bm25_fixed.pkl").write_bytes(b"x")
    (tmp_path / "bm25_structural.pkl").write_bytes(b"x")


def test_probe_ready_when_index_and_keys_present(tmp_path):
    _make_bm25(tmp_path)
    r = probe_readiness(_settings(tmp_path), FakeIndexer(count=5))
    assert r["ready"] is True
    assert r["index_ok"] is True and r["keys_ok"] is True
    assert r["strategies"]["structural"] == {"chunks": 5, "bm25": True}


def test_probe_not_ready_when_bm25_missing(tmp_path):
    r = probe_readiness(_settings(tmp_path), FakeIndexer(count=5))  # 未建 bm25 文件
    assert r["index_ok"] is False
    assert r["strategies"]["fixed"]["bm25"] is False


def test_probe_not_ready_when_collection_empty(tmp_path):
    _make_bm25(tmp_path)
    r = probe_readiness(_settings(tmp_path), FakeIndexer(count=0))
    assert r["index_ok"] is False  # 空 collection = 索引缺失，显性暴露


def test_probe_keys_missing_marks_not_ready_but_index_ok(tmp_path):
    _make_bm25(tmp_path)
    r = probe_readiness(_settings(tmp_path, keys=False), FakeIndexer(count=5))
    assert r["index_ok"] is True  # 索引本身没问题（不阻塞启动）
    assert r["keys_ok"] is False and r["ready"] is False  # 但整体未就绪


def test_probe_survives_indexer_exception(tmp_path):
    class Boom:
        def collection(self, strategy):
            raise RuntimeError("chroma down")

    r = probe_readiness(_settings(tmp_path), Boom())
    assert r["index_ok"] is False
    assert "chroma down" in r["error"]


def test_ready_endpoint_200_when_ready(tmp_path, monkeypatch):
    _make_bm25(tmp_path)
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "x")
    monkeypatch.setenv("SILICONFLOW_API_KEY", "y")
    get_settings.cache_clear()
    app.dependency_overrides[get_indexer] = lambda: FakeIndexer(count=5)
    try:
        resp = client.get("/ready")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_ready_endpoint_503_when_not_ready(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))  # 无 bm25 文件 → 未就绪
    get_settings.cache_clear()
    app.dependency_overrides[get_indexer] = lambda: FakeIndexer(count=0)
    try:
        resp = client.get("/ready")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json()["detail"]["ready"] is False


def test_health_stays_shallow():
    assert client.get("/health").json() == {"status": "ok"}  # 不受索引状态影响


def test_lifespan_fail_fast_when_required(tmp_path, monkeypatch):
    """STARTUP_REQUIRE_INDEX=true 且索引空 → 启动直接抛错，容器起不来。"""
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("STARTUP_REQUIRE_INDEX", "true")
    monkeypatch.setattr(routes, "get_indexer", lambda: FakeIndexer(count=0))
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="索引未就绪"), TestClient(app):
        pass  # 进入 context 触发 lifespan → fail-fast


def test_lifespan_starts_when_not_required(tmp_path, monkeypatch):
    """默认不强制：索引空只告警，服务照常起（本地/CI 场景）。"""
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("STARTUP_REQUIRE_INDEX", raising=False)
    monkeypatch.setattr(routes, "get_indexer", lambda: FakeIndexer(count=0))
    get_settings.cache_clear()
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200
