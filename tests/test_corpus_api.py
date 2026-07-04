"""语料 API：/corpus 总览与 /corpus/{product}/chunks 切片浏览数据。"""

import json

from fastapi.testclient import TestClient

from app.api.routes import get_indexer
from app.main import app

client = TestClient(app)

REPORT = {
    "product": "Demo",
    "file": "Demo_tc.pdf",
    "sha256": "ab12",
    "generated_at": "2026-07-03T12:00:00+00:00",
    "total_pages": 2,
    "parsed_pages": 1,
    "empty_pages": [2],
    "pages": [{"page": 1, "chars": 7, "cjk_ratio": 0.9, "two_column": False}],
    "strategies": {
        "structural": {
            "n_chunks": 2,
            "clause_coverage": 0.5,
            "flag_counts": {"no_clause": 1, "cut_midsentence": 1},
            "chunks": [
                {
                    "chunk_id": "Demo:structural:0000",
                    "seq": 0,
                    "page_start": 1,
                    "page_end": 1,
                    "section": "第1条 等候期",
                    "clause": "第1条 等候期",
                    "chars": 12,
                    "flags": ["cut_midsentence"],
                    "boundary": {"tail": "期内确诊", "next_head": "不予赔付。", "cut": True},
                },
                {
                    "chunk_id": "Demo:structural:0001",
                    "seq": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "section": "",
                    "clause": "",
                    "chars": 5,
                    "flags": ["no_clause"],
                    "boundary": None,
                },
            ],
        }
    },
}


class FakeCollection:
    def get(self, ids, include):
        return {"ids": list(ids), "documents": [f"text-of-{i}" for i in ids]}


class FakeIndexer:
    def collection(self, strategy):
        return FakeCollection()


def _write_report(tmp_path, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    reports = tmp_path / "index" / "reports"
    reports.mkdir(parents=True)
    (reports / "Demo.json").write_text(json.dumps(REPORT, ensure_ascii=False), encoding="utf-8")


def test_corpus_lists_documents_without_chunk_detail(tmp_path, monkeypatch):
    _write_report(tmp_path, monkeypatch)
    resp = client.get("/corpus")
    assert resp.status_code == 200
    [doc] = resp.json()["documents"]
    assert doc["product"] == "Demo"
    assert doc["empty_pages"] == [2]
    assert doc["strategies"]["structural"]["clause_coverage"] == 0.5
    assert "chunks" not in doc["strategies"]["structural"]  # 总览不带逐块明细


def test_corpus_empty_when_no_reports(tmp_path, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    get_settings.cache_clear()
    assert client.get("/corpus").json() == {"documents": []}


def test_corpus_chunks_merges_report_and_texts(tmp_path, monkeypatch):
    _write_report(tmp_path, monkeypatch)
    app.dependency_overrides[get_indexer] = lambda: FakeIndexer()
    try:
        resp = client.get("/corpus/Demo/chunks?strategy=structural")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["clause_coverage"] == 0.5
    assert body["flag_counts"]["cut_midsentence"] == 1
    first = body["chunks"][0]
    assert first["text"] == "text-of-Demo:structural:0000"  # 全文来自 Chroma
    assert first["boundary"]["cut"] is True  # lint 结果来自报告
    assert body["chunks"][1]["flags"] == ["no_clause"]


def test_corpus_chunks_unknown_product_404(tmp_path, monkeypatch):
    _write_report(tmp_path, monkeypatch)
    app.dependency_overrides[get_indexer] = lambda: FakeIndexer()
    try:
        assert client.get("/corpus/Nope/chunks").status_code == 404
        assert client.get("/corpus/../chunks").status_code == 404  # 路径穿越被白名单挡住
    finally:
        app.dependency_overrides.clear()


def test_corpus_chunks_rejects_unknown_strategy(tmp_path, monkeypatch):
    _write_report(tmp_path, monkeypatch)
    app.dependency_overrides[get_indexer] = lambda: FakeIndexer()
    try:
        resp = client.get("/corpus/Demo/chunks?strategy=magic")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 422
