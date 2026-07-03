from fastapi.testclient import TestClient

from app.api.routes import get_pipeline
from app.main import app
from app.rag.pipeline import RagConfig, RagPipeline
from app.rag.retriever import RetrievedChunk
from tests.test_rerank_pipeline import SETTINGS, FakeReranker, FakeRetriever


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="Demo:fixed:0000",
        text="等待期为90天。",
        product="Demo",
        page_start=4,
        page_end=4,
        score=0.9,
        section="住院保障",
    )


class StreamingLLM:
    def stream(self, system, user):
        yield "等待期为"
        yield "90天[1]。"

    def complete(self, system, user):
        return "等待期为90天[1]。"


def _pipeline() -> RagPipeline:
    return RagPipeline(FakeRetriever([_chunk()]), StreamingLLM(), FakeReranker(), SETTINGS)


def test_ask_stream_event_sequence():
    events = list(_pipeline().ask_stream("等待期多少天", RagConfig()))
    names = [name for name, _ in events]
    assert names == ["chunks", "delta", "delta", "final"]
    final = events[-1][1]
    assert final["answer"] == "等待期为90天[Demo-住院保障-第4页]。"
    assert final["citations"][0]["chunk_id"] == "Demo:fixed:0000"
    assert final["refused"] is False
    deltas = "".join(d["text"] for n, d in events if n == "delta")
    assert deltas == "等待期为90天[1]。"  # 过程流原始编号


def test_ask_stream_premium_refusal_has_no_delta():
    events = list(_pipeline().ask_stream("30岁买每年多少钱", RagConfig()))
    names = [name for name, _ in events]
    assert names == ["chunks", "final"]
    assert events[-1][1]["refused"] is True
    assert events[-1][1]["refuse_reason"] == "premium_intent"


def test_ask_endpoint_streams_sse():
    app.dependency_overrides[get_pipeline] = _pipeline
    try:
        resp = TestClient(app).post(
            "/ask", json={"question": "等待期多少天", "stream": True}
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    body = resp.text
    assert "event: chunks" in body
    assert "event: delta" in body
    assert "event: final" in body
    assert "Demo-住院保障-第4页" in body


def test_ask_endpoint_non_stream_unchanged():
    app.dependency_overrides[get_pipeline] = _pipeline
    try:
        resp = TestClient(app).post("/ask", json={"question": "等待期多少天"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    assert resp.json()["answer"] == "等待期为90天[Demo-住院保障-第4页]。"
