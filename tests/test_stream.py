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
    # /ask 生产配置 rerank=on：FakeReranker 需返回有效精排对，空对会清空 chunks
    reranker = FakeReranker(pairs=[(0, 0.9)])
    return RagPipeline(FakeRetriever([_chunk()]), StreamingLLM(), reranker, SETTINGS)


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


class NoCiteStreamLLM:
    def stream(self, system, user):
        yield "等待期为"
        yield "90天。"

    def complete(self, system, user):
        return "等待期为90天。"


def test_ask_stream_refuses_uncited_answer_in_final():
    """过程流可能已吐出无引用文本，final 事件必须整体替换为拒答话术。"""
    from app.rag.pipeline import REFUSAL_NO_CITATION

    pipe = RagPipeline(FakeRetriever([_chunk()]), NoCiteStreamLLM(), FakeReranker(), SETTINGS)
    events = list(pipe.ask_stream("等待期多少天", RagConfig()))
    final = events[-1][1]
    assert final["refused"] is True
    assert final["refuse_reason"] == "no_citation"
    assert final["answer"] == REFUSAL_NO_CITATION
    assert final["citations"] == []


class ExplodingStreamLLM:
    """第一段增量之后上游断掉：SSE 必须发 error 事件，不许裸断连。"""

    def stream(self, system, user):
        yield "等待期为"
        raise RuntimeError("upstream connection reset")

    def complete(self, system, user):
        raise RuntimeError("upstream connection reset")


def test_sse_emits_error_event_on_midstream_failure():
    pipe = RagPipeline(
        FakeRetriever([_chunk()]), ExplodingStreamLLM(), FakeReranker(pairs=[(0, 0.9)]), SETTINGS
    )
    app.dependency_overrides[get_pipeline] = lambda: pipe
    try:
        resp = TestClient(app).post(
            "/ask",
            json={"question": "等待期多少天", "stream": True},
            headers={"x-request-id": "rid-sse-1"},
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200  # SSE 已发响应头，错误走事件而非状态码
    body = resp.text
    assert "event: delta" in body  # 断掉前的增量已流出
    assert "event: error" in body  # 终结事件必须存在
    assert "rid-sse-1" in body  # error 事件带 request_id，可对日志
    assert "event: final" not in body
    assert resp.headers.get("x-accel-buffering") == "no"  # 反代不缓冲流式


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
