"""/retrieve 只检索端点：路由/检索/重排照常，绝不触发 LLM 生成。"""

from fastapi.testclient import TestClient

from app.api.routes import get_pipeline
from app.main import app
from app.rag.pipeline import RagPipeline
from app.rag.retriever import RetrievedChunk

client = TestClient(app)


class FakeRetriever:
    def __init__(self):
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=None, strategy="fixed", mode="vector"):
        self.calls.append({"strategy": strategy, "mode": mode})
        return [
            RetrievedChunk(
                chunk_id="Demo:fixed:0000",
                text="等待期为90天。",
                product="Demo",
                page_start=3,
                page_end=3,
                score=0.9,
                vector_rank=1,
                vector_score=0.9,
                retrieval_rank=1,
            )
        ]


class ExplodingLLM:
    """只检索模式绝不能触发生成——任何调用即测试失败。"""

    def complete(self, system, user):
        raise AssertionError("LLM must not be called in retrieve-only mode")

    def stream(self, system, user):
        raise AssertionError("LLM must not be called in retrieve-only mode")


def _override(retriever):
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(retriever, ExplodingLLM())


def test_retrieve_returns_chunks_without_generation():
    retriever = FakeRetriever()
    _override(retriever)
    try:
        resp = client.post(
            "/retrieve",
            json={
                "question": "等待期多少天？",
                "config": {"chunking": "structural", "retrieval": "hybrid"},
            },
        )
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert body["answer"] == ""  # 无生成内容
    assert "generate_ms" not in body["timings"]
    assert body["chunks"][0]["retrieval_rank"] == 1  # 溯源字段透出
    assert retriever.calls == [{"strategy": "structural", "mode": "hybrid"}]  # config 生效


def test_retrieve_premium_question_still_routed():
    retriever = FakeRetriever()
    _override(retriever)
    try:
        resp = client.post("/retrieve", json={"question": "30岁买每年多少钱"})
    finally:
        app.dependency_overrides.clear()
    body = resp.json()
    assert body["refused"] is True
    assert body["refuse_reason"] == "premium_intent"
    assert body["chunks"] == []
    assert body["answer"]  # 拒答话术随响应返回，前端可直接展示
    assert retriever.calls == []  # 路由拦截在检索之前


def test_retrieve_validates_question():
    assert client.post("/retrieve", json={"question": ""}).status_code == 422
