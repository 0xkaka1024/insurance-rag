from fastapi.testclient import TestClient

from app.api.routes import get_pipeline
from app.main import app
from app.rag.pipeline import RagPipeline, build_user_prompt
from app.rag.retriever import RetrievedChunk

client = TestClient(app)


def _chunk(text: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="Demo:fixed:0000",
        text=text,
        product="Demo",
        page_start=3,
        page_end=3,
        score=score,
    )


class FakeRetriever:
    def retrieve(self, question, top_k=None, strategy="fixed"):
        return [_chunk("等待期为90天。")]


class FakeLLM:
    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        return "等待期为90天。"


def test_build_user_prompt_contains_chunks_and_question():
    prompt = build_user_prompt([_chunk("等待期为90天。")], "等待期多少天？")
    assert "等待期为90天。" in prompt
    assert "（Demo 第3页）" in prompt
    assert "等待期多少天？" in prompt


def test_pipeline_ask_returns_answer_chunks_timings():
    llm = FakeLLM()
    result = RagPipeline(FakeRetriever(), llm).ask("等待期多少天？")
    assert result.answer == "等待期为90天。"
    assert len(result.chunks) == 1
    assert set(result.timings) == {"retrieve_ms", "generate_ms", "total_ms"}
    system, user = llm.prompts[0]
    assert "无法回答" in system  # 拒答指令在 system prompt 中
    assert "等待期为90天。" in user


def test_ask_endpoint(monkeypatch):
    app.dependency_overrides[get_pipeline] = lambda: RagPipeline(FakeRetriever(), FakeLLM())
    try:
        resp = client.post("/ask", json={"question": "等待期多少天？"})
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "等待期为90天。"
    assert body["chunks"][0]["product"] == "Demo"
    assert body["timings"]["total_ms"] >= 0


def test_ask_endpoint_validates_question():
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
