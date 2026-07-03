from app.core.config import Settings
from app.rag.pipeline import (
    REFUSAL_LOW_SCORE,
    REFUSAL_NO_CONTEXT,
    RagConfig,
    RagPipeline,
)
from app.rag.reranker import RerankUnavailable
from app.rag.retriever import RetrievedChunk


def _chunk(seq: int, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"Demo:fixed:{seq:04d}",
        text=text,
        product="Demo",
        page_start=1,
        page_end=1,
        score=score,
    )


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls: list[int] = []

    def retrieve(self, question, top_k=None, strategy="fixed", mode="vector"):
        self.calls.append(top_k)
        return self._chunks[:top_k]


class FakeLLM:
    def complete(self, system, user):
        return "答案。"


class FakeReranker:
    def __init__(self, pairs=None, fail=False):
        self._pairs = pairs or []
        self._fail = fail
        self.calls: list[int] = []

    def rerank(self, query, documents, top_n):
        self.calls.append(len(documents))
        if self._fail:
            raise RerankUnavailable("down")
        return self._pairs


SETTINGS = Settings(_env_file=None, top_k=2, recall_k=5, refuse_threshold=0.3)
CORPUS = [_chunk(i, f"内容{i}", 0.5) for i in range(5)]


def test_rerank_reorders_and_rescores():
    reranker = FakeReranker(pairs=[(3, 0.92), (0, 0.55)])
    pipe = RagPipeline(FakeRetriever(CORPUS), FakeLLM(), reranker, SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=True))
    assert not result.refused
    assert reranker.calls == [5]  # recall_k 全量送精排
    assert [c.chunk_id for c in result.chunks] == ["Demo:fixed:0003", "Demo:fixed:0000"]
    assert result.chunks[0].score == 0.92
    assert "rerank_ms" in result.timings


def test_low_rerank_score_refuses():
    reranker = FakeReranker(pairs=[(0, 0.12), (1, 0.08)])
    pipe = RagPipeline(FakeRetriever(CORPUS), FakeLLM(), reranker, SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=True))
    assert result.refused
    assert result.refuse_reason == "low_score"
    assert result.answer == REFUSAL_LOW_SCORE
    assert "generate_ms" not in result.timings  # 拒答不烧生成 token


def test_rerank_failure_degrades_not_crashes():
    reranker = FakeReranker(fail=True)
    pipe = RagPipeline(FakeRetriever(CORPUS), FakeLLM(), reranker, SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=True))
    assert not result.refused  # 降级后不做阈值拒答（粗排分数无绝对意义）
    assert result.rerank_degraded
    assert len(result.chunks) == SETTINGS.top_k  # 粗排截断到 top_k
    assert result.answer == "答案。"


def test_no_context_refuses():
    pipe = RagPipeline(FakeRetriever([]), FakeLLM(), FakeReranker(), SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=True))
    assert result.refused
    assert result.refuse_reason == "no_context"
    assert result.answer == REFUSAL_NO_CONTEXT


def test_rerank_off_uses_top_k_directly():
    retriever = FakeRetriever(CORPUS)
    pipe = RagPipeline(retriever, FakeLLM(), FakeReranker(), SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=False))
    assert retriever.calls == [SETTINGS.top_k]  # 不放大召回
    assert not result.refused
    assert "rerank_ms" not in result.timings


def test_missing_reranker_degrades():
    pipe = RagPipeline(FakeRetriever(CORPUS), FakeLLM(), reranker=None, settings=SETTINGS)
    result = pipe.ask("q", RagConfig(rerank=True))
    assert result.rerank_degraded
    assert result.answer == "答案。"


class GroundedRefusingLLM:
    def complete(self, system, user):
        return "根据现有条款资料无法回答该问题。片段仅提及可查阅官网[1]。"


def test_llm_grounded_refusal_sets_flag():
    pipe = RagPipeline(FakeRetriever(CORPUS), GroundedRefusingLLM(), FakeReranker(), SETTINGS)
    result = pipe.ask("分红实现率是多少", RagConfig())
    assert result.refused
    assert result.refuse_reason == "no_evidence"
    assert len(result.citations) == 1  # 有据拒答保留指路引用
