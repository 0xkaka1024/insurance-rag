import pytest

from app.rag.pipeline import RagConfig, RagPipeline
from app.rag.preprocess import REFUSAL_PREMIUM, normalize_terms, route
from tests.test_rerank_pipeline import SETTINGS, FakeLLM, FakeReranker, FakeRetriever


@pytest.mark.parametrize(
    "q",
    [
        "30岁买每年多少钱",
        "这个计划保费多少",
        "一年要交几多钱",
        "帮我报价",
        "月缴金额是多少",
        "40 岁投保大概多少钱",
    ],
)
def test_premium_questions_intercepted(q):
    assert route(q).kind == "premium_refuse"


@pytest.mark.parametrize(
    "q",
    [
        "等待期多少天",  # 条款数值，不是钱
        "住院赔偿限额是多少",  # 限额是保障金额，可从条款回答
        "自付费有哪些档位可以选",
        "保费会不会随年龄调整",  # 问机制不问数字
        "身故赔偿的给付条件是什么",
    ],
)
def test_clause_questions_pass_router(q):
    assert route(q).kind == "clause"


def test_normalize_colloquial_terms():
    assert normalize_terms("观察期内确诊怎么办") == "等待期内确诊怎么办"
    assert normalize_terms("有老毛病能买吗") == "有已有病症能买吗"
    assert normalize_terms("免赔额是多少档") == "自付费是多少档"


def test_route_normalizes_clause_question():
    r = route("免责期内住院费能赔吗")
    assert r.kind == "clause"
    assert "等待期" in r.question
    assert "住院保障" in r.question


def test_pipeline_premium_refusal_skips_retrieval():
    retriever = FakeRetriever([])
    pipe = RagPipeline(retriever, FakeLLM(), FakeReranker(), SETTINGS)
    result = pipe.ask("30岁买每年多少钱", RagConfig())
    assert result.refused
    assert result.refuse_reason == "premium_intent"
    assert result.answer == REFUSAL_PREMIUM
    assert retriever.calls == []  # 检索都没发生，零成本拦截
