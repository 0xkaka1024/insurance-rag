from app.rag.citations import citation_label, render_citations
from app.rag.pipeline import RagConfig, RagPipeline
from app.rag.retriever import RetrievedChunk
from tests.test_rerank_pipeline import SETTINGS, FakeReranker, FakeRetriever


def _chunk(seq: int, section: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"Demo:structural:{seq:04d}",
        text=f"内容{seq}",
        product="newVHISmedical",
        page_start=4,
        page_end=4,
        score=0.9,
        section=section,
    )


def test_label_with_section_and_without():
    assert citation_label(_chunk(0, "住院保障")) == "newVHISmedical-住院保障-第4页"
    assert citation_label(_chunk(0, "第三条")) == "newVHISmedical-第三条-第4页"
    assert citation_label(_chunk(0)) == "newVHISmedical-第4页"


def test_label_page_range():
    c = _chunk(0, "住院保障")
    c.page_end = 6
    assert citation_label(c) == "newVHISmedical-住院保障-第4-6页"


def test_render_substitutes_labels_and_collects():
    chunks = [_chunk(0, "住院保障"), _chunk(1, "诊断保障")]
    answer, cites = render_citations("等待期90天[1]。CT可赔[2][1]。", chunks)
    assert "[newVHISmedical-住院保障-第4页]" in answer
    assert "[newVHISmedical-诊断保障-第4页]" in answer
    assert [c.index for c in cites] == [1, 2]  # 去重且按首次出现排序
    assert cites[0].chunk_id == "Demo:structural:0000"


def test_render_strips_hallucinated_index():
    answer, cites = render_citations("论断[7]。", [_chunk(0)])
    assert "[7]" not in answer
    assert cites == []


def test_render_no_markers():
    answer, cites = render_citations("没有引用的回答。", [_chunk(0)])
    assert answer == "没有引用的回答。"
    assert cites == []


class CitingLLM:
    def complete(self, system, user):
        assert "[1]" in user  # 片段编号出现在 prompt 中
        assert "禁止编造编号" in system
        return "等待期为90天[1]。"


def test_pipeline_renders_citations():
    chunks = [_chunk(0, "住院保障")]
    pipe = RagPipeline(FakeRetriever(chunks), CitingLLM(), FakeReranker(), SETTINGS)
    result = pipe.ask("等待期多少天", RagConfig())
    assert result.answer == "等待期为90天[newVHISmedical-住院保障-第4页]。"
    assert len(result.citations) == 1
    assert result.citations[0].label == "newVHISmedical-住院保障-第4页"
