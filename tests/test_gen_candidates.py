from app.ingest.chunker import Chunk
from scripts.gen_eval_candidates import (
    candidates_from_chunk,
    parse_llm_json,
    sample_chunks,
)


def _chunk(seq: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"Demo:structural:{seq:04d}",
        product="Demo",
        strategy="structural",
        text=text,
        page_start=2,
        page_end=3,
        meta={"section": "住院保障"},
    )


def test_sample_chunks_deterministic_and_filters_short():
    chunks = [_chunk(i, "长内容" * (30 + i)) for i in range(10)] + [_chunk(99, "短")]
    a = sample_chunks(chunks, max_chunks=5)
    b = sample_chunks(chunks, max_chunks=5)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(a) == 5
    assert all(len(c.text) >= 80 for c in a)


def test_parse_llm_json_tolerates_fences():
    text = '好的，以下是结果：\n```json\n[{"question": "等待期多少天？", "type": "fact"}]\n```'
    items = parse_llm_json(text)
    assert items == [{"question": "等待期多少天？", "type": "fact"}]


def test_parse_llm_json_garbage_returns_empty():
    assert parse_llm_json("说不出 JSON") == []
    assert parse_llm_json("[not json]") == []


class FakeLLM:
    def complete(self, system, user):
        assert "住院保障" in user  # 章节上下文进 prompt
        return (
            '[{"question": "住院前门诊能赔吗？", "type": "fact", '
            '"difficulty": "easy", "draft_answer": "可赔，见片段"}]'
        )


def test_candidates_from_chunk_carries_source_fields():
    [cand] = candidates_from_chunk(FakeLLM(), _chunk(0, "内容" * 50), per_chunk=1)
    assert cand["question"] == "住院前门诊能赔吗？"
    assert cand["ground_truth"] == ""  # 必须人工核对，不预填
    assert cand["source_chunk_id"] == "Demo:structural:0000"
    assert cand["source_pages"] == "2-3"
    assert "人工" in cand["note"]
