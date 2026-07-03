from app.ingest.bm25_index import BM25Index, tokenize
from app.ingest.chunker import Chunk
from app.rag.fusion import rrf_fuse


def _chunk(seq: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"Demo:fixed:{seq:04d}",
        product="Demo",
        strategy="fixed",
        text=text,
        page_start=1,
        page_end=1,
    )


def test_tokenize_chinese_and_latin():
    tokens = tokenize("等待期为90天 AVPU Scheme")
    assert "等待期" in tokens
    assert "avpu" in tokens  # 统一小写


def test_search_hits_exact_term(tmp_path):
    idx = BM25Index(tmp_path, "fixed")
    idx.upsert(
        [
            _chunk(0, "等待期为90天，期内确诊不予赔付。"),
            _chunk(1, "产品代号 AVPU 适用特别条款。"),
            _chunk(2, "住院现金保障每日限额。"),
        ]
    )
    top_id, text, meta, score = idx.search("AVPU", k=1)[0]
    assert top_id == "Demo:fixed:0001"
    assert meta["product"] == "Demo"
    assert score > 0


def test_persistence_roundtrip(tmp_path):
    BM25Index(tmp_path, "fixed").upsert([_chunk(0, "等待期为90天。")])
    reloaded = BM25Index(tmp_path, "fixed")
    assert len(reloaded) == 1
    assert reloaded.search("等待期", k=1)[0][0] == "Demo:fixed:0000"


def test_upsert_replaces_same_id(tmp_path):
    idx = BM25Index(tmp_path, "fixed")
    idx.upsert([_chunk(0, "旧内容关于住院。")])
    idx.upsert([_chunk(0, "新内容关于等待期。")])
    assert len(idx) == 1
    assert idx.search("等待期", k=1)[0][1] == "新内容关于等待期。"


def test_empty_index_search(tmp_path):
    assert BM25Index(tmp_path, "fixed").search("等待期", k=5) == []


def test_rrf_doc_in_both_lists_wins():
    fused = rrf_fuse([["a", "x"], ["b", "x"]])
    assert fused["x"] > fused["a"]
    assert fused["x"] > fused["b"]


def test_rrf_rank_order_matters():
    fused = rrf_fuse([["a", "b", "c"]])
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_empty():
    assert rrf_fuse([]) == {}
