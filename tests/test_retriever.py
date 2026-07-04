import pytest

from app.core.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.indexer import Indexer
from app.rag.retriever import Retriever


class KeywordEmbedder:
    """确定性向量：按关键词命中构造，可控相似度、不联网。"""

    KEYWORDS = ("等待期", "住院", "身故")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 if k in t else 0.0 for k in self.KEYWORDS] + [0.1] for t in texts]


def _chunk(seq: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"Demo:fixed:{seq:04d}",
        product="Demo",
        strategy="fixed",
        text=text,
        page_start=seq + 1,
        page_end=seq + 1,
    )


CORPUS = [
    "等待期为90天，期内确诊不予赔付。",
    "住院现金保障每日限额。",
    "身故赔偿金给付条件。",
    "产品代号 AVPU 适用特别条款。",
]


@pytest.fixture
def retriever(tmp_path) -> Retriever:
    settings = Settings(_env_file=None, index_dir=tmp_path / "index", top_k=2, recall_k=4)
    indexer = Indexer(settings)
    embedder = KeywordEmbedder()
    indexer.index([_chunk(i, t) for i, t in enumerate(CORPUS)], embedder)
    return Retriever(indexer, embedder, settings)


def test_vector_ranks_relevant_first(retriever):
    chunks = retriever.retrieve("等待期多少天", mode="vector")
    assert chunks[0].chunk_id == "Demo:fixed:0000"
    assert chunks[0].score > chunks[1].score


def test_vector_respects_top_k(retriever):
    assert len(retriever.retrieve("等待期")) == 2  # settings.top_k = 2
    assert len(retriever.retrieve("等待期", top_k=1)) == 1


def test_citation_fields(retriever):
    top = retriever.retrieve("等待期多少天")[0]
    assert top.product == "Demo"
    assert top.page_start == 1
    assert 0 <= top.score <= 1


def test_hybrid_agreement_ranks_first(retriever):
    chunks = retriever.retrieve("等待期", mode="hybrid")
    assert chunks[0].chunk_id == "Demo:fixed:0000"  # 向量与 BM25 都指向它


def test_hybrid_surfaces_exact_term_match(retriever):
    """向量语义弱、关键词精确的 query（产品代号），hybrid 必须捞回来。"""
    chunks = retriever.retrieve("AVPU", top_k=3, mode="hybrid")
    assert any(c.chunk_id == "Demo:fixed:0003" for c in chunks)


def test_vector_sets_provenance(retriever):
    chunks = retriever.retrieve("等待期多少天", mode="vector")
    top = chunks[0]
    assert top.vector_rank == 1
    assert top.retrieval_rank == 1
    assert top.vector_score == round(top.score, 6)
    assert top.bm25_rank is None and top.bm25_score is None  # 纯向量路无 BM25 溯源
    assert top.rerank_score is None  # 未重排
    assert [c.retrieval_rank for c in chunks] == [1, 2]


def test_hybrid_sets_both_path_provenance(retriever):
    chunks = retriever.retrieve("等待期", mode="hybrid")
    top = chunks[0]  # 向量与 BM25 双路命中的块，两路名次都应可见
    assert top.chunk_id == "Demo:fixed:0000"
    assert top.vector_rank == 1
    assert top.bm25_rank == 1
    assert top.bm25_score > 0
    assert top.vector_score is not None
    assert [c.retrieval_rank for c in chunks] == [1, 2]  # RRF 融合序


def test_unknown_mode_raises(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("等待期", mode="magic")


def test_vector_on_empty_collection(tmp_path):
    settings = Settings(_env_file=None, index_dir=tmp_path / "empty")
    retriever = Retriever(Indexer(settings), KeywordEmbedder(), settings)
    assert retriever.retrieve("等待期") == []
