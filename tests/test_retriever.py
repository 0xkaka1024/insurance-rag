import pytest

from app.core.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.indexer import ChromaIndexer
from app.rag.retriever import VectorRetriever


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


@pytest.fixture
def retriever(tmp_path) -> VectorRetriever:
    settings = Settings(_env_file=None, index_dir=tmp_path / "index", top_k=2)
    indexer = ChromaIndexer(settings)
    embedder = KeywordEmbedder()
    indexer.index(
        [
            _chunk(0, "等待期为90天，期内确诊不予赔付。"),
            _chunk(1, "住院现金保障每日限额。"),
            _chunk(2, "身故赔偿金给付条件。"),
        ],
        embedder,
    )
    return VectorRetriever(indexer, embedder, settings)


def test_retrieve_ranks_relevant_first(retriever):
    chunks = retriever.retrieve("等待期多少天")
    assert chunks[0].chunk_id == "Demo:fixed:0000"
    assert "等待期" in chunks[0].text
    assert chunks[0].score > chunks[1].score


def test_retrieve_respects_top_k(retriever):
    assert len(retriever.retrieve("等待期")) == 2  # settings.top_k = 2
    assert len(retriever.retrieve("等待期", top_k=1)) == 1


def test_retrieved_chunk_carries_citation_fields(retriever):
    top = retriever.retrieve("等待期多少天")[0]
    assert top.product == "Demo"
    assert top.page_start == 1
    assert 0 <= top.score <= 1
