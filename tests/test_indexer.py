import pytest

from app.core.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.indexer import ChromaIndexer


class FakeEmbedder:
    """确定性向量：同文本同向量，不联网。"""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), float(hash(t) % 11), 1.0] for t in texts]


def _chunk(seq: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"Demo:fixed:{seq:04d}",
        product="Demo",
        strategy="fixed",
        text=text,
        page_start=1,
        page_end=1,
    )


@pytest.fixture
def indexer(tmp_path) -> ChromaIndexer:
    return ChromaIndexer(settings=Settings(_env_file=None, index_dir=tmp_path / "index"))


def test_index_writes_documents_and_metadata(indexer):
    n = indexer.index([_chunk(0, "等待期为90天。"), _chunk(1, "住院赔偿限额。")], FakeEmbedder())
    assert n == 2
    col = indexer.collection("fixed")
    assert col.count() == 2
    got = col.get(ids=["Demo:fixed:0000"], include=["metadatas", "documents"])
    assert got["documents"][0] == "等待期为90天。"
    assert got["metadatas"][0]["product"] == "Demo"
    assert got["metadatas"][0]["page_start"] == 1


def test_reingest_is_idempotent(indexer):
    chunks = [_chunk(0, "等待期为90天。")]
    indexer.index(chunks, FakeEmbedder())
    indexer.index(chunks, FakeEmbedder())
    assert indexer.collection("fixed").count() == 1


def test_query_returns_nearest(indexer):
    embedder = FakeEmbedder()
    chunks = [_chunk(0, "等待期为90天。"), _chunk(1, "完全不同的很长很长的文本内容")]
    indexer.index(chunks, embedder)
    res = indexer.collection("fixed").query(
        query_embeddings=embedder.embed(["等待期为90天。"]), n_results=1
    )
    assert res["ids"][0][0] == "Demo:fixed:0000"


def test_index_empty_is_noop(indexer):
    assert indexer.index([], FakeEmbedder()) == 0
