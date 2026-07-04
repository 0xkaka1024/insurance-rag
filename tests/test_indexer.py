import pytest

from app.core.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.indexer import Indexer


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
def indexer(tmp_path) -> Indexer:
    # 夹具产品 "Demo" 不在白名单：关闭索引口白名单断言，专测索引机制本身
    return Indexer(
        settings=Settings(
            _env_file=None, index_dir=tmp_path / "index", whitelist_enforce_at_index=False
        )
    )


def test_index_rejects_non_whitelisted_product_by_default(tmp_path):
    """纵深防御：默认设置下白名单外产品写不进索引（Indexer 入口二次断言）。"""
    strict = Indexer(settings=Settings(_env_file=None, index_dir=tmp_path / "strict"))
    with pytest.raises(ValueError, match="白名单外"):
        strict.index([_chunk(0, "内部材料内容")], FakeEmbedder())


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
