"""索引门面：向量（Chroma）+ 关键词（BM25）双存储，每种切片策略各一份。

chunk_id 确定性生成（产品+策略+序号），两个存储都是 upsert 语义、天然幂等；
文件级 hash 判重（跳过未变更文件）在入库服务层实现。
"""

import chromadb

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.ingest.bm25_index import BM25Index
from app.ingest.chunker import Chunk


class Indexer:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        s.index_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir = s.index_dir
        self._client = chromadb.PersistentClient(path=str(s.index_dir / "chroma"))
        self._bm25: dict[str, BM25Index] = {}

    def collection(self, strategy: str):
        return self._client.get_or_create_collection(
            f"clauses_{strategy}",
            metadata={"hnsw:space": "cosine"},  # bge-m3 用余弦相似度
        )

    def bm25(self, strategy: str) -> BM25Index:
        if strategy not in self._bm25:
            self._bm25[strategy] = BM25Index(self._index_dir, strategy)
        return self._bm25[strategy]

    def index(self, chunks: list[Chunk], embedder: EmbeddingClient) -> int:
        if not chunks:
            return 0
        by_strategy: dict[str, list[Chunk]] = {}
        for c in chunks:
            by_strategy.setdefault(c.strategy, []).append(c)
        total = 0
        for strategy, group in by_strategy.items():
            embeddings = embedder.embed([c.text for c in group])
            self.collection(strategy).upsert(
                ids=[c.chunk_id for c in group],
                documents=[c.text for c in group],
                embeddings=embeddings,
                metadatas=[
                    {
                        "product": c.product,
                        "page_start": c.page_start,
                        "page_end": c.page_end,
                        **c.meta,
                    }
                    for c in group
                ],
            )
            self.bm25(strategy).upsert(group)
            total += len(group)
        return total
