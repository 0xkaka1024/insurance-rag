"""Chroma 索引：每种切片策略一个 collection（clauses_fixed / clauses_structural ...）。

chunk_id 确定性生成（产品+策略+序号），upsert 语义天然幂等；
文件级 hash 判重（跳过未变更文件）在 D2 入库脚本层实现。
"""

import chromadb

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.ingest.chunker import Chunk


class ChromaIndexer:
    def __init__(self, settings: Settings | None = None):
        s = settings or get_settings()
        s.index_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(s.index_dir / "chroma"))

    def collection(self, strategy: str):
        return self._client.get_or_create_collection(
            f"clauses_{strategy}",
            metadata={"hnsw:space": "cosine"},  # bge-m3 用余弦相似度
        )

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
            total += len(group)
        return total
