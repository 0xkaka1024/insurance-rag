"""检索器。D1 仅向量检索；BM25 + RRF 混合在 D2 加入，接口保持不变。"""

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.ingest.indexer import ChromaIndexer


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    product: str
    page_start: int
    page_end: int
    score: float  # 余弦相似度，越大越相关


class VectorRetriever:
    def __init__(
        self,
        indexer: ChromaIndexer,
        embedder: EmbeddingClient,
        settings: Settings | None = None,
    ):
        self._indexer = indexer
        self._embedder = embedder
        self._settings = settings or get_settings()

    def retrieve(self, question: str, top_k: int | None = None, strategy: str = "fixed"):
        k = top_k or self._settings.top_k
        [query_vec] = self._embedder.embed([question])
        res = self._indexer.collection(strategy).query(
            query_embeddings=[query_vec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
            strict=True,
        ):
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    product=str(meta["product"]),
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    score=1.0 - dist,  # chroma cosine distance = 1 - 相似度
                )
            )
        return chunks
