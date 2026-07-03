"""检索器：vector（Chroma 余弦）与 hybrid（向量 + BM25 + RRF 融合）两种模式。

score 语义随模式而变：vector 为余弦相似度，hybrid 为 RRF 融合分（只可比排序，
不可跨模式比大小）；拒答阈值统一用 rerank 分数（D3），不依赖这里的 score。
"""

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.ingest.indexer import Indexer
from app.rag.fusion import rrf_fuse


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    product: str
    page_start: int
    page_end: int
    score: float


class Retriever:
    def __init__(
        self,
        indexer: Indexer,
        embedder: EmbeddingClient,
        settings: Settings | None = None,
    ):
        self._indexer = indexer
        self._embedder = embedder
        self._settings = settings or get_settings()

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        strategy: str = "fixed",
        mode: str = "vector",
    ) -> list[RetrievedChunk]:
        k = top_k or self._settings.top_k
        if mode == "vector":
            return self._vector(question, k, strategy)
        if mode == "hybrid":
            return self._hybrid(question, k, strategy)
        raise ValueError(f"unknown retrieval mode: {mode}")

    def _vector(self, question: str, k: int, strategy: str) -> list[RetrievedChunk]:
        collection = self._indexer.collection(strategy)
        n = min(k, collection.count())
        if n == 0:
            return []
        [query_vec] = self._embedder.embed([question])
        res = collection.query(
            query_embeddings=[query_vec],
            n_results=n,
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

    def _hybrid(self, question: str, k: int, strategy: str) -> list[RetrievedChunk]:
        recall = max(self._settings.recall_k, k)
        vec = self._vector(question, recall, strategy)
        kw = self._indexer.bm25(strategy).search(question, recall)

        payload: dict[str, tuple[str, dict]] = {
            c.chunk_id: (
                c.text,
                {"product": c.product, "page_start": c.page_start, "page_end": c.page_end},
            )
            for c in vec
        }
        for cid, text, meta, _score in kw:
            payload.setdefault(cid, (text, meta))

        fused = rrf_fuse([[c.chunk_id for c in vec], [cid for cid, *_ in kw]])
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for cid, score in ranked:
            text, meta = payload[cid]
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    product=str(meta["product"]),
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    score=round(score, 6),
                )
            )
        return out
