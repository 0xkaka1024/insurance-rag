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
    section: str = ""  # 条号（structural 条款模式）或章节标题（简介模式），fixed 无
    # ── 检索过程溯源（Playground 透明化）：None = 该路未命中 / 该阶段未执行 ──
    vector_rank: int | None = None  # 向量路名次（1 起）
    vector_score: float | None = None  # 余弦相似度
    bm25_rank: int | None = None  # BM25 路名次（1 起）
    bm25_score: float | None = None  # BM25 原始分
    retrieval_rank: int | None = None  # 进入重排前的粗排位次（vector 序或 RRF 融合序）
    rerank_score: float | None = None  # rerank 成功后回填（与 score 同值）


def _section_of(meta: dict) -> str:
    return str(meta.get("clause") or meta.get("section") or "")


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
        rows = zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
            strict=True,
        )
        for rank, (chunk_id, doc, meta, dist) in enumerate(rows, start=1):
            score = 1.0 - dist  # chroma cosine distance = 1 - 相似度
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=doc,
                    product=str(meta["product"]),
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    score=score,
                    section=_section_of(meta),
                    vector_rank=rank,
                    vector_score=round(score, 6),
                    retrieval_rank=rank,
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
                {
                    "product": c.product,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "section": c.section,
                },
            )
            for c in vec
        }
        for cid, text, meta, _score in kw:
            payload.setdefault(cid, (text, meta))

        vec_pos = {c.chunk_id: (i + 1, round(c.score, 6)) for i, c in enumerate(vec)}
        kw_pos = {cid: (i + 1, round(s, 6)) for i, (cid, _t, _m, s) in enumerate(kw)}

        fused = rrf_fuse([[c.chunk_id for c in vec], [cid for cid, *_ in kw]])
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        out: list[RetrievedChunk] = []
        for fused_rank, (cid, score) in enumerate(ranked, start=1):
            text, meta = payload[cid]
            v, b = vec_pos.get(cid), kw_pos.get(cid)
            out.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    product=str(meta["product"]),
                    page_start=int(meta["page_start"]),
                    page_end=int(meta["page_end"]),
                    score=round(score, 6),
                    section=_section_of(meta),
                    vector_rank=v[0] if v else None,
                    vector_score=v[1] if v else None,
                    bm25_rank=b[0] if b else None,
                    bm25_score=b[1] if b else None,
                    retrieval_rank=fused_rank,
                )
            )
        return out
