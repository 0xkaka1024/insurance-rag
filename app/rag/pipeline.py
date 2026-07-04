"""问答 pipeline（D1 最小版）：向量检索 top-k → LLM 生成。

引用强制格式、拒答阈值、意图路由在 D3 引入；config 驱动的策略切换在 D2/D4 引入。
"""

import logging
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.core.llm import LLMClient
from app.ingest.indexer import Indexer
from app.rag.citations import Citation, render_citations
from app.rag.preprocess import REFUSAL_PREMIUM, route
from app.rag.reranker import RerankClient, RerankUnavailable
from app.rag.retriever import RetrievedChunk, Retriever

logger = logging.getLogger("rag")


class RagConfig(BaseModel):
    """Playground 可插拔维度（查询期实时切换；入库期维度决定读哪个 collection）。"""

    chunking: Literal["fixed", "structural"] = "fixed"
    retrieval: Literal["vector", "hybrid"] = "vector"
    rerank: bool = False


REFUSAL_LOW_SCORE = (
    "根据现有条款资料，我无法为这个问题找到足够可靠的依据，不能回答。"
    "建议确认相关产品文档是否已入库，或换一种问法。"
)
REFUSAL_NO_CONTEXT = "知识库中没有可用的条款资料，无法回答。请先上传相关产品文档。"

# LLM 按 system prompt 的指令做出的「有据拒答」开头话术；检测它以正确置位 refused
GROUNDED_REFUSAL_MARKER = "无法回答"


def _detect_grounded_refusal(p: "_Prepared", answer: str) -> None:
    if GROUNDED_REFUSAL_MARKER in answer[:40]:
        p.refused, p.refuse_reason = True, "no_evidence"

SYSTEM_PROMPT = (
    "你是保险条款问答助手，服务对象是保险代理人。"
    "只能依据【条款片段】中的内容回答问题，禁止编造或使用片段之外的知识。"
    "每个事实论断的句末必须标注来源片段编号，格式如 [1]，多个来源写作 [1][2]；"
    "只允许引用已提供的编号，禁止编造编号。"
    "如果片段中没有足够依据，直接回答：根据现有条款资料无法回答该问题。"
)


def build_user_prompt(chunks: list[RetrievedChunk], question: str) -> str:
    parts = ["【条款片段】"]
    for i, c in enumerate(chunks, start=1):
        if c.page_start == c.page_end:
            loc = f"第{c.page_start}页"
        else:
            loc = f"第{c.page_start}-{c.page_end}页"
        head = f"{c.product}·{c.section}·{loc}" if c.section else f"{c.product}·{loc}"
        parts.append(f"[{i}]（{head}）\n{c.text}")
    parts.append(f"\n【问题】\n{question}")
    return "\n\n".join(parts)


@dataclass
class AskResult:
    answer: str
    chunks: list[RetrievedChunk]
    timings: dict[str, float]
    config: RagConfig
    refused: bool = False
    refuse_reason: str = ""  # premium_intent | low_score | no_context
    rerank_degraded: bool = False
    citations: list[Citation] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMClient,
        reranker: RerankClient | None = None,
        settings: Settings | None = None,
    ):
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._settings = settings or get_settings()

    def _rerank(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool]:
        """rerank 成功 → (精排后 top_k, False)；失败降级 → (粗排截断, True)。"""
        s = self._settings
        if self._reranker is None:
            raise RerankUnavailable("no reranker configured")
        pairs = self._reranker.rerank(question, [c.text for c in chunks], top_n=s.top_k)
        return [replace(chunks[i], score=score, rerank_score=score) for i, score in pairs], False

    def _prepare(self, question: str, config: RagConfig | None) -> "_Prepared":
        """路由 → 检索 → 重排/阈值：ask 与 ask_stream 共用的生成前阶段。"""
        cfg = config or RagConfig()
        s = self._settings
        timings: dict[str, float] = {}
        t0 = perf_counter()

        routed = route(question)
        if routed.kind == "premium_refuse":
            logger.info(
                "ask refused by router",
                extra={"extra_fields": {"matched": routed.matched, **cfg.model_dump()}},
            )
            return _Prepared(
                cfg=cfg, t0=t0, timings=timings, chunks=[], question=question,
                refused=True, refuse_reason="premium_intent", refusal_answer=REFUSAL_PREMIUM,
            )
        question = routed.question  # 术语归一后的问题参与检索与生成

        recall_k = s.recall_k if cfg.rerank else s.top_k
        chunks = self._retriever.retrieve(
            question, top_k=recall_k, strategy=cfg.chunking, mode=cfg.retrieval
        )
        t1 = perf_counter()
        timings["retrieve_ms"] = round((t1 - t0) * 1000, 1)

        if not chunks:
            return _Prepared(
                cfg=cfg, t0=t0, timings=timings, chunks=[], question=question,
                refused=True, refuse_reason="no_context", refusal_answer=REFUSAL_NO_CONTEXT,
            )

        degraded = False
        if cfg.rerank:
            try:
                chunks, degraded = self._rerank(question, chunks)
            except RerankUnavailable as exc:
                logger.warning("rerank degraded, falling back to retrieval order: %s", exc)
                chunks, degraded = chunks[: s.top_k], True
            timings["rerank_ms"] = round((perf_counter() - t1) * 1000, 1)
            # 拒答阈值只在 rerank 分数可用时生效（cross-encoder 分数才有绝对意义）
            if not degraded and chunks and chunks[0].score < s.refuse_threshold:
                return _Prepared(
                    cfg=cfg, t0=t0, timings=timings, chunks=chunks, question=question,
                    refused=True, refuse_reason="low_score",
                    refusal_answer=REFUSAL_LOW_SCORE, degraded=degraded,
                )
        return _Prepared(
            cfg=cfg, t0=t0, timings=timings, chunks=chunks, question=question,
            degraded=degraded,
        )

    def _log_served(self, p: "_Prepared") -> None:
        top_score = p.chunks[0].score if p.chunks else None
        logger.info(
            "ask served",
            extra={
                "extra_fields": {
                    **p.timings,
                    "top_score": top_score,
                    "refused": p.refused,
                    "refuse_reason": p.refuse_reason,
                    "rerank_degraded": p.degraded,
                    **p.cfg.model_dump(),
                }
            },
        )

    def _finish(self, p: "_Prepared", answer: str, citations: list[Citation]) -> AskResult:
        p.timings["total_ms"] = round((perf_counter() - p.t0) * 1000, 1)
        self._log_served(p)
        return AskResult(
            answer=answer,
            chunks=p.chunks,
            timings=p.timings,
            config=p.cfg,
            refused=p.refused,
            refuse_reason=p.refuse_reason,
            rerank_degraded=p.degraded,
            citations=citations,
        )

    def ask(self, question: str, config: RagConfig | None = None) -> AskResult:
        p = self._prepare(question, config)
        if p.refused:
            return self._finish(p, p.refusal_answer, [])
        t2 = perf_counter()
        usage: dict = {}
        with_usage = getattr(self._llm, "complete_with_usage", None)
        prompt = build_user_prompt(p.chunks, p.question)
        if with_usage:
            raw, usage = with_usage(SYSTEM_PROMPT, prompt)
        else:  # 测试用 Fake 只实现 complete
            raw = self._llm.complete(SYSTEM_PROMPT, prompt)
        answer, citations = render_citations(raw, p.chunks)
        p.timings["generate_ms"] = round((perf_counter() - t2) * 1000, 1)
        _detect_grounded_refusal(p, answer)
        if not citations and not p.refused:
            # 非拒答回答不带任何有效引用：红线告警，评测统计 citations==[]
            logger.warning("answer without citations", extra={"extra_fields": {}})
        result = self._finish(p, answer, citations)
        result.meta["usage"] = usage
        return result

    def ask_stream(self, question: str, config: RagConfig | None = None):
        """SSE 事件流：chunks →（delta ×N）→ final。

        流式与服务端引用回填的矛盾解法：过程流原始 [n] 增量，final 事件
        携带标签渲染后的全文与结构化引用，前端收到后整体替换。
        """
        p = self._prepare(question, config)
        yield (
            "chunks",
            {
                "chunks": [vars(c) for c in p.chunks],
                "config": p.cfg.model_dump(),
                "rerank_degraded": p.degraded,
            },
        )
        if p.refused:
            result = self._finish(p, p.refusal_answer, [])
        else:
            t2 = perf_counter()
            parts: list[str] = []
            for delta in self._llm.stream(SYSTEM_PROMPT, build_user_prompt(p.chunks, p.question)):
                parts.append(delta)
                yield "delta", {"text": delta}
            answer, citations = render_citations("".join(parts), p.chunks)
            p.timings["generate_ms"] = round((perf_counter() - t2) * 1000, 1)
            _detect_grounded_refusal(p, answer)
            if not citations and not p.refused:
                logger.warning("answer without citations", extra={"extra_fields": {}})
            result = self._finish(p, answer, citations)
        yield (
            "final",
            {
                "answer": result.answer,
                "citations": [vars(c) for c in result.citations],
                "timings": result.timings,
                "config": result.config.model_dump(),
                "refused": result.refused,
                "refuse_reason": result.refuse_reason,
                "rerank_degraded": result.rerank_degraded,
            },
        )


@dataclass
class _Prepared:
    cfg: RagConfig
    t0: float
    timings: dict[str, float]
    chunks: list[RetrievedChunk]
    question: str
    refused: bool = False
    refuse_reason: str = ""
    refusal_answer: str = ""
    degraded: bool = False


def build_pipeline(settings: Settings | None = None) -> RagPipeline:
    """组装完整管线；API 层与评测层共用同一工厂，保证行为一致。"""
    s = settings or get_settings()
    indexer = Indexer(s)
    return RagPipeline(
        retriever=Retriever(indexer, EmbeddingClient(s), s),
        llm=LLMClient(s),
        reranker=RerankClient(s),
        settings=s,
    )
