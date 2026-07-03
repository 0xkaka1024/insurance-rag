"""问答 pipeline（D1 最小版）：向量检索 top-k → LLM 生成。

引用强制格式、拒答阈值、意图路由在 D3 引入；config 驱动的策略切换在 D2/D4 引入。
"""

import logging
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.core.llm import LLMClient
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

SYSTEM_PROMPT = (
    "你是保险条款问答助手，服务对象是保险代理人。"
    "只能依据【条款片段】中的内容回答问题，禁止编造或使用片段之外的知识。"
    "如果片段中没有足够依据，直接回答：根据现有条款资料无法回答该问题。"
)


def build_user_prompt(chunks: list[RetrievedChunk], question: str) -> str:
    parts = ["【条款片段】"]
    for i, c in enumerate(chunks, start=1):
        if c.page_start == c.page_end:
            loc = f"第{c.page_start}页"
        else:
            loc = f"第{c.page_start}-{c.page_end}页"
        parts.append(f"[{i}] （{c.product} {loc}）\n{c.text}")
    parts.append(f"\n【问题】\n{question}")
    return "\n\n".join(parts)


@dataclass
class AskResult:
    answer: str
    chunks: list[RetrievedChunk]
    timings: dict[str, float]
    config: RagConfig
    refused: bool = False
    refuse_reason: str = ""  # low_score | no_context |（D3 路由）premium_intent
    rerank_degraded: bool = False
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
        return [replace(chunks[i], score=score) for i, score in pairs], False

    def ask(self, question: str, config: RagConfig | None = None) -> AskResult:
        cfg = config or RagConfig()
        s = self._settings
        timings: dict[str, float] = {}

        t0 = perf_counter()
        routed = route(question)
        if routed.kind == "premium_refuse":
            timings["total_ms"] = round((perf_counter() - t0) * 1000, 1)
            logger.info(
                "ask refused by router",
                extra={"extra_fields": {"matched": routed.matched, **cfg.model_dump()}},
            )
            return AskResult(
                answer=REFUSAL_PREMIUM,
                chunks=[],
                timings=timings,
                config=cfg,
                refused=True,
                refuse_reason="premium_intent",
            )
        question = routed.question  # 术语归一后的问题参与检索与生成

        recall_k = s.recall_k if cfg.rerank else s.top_k
        chunks = self._retriever.retrieve(
            question, top_k=recall_k, strategy=cfg.chunking, mode=cfg.retrieval
        )
        t1 = perf_counter()
        timings["retrieve_ms"] = round((t1 - t0) * 1000, 1)

        refused = False
        refuse_reason = ""
        degraded = False
        answer = ""

        if not chunks:
            refused, refuse_reason, answer = True, "no_context", REFUSAL_NO_CONTEXT

        if not refused and cfg.rerank:
            try:
                chunks, degraded = self._rerank(question, chunks)
            except RerankUnavailable as exc:
                logger.warning("rerank degraded, falling back to retrieval order: %s", exc)
                chunks, degraded = chunks[: s.top_k], True
            timings["rerank_ms"] = round((perf_counter() - t1) * 1000, 1)
            # 拒答阈值只在 rerank 分数可用时生效（cross-encoder 分数才有绝对意义）
            if not degraded and chunks and chunks[0].score < s.refuse_threshold:
                refused, refuse_reason, answer = True, "low_score", REFUSAL_LOW_SCORE

        if not refused:
            t2 = perf_counter()
            answer = self._llm.complete(SYSTEM_PROMPT, build_user_prompt(chunks, question))
            timings["generate_ms"] = round((perf_counter() - t2) * 1000, 1)

        timings["total_ms"] = round((perf_counter() - t0) * 1000, 1)
        top_score = chunks[0].score if chunks else None
        logger.info(
            "ask served",
            extra={
                "extra_fields": {
                    **timings,
                    "top_score": top_score,
                    "refused": refused,
                    "refuse_reason": refuse_reason,
                    "rerank_degraded": degraded,
                    **cfg.model_dump(),
                }
            },
        )
        return AskResult(
            answer=answer,
            chunks=chunks,
            timings=timings,
            config=cfg,
            refused=refused,
            refuse_reason=refuse_reason,
            rerank_degraded=degraded,
        )
