"""问答 pipeline（D1 最小版）：向量检索 top-k → LLM 生成。

引用强制格式、拒答阈值、意图路由在 D3 引入；config 驱动的策略切换在 D2/D4 引入。
"""

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from pydantic import BaseModel

from app.core.llm import LLMClient
from app.rag.retriever import RetrievedChunk, Retriever

logger = logging.getLogger("rag")


class RagConfig(BaseModel):
    """Playground 可插拔维度（查询期实时切换；入库期维度决定读哪个 collection）。"""

    chunking: Literal["fixed", "structural"] = "fixed"
    retrieval: Literal["vector", "hybrid"] = "vector"

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


class RagPipeline:
    def __init__(self, retriever: Retriever, llm: LLMClient):
        self._retriever = retriever
        self._llm = llm

    def ask(self, question: str, config: RagConfig | None = None) -> AskResult:
        cfg = config or RagConfig()
        t0 = perf_counter()
        chunks = self._retriever.retrieve(
            question, strategy=cfg.chunking, mode=cfg.retrieval
        )
        t1 = perf_counter()
        answer = self._llm.complete(SYSTEM_PROMPT, build_user_prompt(chunks, question))
        t2 = perf_counter()
        timings = {
            "retrieve_ms": round((t1 - t0) * 1000, 1),
            "generate_ms": round((t2 - t1) * 1000, 1),
            "total_ms": round((t2 - t0) * 1000, 1),
        }
        top_score = chunks[0].score if chunks else None
        logger.info(
            "ask served",
            extra={
                "extra_fields": {**timings, "top_score": top_score, **cfg.model_dump()}
            },
        )
        return AskResult(answer=answer, chunks=chunks, timings=timings, config=cfg)
