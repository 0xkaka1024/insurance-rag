"""引用构建与回填。

红线：回答中的事实论断必须带引用。可靠性设计——LLM 只输出 [1][2] 这类
片段编号（复述长标签极易出错），服务端把编号替换为完整标签
[产品-条号/章节-页码]，并返回结构化 citations 供前端点开原文。
"""

import re
from dataclasses import dataclass

from app.rag.retriever import RetrievedChunk

_CIT_RE = re.compile(r"\[(\d+)\]")


@dataclass
class Citation:
    index: int  # 片段编号（1-based，与 prompt 中一致）
    label: str  # 展示标签：产品-条号/章节-页码
    chunk_id: str


def citation_label(chunk: RetrievedChunk) -> str:
    if chunk.page_start == chunk.page_end:
        loc = f"第{chunk.page_start}页"
    else:
        loc = f"第{chunk.page_start}-{chunk.page_end}页"
    if chunk.section:
        return f"{chunk.product}-{chunk.section}-{loc}"
    return f"{chunk.product}-{loc}"


def render_citations(
    answer: str, chunks: list[RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """把回答中的 [n] 替换为完整标签；返回（渲染后回答, 有效引用列表）。

    无效编号（超出片段数 / 非法）直接剔除，不让幻觉编号流出。
    """
    citations: list[Citation] = []
    seen: set[int] = set()

    def _sub(m: re.Match) -> str:
        idx = int(m.group(1))
        if not 1 <= idx <= len(chunks):
            return ""  # 幻觉编号，剔除
        if idx not in seen:
            seen.add(idx)
            citations.append(
                Citation(
                    index=idx,
                    label=citation_label(chunks[idx - 1]),
                    chunk_id=chunks[idx - 1].chunk_id,
                )
            )
        return f"[{citation_label(chunks[idx - 1])}]"

    rendered = _CIT_RE.sub(_sub, answer)
    return rendered, citations
