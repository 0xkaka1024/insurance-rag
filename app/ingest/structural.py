"""StructuralChunker：层级感知切分。

双轨模式（对应混合语料决策，见 SPEC 开放问题 2026-07-03）：
- 条款模式：「第X章」维护章上下文，「第X条 / X.Y / X.」开启新条目
- 简介模式：无编号语料按标题启发式切段（短行、无句读、非数字表格行）

过长段落用 FixedChunker 二次切分，子块继承段落层级元数据。
"""

import re
from dataclasses import dataclass

from app.ingest.chunker import Chunk, FixedChunker
from app.ingest.parser import Page

CHAPTER_RE = re.compile(r"^第\s*[一二三四五六七八九十百0-9]+\s*章")
CLAUSE_RE = re.compile(r"^(第\s*[一二三四五六七八九十百0-9]+\s*条|\d+(\.\d+)+\s|\d+\.\s)")

_HEADING_MAX_LEN = 20
_SENTENCE_PUNCT = "。；，：、"
_MIN_SECTION_LEN = 30


def _is_heading(line: str) -> bool:
    """简介模式标题启发式：短、无句读、且不是数字为主的表格行。"""
    line = line.strip()
    if not line or len(line) > _HEADING_MAX_LEN:
        return False
    if any(p in line for p in _SENTENCE_PUNCT):
        return False
    digit_like = sum(c.isdigit() or c in ",.%$ " for c in line)
    return digit_like / len(line) <= 0.5


@dataclass
class _Section:
    heading: str
    chapter: str
    clause: str
    lines: list[tuple[int, str]]  # (page_no, line)

    @property
    def text(self) -> str:
        return "\n".join(line for _, line in self.lines)


class StructuralChunker:
    name = "structural"

    def __init__(self, max_chunk_size: int = 768):
        self.max_chunk_size = max_chunk_size
        self._splitter = FixedChunker(chunk_size=512, overlap_ratio=0.1)

    def _sections(self, pages: list[Page]) -> list[_Section]:
        sections: list[_Section] = []
        chapter = ""
        for pg in pages:
            for line in pg.text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if CHAPTER_RE.match(stripped):
                    chapter = stripped
                    sections.append(
                        _Section(heading=stripped, chapter=chapter, clause="", lines=[])
                    )
                    continue
                clause_hit = bool(CLAUSE_RE.match(stripped))
                if clause_hit or _is_heading(stripped):
                    sections.append(
                        _Section(
                            heading=stripped,
                            chapter=chapter,
                            clause=stripped if clause_hit else "",
                            lines=[],
                        )
                    )
                if not sections:
                    sections.append(_Section(heading="", chapter=chapter, clause="", lines=[]))
                sections[-1].lines.append((pg.page_no, stripped))
        # 过短的段并入前一段，避免碎片
        merged: list[_Section] = []
        for sec in sections:
            if merged and len(sec.text) < _MIN_SECTION_LEN:
                merged[-1].lines.extend(sec.lines)
            else:
                merged.append(sec)
        return [s for s in merged if s.lines]

    def split(self, pages: list[Page]) -> list[Chunk]:
        if not pages:
            return []
        product = pages[0].product
        chunks: list[Chunk] = []

        def add(text: str, page_start: int, page_end: int, sec: _Section) -> None:
            meta = {"section": sec.heading}
            if sec.chapter:
                meta["chapter"] = sec.chapter
            if sec.clause:
                meta["clause"] = sec.clause
            chunks.append(
                Chunk(
                    chunk_id=f"{product}:{self.name}:{len(chunks):04d}",
                    product=product,
                    strategy=self.name,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    meta=meta,
                )
            )

        for sec in self._sections(pages):
            page_start = sec.lines[0][0]
            page_end = sec.lines[-1][0]
            if len(sec.text) <= self.max_chunk_size:
                add(sec.text, page_start, page_end, sec)
            else:
                # 过长段落二次切分；子块共享段落层级元数据与页码范围
                synthetic = Page(
                    product=product, page_no=page_start, text=sec.text, raw_text=sec.text
                )
                for sub in self._splitter.split([synthetic]):
                    add(sub.text, page_start, page_end, sec)
        return chunks
