"""切片策略。所有策略实现同一接口：split(pages) -> list[Chunk]。

长度单位说明：bge-m3 的 sentencepiece 对 CJK 约 1 字符 ≈ 1 token，
v1 用字符数作 token 代理（chunk_size=512 字符）；如需精确 token 数，
注入 len_fn 换成真实 tokenizer 计数即可，切分算法不变。
"""

from bisect import bisect_right
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from app.ingest.parser import Page

_SENT_END = "。；！？"


@dataclass
class Chunk:
    chunk_id: str  # {product}:{strategy}:{seq}，确定性 id 支撑幂等 upsert
    product: str
    strategy: str
    text: str
    page_start: int
    page_end: int
    meta: dict = field(default_factory=dict)


class Chunker(Protocol):
    name: str

    def split(self, pages: list[Page]) -> list[Chunk]: ...


def _atomize(text: str, max_len: int) -> list[str]:
    """切成句子级原子片段（保留分隔符）；超长片段硬切，保证单片 <= max_len。"""
    pieces: list[str] = []
    buf: list[str] = []
    for ch in text:
        buf.append(ch)
        if ch in _SENT_END or ch == "\n":
            pieces.append("".join(buf))
            buf = []
    if buf:
        pieces.append("".join(buf))
    out: list[str] = []
    for p in pieces:
        while len(p) > max_len:
            out.append(p[:max_len])
            p = p[max_len:]
        if p:
            out.append(p)
    return out


class FixedChunker:
    """固定长度 + 重叠的 baseline 策略：句子级贪心装包，切点尽量落在句边界。

    重叠通过回携上一 chunk 的尾部句子实现，因此实际 chunk 长度上限为
    chunk_size + overlap（回携片段不挤掉新内容）。
    """

    name = "fixed"

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_ratio: float = 0.15,
        len_fn: Callable[[str], int] = len,
    ):
        self.chunk_size = chunk_size
        self.overlap = int(chunk_size * overlap_ratio)
        self.len_fn = len_fn

    def split(self, pages: list[Page]) -> list[Chunk]:
        if not pages:
            return []
        product = pages[0].product

        page_starts: list[int] = []
        page_nos: list[int] = []
        parts: list[str] = []
        pos = 0
        for pg in pages:
            page_starts.append(pos)
            page_nos.append(pg.page_no)
            parts.append(pg.text)
            pos += len(pg.text) + 1  # +1: "\n" 连接符
        text = "\n".join(parts)

        pieces = _atomize(text, self.chunk_size)
        piece_offsets: list[int] = []
        off = 0
        for p in pieces:
            piece_offsets.append(off)
            off += len(p)

        chunks: list[Chunk] = []
        cur: list[tuple[int, str]] = []  # (绝对偏移, 片段)
        cur_len = 0
        cur_has_fresh = False  # cur 中是否含本 chunk 的新内容（而非纯重叠回携）

        def emit() -> None:
            start = cur[0][0]
            end = cur[-1][0] + len(cur[-1][1])
            pg_i_start = bisect_right(page_starts, start) - 1
            pg_i_end = bisect_right(page_starts, end - 1) - 1
            chunks.append(
                Chunk(
                    chunk_id=f"{product}:{self.name}:{len(chunks):04d}",
                    product=product,
                    strategy=self.name,
                    text="".join(p for _, p in cur),
                    page_start=page_nos[pg_i_start],
                    page_end=page_nos[pg_i_end],
                )
            )

        for piece, p_off in zip(pieces, piece_offsets, strict=True):
            plen = self.len_fn(piece)
            if cur_has_fresh and cur_len + plen > self.chunk_size:
                emit()
                tail: list[tuple[int, str]] = []
                tail_len = 0
                for item in reversed(cur):
                    ilen = self.len_fn(item[1])
                    if tail_len + ilen > self.overlap:
                        break
                    tail.insert(0, item)
                    tail_len += ilen
                cur = tail
                cur_len = tail_len
                cur_has_fresh = False
            cur.append((p_off, piece))
            cur_len += plen
            cur_has_fresh = True
        if cur_has_fresh:
            emit()
        return chunks
