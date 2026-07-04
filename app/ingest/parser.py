"""PDF 解析：pdfplumber 逐页提取文本，双栏检测分栏，opencc 繁→简归一。

产品简介类 PDF 多为双栏排版，pdfplumber 默认按行水平拼接会把左右两栏交错；
这里按「行带」检测：整页先聚成文本行，跨中缝的行是通栏（标题等），连续的
左/右半区行构成栏区带，栏区带内先输出整个左栏再输出右栏。

保留每页原文（raw_text）：引用溯源展示条款原文时不应吐出被归一改写过的文本。

表格页 / 低质量页的 VLM fallback（SPEC R9）：文本流拍平表格会造成事实性误引
（真实案例：VHIS 保障表「(等候期：300日)」是单项标注，拍平后被引用为整体等待期）。
parse_pdf 接受可注入的 vlm callable（页 → Markdown），触发条件是检出表格线框或
提取质量差（cid 占位符）；未注入时只打标记，报告可见。Qwen-VL 真实客户端属 P1。
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import opencc
import pdfplumber

logger = logging.getLogger("ingest")

# VLM 接口缝：入参为 pdfplumber 页对象（实现方可自行 page.to_image() 送图），
# 返回该页 Markdown（保持原文字系，繁简归一由 parse_pdf 统一做）。
PageVLM = Callable[[pdfplumber.page.Page], str]

_t2s = opencc.OpenCC("t2s")

_Y_MARGIN = 2.0  # 分割带上下的余量（pt）


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def extract_page_text(page: pdfplumber.page.Page, info: dict | None = None) -> str:
    """双栏感知的整页文本提取；单栏页行为与 extract_text 一致。

    以「跨中缝的词」（通栏标题等）的 y 区间为分割线，把页面切成若干水平带；
    带内若左右两侧都有词且无词跨中缝，则先提取整个左半再提取右半。
    传入 info dict 时回填页级判定（two_column），供语料质量报告使用。
    """
    words = page.extract_words()
    if not words:
        return page.extract_text() or ""

    px0, py0, px1, py1 = page.bbox
    center = (px0 + px1) / 2
    gutter = (px1 - px0) * 0.02

    spanning = [
        (max(py0, w["top"] - _Y_MARGIN), min(py1, w["bottom"] + _Y_MARGIN))
        for w in words
        if w["x0"] < center - gutter and w["x1"] > center + gutter
    ]
    full_bands = _merge_intervals(spanning)

    # 通栏带之间的空隙是候选栏区带
    bands: list[tuple[float, float, str]] = []  # (y0, y1, kind)
    cursor = py0
    for start, end in full_bands:
        if start > cursor:
            bands.append((cursor, start, "columnar"))
        bands.append((start, end, "full"))
        cursor = end
    if cursor < py1:
        bands.append((cursor, py1, "columnar"))

    parts: list[str] = []
    for y0, y1, kind in bands:
        in_band = [w for w in words if w["top"] >= y0 and w["bottom"] <= y1]
        has_left = any(w["x1"] <= center + gutter for w in in_band)
        has_right = any(w["x0"] >= center - gutter for w in in_band)
        if kind == "columnar" and has_left and has_right:
            if info is not None:
                info["two_column"] = True
            left = page.crop((px0, y0, center, y1)).extract_text() or ""
            right = page.crop((center, y0, px1, y1)).extract_text() or ""
            parts.extend(t for t in (left, right) if t.strip())
        else:
            text = page.crop((px0, y0, px1, y1)).extract_text() or ""
            if text.strip():
                parts.append(text)
    return "\n".join(parts)


@dataclass
class Page:
    product: str
    page_no: int  # 1-based，与 PDF 阅读器页码一致，供引用定位
    text: str  # 简体归一后文本，用于切片与 embedding
    raw_text: str  # 原始文本（多为繁体），用于引用展示
    two_column: bool = False  # 该页是否触发过分栏提取，供语料质量报告抽查
    has_table: bool = False  # 检出表格线框：拍平文本有误引风险（SPEC R9）
    vlm_used: bool = False  # 该页文本是否来自 VLM 转写（has_table 且未转写 = 报告红旗）


def _is_low_quality(raw: str) -> bool:
    """提取质量差的启发式：字体缺 unicode 映射时 pdfplumber 吐 (cid:N) 占位符。"""
    return "(cid:" in raw


def normalize(text: str) -> str:
    """繁→简归一。embedding 与 BM25 统一走简体，消除繁简检索鸿沟。"""
    return _t2s.convert(text)


def product_from_filename(path: Path) -> str:
    """从文件名推导产品标识：去掉繁体标记后缀 _tc / -tc。"""
    stem = path.stem
    for suffix in ("_tc", "-tc"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def parse_pdf(path: Path, vlm: PageVLM | None = None) -> list[Page]:
    """逐页解析。注入 vlm 后，表格页/低质量页改走 VLM 转 Markdown（SPEC R9）。

    VLM 输出同时作为 text（归一后）与 raw_text：表格页的 pdfplumber 拍平文本
    本就不可信，引用展示也应基于转写结果。VLM 调用失败或返回空 → 回退
    pdfplumber 文本并保留 has_table 标记，报告红旗可见，不静默吞错。
    """
    product = product_from_filename(path)
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            info: dict = {}
            raw = extract_page_text(page, info)
            if not raw.strip():
                continue
            has_table = bool(page.find_tables())
            vlm_used = False
            if vlm is not None and (has_table or _is_low_quality(raw)):
                try:
                    md = vlm(page)
                except Exception:
                    logger.exception("VLM fallback failed for %s p%d", path.name, i)
                    md = ""
                else:
                    if not md.strip():
                        logger.warning(
                            "VLM fallback empty for %s p%d, keeping pdfplumber text",
                            path.name, i,
                        )
                if md.strip():
                    raw = md
                    vlm_used = True
            pages.append(
                Page(
                    product=product,
                    page_no=i,
                    text=normalize(raw),
                    raw_text=raw,
                    two_column=info.get("two_column", False),
                    has_table=has_table,
                    vlm_used=vlm_used,
                )
            )
    return pages


def page_count(path: Path) -> int:
    """PDF 总页数（含未能提取文本的页），供报告对照 parsed_pages 暴露空页。"""
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)
