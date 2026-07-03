"""PDF 解析：pdfplumber 逐页提取文本，opencc 繁→简归一。

保留每页原文（raw_text）：引用溯源展示条款原文时不应吐出被归一改写过的文本。
表格页 / 低质量页的 VLM fallback 属 P1，见 SPEC R9。
"""

from dataclasses import dataclass
from pathlib import Path

import opencc
import pdfplumber

_t2s = opencc.OpenCC("t2s")


@dataclass
class Page:
    product: str
    page_no: int  # 1-based，与 PDF 阅读器页码一致，供引用定位
    text: str  # 简体归一后文本，用于切片与 embedding
    raw_text: str  # 原始文本（多为繁体），用于引用展示


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


def parse_pdf(path: Path) -> list[Page]:
    product = product_from_filename(path)
    pages: list[Page] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            if not raw.strip():
                continue
            pages.append(Page(product=product, page_no=i, text=normalize(raw), raw_text=raw))
    return pages
