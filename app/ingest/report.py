"""语料质量报告：入库时生成静态 report.json，页级解析质量 + 块级边界质检（lint）。

设计动机（docs/REVIEW-2026-07.md「Playground 优化方案」第 3 条）：切分质量缺陷
藏在检索看不到的块里，抽查靠人翻不现实；lint 把「随机抽 10 个人工检查」升级为
全量机检 + 人工只看红旗。报告落盘 data/index/reports/{product}.json，
是「语料」视图的数据源，也兼作解析/切分器输出的审计留档。
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.ingest.chunker import Chunk
from app.ingest.parser import Page
from app.ingest.structural import CLAUSE_RE

# 句末符号：中文为主，兼容少量西文标点；收尾括号/引号之前的句末符同样算正常结束
_SENT_END = "。！？；…!?.;"
_CLOSERS = "」』】》）)\"'"

OVERSHORT = 40  # 与 structural 短段合并阈值同数量级：更短说明碎片漏网
OVERLONG = 1200  # fixed(512+回携) 与 structural(768) 之上的宽松上限
_SNIPPET = 20  # 边界上下文条展示的前块尾 / 后块首字数

FLAGS = ("cut_midsentence", "merged_clauses", "no_clause", "overshort", "overlong")


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _ends_mid_sentence(text: str) -> bool:
    t = text.rstrip()
    while t and t[-1] in _CLOSERS:
        t = t[:-1].rstrip()
    return bool(t) and t[-1] not in _SENT_END


def _clause_head_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if CLAUSE_RE.match(line.strip()))


def lint_chunks(chunks: list[Chunk]) -> list[dict]:
    """逐块质检。boundary 描述本块与下一块的接缝（末块为 None）。

    - cut_midsentence：块结尾非句末标点，边界疑似把句子拦腰截断
    - merged_clauses：structural 块内出现多于一个条号行，疑似短条款被吞并
    - no_clause：structural 块无条号元数据，引用只能走降级轨
    - overshort / overlong：长度离群，碎片或二次切分失效
    """
    records: list[dict] = []
    for i, c in enumerate(chunks):
        clause = str(c.meta.get("clause", ""))
        flags: list[str] = []
        if _clause_head_lines(c.text) > 1 and c.strategy == "structural":
            flags.append("merged_clauses")
        if c.strategy == "structural" and not clause:
            flags.append("no_clause")
        if len(c.text) < OVERSHORT:
            flags.append("overshort")
        elif len(c.text) > OVERLONG:
            flags.append("overlong")

        boundary = None
        if i + 1 < len(chunks):
            cut = _ends_mid_sentence(c.text)
            boundary = {
                "tail": _squash(c.text)[-_SNIPPET:],
                "next_head": _squash(chunks[i + 1].text)[:_SNIPPET],
                "cut": cut,
            }
            if cut:
                flags.append("cut_midsentence")

        records.append(
            {
                "chunk_id": c.chunk_id,
                "seq": i,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "section": str(c.meta.get("section", "")),
                "clause": clause,
                "chars": len(c.text),
                "flags": flags,
                "boundary": boundary,
            }
        )
    return records


def _page_records(pages: list[Page]) -> list[dict]:
    out = []
    for p in pages:
        nonspace = [ch for ch in p.text if not ch.isspace()]
        cjk = sum(1 for ch in nonspace if "一" <= ch <= "鿿")
        out.append(
            {
                "page": p.page_no,
                "chars": len(p.text),
                "cjk_ratio": round(cjk / len(nonspace), 3) if nonspace else 0.0,
                "two_column": p.two_column,
            }
        )
    return out


def build_report(
    product: str,
    filename: str,
    sha256: str,
    total_pages: int,
    pages: list[Page],
    chunks_by_strategy: dict[str, list[Chunk]],
) -> dict:
    strategies: dict[str, dict] = {}
    for strategy, chunks in chunks_by_strategy.items():
        recs = lint_chunks(chunks)
        flag_counts: dict[str, int] = {}
        for r in recs:
            for f in r["flags"]:
                flag_counts[f] = flag_counts.get(f, 0) + 1
        entry: dict = {"n_chunks": len(recs), "flag_counts": flag_counts, "chunks": recs}
        if strategy == "structural":
            # 条号主轨覆盖率：决定引用走 [产品-条号-页码] 主轨还是章节降级轨
            with_clause = sum(1 for r in recs if r["clause"])
            entry["clause_coverage"] = round(with_clause / len(recs), 3) if recs else 0.0
        strategies[strategy] = entry

    parsed_nos = {p.page_no for p in pages}
    return {
        "product": product,
        "file": filename,
        "sha256": sha256,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_pages": total_pages,
        "parsed_pages": len(pages),
        "empty_pages": sorted(set(range(1, total_pages + 1)) - parsed_nos),
        "pages": _page_records(pages),
        "strategies": strategies,
    }


def reports_dir(settings: Settings) -> Path:
    return settings.index_dir / "reports"


def save_report(report: dict, settings: Settings) -> Path:
    d = reports_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{report['product']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)  # 原子替换，避免写一半的报告被读到
    return path
