"""入库服务：数据治理校验 → 解析 → 切片 → 索引。

红线（代码级强制，不依赖人工自觉）：
- 内部培训材料（文件名含 training deck）一律拒绝
- 费率表（premiumtable / premium-table）属 v2 查表功能原料，v1 拒绝
- 白名单外的产品不入公开部署的索引
"""

import logging
from pathlib import Path

from app.core.embedding import EmbeddingClient
from app.ingest.chunker import Chunker, FixedChunker
from app.ingest.indexer import ChromaIndexer
from app.ingest.parser import parse_pdf, product_from_filename

logger = logging.getLogger("ingest")

DENY_SUBSTRINGS = ("training deck", "premiumtable", "premium-table")

# v1 入库白名单（CLAUDE.md 红线），按 product_from_filename 推导的产品标识精确匹配。
# 危疾类条款一款待 kaka 选定后加入（SPEC 开放问题）。
INGEST_WHITELIST = frozenset({"newVHISmedical", "GlobalFlexiSavingsInsurancePlan"})


def check_ingestable(path: Path) -> tuple[bool, str]:
    name = path.name.lower()
    for pattern in DENY_SUBSTRINGS:
        if pattern in name:
            return False, f"红线拒绝：文件名含 '{pattern}'（内部材料/费率表不入库）"
    product = product_from_filename(path)
    if product not in INGEST_WHITELIST:
        return False, f"白名单外：产品 '{product}' 不在 v1 入库白名单 {sorted(INGEST_WHITELIST)}"
    return True, ""


def ingest_files(
    paths: list[Path],
    chunkers: list[Chunker] | None = None,
    indexer: ChromaIndexer | None = None,
    embedder: EmbeddingClient | None = None,
) -> dict[str, int | list[str]]:
    """逐文件入库；被拒文件记入 rejected 并说明原因，不中断其余文件。"""
    chunkers = chunkers or [FixedChunker()]
    indexer = indexer or ChromaIndexer()
    embedder = embedder or EmbeddingClient()

    indexed_chunks = 0
    ingested: list[str] = []
    rejected: list[str] = []
    for path in paths:
        ok, reason = check_ingestable(path)
        if not ok:
            logger.warning("reject %s: %s", path.name, reason)
            rejected.append(f"{path.name}: {reason}")
            continue
        pages = parse_pdf(path)
        for chunker in chunkers:
            chunks = chunker.split(pages)
            indexed_chunks += indexer.index(chunks, embedder)
        ingested.append(path.name)
        logger.info(
            "ingested %s",
            path.name,
            extra={"extra_fields": {"pages": len(pages), "strategies": len(chunkers)}},
        )
    return {"files": len(ingested), "chunks": indexed_chunks, "ingested": ingested,
            "rejected": rejected}
