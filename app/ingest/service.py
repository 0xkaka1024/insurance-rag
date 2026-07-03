"""入库服务：数据治理校验 → 解析 → 切片 → 索引。

红线（代码级强制，不依赖人工自觉）：
- 内部培训材料（文件名含 training deck）一律拒绝
- 费率表（premiumtable / premium-table）属 v2 查表功能原料，v1 拒绝
- 白名单外的产品不入公开部署的索引
"""

import hashlib
import json
import logging
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.embedding import EmbeddingClient
from app.ingest.chunker import Chunker, FixedChunker
from app.ingest.indexer import ChromaIndexer
from app.ingest.parser import parse_pdf, product_from_filename
from app.ingest.structural import StructuralChunker

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


def ingest_files(
    paths: list[Path],
    chunkers: list[Chunker] | None = None,
    indexer: ChromaIndexer | None = None,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
    force: bool = False,
) -> dict[str, int | list[str]]:
    """逐文件入库；被拒/未变更文件分别记入 rejected/skipped，不中断其余文件。

    幂等：文件内容 sha256 记入 manifest，同 hash 直接跳过（embedding 不花冤枉钱）；
    切片逻辑升级后需要重建索引时用 force=True。
    """
    s = settings or get_settings()
    chunkers = chunkers or [FixedChunker(), StructuralChunker()]
    indexer = indexer or ChromaIndexer(s)
    embedder = embedder or EmbeddingClient(s)

    manifest_path = s.index_dir / "ingest_manifest.json"
    manifest = _load_manifest(manifest_path)

    indexed_chunks = 0
    ingested: list[str] = []
    rejected: list[str] = []
    skipped: list[str] = []
    for path in paths:
        ok, reason = check_ingestable(path)
        if not ok:
            logger.warning("reject %s: %s", path.name, reason)
            rejected.append(f"{path.name}: {reason}")
            continue
        digest = _sha256(path)
        if not force and manifest.get(path.name) == digest:
            logger.info("skip unchanged %s", path.name)
            skipped.append(path.name)
            continue
        pages = parse_pdf(path)
        for chunker in chunkers:
            chunks = chunker.split(pages)
            indexed_chunks += indexer.index(chunks, embedder)
        manifest[path.name] = digest
        _save_manifest(manifest_path, manifest)  # 每文件落盘，中断后已完成的不重做
        ingested.append(path.name)
        logger.info(
            "ingested %s",
            path.name,
            extra={"extra_fields": {"pages": len(pages), "strategies": len(chunkers)}},
        )
    return {"files": len(ingested), "chunks": indexed_chunks, "ingested": ingested,
            "rejected": rejected, "skipped": skipped}
