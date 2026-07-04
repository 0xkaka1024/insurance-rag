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
from app.ingest.chunker import Chunk, Chunker, FixedChunker

# 治理规则集中在 governance.py（deny 归一化 + 白名单 + 内容指纹）；re-export 维持既有导入路径
from app.ingest.governance import INGEST_WHITELIST, check_ingestable  # noqa: F401
from app.ingest.indexer import Indexer
from app.ingest.parser import PageVLM, page_count, parse_pdf, product_from_filename
from app.ingest.report import build_report, save_report
from app.ingest.structural import StructuralChunker

logger = logging.getLogger("ingest")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_vlm(s: Settings, vlm: PageVLM | None) -> None:
    """开关与注入一致性：宣称启用 VLM fallback 却没有客户端 → 报错而非静默降级。

    表格页拍平入库是事实性误引源头（SPEC R9 的直接依据），配置说转写了
    实际没转写，报告与评测都会被误导。
    """
    if s.parse_vlm_fallback and vlm is None:
        raise ValueError(
            "parse_vlm_fallback=true 但未注入 vlm 客户端：Qwen-VL 接入属 P1（SPEC R9），"
            "请传入 vlm 回调或关闭开关"
        )


def _load_manifest(path: Path) -> dict[str, str]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_manifest(path: Path, manifest: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    tmp.replace(path)  # 原子替换：损坏的 manifest 会让所有后续入库不可用


def ingest_files(
    paths: list[Path],
    chunkers: list[Chunker] | None = None,
    indexer: Indexer | None = None,
    embedder: EmbeddingClient | None = None,
    settings: Settings | None = None,
    force: bool = False,
    fingerprints: dict[str, frozenset[str]] | None = None,
    vlm: PageVLM | None = None,
) -> dict[str, int | list[str]]:
    """逐文件入库；被拒/未变更文件分别记入 rejected/skipped，不中断其余文件。

    治理双因子：文件名白名单 + 内容 sha256 指纹（governance.FINGERPRINTS 登记制，
    重命名穿透不了）；fingerprints 参数供测试注入夹具指纹。
    幂等：文件内容 sha256 记入 manifest，同 hash 直接跳过（embedding 不花冤枉钱）；
    切片逻辑升级后需要重建索引时用 force=True。
    vlm：表格页/低质量页转 Markdown 的回调（SPEC R9），是否启用记入报告供评测归因。
    """
    s = settings or get_settings()
    _check_vlm(s, vlm)
    chunkers = chunkers or [FixedChunker(), StructuralChunker()]
    indexer = indexer or Indexer(s)
    embedder = embedder or EmbeddingClient(s)

    manifest_path = s.index_dir / "ingest_manifest.json"
    manifest = _load_manifest(manifest_path)

    indexed_chunks = 0
    ingested: list[str] = []
    rejected: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    for path in paths:
        digest = _sha256(path)
        ok, reason = check_ingestable(path, digest=digest, fingerprints=fingerprints)
        if not ok:
            logger.warning("reject %s: %s", path.name, reason)
            rejected.append(f"{path.name}: {reason}")
            continue
        if not force and manifest.get(path.name) == digest:
            logger.info("skip unchanged %s", path.name)
            skipped.append(path.name)
            continue
        # 失败通道：单文件异常/零解析只记 failed，不炸整批；
        # 不写 manifest（下次运行重试），不写报告（避免"成功"假象）。
        try:
            pages = parse_pdf(path, vlm=vlm)
            if not pages:
                logger.error("parse produced no text for %s", path.name)
                failed.append(f"{path.name}: 解析 0 页文本（扫描件/加密/空文件？），未入库")
                continue
            by_strategy: dict[str, list[Chunk]] = {ch.name: ch.split(pages) for ch in chunkers}
            # 清场式重入库（红线）：先删该产品旧块再写，新版块数少于旧版时
            # 尾部旧 chunk 不再残留（残留会引用已废止条款）。
            # 注：purge 与 index 之间无事务，embedding 失败会短暂缺该产品——
            # 蓝绿索引切换在 backlog（G4 collection 版本 manifest）。
            indexer.purge_product(product_from_filename(path))
            for chunks in by_strategy.values():
                indexed_chunks += indexer.index(chunks, embedder)
            save_report(
                build_report(
                    product=product_from_filename(path),
                    filename=path.name,
                    sha256=digest,
                    total_pages=page_count(path),
                    pages=pages,
                    chunks_by_strategy=by_strategy,
                    vlm_fallback=vlm is not None,
                ),
                s,
            )
        except Exception as exc:
            logger.exception("ingest failed for %s", path.name)
            failed.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        manifest[path.name] = digest
        _save_manifest(manifest_path, manifest)  # 每文件落盘，中断后已完成的不重做
        ingested.append(path.name)
        logger.info(
            "ingested %s",
            path.name,
            extra={"extra_fields": {"pages": len(pages), "strategies": len(chunkers)}},
        )
    return {"files": len(ingested), "chunks": indexed_chunks, "ingested": ingested,
            "rejected": rejected, "skipped": skipped, "failed": failed}


def rebuild_reports(
    paths: list[Path],
    chunkers: list[Chunker] | None = None,
    settings: Settings | None = None,
    fingerprints: dict[str, frozenset[str]] | None = None,
    vlm: PageVLM | None = None,
) -> dict[str, int | list[str]]:
    """只重建语料质量报告：解析 + 切片 + lint，不写索引、不调 embedding。

    用途：给报告机制上线前已入库的文件补报告（manifest hash 会让 ingest 跳过它们）。
    治理校验照常生效（含内容指纹）——报告与索引同属公开面。
    vlm 与索引入库时保持同配（否则报告与索引内容对不上）；注入后不再零成本。
    """
    s = settings or get_settings()
    _check_vlm(s, vlm)
    chunkers = chunkers or [FixedChunker(), StructuralChunker()]
    written: list[str] = []
    rejected: list[str] = []
    failed: list[str] = []
    for path in paths:
        digest = _sha256(path)
        ok, reason = check_ingestable(path, digest=digest, fingerprints=fingerprints)
        if not ok:
            logger.warning("reject %s: %s", path.name, reason)
            rejected.append(f"{path.name}: {reason}")
            continue
        try:
            pages = parse_pdf(path, vlm=vlm)
            if not pages:
                failed.append(f"{path.name}: 解析 0 页文本（扫描件/加密/空文件？）")
                continue
            by_strategy = {ch.name: ch.split(pages) for ch in chunkers}
            save_report(
                build_report(
                    product=product_from_filename(path),
                    filename=path.name,
                    sha256=digest,
                    total_pages=page_count(path),
                    pages=pages,
                    chunks_by_strategy=by_strategy,
                    vlm_fallback=vlm is not None,
                ),
                s,
            )
        except Exception as exc:
            logger.exception("report rebuild failed for %s", path.name)
            failed.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        written.append(path.name)
    return {"reports": len(written), "written": written, "rejected": rejected,
            "failed": failed}
