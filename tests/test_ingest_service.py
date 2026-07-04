import hashlib
import json
from pathlib import Path

import pytest

from app.ingest.parser import product_from_filename
from app.ingest.service import check_ingestable, ingest_files, rebuild_reports
from tests.pdf_fixture import make_pdf, table_grid


def _fp(*paths: Path) -> dict[str, frozenset[str]]:
    """夹具文件的内容指纹表：white 名单产品 + 测试 PDF 的真实 sha256。"""
    out: dict[str, set[str]] = {}
    for p in paths:
        out.setdefault(product_from_filename(p), set()).add(
            hashlib.sha256(p.read_bytes()).hexdigest()
        )
    return {k: frozenset(v) for k, v in out.items()}


def test_denylist_blocks_training_deck():
    ok, reason = check_ingestable(Path("07X_AVPU - training deck_SC_072024.pdf"))
    assert not ok
    assert "trainingdeck" in reason  # deny 按归一化后的形态报告


def test_denylist_blocks_premium_tables():
    for name in ("newVHISmedicalpremiumtable-tc.pdf", "vhis-selectwise-premium-table-tc.pdf"):
        ok, reason = check_ingestable(Path(name))
        assert not ok, name


def test_denylist_blocks_renamed_variants():
    """deny 归一化：下划线/无空格/空格变体全部拦截，重命名绕不过。"""
    for name in (
        "07X_Training_Deck_SC.pdf",
        "TrainingDeck.pdf",
        "vhis premium table 2024.pdf",
        "premium_table_final.pdf",
    ):
        ok, reason = check_ingestable(Path(name))
        assert not ok, name
        assert "红线拒绝" in reason


def test_unregistered_fingerprint_rejected(tmp_path):
    """双因子准入：文件名合法但内容 sha256 未登记 → 拒绝（重命名穿透防线）。"""
    fake = tmp_path / "newVHISmedical-tc.pdf"  # 假装是白名单产品的任意内容
    fake.write_bytes(make_pdf("actually internal training material"))

    ok, reason = check_ingestable(fake, digest="deadbeef")
    assert not ok
    assert "指纹未登记" in reason

    result = ingest_files(  # 不注入夹具指纹 → 走真实登记表 → 必拒
        [fake], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=_settings(tmp_path)
    )
    assert result["ingested"] == []
    assert len(result["rejected"]) == 1
    assert "指纹未登记" in result["rejected"][0]


def test_whitelist_blocks_unknown_product():
    ok, reason = check_ingestable(Path("SomeRandomPlan_tc.pdf"))
    assert not ok
    assert "白名单" in reason


def test_whitelisted_products_pass():
    assert check_ingestable(Path("newVHISmedical-tc.pdf")) == (True, "")
    assert check_ingestable(Path("GlobalFlexiSavingsInsurancePlan_tc.pdf")) == (True, "")
    assert check_ingestable(Path("OnYourSideInsurancePlan2_tc.pdf")) == (True, "")


class RecordingIndexer:
    def __init__(self):
        self.indexed: list[str] = []
        self.purged: list[str] = []

    def index(self, chunks, embedder) -> int:
        self.indexed.extend(c.chunk_id for c in chunks)
        return len(chunks)

    def purge_product(self, product: str, strategies=("fixed", "structural")) -> None:
        self.purged.append(product)


class NullEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


def _settings(tmp_path):
    from app.core.config import Settings

    return Settings(_env_file=None, index_dir=tmp_path / "index")


def test_ingest_files_rejects_and_ingests(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    bad = tmp_path / "x - training deck.pdf"
    bad.write_bytes(make_pdf("internal only"))

    indexer = RecordingIndexer()
    result = ingest_files(
        [good, bad], indexer=indexer, embedder=NullEmbedder(), settings=_settings(tmp_path),
        fingerprints=_fp(good),
    )

    assert result["files"] == 1
    assert result["ingested"] == ["newVHISmedical-tc.pdf"]
    assert len(result["rejected"]) == 1
    assert "training deck" in result["rejected"][0]
    assert result["chunks"] == len(indexer.indexed) > 0


def test_ingest_builds_both_strategies_by_default(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    indexer = RecordingIndexer()
    ingest_files([good], indexer=indexer, embedder=NullEmbedder(), settings=_settings(tmp_path),
                 fingerprints=_fp(good))
    strategies = {cid.split(":")[1] for cid in indexer.indexed}
    assert strategies == {"fixed", "structural"}


def test_reingest_skipped_by_hash_and_forced(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)

    fp = _fp(good)
    first = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        fingerprints=fp,
    )
    assert first["ingested"] == ["newVHISmedical-tc.pdf"]

    second_indexer = RecordingIndexer()
    second = ingest_files(
        [good], indexer=second_indexer, embedder=NullEmbedder(), settings=settings,
        fingerprints=fp,
    )
    assert second["skipped"] == ["newVHISmedical-tc.pdf"]
    assert second_indexer.indexed == []  # 没有花任何 embedding

    forced = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        force=True, fingerprints=fp,
    )
    assert forced["ingested"] == ["newVHISmedical-tc.pdf"]


def test_corrupt_file_fails_without_aborting_batch(tmp_path):
    """失败通道：坏文件记入 failed，不炸整批；不写 manifest（下次重试）。"""
    bad = tmp_path / "newVHISmedical-tc.pdf"
    bad.write_bytes(b"not a pdf at all")
    good = tmp_path / "OnYourSideInsurancePlan2_tc.pdf"
    good.write_bytes(make_pdf("Critical illness benefit terms."))
    settings = _settings(tmp_path)

    result = ingest_files(
        [bad, good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        fingerprints={**_fp(bad), **_fp(good)},
    )
    assert result["ingested"] == ["OnYourSideInsurancePlan2_tc.pdf"]  # 好文件照常入库
    assert len(result["failed"]) == 1
    assert "newVHISmedical-tc.pdf" in result["failed"][0]
    manifest = json.loads((settings.index_dir / "ingest_manifest.json").read_text())
    assert "newVHISmedical-tc.pdf" not in manifest  # 失败不写 manifest


def test_zero_page_parse_recorded_as_failed(tmp_path):
    """0 页解析不算成功：不写 manifest、不写报告、计入 failed。"""
    empty = tmp_path / "newVHISmedical-tc.pdf"
    empty.write_bytes(make_pdf([]))  # 一页但无任何文本
    settings = _settings(tmp_path)

    result = ingest_files(
        [empty], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        fingerprints=_fp(empty),
    )
    assert result["ingested"] == []
    assert len(result["failed"]) == 1
    assert "0 页" in result["failed"][0]
    assert not (settings.index_dir / "ingest_manifest.json").exists()
    assert not (settings.index_dir / "reports" / "newVHISmedical.json").exists()

    retry = ingest_files(  # 未写 manifest → 下次运行不会被 hash 跳过
        [empty], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        fingerprints=_fp(empty),
    )
    assert retry["skipped"] == []
    assert len(retry["failed"]) == 1


def test_reingest_shrinking_file_leaves_no_stale_chunks(tmp_path):
    """清场式重入库：新版块数少于旧版时，尾部旧 chunk 必须消失（真索引验证）。"""
    from app.core.config import Settings
    from app.ingest.indexer import Indexer

    good = tmp_path / "newVHISmedical-tc.pdf"
    # 多行 spec（单超长行会画出页面边界被裁掉）：14 行 ≈ 650+ 字符 → 多个 fixed 块
    lines = [(72, 740 - i * 20, f"Waiting period is 90 days for plan number {i}.")
             for i in range(14)]
    good.write_bytes(make_pdf(lines))
    settings = Settings(_env_file=None, index_dir=tmp_path / "index")
    indexer = Indexer(settings)
    ingest_files([good], indexer=indexer, embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))
    n_v1 = indexer.collection("fixed").count()
    assert n_v1 > 1

    good.write_bytes(make_pdf("Waiting period is 30 days."))  # 新版大幅缩短
    ingest_files([good], indexer=indexer, embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))
    n_v2 = indexer.collection("fixed").count()
    assert n_v2 < n_v1  # 旧尾块被清场，而非永久残留
    assert len(indexer.bm25("fixed")) == n_v2  # 双存储一致
    texts = indexer.collection("fixed").get(include=["documents"])["documents"]
    assert all("30 days" in t for t in texts)  # 无任何旧版内容
    assert all("90 days" not in t for t in texts)


def test_ingest_purges_product_before_indexing(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    indexer = RecordingIndexer()
    ingest_files([good], indexer=indexer, embedder=NullEmbedder(),
                 settings=_settings(tmp_path), fingerprints=_fp(good))
    assert indexer.purged == ["newVHISmedical"]


def test_index_dir_has_no_tmp_residue_after_ingest(tmp_path):
    """原子写：所有落盘走临时文件 + replace，成功后不留 *.tmp。"""
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))
    assert list(settings.index_dir.rglob("*.tmp")) == []
    assert (settings.index_dir / "ingest_manifest.json").exists()


def test_ingest_writes_corpus_report(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))

    report_path = settings.index_dir / "reports" / "newVHISmedical.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report["strategies"]) == {"fixed", "structural"}
    assert report["parsed_pages"] >= 1
    assert report["total_pages"] >= report["parsed_pages"]
    assert report["strategies"]["structural"]["n_chunks"] >= 1
    assert "clause_coverage" in report["strategies"]["structural"]


def test_rebuild_reports_writes_report_without_indexing(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    bad = tmp_path / "x - training deck.pdf"
    bad.write_bytes(make_pdf("internal only"))
    settings = _settings(tmp_path)

    result = rebuild_reports([good, bad], settings=settings, fingerprints=_fp(good))

    assert result["written"] == ["newVHISmedical-tc.pdf"]
    assert len(result["rejected"]) == 1  # 治理校验对报告同样生效
    assert (settings.index_dir / "reports" / "newVHISmedical.json").exists()
    assert not (settings.index_dir / "chroma").exists()  # 未触碰索引


def _table_pdf() -> bytes:
    return make_pdf({"texts": [(80.0, 680.0, "Benefit waiting 300 days")], "lines": table_grid()})


def test_vlm_flag_without_client_fails_loud(tmp_path):
    """开关声明启用 VLM fallback 却未注入客户端 → 报错，不静默拍平表格入库。"""
    from app.core.config import Settings

    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = Settings(_env_file=None, index_dir=tmp_path / "index", parse_vlm_fallback=True)

    with pytest.raises(ValueError, match="parse_vlm_fallback"):
        ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(),
                     settings=settings, fingerprints=_fp(good))
    with pytest.raises(ValueError, match="parse_vlm_fallback"):
        rebuild_reports([good], settings=settings, fingerprints=_fp(good))


def test_ingest_with_vlm_records_attribution_in_report(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(_table_pdf())
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good), vlm=lambda page: "| Benefit | 300 days waiting |")

    report = json.loads(
        (settings.index_dir / "reports" / "newVHISmedical.json").read_text(encoding="utf-8")
    )
    assert report["vlm_fallback"] is True
    assert report["table_pages_flat"] == []  # 表格页已转写，无红旗
    assert report["pages"][0]["has_table"] is True
    assert report["pages"][0]["vlm_used"] is True


def test_ingest_without_vlm_red_flags_flattened_table_pages(tmp_path):
    """未启用 VLM 时表格页照常入库（现状），但报告必须亮红旗可追查。"""
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(_table_pdf())
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))

    report = json.loads(
        (settings.index_dir / "reports" / "newVHISmedical.json").read_text(encoding="utf-8")
    )
    assert report["vlm_fallback"] is False
    assert report["table_pages_flat"] == [1]
    assert report["pages"][0]["vlm_used"] is False


def test_changed_file_reingested(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
                 fingerprints=_fp(good))

    good.write_bytes(make_pdf("Waiting period is 30 days."))
    result = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        fingerprints=_fp(good),  # 新版本文件需要登记新指纹（模拟登记流程）
    )
    assert result["ingested"] == ["newVHISmedical-tc.pdf"]
    assert result["skipped"] == []
