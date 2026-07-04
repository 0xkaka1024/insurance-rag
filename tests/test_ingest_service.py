import hashlib
import json
from pathlib import Path

from app.ingest.parser import product_from_filename
from app.ingest.service import check_ingestable, ingest_files, rebuild_reports
from tests.pdf_fixture import make_pdf


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

    def index(self, chunks, embedder) -> int:
        self.indexed.extend(c.chunk_id for c in chunks)
        return len(chunks)


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
