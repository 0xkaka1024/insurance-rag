import json
from pathlib import Path

from app.ingest.service import check_ingestable, ingest_files, rebuild_reports
from tests.pdf_fixture import make_pdf


def test_denylist_blocks_training_deck():
    ok, reason = check_ingestable(Path("07X_AVPU - training deck_SC_072024.pdf"))
    assert not ok
    assert "training deck" in reason


def test_denylist_blocks_premium_tables():
    for name in ("newVHISmedicalpremiumtable-tc.pdf", "vhis-selectwise-premium-table-tc.pdf"):
        ok, reason = check_ingestable(Path(name))
        assert not ok, name


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
        [good, bad], indexer=indexer, embedder=NullEmbedder(), settings=_settings(tmp_path)
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
    ingest_files([good], indexer=indexer, embedder=NullEmbedder(), settings=_settings(tmp_path))
    strategies = {cid.split(":")[1] for cid in indexer.indexed}
    assert strategies == {"fixed", "structural"}


def test_reingest_skipped_by_hash_and_forced(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)

    first = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings
    )
    assert first["ingested"] == ["newVHISmedical-tc.pdf"]

    second_indexer = RecordingIndexer()
    second = ingest_files(
        [good], indexer=second_indexer, embedder=NullEmbedder(), settings=settings
    )
    assert second["skipped"] == ["newVHISmedical-tc.pdf"]
    assert second_indexer.indexed == []  # 没有花任何 embedding

    forced = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings,
        force=True,
    )
    assert forced["ingested"] == ["newVHISmedical-tc.pdf"]


def test_ingest_writes_corpus_report(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings)

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

    result = rebuild_reports([good, bad], settings=settings)

    assert result["written"] == ["newVHISmedical-tc.pdf"]
    assert len(result["rejected"]) == 1  # 治理校验对报告同样生效
    assert (settings.index_dir / "reports" / "newVHISmedical.json").exists()
    assert not (settings.index_dir / "chroma").exists()  # 未触碰索引


def test_changed_file_reingested(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    settings = _settings(tmp_path)
    ingest_files([good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings)

    good.write_bytes(make_pdf("Waiting period is 30 days."))
    result = ingest_files(
        [good], indexer=RecordingIndexer(), embedder=NullEmbedder(), settings=settings
    )
    assert result["ingested"] == ["newVHISmedical-tc.pdf"]
    assert result["skipped"] == []
