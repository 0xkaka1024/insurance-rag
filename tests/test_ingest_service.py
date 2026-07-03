from pathlib import Path

from app.ingest.service import check_ingestable, ingest_files
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


class RecordingIndexer:
    def __init__(self):
        self.indexed: list[str] = []

    def index(self, chunks, embedder) -> int:
        self.indexed.extend(c.chunk_id for c in chunks)
        return len(chunks)


class NullEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


def test_ingest_files_rejects_and_ingests(tmp_path):
    good = tmp_path / "newVHISmedical-tc.pdf"
    good.write_bytes(make_pdf("Waiting period is 90 days."))
    bad = tmp_path / "x - training deck.pdf"
    bad.write_bytes(make_pdf("internal only"))

    indexer = RecordingIndexer()
    result = ingest_files([good, bad], indexer=indexer, embedder=NullEmbedder())

    assert result["files"] == 1
    assert result["ingested"] == ["newVHISmedical-tc.pdf"]
    assert len(result["rejected"]) == 1
    assert "training deck" in result["rejected"][0]
    assert result["chunks"] == len(indexer.indexed) > 0
