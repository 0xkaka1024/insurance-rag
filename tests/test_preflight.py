"""HF 部署前自检：索引产物齐全性判定。"""

from scripts.preflight_hf import check_index_artifacts


def _make_index(tmp_path):
    for name in ("bm25_fixed.pkl", "bm25_structural.pkl", "ingest_manifest.json"):
        (tmp_path / name).write_bytes(b"x")
    chroma = tmp_path / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"x")


def test_all_artifacts_present_pass(tmp_path):
    _make_index(tmp_path)
    checks = check_index_artifacts(tmp_path)
    assert all(c.ok for c in checks)


def test_missing_bm25_file_flagged(tmp_path):
    _make_index(tmp_path)
    (tmp_path / "bm25_structural.pkl").unlink()
    checks = {c.name: c for c in check_index_artifacts(tmp_path)}
    assert checks["索引文件 bm25_structural.pkl"].ok is False
    assert checks["索引文件 bm25_structural.pkl"].hard is True  # 硬失败


def test_empty_chroma_dir_flagged(tmp_path):
    _make_index(tmp_path)
    for f in (tmp_path / "chroma").iterdir():
        f.unlink()
    checks = {c.name: c for c in check_index_artifacts(tmp_path)}
    assert checks["chroma 目录非空"].ok is False


def test_missing_index_dir_flagged(tmp_path):
    checks = {c.name: c for c in check_index_artifacts(tmp_path / "nope")}
    assert checks["data/index 目录存在"].ok is False
