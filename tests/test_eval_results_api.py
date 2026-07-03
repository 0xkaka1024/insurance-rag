import json

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app

client = TestClient(app)

SAMPLE = {
    "run_at": "2026-07-03T20:00:00",
    "git": "abc1234",
    "n_questions": 2,
    "requested_metrics": ["faithfulness"],
    "total_cost_cny": 0.02,
    "total_duration_s": 3.5,
    "configs": [
        {
            "config": {"chunking": "fixed", "retrieval": "vector", "rerank": False},
            "metrics": {"faithfulness": 0.91},
            "refusal_accuracy": 1.0,
            "cost_cny": 0.02,
            "duration_s": 3.5,
        }
    ],
}


def _prep(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_RESULTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    (tmp_path / "20260703_abc1234.json").write_text(
        json.dumps(SAMPLE, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "20260702_older00.json").write_text("{}", encoding="utf-8")


def test_list_results_latest_first(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    resp = client.get("/eval_results")
    assert resp.status_code == 200
    assert resp.json()["files"] == ["20260703_abc1234.json", "20260702_older00.json"]


def test_get_result_payload(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    resp = client.get("/eval_results/20260703_abc1234.json")
    assert resp.status_code == 200
    assert resp.json()["configs"][0]["metrics"]["faithfulness"] == 0.91


def test_get_result_unknown_name_404(tmp_path, monkeypatch):
    _prep(tmp_path, monkeypatch)
    assert client.get("/eval_results/nope.json").status_code == 404
    assert client.get("/eval_results/%2e%2e%2fsecret.json").status_code == 404


def test_list_results_empty_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_RESULTS_DIR", str(tmp_path / "none"))
    get_settings.cache_clear()
    assert client.get("/eval_results").json() == {"files": []}
