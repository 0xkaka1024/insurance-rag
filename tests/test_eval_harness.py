import json

import pytest

from app.rag.pipeline import AskResult, RagConfig
from app.rag.retriever import RetrievedChunk
from eval.harness import (
    evaluate_config,
    expand_configs,
    load_dataset,
    result_filename,
    save_result,
    validate_metrics,
)


def test_expand_default_eight_unique():
    configs = expand_configs()
    assert len(configs) == 8
    assert len({tuple(c.model_dump().items()) for c in configs}) == 8


def test_expand_single_config():
    [cfg] = expand_configs("structural", "hybrid", True)
    assert cfg.model_dump() == {"chunking": "structural", "retrieval": "hybrid", "rerank": True}


def test_expand_partial_pin():
    assert len(expand_configs(chunking="fixed")) == 4


def test_load_dataset_and_validation(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"question": "等待期多少天", "type": "fact", "ground_truth": "90天"}\n'
        '{"question": "30岁多少钱", "type": "refuse"}\n',
        encoding="utf-8",
    )
    rows = load_dataset(p)
    assert len(rows) == 2
    validate_metrics(["faithfulness", "context_recall"], rows)  # gt 齐全，不抛


def test_validate_metrics_requires_ground_truth(tmp_path):
    rows = [{"question": "q", "type": "fact"}]  # 无 ground_truth
    with pytest.raises(ValueError, match="ground_truth"):
        validate_metrics(["context_recall"], rows)
    validate_metrics(["faithfulness", "answer_relevancy"], rows)  # 免参照指标可跑


def test_validate_metrics_rejects_unknown():
    with pytest.raises(ValueError, match="未知指标"):
        validate_metrics(["bleu"], [])


class FakePipeline:
    """按题型返回：refuse 题拒答，其余给带 usage 的正常回答。"""

    def ask(self, question: str, cfg: RagConfig) -> AskResult:
        refused = "多少钱" in question
        result = AskResult(
            answer="拒答" if refused else "答案[1]。",
            chunks=[] if refused else [
                RetrievedChunk("Demo:fixed:0000", "内容", "Demo", 1, 1, 0.9)
            ],
            timings={"total_ms": 1.0},
            config=cfg,
            refused=refused,
            refuse_reason="premium_intent" if refused else "",
        )
        result.meta["usage"] = {"prompt_tokens": 1000, "completion_tokens": 500}
        return result


def test_evaluate_config_metrics_and_cost():
    rows = [
        {"question": "等待期多少天", "type": "fact", "ground_truth": "90天"},
        {"question": "30岁多少钱", "type": "refuse"},
        {"question": "又一个多少钱的问题", "type": "refuse"},
    ]
    entry = evaluate_config(FakePipeline(), RagConfig(), rows, llm_model="deepseek-chat")
    assert entry["refusal_accuracy"] == 1.0
    assert len(entry["records"]) == 3
    # 3 题 × (1000 in × ¥2/M + 500 out × ¥8/M) = 3 × ¥0.006 = ¥0.018
    assert entry["cost_cny"] == pytest.approx(0.018)
    assert entry["records"][0]["contexts"] == ["内容"]
    assert entry["records"][1]["refused"] is True
    assert entry["duration_s"] >= 0


def test_refusal_accuracy_none_without_refuse_rows():
    rows = [{"question": "等待期多少天", "type": "fact", "ground_truth": "90"}]
    entry = evaluate_config(FakePipeline(), RagConfig(), rows)
    assert entry["refusal_accuracy"] is None


def test_result_filename_format():
    assert result_filename("20260703", "abc1234") == "20260703_abc1234.json"


def test_save_result_roundtrip(tmp_path):
    path = save_result(tmp_path / "results", "20260703_abc.json", {"k": "值"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": "值"}
