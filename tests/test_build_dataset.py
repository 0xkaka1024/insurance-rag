"""评测集装配：配额/去重/产品轮转，与 loader 的人工核对门禁。"""

import json

import pytest

from eval.build_dataset import SUPPLEMENTS, assemble, dedup_candidates, pick_by_quota
from eval.harness import DATASET_TYPES, load_dataset


def _cand(i: int, qtype: str = "fact", product: str = "A", diff: str = "easy",
          chunk: str = "", question: str = "") -> dict:
    return {
        "id": i, "question": question or f"问题{i}是什么", "type": qtype,
        "difficulty": diff, "source_product": product,
        "source_chunk_id": chunk or f"{product}:structural:{i:04d}",
    }


def test_dedup_drops_same_chunk_same_prefix():
    q1 = "等候期是多少天以及等候期内确诊了重疾会怎么处理呢"
    q2 = "等候期是多少天以及等候期内确诊了重疾会怎么处理呀"  # 前 16 字相同 → 近重复
    q3 = "冷静期退保能拿回已缴保费吗手续费怎么算的呢"
    a = _cand(1, chunk="A:structural:0001", question=q1)
    b = _cand(2, chunk="A:structural:0001", question=q2)  # 同块同前缀 → 滤除
    c = _cand(3, chunk="A:structural:0002", question=q1)  # 块不同 → 保留
    d = _cand(4, chunk="A:structural:0001", question=q3)  # 同块不同问题 → 保留
    assert dedup_candidates([a, b, c, d]) == [a, c, d]


def test_pick_by_quota_rotates_products_and_reports_shortfall():
    cands = [_cand(i, product=p) for i, p in enumerate(["A"] * 6 + ["B"] * 6)]
    picked, shortfall = pick_by_quota(cands, {"fact": 4, "table": 2})
    assert len(picked) == 4
    assert {c["source_product"] for c in picked} == {"A", "B"}  # 产品轮转不偏科
    assert shortfall == {"table": 2}  # 候选池没有 table 题：如实报缺，不静默凑数


def test_assemble_forces_supplements_within_quota():
    cands = [_cand(i, qtype="unanswerable", product="A") for i in range(20)]
    rows, _ = assemble(cands, {"unanswerable": 10})
    supp_n = sum(1 for s in SUPPLEMENTS if s["type"] == "unanswerable")
    from_pool = sum(1 for r in rows if r["type"] == "unanswerable") - supp_n
    assert from_pool == 10 - supp_n  # 补充模板占配额，不超编


def test_supplements_are_valid_types():
    assert all(s["type"] in DATASET_TYPES for s in SUPPLEMENTS)
    assert any(s["type"] == "comparison" for s in SUPPLEMENTS)  # 单 chunk 生成不了的类型
    assert any(s["source_product"] == "跨产品" and s["type"] == "unanswerable"
               for s in SUPPLEMENTS)


def test_loader_rejects_needs_review(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({
        "question": "等候期多少天", "type": "fact",
        "ground_truth": "草稿", "needs_review": True,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs_review"):
        load_dataset(p)


def test_loader_rejects_unknown_type(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"question": "q", "type": "unanserable"}\n', encoding="utf-8")  # 拼写错误
    with pytest.raises(ValueError, match="未知题型"):
        load_dataset(p)
