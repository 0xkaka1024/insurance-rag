"""从候选题装配评测集草稿：类型配额 + 产品轮转 + 去重；产出带 needs_review 标记的草稿。

人工核对流程（G2：这一步不能省，也省不掉——loader 会拒绝未核对的行）：
  1. python eval/build_dataset.py                # candidates.jsonl → eval/dataset.draft.jsonl
  2. 逐条对照 PDF 原文核对/改写 ground_truth（draft_answer 是 LLM 草稿，不可直接采信）
  3. 核对完成的行删除 needs_review 字段；全部完成后改名为 eval/dataset.jsonl
  4. python eval/run_eval.py                     # 任何仍带 needs_review 的行会被 loader 拒绝

跨产品混淆拒答题与 comparison 题由 SUPPLEMENTS 人工模板补充（单 chunk 生成机制
出不了这两类题——评测集构造偏置的对策，见 docs/REVIEW-2026-07.md P1-4）。
"""

import argparse
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import DATASET_TYPES  # noqa: E402

# 50 题目标配额（SPEC R6 五类）；候选不足的类型如实报缺，人工补写
DEFAULT_QUOTA = {"fact": 18, "synthesis": 10, "table": 6, "comparison": 6, "unanswerable": 10}

_DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2}

# 人工补充模板：跨产品混淆（最危险的拒答场景——A 产品概念安到 B 产品头上）
# 与 comparison（跨文档综合）。ground_truth 留白，kaka 依据两份原文核写。
SUPPLEMENTS: list[dict] = [
    {"question": "「爱伴航2」重疾险可以像自愿医保那样申请税务扣减吗？",
     "type": "unanswerable", "difficulty": "hard", "ground_truth": "",
     "draft_answer": "考察跨产品混淆：税务扣减是 VHIS 认可产品特性，危疾语料无据→应拒答或有据澄清",
     "source_product": "跨产品"},
    {"question": "GlobalFlexi 储蓄计划的等候期是多少天？",
     "type": "unanswerable", "difficulty": "hard", "ground_truth": "",
     "draft_answer": "储蓄计划语料无等候期概念，不得套用医疗/危疾条目→应拒答",
     "source_product": "跨产品"},
    {"question": "自愿医保 VHIS 的危疾赔偿是保额的多少倍？",
     "type": "unanswerable", "difficulty": "medium", "ground_truth": "",
     "draft_answer": "危疾倍数赔付是爱伴航2 概念，VHIS 语料无据→应拒答",
     "source_product": "跨产品"},
    {"question": "爱伴航2 的每年保障限额是多少？",
     "type": "unanswerable", "difficulty": "medium", "ground_truth": "",
     "draft_answer": "年度保障限额是 VHIS 医疗险概念，危疾按保额给付→应拒答",
     "source_product": "跨产品"},
    {"question": "爱伴航2 和自愿医保的等候期规定分别是什么？有什么不同？",
     "type": "comparison", "difficulty": "hard", "ground_truth": "",
     "draft_answer": "需人工依据两份文档核写：分别列出两款等候期条目并对比",
     "source_product": "跨产品"},
    {"question": "这三款产品里哪些有冷静期条款？各是多久？",
     "type": "comparison", "difficulty": "medium", "ground_truth": "",
     "draft_answer": "需人工依据三份文档核写", "source_product": "跨产品"},
    {"question": "自愿医保和爱伴航2 在保证续保或保证更新上的说法有什么区别？",
     "type": "comparison", "difficulty": "hard", "ground_truth": "",
     "draft_answer": "需人工依据两份文档核写", "source_product": "跨产品"},
]


def dedup_candidates(cands: list[dict]) -> list[dict]:
    """近重复去重：相邻/重叠 chunk 常生成同义问题，按（来源块, 问题前缀）滤重。"""
    seen: set[tuple] = set()
    out = []
    for c in cands:
        key = (c.get("source_chunk_id") or "", re.sub(r"\s", "", c["question"])[:16])
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def pick_by_quota(
    cands: list[dict], quota: dict[str, int]
) -> tuple[list[dict], dict[str, int]]:
    """按类型配额确定性挑选：类型内产品轮转（均衡覆盖）、难度升序交错。

    返回（选中列表, 缺额表）。缺额如实上报而非静默凑数——评测集偏科要可见。
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_type[str(c.get("type", "fact"))].append(c)

    picked: list[dict] = []
    shortfall: dict[str, int] = {}
    for qtype, want in quota.items():
        pool = sorted(
            by_type.get(qtype, []),
            key=lambda c: (
                _DIFF_ORDER.get(str(c.get("difficulty")), 1),
                str(c.get("source_product")),
                c.get("id", 0),
            ),
        )
        groups: dict[str, deque] = defaultdict(deque)
        for c in pool:
            groups[str(c.get("source_product"))].append(c)
        chosen: list[dict] = []
        while len(chosen) < want and any(groups.values()):
            for product in sorted(groups):
                if groups[product] and len(chosen) < want:
                    chosen.append(groups[product].popleft())
        picked.extend(chosen)
        if len(chosen) < want:
            shortfall[qtype] = want - len(chosen)
    return picked, shortfall


def assemble(candidates: list[dict], quota: dict[str, int]) -> tuple[list[dict], dict[str, int]]:
    """人工补充模板必选；剩余配额从候选池按类型/产品/难度装配。"""
    remaining = dict(quota)
    for s in SUPPLEMENTS:
        t = s["type"]
        if remaining.get(t, 0) > 0:
            remaining[t] -= 1
    picked, shortfall = pick_by_quota(dedup_candidates(candidates), remaining)
    rows = list(SUPPLEMENTS) + picked
    return rows, shortfall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("eval/candidates.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("eval/dataset.draft.jsonl"))
    args = parser.parse_args()

    if not args.candidates.exists():
        print(f"候选文件不存在：{args.candidates}（先跑 scripts/gen_eval_candidates.py）",
              file=sys.stderr)
        return 2
    candidates = [
        json.loads(line)
        for line in args.candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows, shortfall = assemble(candidates, DEFAULT_QUOTA)

    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, start=1):
            record = {"id": i, **row, "needs_review": True}
            assert record["type"] in DATASET_TYPES, record["type"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[r["type"]] += 1
    print(f"草稿 {len(rows)} 题写入 {args.out}（全部带 needs_review，核对后才可用）")
    print(f"类型分布：{dict(counts)}")
    if shortfall:
        print(f"⚠ 候选池缺额（需人工补写）：{shortfall}")
    print("下一步：逐条对照 PDF 原文核对 ground_truth → 删除 needs_review → "
          "改名 eval/dataset.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
