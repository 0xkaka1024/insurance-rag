"""评测集候选题生成：从已切片的条款 chunk 批量生成问题草稿。

产出 eval/candidates.jsonl（**草稿，不入 git**）——每条带来源 chunk 与页码，
供人工逐条核对 ground_truth 后誊入 eval/dataset.jsonl（那一步不能省）。

用法：
    python scripts/gen_eval_candidates.py                 # 白名单全部 PDF
    python scripts/gen_eval_candidates.py --max-chunks 20 # 控制成本的小批量
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.llm import LLMClient  # noqa: E402
from app.ingest.chunker import Chunk  # noqa: E402
from app.ingest.parser import parse_pdf  # noqa: E402
from app.ingest.service import check_ingestable  # noqa: E402
from app.ingest.structural import StructuralChunker  # noqa: E402

GEN_SYSTEM = (
    "你是保险产品评测集构建助手。基于给定的条款/产品简介片段，"
    "生成保险代理人在真实工作中会问的问题。输出 JSON 数组，每个元素形如："
    '{"question": "...", "type": "fact|synthesis|table", "difficulty": "easy|medium|hard", '
    '"draft_answer": "仅依据片段可得的答案草稿"}。'
    "问题必须能从片段回答；口语化提问；不要问保费价格。只输出 JSON。"
)

# 拒答类候选不依赖 LLM：直接给模板，人工挑选改写
REFUSAL_CANDIDATES = [
    {"question": "这款产品今年的分红实现率是多少？", "type": "unanswerable",
     "difficulty": "easy", "draft_answer": "应拒答：条款/简介不含分红实现率数据"},
    {"question": "30岁男性买这款每年保费多少钱？", "type": "unanswerable",
     "difficulty": "easy", "draft_answer": "应路由拦截：保费数值属计划书/费率表范畴"},
    {"question": "隔壁公司同类产品比这款好吗？", "type": "unanswerable",
     "difficulty": "medium", "draft_answer": "应拒答：知识库无对方产品资料，且涉主观推荐"},
    {"question": "帮我预测下这款产品未来的现金价值走势", "type": "unanswerable",
     "difficulty": "medium", "draft_answer": "应拒答：非保证利益不可预测，条款无依据"},
]


def sample_chunks(chunks: list[Chunk], max_chunks: int, seed: int = 42) -> list[Chunk]:
    """确定性抽样：优先长 chunk（信息量足），同长度按 id 稳定。"""
    pool = sorted(chunks, key=lambda c: (-len(c.text), c.chunk_id))
    pool = [c for c in pool if len(c.text) >= 80]  # 太短的片段出不了好题
    random.Random(seed).shuffle(pool)
    return pool[:max_chunks]


def build_user_prompt(chunk: Chunk, per_chunk: int) -> str:
    loc = f"{chunk.product} 第{chunk.page_start}-{chunk.page_end}页"
    section = chunk.meta.get("section", "")
    return (
        f"【片段】（{loc}·{section}）\n{chunk.text}\n\n"
        f"基于该片段生成 {per_chunk} 个问题（JSON 数组）。"
    )


def parse_llm_json(text: str) -> list[dict]:
    """容忍 ```json 围栏与前后杂文，取第一个 JSON 数组。"""
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict) and i.get("question")]


def candidates_from_chunk(llm: LLMClient, chunk: Chunk, per_chunk: int) -> list[dict]:
    raw = llm.complete(GEN_SYSTEM, build_user_prompt(chunk, per_chunk))
    out = []
    for item in parse_llm_json(raw):
        out.append(
            {
                "question": item["question"],
                "type": item.get("type", "fact"),
                "difficulty": item.get("difficulty", "medium"),
                "ground_truth": "",  # 人工核对后填写，草稿见 draft_answer
                "draft_answer": item.get("draft_answer", ""),
                "source_chunk_id": chunk.chunk_id,
                "source_pages": f"{chunk.page_start}-{chunk.page_end}",
                "source_product": chunk.product,
                "note": "候选草稿：ground_truth 必须人工依据原文核对后定稿",
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/candidates.jsonl"))
    parser.add_argument("--max-chunks", type=int, default=25, help="每份 PDF 抽样上限")
    parser.add_argument("--per-chunk", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    raw_dir = args.raw_dir or settings.raw_dir
    llm = LLMClient(settings)

    candidates: list[dict] = []
    for pdf in sorted(raw_dir.glob("*.pdf")):
        ok, _ = check_ingestable(pdf)
        if not ok:
            continue
        chunks = StructuralChunker().split(parse_pdf(pdf))
        picked = sample_chunks(chunks, args.max_chunks)
        print(f"{pdf.name}: {len(chunks)} chunks，抽样 {len(picked)}", flush=True)
        for chunk in picked:
            candidates.extend(candidates_from_chunk(llm, chunk, args.per_chunk))

    candidates.extend(REFUSAL_CANDIDATES)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, c in enumerate(candidates, start=1):
            f.write(json.dumps({"id": i, **c}, ensure_ascii=False) + "\n")
    print(f"共 {len(candidates)} 条候选写入 {args.out}（草稿不入 git，核对后誊入 dataset.jsonl）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
