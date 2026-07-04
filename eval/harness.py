"""评测编排：纯逻辑与持久化（离线可单测）；RAGAS 打分延迟导入、显式配 DeepSeek judge。

设计（SPEC R6，2026-07-03 定稿）：
- 缺省 2(切片)×2(检索)×2(重排) = 8 套配置全组合，CLI 可锁单套
- 结果持久化 eval/results/{YYYYMMDD}_{git短hash}.json，目录入 git
- --metrics 支持只跑无需 ground_truth 的子集（faithfulness / answer_relevancy）
"""

import hashlib
import itertools
import json
import logging
import subprocess
import time
from pathlib import Path

from app.core.config import Settings
from app.rag.pipeline import RagConfig, RagPipeline

logger = logging.getLogger("eval")

GT_FREE_METRICS = ("faithfulness", "answer_relevancy")
ALL_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# 期望拒答的题型（种子题模板用 unanswerable，兼容简写 refuse）
REFUSE_TYPES = frozenset({"unanswerable", "refuse"})
# 合法题型全集（SPEC R6 五类）：拼写错误的 type 会静默逃过拒答统计，必须白名单校验
DATASET_TYPES = frozenset({"fact", "synthesis", "table", "comparison"}) | REFUSE_TYPES

# DeepSeek 定价（CNY / 1M tokens，2026-07 官网价，调价请更新此表）
PRICE_PER_MTOK = {"deepseek-chat": {"in": 2.0, "out": 8.0}}


def expand_configs(
    chunking: str | None = None,
    retrieval: str | None = None,
    rerank: bool | None = None,
) -> list[RagConfig]:
    cs = [chunking] if chunking else ["fixed", "structural"]
    rs = [retrieval] if retrieval else ["vector", "hybrid"]
    ks = [rerank] if rerank is not None else [False, True]
    return [
        RagConfig(chunking=c, retrieval=r, rerank=k)
        for c, r, k in itertools.product(cs, rs, ks)
    ]


def load_dataset(path: Path) -> list[dict]:
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "question" not in row or "type" not in row:
            raise ValueError(f"{path}:{i} 缺少 question/type 字段")
        if row["type"] not in DATASET_TYPES:
            raise ValueError(
                f"{path}:{i} 未知题型 '{row['type']}'，可选 {sorted(DATASET_TYPES)}"
            )
        if row.get("needs_review"):
            # 人工核对门禁（红线）：未经核对的 ground_truth 跑出的数字不可对外
            raise ValueError(
                f"{path}:{i} 仍标记 needs_review：ground_truth 未经人工核对。"
                "对照 PDF 原文核对后删除该字段再运行（流程见 eval/build_dataset.py）。"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"{path} 为空数据集")
    return rows


def validate_metrics(metrics: list[str], rows: list[dict]) -> None:
    unknown = set(metrics) - set(ALL_METRICS)
    if unknown:
        raise ValueError(f"未知指标：{sorted(unknown)}，可选 {ALL_METRICS}")
    needs_gt = set(metrics) - set(GT_FREE_METRICS)
    if needs_gt:
        missing = [
            r["question"][:20]
            for r in rows
            if r["type"] not in REFUSE_TYPES and not r.get("ground_truth")
        ]
        if missing:
            raise ValueError(
                f"指标 {sorted(needs_gt)} 需要 ground_truth，"
                f"但 {len(missing)} 条问题缺失（如：{missing[:3]}）。"
                f"可先用 --metrics {' '.join(GT_FREE_METRICS)} 跑无参照指标。"
            )


def scorable(record: dict) -> bool:
    """可进 RAGAS 打分池的 record：可答题、未被拒答、pipeline 未出错。"""
    return (
        record["type"] not in REFUSE_TYPES
        and not record["refused"]
        and not record.get("error")
    )


def penalized_means(means: dict, n_scored: int, n_answerable: int) -> dict:
    """误拒/出错题按 0 分计入的均值。

    RAGAS 只对实际作答的题打分（拒答文本没有可评内容），单看 scored 均值
    会给「多拒答少作答」的配置博弈空间——阈值一收紧，难题被藏出打分池，
    均值反而上升。惩罚后均值以全部可答题为分母，两个数并列展示。
    """
    if not n_answerable:
        return {m: None for m in means}
    return {
        m: (None if v is None else round(v * n_scored / n_answerable, 4))
        for m, v in means.items()
    }


def _gold_span(row: dict) -> tuple[str, int, int] | None:
    """金标位置：候选题生成时记录的 (source_product, source_pages)。

    用「产品 + 页码区间重叠」而非 chunk_id 相等判命中：chunk_id 绑定切片
    策略与语料版本（重建索引 seq 全变），页码金标对 fixed/structural 一视同仁。
    """
    product, pages = row.get("source_product"), row.get("source_pages")
    if not product or not pages:
        return None
    start, _, end = str(pages).partition("-")
    try:
        p0 = int(start)
        p1 = int(end) if end else p0
    except ValueError:
        return None
    return str(product), p0, p1


def _overlaps(chunk, gold: tuple[str, int, int]) -> bool:
    product, p0, p1 = gold
    return chunk.product == product and chunk.page_start <= p1 and chunk.page_end >= p0


def evaluate_config(
    pipeline: RagPipeline,
    cfg: RagConfig,
    rows: list[dict],
    llm_model: str = "deepseek-chat",
) -> dict:
    """逐题跑 pipeline，产出 RAGAS 输入 records 与拒答/误拒/命中/成本/耗时指标。

    retrieval_hit / citation_hit 是不依赖 judge 的确定性指标：零成本、
    可逐 commit 回归，度量「检索到没有」与「引用的位置对不对」——后者
    正是 RAGAS faithfulness 不覆盖的合规命门（内容对但页码张冠李戴不扣分）。
    """
    price = PRICE_PER_MTOK.get(llm_model)
    if price is None:
        # 静默记 0 会让成本报表失真；记 None 并告警，提示更新价格表
        logger.warning("模型 %s 不在 PRICE_PER_MTOK 价格表中，cost_cny 记为 None", llm_model)
    records: list[dict] = []
    refusal_hits = 0
    refusal_total = 0
    n_answerable = 0
    false_refusals = 0
    errors = 0
    cost = 0.0
    t0 = time.perf_counter()
    for row in rows:
        # 逐题容错：400 次真金调用的批次不能被一次限流/超时整场作废；
        # 出错题记 error 字段（不进任何指标分母），批次继续
        try:
            result = pipeline.ask(row["question"], cfg)
        except Exception as exc:
            logger.exception("eval question failed: %s", row["question"][:30])
            errors += 1
            records.append(
                {
                    "question": row["question"], "type": row["type"],
                    "answer": "", "contexts": [],
                    "ground_truth": row.get("ground_truth", ""),
                    "refused": False, "refuse_reason": "", "citations": 0,
                    "retrieved_chunk_ids": [], "cited_chunk_ids": [],
                    "retrieval_hit": None, "citation_hit": None,
                    "timings": {}, "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        is_refuse_type = row["type"] in REFUSE_TYPES
        if is_refuse_type:
            refusal_total += 1
            refusal_hits += int(result.refused)
        else:
            n_answerable += 1
            false_refusals += int(result.refused)
        usage = result.meta.get("usage", {})
        if price is not None:
            cost += usage.get("prompt_tokens", 0) / 1e6 * price["in"]
            cost += usage.get("completion_tokens", 0) / 1e6 * price["out"]

        gold = None if is_refuse_type else _gold_span(row)
        retrieval_hit = citation_hit = None
        cited_ids = [c.chunk_id for c in result.citations]
        if gold:
            retrieval_hit = any(_overlaps(c, gold) for c in result.chunks)
            if not result.refused:
                cited = [c for c in result.chunks if c.chunk_id in set(cited_ids)]
                citation_hit = any(_overlaps(c, gold) for c in cited)
        records.append(
            {
                "question": row["question"],
                "type": row["type"],
                "answer": result.answer,
                "contexts": [c.text for c in result.chunks],
                "ground_truth": row.get("ground_truth", ""),
                "refused": result.refused,
                "refuse_reason": result.refuse_reason,
                "citations": len(result.citations),
                "retrieved_chunk_ids": [c.chunk_id for c in result.chunks],
                "cited_chunk_ids": cited_ids,
                "retrieval_hit": retrieval_hit,
                "citation_hit": citation_hit,
                "timings": result.timings,
            }
        )
    n_scored = sum(1 for r in records if scorable(r))
    r_hits = [r["retrieval_hit"] for r in records if r["retrieval_hit"] is not None]
    c_hits = [r["citation_hit"] for r in records if r["citation_hit"] is not None]
    return {
        "config": cfg.model_dump(),
        "records": records,
        "refusal_accuracy": (refusal_hits / refusal_total) if refusal_total else None,
        # 误拒率：可答题被拒答的比例。没有它，"拒答一切"的配置能拿满分拒答准确率
        "false_refusal_rate": round(false_refusals / n_answerable, 4) if n_answerable else None,
        "n_answerable": n_answerable,
        "n_scored": n_scored,
        "errors": errors,
        "n_gold": len(r_hits),
        "retrieval_hit_rate": round(sum(r_hits) / len(r_hits), 4) if r_hits else None,
        "citation_hit_rate": round(sum(c_hits) / len(c_hits), 4) if c_hits else None,
        "cost_cny": round(cost, 4) if price is not None else None,
        "duration_s": round(time.perf_counter() - t0, 2),
    }


def score_with_ragas(records: list[dict], metrics: list[str], settings: Settings) -> dict:
    """RAGAS 打分（真实调用 judge / embedding，只手动触发）。

    judge 显式配 DeepSeek（默认 OpenAI 会因无 key 报错，见 TODO 风险备忘）；
    answer_relevancy 的 embedding 走 SiliconFlow bge-m3。
    """
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "RAGAS 依赖未安装：pip install -r eval/requirements.txt"
        ) from exc

    metric_objs = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    judge = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.judge_model,  # 与被评模型解耦：--llm 只换生成侧
            api_key=settings.judge_api_key or settings.deepseek_api_key,
            base_url=settings.judge_base_url or settings.deepseek_base_url,
            temperature=0,
        )
    )
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.siliconflow_api_key,
            base_url=settings.siliconflow_base_url,
            check_embedding_ctx_length=False,
        )
    )
    # 拒答/出错题不进 judge（无生成内容可评、也不烧 judge 钱）；
    # 其代价通过 false_refusal_rate 与 metrics_penalized 显性呈现，
    # 不再是从打分池里静默消失（幸存者偏差）。
    scored = [r for r in records if scorable(r)]
    rows = [
        {
            "user_input": r["question"],
            "response": r["answer"],
            "retrieved_contexts": r["contexts"],
            "reference": r["ground_truth"],
        }
        for r in scored
    ]
    if not rows:
        return {m: None for m in metrics}
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[metric_objs[m] for m in metrics],
        llm=judge,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    # 逐题分数回填 records：两次运行可逐题 diff（"哪道题退步了"可回答）
    for r, (_, df_row) in zip(scored, df.iterrows(), strict=True):
        r["scores"] = {
            m: (None if df_row[m] != df_row[m] else round(float(df_row[m]), 4))
            for m in metrics  # NaN != NaN，judge 偶发解析失败的题记 None
        }
    return {m: round(float(df[m].mean()), 4) for m in metrics}


def git_short_hash() -> str:
    """短 hash；工作区有未提交改动时带 -dirty 后缀（结果不可假装来自干净 commit）。"""
    try:
        h = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{h}-dirty" if dirty else h
    except Exception:  # noqa: BLE001 - 无 git 环境（如容器内）用占位
        return "nogit"


def result_filename(date_str: str, time_str: str, git_hash: str) -> str:
    """含时分秒：同日同 commit 多次运行不再互相覆盖。"""
    return f"{date_str}_{time_str}_{git_hash}.json"


def dataset_sha256(path: Path) -> str:
    """数据集内容指纹（短）：结果文件只记路径会在题库悄悄变化后误导对比。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def corpus_fingerprint(settings: Settings) -> dict:
    """语料指纹：来自 ingest 质量报告。语料重建后跑分，结果必须能看出差别。"""
    from app.ingest.report import reports_dir

    d = reports_dir(settings)
    out: dict[str, dict] = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        out[r.get("product", p.stem)] = {
            "sha8": str(r.get("sha256", ""))[:8],
            "chunks": {k: v.get("n_chunks") for k, v in r.get("strategies", {}).items()},
        }
    return out


def save_result(results_dir: Path, filename: str, payload: dict) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
