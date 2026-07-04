"""RAGAS 评测 CLI（只手动跑，不进 CI；调用真实 LLM，花钱）。

用法：
    python eval/run_eval.py                                   # 8 套全组合
    python eval/run_eval.py --chunking structural --retrieval hybrid --rerank on
    python eval/run_eval.py --metrics faithfulness answer_relevancy   # 无需 ground_truth
    python eval/run_eval.py --dry-run --limit 3               # 冒烟：只跑 pipeline 不跑 judge
    python eval/run_eval.py --dataset eval/smoke.jsonl

容错与恢复：逐题异常记入 record.error 不中断；每套配置评完立即落盘
（partial=true），中途被杀已花的钱不白烧；文件名含时分秒不互相覆盖。
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.rag.pipeline import build_pipeline  # noqa: E402
from eval.harness import (  # noqa: E402
    ALL_METRICS,
    GT_FREE_METRICS,
    corpus_fingerprint,
    dataset_sha256,
    evaluate_config,
    expand_configs,
    git_short_hash,
    load_dataset,
    penalized_means,
    result_filename,
    save_result,
    score_with_ragas,
    validate_metrics,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunking", choices=["fixed", "structural"])
    parser.add_argument("--retrieval", choices=["vector", "hybrid"])
    parser.add_argument("--rerank", choices=["on", "off"])
    parser.add_argument("--llm", default=None, help="覆盖生成模型（如 qwen-plus，P1）")
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.jsonl"))
    parser.add_argument(
        "--metrics", nargs="+", default=list(ALL_METRICS),
        help=f"可选 {ALL_METRICS}；无 ground_truth 时用 {GT_FREE_METRICS}",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟省钱）")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只跑 pipeline 不调 RAGAS judge、不落盘：验证链路与拒答/命中指标",
    )
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    if args.limit:
        rows = rows[: args.limit]
        print(f"--limit {args.limit}：只跑前 {len(rows)} 题（冒烟模式）")
    if not args.dry_run:
        validate_metrics(args.metrics, rows)
    configs = expand_configs(
        args.chunking, args.retrieval, None if args.rerank is None else args.rerank == "on"
    )

    settings = get_settings()
    if args.llm:
        settings = settings.model_copy(update={"llm_model": args.llm})
    pipeline = build_pipeline(settings)

    now = datetime.now()
    filename = result_filename(
        now.strftime("%Y%m%d"), now.strftime("%H%M%S"), git_short_hash()
    )
    base = {
        "run_at": now.isoformat(timespec="seconds"),
        "git": git_short_hash(),
        "dataset": str(args.dataset),
        "dataset_sha": dataset_sha256(args.dataset),
        "n_questions": len(rows),
        "requested_metrics": args.metrics,
        "llm": settings.llm_model,
        "models": {
            "llm": settings.llm_model,
            "embedding": settings.embedding_model,
            "rerank": settings.rerank_model,
        },
        "judge": {
            "model": settings.judge_model,
            "base_url": settings.judge_base_url or settings.deepseek_base_url,
        },
        "params": {
            "top_k": settings.top_k,
            "recall_k": settings.recall_k,
            "refuse_threshold": settings.refuse_threshold,
            "vector_floor": settings.vector_floor,
        },
        "corpus": corpus_fingerprint(settings),  # 语料指纹：换语料重建后结果可区分
    }

    print(f"数据集 {args.dataset}（{len(rows)} 题）× {len(configs)} 套配置，"
          f"指标 {args.metrics}{'（dry-run 不打分）' if args.dry_run else ''}")
    entries = []
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg.model_dump()} ...", flush=True)
        entry = evaluate_config(pipeline, cfg, rows, llm_model=settings.llm_model)
        entry["metrics"] = (
            {m: None for m in args.metrics}
            if args.dry_run
            else score_with_ragas(entry["records"], args.metrics, settings)
        )
        entry["metrics_penalized"] = penalized_means(
            entry["metrics"], entry["n_scored"], entry["n_answerable"]
        )
        entries.append(entry)
        print(f"    metrics={entry['metrics']} penalized={entry['metrics_penalized']} "
              f"refusal={entry['refusal_accuracy']} false_refusal={entry['false_refusal_rate']} "
              f"hit={entry['retrieval_hit_rate']} cite_hit={entry['citation_hit_rate']} "
              f"scored={entry['n_scored']}/{entry['n_answerable']} errors={entry['errors']} "
              f"cost=¥{entry['cost_cny']} {entry['duration_s']}s")
        if not args.dry_run:
            # 每配置立即落盘（partial 标记）：中途被杀已花的钱不白烧
            save_result(RESULTS_DIR, filename, {**base, "partial": True, "configs": entries})

    totals = {
        "total_cost_cny": (
            None  # 任一配置成本未知（价格表缺模型）→ 总额不假装精确
            if any(e["cost_cny"] is None for e in entries)
            else round(sum(e["cost_cny"] for e in entries), 4)
        ),
        "total_duration_s": round(sum(e["duration_s"] for e in entries), 2),
    }
    if args.dry_run:
        print(f"dry-run 完成（耗时 {totals['total_duration_s']}s，不落盘、不计 judge 成本）")
        return 0
    path = save_result(
        RESULTS_DIR, filename, {**base, "partial": False, **totals, "configs": entries}
    )
    print(f"结果已写入 {path}（记得 git add 提交，评测历史入库）")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
