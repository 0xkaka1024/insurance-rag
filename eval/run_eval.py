"""RAGAS 评测 CLI（只手动跑，不进 CI；调用真实 LLM，花钱）。

用法：
    python eval/run_eval.py                                   # 8 套全组合
    python eval/run_eval.py --chunking structural --retrieval hybrid --rerank on
    python eval/run_eval.py --metrics faithfulness answer_relevancy   # 无需 ground_truth
    python eval/run_eval.py --dataset eval/smoke.jsonl
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
    args = parser.parse_args()

    rows = load_dataset(args.dataset)
    validate_metrics(args.metrics, rows)
    configs = expand_configs(
        args.chunking, args.retrieval, None if args.rerank is None else args.rerank == "on"
    )

    settings = get_settings()
    if args.llm:
        settings = settings.model_copy(update={"llm_model": args.llm})
    pipeline = build_pipeline(settings)

    print(f"数据集 {args.dataset}（{len(rows)} 题）× {len(configs)} 套配置，"
          f"指标 {args.metrics}")
    entries = []
    for i, cfg in enumerate(configs, start=1):
        print(f"[{i}/{len(configs)}] {cfg.model_dump()} ...", flush=True)
        entry = evaluate_config(pipeline, cfg, rows, llm_model=settings.llm_model)
        entry["metrics"] = score_with_ragas(entry["records"], args.metrics, settings)
        entry["metrics_penalized"] = penalized_means(
            entry["metrics"], entry["n_scored"], entry["n_answerable"]
        )
        entries.append(entry)
        print(f"    metrics={entry['metrics']} penalized={entry['metrics_penalized']} "
              f"refusal={entry['refusal_accuracy']} false_refusal={entry['false_refusal_rate']} "
              f"scored={entry['n_scored']}/{entry['n_answerable']} "
              f"cost=¥{entry['cost_cny']} {entry['duration_s']}s")

    now = datetime.now()
    payload = {
        "run_at": now.isoformat(timespec="seconds"),
        "git": git_short_hash(),
        "dataset": str(args.dataset),
        "n_questions": len(rows),
        "requested_metrics": args.metrics,
        "llm": settings.llm_model,
        "judge": {
            "model": settings.judge_model,
            "base_url": settings.judge_base_url or settings.deepseek_base_url,
        },
        "total_cost_cny": (
            None  # 任一配置成本未知（价格表缺模型）→ 总额不假装精确
            if any(e["cost_cny"] is None for e in entries)
            else round(sum(e["cost_cny"] for e in entries), 4)
        ),
        "total_duration_s": round(sum(e["duration_s"] for e in entries), 2),
        "configs": entries,
    }
    filename = result_filename(now.strftime("%Y%m%d"), payload["git"])
    path = save_result(RESULTS_DIR, filename, payload)
    print(f"结果已写入 {path}（记得 git add 提交，评测历史入库）")
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
