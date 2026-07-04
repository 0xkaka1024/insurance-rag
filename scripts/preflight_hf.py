"""HF Spaces 部署前自检：确保索引齐备且会随镜像上去，Docker COPY data/index 不会失败。

用法：
    python scripts/preflight_hf.py        # 退出非零 = 有硬性问题，先别 push

检查项：索引文件齐全、collection 非空（复用启动就绪逻辑）、data/index 是否会进
HF push（被 gitignore 且未 force-add → 构建必失败）、Dockerfile 端口与 COPY 目标。
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import get_settings  # noqa: E402

REQUIRED_FILES = ("bm25_fixed.pkl", "bm25_structural.pkl", "ingest_manifest.json")


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    hard: bool = True  # 硬性失败 → 退出非零；软性失败只告警


def check_index_artifacts(index_dir: Path) -> list[Check]:
    """索引落盘产物齐全性（纯文件系统检查，可单测）。"""
    checks = [Check("data/index 目录存在", index_dir.is_dir(), str(index_dir))]
    for name in REQUIRED_FILES:
        p = index_dir / name
        checks.append(Check(f"索引文件 {name}", p.is_file(), str(p)))
    chroma = index_dir / "chroma"
    has_chroma = chroma.is_dir() and any(chroma.iterdir())
    checks.append(Check("chroma 目录非空", has_chroma, str(chroma)))
    return checks


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - 无 git 环境降级为空
        return ""


def check_pushable(rel_path: str) -> Check:
    """data/index 是否会随 push 上传：被 ignore 且 ls-files 为空 → 构建会失败。"""
    tracked = bool(_git("ls-files", rel_path))
    return Check(
        f"{rel_path} 会随 push 上传",
        tracked,
        "被 gitignore：直接 git push 不含索引，HF 构建会在 COPY data/index 处失败。\n"
        "      标准发版路径：bash scripts/deploy_hf.sh（临时 worktree 自动经 LFS 打包索引）",
        hard=False,  # deploy_hf.sh 流程下该告警属预期，不算硬失败
    )


def run_checks(settings) -> list[Check]:
    checks = check_index_artifacts(settings.index_dir)
    try:
        from app.ingest.indexer import Indexer

        indexer = Indexer(settings)
        for strategy in ("fixed", "structural"):
            count = indexer.collection(strategy).count()
            checks.append(Check(f"{strategy} collection 非空", count > 0, f"{count} 块"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("collection 可读", False, repr(exc)))
    checks.append(check_pushable("data/index"))
    checks.append(
        Check("eval/results 目录存在（COPY 目标）", Path("eval/results").is_dir(),
              hard=False)
    )
    df = Path("Dockerfile")
    dockerfile = df.read_text(encoding="utf-8") if df.is_file() else ""
    checks.append(Check("Dockerfile 端口 7860", "7860" in dockerfile, hard=False))
    return checks


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    checks = run_checks(get_settings())

    hard_fail = False
    for c in checks:
        mark = "✓" if c.ok else ("✗" if c.hard else "⚠")
        line = f"  {mark} {c.name}"
        if c.detail and not c.ok:
            line += f"\n      {c.detail}"
        elif c.detail:
            line += f"（{c.detail}）"
        print(line)
        if not c.ok and c.hard:
            hard_fail = True

    if hard_fail:
        print("\n✗ 存在硬性问题，修复后再 push（否则 HF 构建会失败）")
        return 1
    if any(not c.ok for c in checks):
        print("\n⚠ 有告警项，确认无碍后再 push")
    else:
        print("\n✓ 全部通过，可以 push 到 HF Space")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
