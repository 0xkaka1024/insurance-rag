"""批量入库 CLI。

用法：
    python scripts/ingest.py data/raw/newVHISmedical-tc.pdf
    python scripts/ingest.py data/raw/          # 目录下所有 PDF（白名单外自动拒绝）
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.logging import setup_logging  # noqa: E402
from app.ingest.service import ingest_files  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="解析条款 PDF 并写入检索索引")
    parser.add_argument("paths", nargs="+", type=Path, help="PDF 文件或目录")
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.pdf")))
        elif p.exists():
            files.append(p)
        else:
            print(f"路径不存在：{p}", file=sys.stderr)
            return 2
    if not files:
        print("未找到任何 PDF", file=sys.stderr)
        return 2

    result = ingest_files(files)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    setup_logging()
    raise SystemExit(main())
