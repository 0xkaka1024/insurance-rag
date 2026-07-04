"""数据治理（CLAUDE.md 红线的代码级强制）：deny 规则 + 白名单 + 内容指纹。

设计原则（docs/REVIEW-2026-07.md P0-3）：
- 文件名不可信——deny 匹配前先归一化（去空格/下划线/连字符），
  白名单在名称之外再校验内容 sha256，重命名穿透不了；
- fail-closed：名单外、指纹未登记一律拒绝；
- 新文件/新版本入库需在 FINGERPRINTS 登记指纹——改代码走 git 审阅，
  这次「多一步」就是治理红线要的审批留痕。
"""

import re
from pathlib import Path

from app.ingest.parser import product_from_filename

__all__ = ["INGEST_WHITELIST", "FINGERPRINTS", "check_ingestable", "product_from_filename"]

# 归一化后匹配：Training_Deck / premium table / premium-table 等变体全部覆盖
_DENY_NORMALIZED = ("trainingdeck", "premiumtable")
_NORM_RE = re.compile(r"[\s_\-]+")

# v1 入库白名单：三款结构差异大的产品（医疗 VHIS + 储蓄 + 危疾爱伴航2）
INGEST_WHITELIST = frozenset(
    {"newVHISmedical", "GlobalFlexiSavingsInsurancePlan", "OnYourSideInsurancePlan2"}
)

# 产品 → 已登记的文件内容 sha256（允许多版本并存）。
# 登记方法：shasum -a 256 <file>，把摘要加进对应产品的集合。
FINGERPRINTS: dict[str, frozenset[str]] = {
    "newVHISmedical": frozenset(
        {"db5c8094d7ebd0c617af8ef32b790cd5e5e29cc47134af591e61ba72f66ac4ea"}
    ),
    "GlobalFlexiSavingsInsurancePlan": frozenset(
        {"be64894dc4c297ed191e3a2f32e2bf2bee4f7f0c6b29c7cdee36fa920a8668a4"}
    ),
    "OnYourSideInsurancePlan2": frozenset(
        {"8d7302386710efbc4bb5d073e4cc86e5141c9bd075cb8de771dc2ae6892ceb0a"}
    ),
}


def check_ingestable(
    path: Path,
    digest: str | None = None,
    fingerprints: dict[str, frozenset[str]] | None = None,
) -> tuple[bool, str]:
    """名称级 + 内容级双重校验。digest=None 时只做名称级（快速预检）。

    fingerprints 可注入（测试用夹具指纹）；缺省用登记表 FINGERPRINTS。
    """
    normalized = _NORM_RE.sub("", path.name.lower())
    for pattern in _DENY_NORMALIZED:
        if pattern in normalized:
            return False, f"红线拒绝：文件名归一化后含 '{pattern}'（内部材料/费率表不入库）"
    product = product_from_filename(path)
    if product not in INGEST_WHITELIST:
        return False, f"白名单外：产品 '{product}' 不在 v1 入库白名单 {sorted(INGEST_WHITELIST)}"
    if digest is not None:
        registered = (fingerprints if fingerprints is not None else FINGERPRINTS).get(
            product, frozenset()
        )
        if digest not in registered:
            return False, (
                f"内容指纹未登记：'{path.name}' 的 sha256 不在产品 '{product}' 的登记表中。"
                "确认文件来源后，用 shasum -a 256 计算摘要并登记到 governance.FINGERPRINTS。"
            )
    return True, ""
