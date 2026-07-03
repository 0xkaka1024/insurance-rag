"""Query 预处理：术语归一 + 规则意图路由。

路由是防幻觉第一道闸门（红线）：保费/费率类数值问题在检索之前直接拦截，
不允许 RAG 生成任何保费数字；引导用户以官方报价为准（v2 接费率表查询）。

注意区分两类问题：
- 「30岁买每年多少钱」→ 数值问题，拦截
- 「等待期多少天」「赔偿限额多少」→ 条款数值，正常回答（锚定金钱名词而非“多少”）
"""

import re
from dataclasses import dataclass

# 口语 → 条款用语；按 key 长度降序替换，避免短词截胡长词
# 注意：香港条款用「等候期」（v1 语料），大陆习惯说「等待期」——归一到语料词汇
TERM_MAP: dict[str, str] = {
    "等待期": "等候期",
    "观察期": "等候期",
    "免责期": "等候期",
    "犹豫期可以退吗": "冷静期",
    "反悔期": "冷静期",
    "免赔额": "自付费",
    "起付线": "自付费",
    "垫底费": "自付费",
    "老毛病": "已有病症",
    "既往病史": "已有病症",
    "以前得过的病": "已有病症",
    "去世": "身故",
    "过世": "身故",
    "死了": "身故",
    "身故金": "身故赔偿",
    "赔钱": "赔偿",
    "打钱": "赔偿",
    "给付钱": "给付",
    "动手术": "外科手术",
    "做手术": "外科手术",
    "住院费": "住院保障",
    "买保险不告知": "未如实告知",
    "隐瞒病情": "未如实告知",
    "断保": "保单终止",
    "停保": "保单终止",
    "退保拿钱": "退保价值",
    "锁定期": "保单年期",
}

# 保费数值意图：锚定金钱名词/句式，"多少天""限额多少"等条款数值不受影响
_PREMIUM_PATTERNS = [
    r"多少钱|几多钱|几钱|什么价|咩价",
    r"保费.{0,6}(多少|几多|贵|平|便宜)",
    r"(多少|几多).{0,4}保费",
    r"价格|报价|费率表",
    r"(年缴|月缴|年交|月交|供款|缴费).{0,6}(多少|几多|金额)",
    r"\d+\s*岁.{0,12}(买|投保).{0,8}(多少|几多|钱)",
]
_PREMIUM_RE = re.compile("|".join(f"(?:{p})" for p in _PREMIUM_PATTERNS))

REFUSAL_PREMIUM = (
    "涉及保费/费率的具体数字不在本助手回答范围：条款资料无法可靠推算个人保费，"
    "为避免误导，请以官方计划书或报价系统为准。（v2 将支持费率表精确查询）"
)


@dataclass
class Route:
    kind: str  # clause | premium_refuse
    question: str  # 归一后的问题
    matched: str = ""  # 命中的规则（日志/调试用）


def normalize_terms(question: str) -> str:
    for src in sorted(TERM_MAP, key=len, reverse=True):
        question = question.replace(src, TERM_MAP[src])
    return question


def route(question: str) -> Route:
    m = _PREMIUM_RE.search(question)
    if m:
        return Route(kind="premium_refuse", question=question, matched=m.group(0))
    return Route(kind="clause", question=normalize_terms(question))
