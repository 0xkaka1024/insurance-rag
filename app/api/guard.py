"""公开部署防滥用（G3）：鉴权 → 每 IP 限流 → 每日额度熔断，三道闸依次过。

昂贵端点（/ask、/playground/ask、/retrieve 都会触发付费 API 调用）挂此依赖。
三道闸全部默认关闭（Settings 各字段为 0/空），生产由 Dockerfile / Space
变量开启——与 STARTUP_REQUIRE_INDEX 同一模式，测试与本地开发零负担。

顺序讲究：被限流拒掉的请求不应消耗每日额度；额度熔断宁可拒绝服务，
也不让公网脚本在无人值守时烧完账单。
"""

import logging
import threading
import time
from collections.abc import Callable

from fastapi import HTTPException, Request

from app.core.config import Settings, get_settings

logger = logging.getLogger("guard")

_PRUNE_THRESHOLD = 4096  # 限流表超过此条目数时清理过期窗口，防内存缓慢膨胀


def client_ip(request: Request) -> str:
    """HF/反代之后取真实来源：x-forwarded-for 首跳，否则直连地址。"""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class AbuseGuard:
    def __init__(self, now: Callable[[], float] = time.time):
        self._now = now
        self._lock = threading.Lock()
        self._minute_hits: dict[str, tuple[int, int]] = {}  # ip -> (分钟窗, 计数)
        self._day = ("", 0)  # (UTC 日期, 计数)

    def check(self, request: Request, settings: Settings) -> None:
        if settings.api_auth_token:
            auth = request.headers.get("authorization", "")
            if auth != f"Bearer {settings.api_auth_token}":
                raise HTTPException(status_code=401, detail="未授权：缺少或错误的 API token")

        if settings.rate_limit_per_minute > 0:
            ip = client_ip(request)
            minute = int(self._now() // 60)
            with self._lock:
                win, n = self._minute_hits.get(ip, (minute, 0))
                if win != minute:
                    win, n = minute, 0
                if n >= settings.rate_limit_per_minute:
                    logger.warning("rate limited", extra={"extra_fields": {"ip": ip}})
                    raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试。")
                self._minute_hits[ip] = (win, n + 1)
                if len(self._minute_hits) > _PRUNE_THRESHOLD:
                    self._minute_hits = {
                        k: v for k, v in self._minute_hits.items() if v[0] == minute
                    }

        if settings.daily_request_budget > 0:
            day = time.strftime("%Y-%m-%d", time.gmtime(self._now()))
            with self._lock:
                if self._day[0] != day:
                    self._day = (day, 0)
                if self._day[1] >= settings.daily_request_budget:
                    logger.error(
                        "daily budget exhausted",
                        extra={"extra_fields": {"budget": settings.daily_request_budget}},
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="今日调用额度已用完（成本熔断），请明天再试。",
                    )
                self._day = (day, self._day[1] + 1)


_guard = AbuseGuard()


def abuse_guard(request: Request) -> None:
    """FastAPI 依赖入口；进程级单例状态（单容器部署下即全局状态）。"""
    _guard.check(request, get_settings())
