"""SiliconFlow rerank API（bge-reranker-v2-m3）：cross-encoder 精排。

容错策略（ARCHITECTURE 非功能性要求 + G3 加固）：
- 独立短超时（rerank_timeout_s）+ 至多 rerank_max_retries 次重试：pipeline
  有零成本降级兜底，慢降级比失败更伤（旧全局配置最坏 ~127s 才降级）
- 熔断：连续 _BREAKER_THRESHOLD 次失败后开路 cooldown 秒，期间直接抛
  RerankUnavailable（立即降级），不再逐请求耗完整个重试周期
- 只重试可恢复错误（网络、429、5xx）；RerankUnavailable 由 pipeline 降级
"""

import logging
import threading
import time
from collections.abc import Callable

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger("rerank")

_BREAKER_THRESHOLD = 3  # 连续失败次数达到即熔断


class RerankUnavailable(Exception):
    """rerank 服务不可用（重试耗尽/熔断中）；调用方应降级而非崩溃。"""


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):  # 连接失败、超时等
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class RerankClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ):
        s = settings or get_settings()
        self._url = s.siliconflow_base_url.rstrip("/") + "/rerank"
        self._key = s.siliconflow_api_key
        self.model = s.rerank_model
        self._max_retries = s.rerank_max_retries
        self._client = client or httpx.Client(timeout=s.rerank_timeout_s)
        self._sleep = sleep
        # 熔断状态（进程级单例共享；now 可注入供测试拨表）
        self._now = now
        self._cooldown = s.rerank_breaker_cooldown_s
        self._lock = threading.Lock()
        self._fail_streak = 0
        self._open_until = 0.0

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 [(原列表下标, 相关性分)]，按分数降序，长度 <= top_n。"""
        with self._lock:
            if self._now() < self._open_until:
                raise RerankUnavailable(
                    f"circuit open（连续失败 {self._fail_streak} 次，冷却中直接降级）"
                )
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        last_err: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post(
                    self._url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._key}"},
                )
                resp.raise_for_status()
                results = resp.json()["results"]
                with self._lock:
                    self._fail_streak = 0  # 成功即复位熔断计数
                return [(r["index"], float(r["relevance_score"])) for r in results]
            except Exception as exc:  # noqa: BLE001 - 分类后决定重试或快速失败
                last_err = exc
                if not _retryable(exc):
                    break
                if attempt < self._max_retries:
                    backoff = 2**attempt
                    logger.warning(
                        "rerank retry %s/%s in %ss: %s",
                        attempt + 1,
                        self._max_retries,
                        backoff,
                        exc,
                    )
                    self._sleep(backoff)
        with self._lock:
            self._fail_streak += 1
            if self._fail_streak >= _BREAKER_THRESHOLD:
                self._open_until = self._now() + self._cooldown
                logger.warning(
                    "rerank circuit opened for %ss after %s consecutive failures",
                    self._cooldown,
                    self._fail_streak,
                )
        raise RerankUnavailable(str(last_err)) from last_err
