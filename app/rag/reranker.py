"""SiliconFlow rerank API（bge-reranker-v2-m3）：cross-encoder 精排。

容错策略（ARCHITECTURE 非功能性要求）：
- 超时 + 指数退避重试（1s/2s/4s...），只重试可恢复错误（网络、429、5xx）
- 重试耗尽抛 RerankUnavailable，由 pipeline 降级为跳过重排，不中断问答
"""

import logging
import time
from collections.abc import Callable

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger("rerank")


class RerankUnavailable(Exception):
    """rerank 服务不可用（重试耗尽）；调用方应降级而非崩溃。"""


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
    ):
        s = settings or get_settings()
        self._url = s.siliconflow_base_url.rstrip("/") + "/rerank"
        self._key = s.siliconflow_api_key
        self.model = s.rerank_model
        self._max_retries = s.max_retries
        self._client = client or httpx.Client(timeout=s.request_timeout_s)
        self._sleep = sleep

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """返回 [(原列表下标, 相关性分)]，按分数降序，长度 <= top_n。"""
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
        raise RerankUnavailable(str(last_err)) from last_err
