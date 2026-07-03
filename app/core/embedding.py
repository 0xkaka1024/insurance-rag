"""SiliconFlow embedding 客户端（OpenAI 兼容协议）。

超时与重试由 openai SDK 内建机制承担（timeout / max_retries 来自 Settings），
批量分批发送避免触发单请求条数上限。
"""

from openai import OpenAI

from app.core.config import Settings, get_settings

_BATCH_SIZE = 32


class EmbeddingClient:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        s = settings or get_settings()
        self._client = client or OpenAI(
            api_key=s.siliconflow_api_key,
            base_url=s.siliconflow_base_url,
            timeout=s.request_timeout_s,
            max_retries=s.max_retries,
        )
        self.model = s.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in sorted(resp.data, key=lambda d: d.index))
        return out
