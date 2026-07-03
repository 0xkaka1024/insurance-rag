"""LLM 工厂：DeepSeek / Qwen 均为 OpenAI 兼容接口，一个客户端类切换。"""

from openai import OpenAI

from app.core.config import Settings, get_settings


class LLMClient:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        s = settings or get_settings()
        self._client = client or OpenAI(
            api_key=s.deepseek_api_key,
            base_url=s.deepseek_base_url,
            timeout=s.request_timeout_s,
            max_retries=s.max_retries,
        )
        self.model = s.llm_model

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,  # 条款问答要事实性，压低随机性
        )
        return resp.choices[0].message.content or ""
