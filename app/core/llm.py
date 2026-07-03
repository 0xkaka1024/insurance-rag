"""LLM 工厂：DeepSeek / Qwen 均为 OpenAI 兼容接口，一个客户端类切换。"""

import logging

from openai import OpenAI

from app.core.config import Settings, get_settings

logger = logging.getLogger("llm")


class LLMClient:
    def __init__(self, settings: Settings | None = None, client: OpenAI | None = None):
        s = settings or get_settings()
        if client is None and not s.deepseek_api_key:
            # 构造不抛错（否则无密钥环境连应用都起不来），真实调用时会 401
            logger.warning("DEEPSEEK_API_KEY not configured; LLM calls will fail")
        self._client = client or OpenAI(
            api_key=s.deepseek_api_key or "not-configured",
            base_url=s.deepseek_base_url,
            timeout=s.request_timeout_s,
            max_retries=s.max_retries,
        )
        self.model = s.llm_model

    def complete(self, system: str, user: str) -> str:
        text, _ = self.complete_with_usage(system, user)
        return text

    def complete_with_usage(self, system: str, user: str) -> tuple[str, dict]:
        """返回 (文本, token 用量)；评测的成本统计依赖 usage。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            temperature=0.2,  # 条款问答要事实性，压低随机性
        )
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
            "completion_tokens": (
                getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0
            ),
        }
        return resp.choices[0].message.content or "", usage

    def stream(self, system: str, user: str):
        """逐段产出增量文本（SSE 流式用）。"""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=self._messages(system, user),
            temperature=0.2,
            stream=True,
        )
        for chunk in resp:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    @staticmethod
    def _messages(system: str, user: str) -> list[dict]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
