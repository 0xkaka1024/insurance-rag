"""LLMClient 请求参数与 usage 契约（生产必经路径，此前零覆盖）。"""

from app.core.config import Settings
from app.core.llm import LLMClient


class _Msg:
    content = "答案[1]。"


class _Choice:
    message = _Msg()


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5


class _Resp:
    choices = [_Choice()]
    usage = _Usage()


class FakeCompletions:
    def __init__(self, rec: dict):
        self._rec = rec

    def create(self, **kwargs):
        self._rec.update(kwargs)
        if kwargs.get("stream"):
            return iter([])  # 流式路径只验参数，不产增量
        return _Resp()


class FakeOpenAI:
    def __init__(self, rec: dict):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(rec)


def _client(rec: dict) -> LLMClient:
    return LLMClient(Settings(_env_file=None, llm_max_tokens=777), client=FakeOpenAI(rec))


def test_complete_passes_max_tokens_and_returns_usage():
    rec: dict = {}
    text, usage = _client(rec).complete_with_usage("sys", "user")
    assert rec["max_tokens"] == 777  # 单次回答成本上限生效
    assert rec["temperature"] == 0.2
    assert text == "答案[1]。"
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_stream_passes_max_tokens():
    rec: dict = {}
    assert list(_client(rec).stream("sys", "user")) == []
    assert rec["max_tokens"] == 777
    assert rec["stream"] is True
