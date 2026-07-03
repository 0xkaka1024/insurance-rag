"""回归：无任何 API key 的环境（CI）中，客户端构造与应用启动不得抛错。"""

from app.core.config import Settings
from app.core.embedding import EmbeddingClient
from app.core.llm import LLMClient


def test_clients_construct_without_keys():
    s = Settings(_env_file=None)
    assert s.deepseek_api_key == ""
    assert s.siliconflow_api_key == ""
    LLMClient(settings=s)
    EmbeddingClient(settings=s)


def test_ask_validation_does_not_require_keys():
    from fastapi.testclient import TestClient

    from app.main import app

    resp = TestClient(app).post("/ask", json={"question": ""})
    assert resp.status_code == 422
