from pathlib import Path

from app.core.config import Settings


def _isolated(**env: str) -> Settings:
    """构造不读 .env 的 Settings，避免测试依赖本机密钥文件。"""
    return Settings(_env_file=None, **env)


def test_defaults_do_not_contain_secrets():
    s = _isolated()
    assert s.deepseek_api_key == ""
    assert s.siliconflow_api_key == ""


def test_env_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("TOP_K", "7")
    s = Settings(_env_file=None)
    assert s.deepseek_api_key == "test-key"
    assert s.top_k == 7


def test_retrieval_defaults():
    s = _isolated()
    assert s.top_k == 5
    assert s.recall_k == 20
    assert 0 < s.refuse_threshold < 1
    assert s.index_dir == Path("data/index")
