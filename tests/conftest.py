"""让本地测试与 CI 完全一致：屏蔽 .env 里的真实密钥，杜绝「本地绿 CI 红」。

环境变量优先级高于 .env 文件（pydantic-settings 规则），置空即等效于无密钥环境。
"""

import pytest

from app.api.routes import get_pipeline
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for var in ("DEEPSEEK_API_KEY", "SILICONFLOW_API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.setenv(var, "")
    get_settings.cache_clear()
    get_pipeline.cache_clear()
    yield
    get_settings.cache_clear()
    get_pipeline.cache_clear()
