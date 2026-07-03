"""集中配置：全部来自环境变量 / .env，代码中不出现任何密钥字面量。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM（DeepSeek 为主，DashScope qwen-plus 作 P1 对比）
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"
    dashscope_api_key: str = ""

    # Embedding / Rerank（SiliconFlow）
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"

    # 数据与索引路径
    raw_dir: Path = Path("data/raw")
    index_dir: Path = Path("data/index")
    eval_results_dir: Path = Path("eval/results")

    # 检索参数
    top_k: int = 5
    recall_k: int = 20
    refuse_threshold: float = 0.3

    # 外部调用容错
    request_timeout_s: float = 30.0
    max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
