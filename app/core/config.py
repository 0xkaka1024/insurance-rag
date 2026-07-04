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

    # RAGAS judge：与被评生成模型解耦（自评偏差 + `--llm` 换生成模型不得连带改 judge）。
    # 留空的 base/key 回退 DeepSeek 配置；换 judge 供应商时三项一起配。
    judge_model: str = "deepseek-chat"
    judge_base_url: str = ""
    judge_api_key: str = ""

    # 生产入口 /ask 的服务端锁定配置（安全相关开关不由调用方决定；
    # 自由切换是实验能力，走 /playground/ask 与 /retrieve）
    prod_chunking: str = "structural"
    prod_retrieval: str = "hybrid"
    prod_rerank: bool = True

    # 解析：表格页/低质量页 VLM 转 Markdown（SPEC R9，P1）。
    # 开关只声明意图；开着却未注入 VLM 客户端时 ingest 直接报错，不静默降级
    # （表格页拍平入库是事实性误引源头，宁可失败也不能假装转写过）。
    parse_vlm_fallback: bool = False

    # 检索参数
    top_k: int = 5
    recall_k: int = 20
    refuse_threshold: float = 0.3  # rerank(cross-encoder) 分数下限
    # rerank 关闭/降级时的兜底：检索侧最强向量余弦低于此值即拒答。
    # 保守初值，未经评测集校准（SPEC 开放问题）；宁可少量误拒不裸奔。
    vector_floor: float = 0.35

    # 外部调用容错
    request_timeout_s: float = 30.0
    max_retries: int = 3

    # rerank 专属容错：pipeline 有零成本降级兜底（粗排截断），等不起全局
    # 30s×N 重试周期（实测最坏 ~127s 才降级）。短超时 + 至多 1 次重试 +
    # 连续 3 次失败熔断（冷却期内直接降级，不再逐请求耗完重试）。
    rerank_timeout_s: float = 5.0
    rerank_max_retries: int = 1
    rerank_breaker_cooldown_s: float = 60.0

    # 启动自检：为 true 时索引未就绪（collection 空 / BM25 缺）直接 fail-fast，
    # 容器起不来（部署平台显性报错），而非静默把所有问题拒答。
    # 本地/CI 默认 false（测试不需要真实索引）；生产镜像在 Dockerfile 里置 true。
    startup_require_index: bool = False

    # ── 公开部署防滥用（G3）。默认全关：本地/CI 零影响；生产在 Dockerfile 开启。
    # 转 public 前这是唯一挡住「任意脚本烧光 API 账单」的防线。
    rate_limit_per_minute: int = 0  # 每 IP 每分钟对昂贵端点的请求上限；0=关
    daily_request_budget: int = 0  # 全站每日昂贵请求总额度（成本熔断）；0=关
    llm_max_tokens: int = 1024  # 单次回答 token 上限：成本上限的最后一环
    # 可选 Bearer 鉴权：非空时昂贵端点要求 Authorization: Bearer <token>。
    # 值走环境变量/Space Secrets，绝不入代码（红线）
    api_auth_token: str = ""

    # 治理：Indexer.index() 入口的白名单二次断言（纵深防御）。
    # 仅测试非白名单夹具产品时置 False，生产不得关闭。
    whitelist_enforce_at_index: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
