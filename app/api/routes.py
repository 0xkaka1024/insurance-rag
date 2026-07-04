import json
import logging
import threading
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.guard import abuse_guard
from app.core.config import get_settings
from app.core.logging import request_id_var
from app.ingest.indexer import Indexer
from app.ingest.report import reports_dir
from app.rag.pipeline import RagConfig, RagPipeline, build_pipeline

logger = logging.getLogger("api")

router = APIRouter()

# SSE 反缓冲：HF/nginx 类代理可能整段缓冲，流式退化为一次性吐出
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    config: RagConfig = RagConfig()
    stream: bool = False


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    product: str
    page_start: int
    page_end: int
    score: float
    section: str = ""
    # 检索过程溯源（Playground 透明化），与 RetrievedChunk 对应字段同义
    vector_rank: int | None = None
    vector_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    retrieval_rank: int | None = None
    rerank_score: float | None = None


class CitationOut(BaseModel):
    index: int
    label: str
    chunk_id: str


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    config: RagConfig = RagConfig()


class RetrieveResponse(BaseModel):
    chunks: list[ChunkOut]
    timings: dict[str, float]
    config: RagConfig
    refused: bool
    refuse_reason: str
    rerank_degraded: bool
    answer: str = ""  # 拒答时携带话术；只检索模式不含生成内容


class AskResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    timings: dict[str, float]
    config: RagConfig
    refused: bool
    refuse_reason: str
    rerank_degraded: bool
    citations: list[CitationOut]


_build_lock = threading.Lock()


@lru_cache
def get_pipeline() -> RagPipeline:
    # 冷启动时并发首请求（Playground 双配置并发）会竞态构建 Chroma 客户端，
    # chromadb 共享系统注册表非线程安全（实测 KeyError/AttributeError），串行化规避。
    with _build_lock:
        return build_pipeline(get_settings())


@lru_cache
def get_indexer() -> Indexer:
    with _build_lock:  # 与 get_pipeline 同理：冷启动并发防 Chroma 竞态
        return Indexer(get_settings())


def probe_readiness(settings, indexer) -> dict:
    """深度就绪探针：两套 collection 非空 + BM25 pkl 存在 + 关键 key 已配。

    索引缺失时检索静默返回空、全部问题走「礼貌拒答」，监控看一切正常——
    这个探针把它变成显性信号（docs/REVIEW-2026-07.md P0-3）。index_ok 供启动
    fail-fast 用；ready 额外含 key（缺 key 是响亮的 500，不是静默失败）。
    """
    details: dict = {"strategies": {}}
    index_ok = True
    try:
        for strategy in ("fixed", "structural"):
            count = indexer.collection(strategy).count()
            bm25 = (settings.index_dir / f"bm25_{strategy}.pkl").exists()
            details["strategies"][strategy] = {"chunks": count, "bm25": bm25}
            index_ok = index_ok and count > 0 and bm25
    except Exception as exc:  # noqa: BLE001 - 探针不能自爆，异常即视为未就绪
        details["error"] = repr(exc)
        index_ok = False
    keys = {
        "deepseek": bool(settings.deepseek_api_key),
        "siliconflow": bool(settings.siliconflow_api_key),
    }
    details["keys"] = keys
    keys_ok = all(keys.values())
    return {"ready": index_ok and keys_ok, "index_ok": index_ok, "keys_ok": keys_ok, **details}


@router.get("/health")
def health() -> dict:
    """浅活探针：进程起来就返回 ok（不查索引/依赖，供负载均衡判存活）。"""
    return {"status": "ok"}


@router.get("/ready")
def ready(indexer: Annotated[Indexer, Depends(get_indexer)]) -> dict:
    """深就绪探针：索引与 key 齐备才 200，否则 503（供部署门禁与监控）。"""
    result = probe_readiness(get_settings(), indexer)
    if not result["ready"]:
        raise HTTPException(status_code=503, detail=result)
    return result


@router.get("/corpus")
def corpus() -> dict:
    """语料总览：每个产品的解析质量与切片统计（不含逐块明细）。"""
    d = reports_dir(get_settings())
    docs = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        docs.append(
            {
                "product": r.get("product", p.stem),
                "file": r.get("file", ""),
                "total_pages": r.get("total_pages", 0),
                "parsed_pages": r.get("parsed_pages", 0),
                "empty_pages": r.get("empty_pages", []),
                "generated_at": r.get("generated_at", ""),
                "strategies": {
                    name: {k: v for k, v in entry.items() if k != "chunks"}
                    for name, entry in r.get("strategies", {}).items()
                },
            }
        )
    return {"documents": docs}


@router.get("/corpus/{product}/chunks")
def corpus_chunks(
    product: str,
    indexer: Annotated[Indexer, Depends(get_indexer)],
    strategy: str = "structural",
) -> dict:
    """切片浏览器数据：报告里的顺序与 lint 标记 + Chroma 里的块全文。"""
    if strategy not in ("fixed", "structural"):
        raise HTTPException(status_code=422, detail=f"unknown strategy: {strategy}")
    d = reports_dir(get_settings())
    valid = {p.stem: p for p in d.glob("*.json")} if d.is_dir() else {}
    if product not in valid:  # 白名单校验，天然阻断路径穿越
        raise HTTPException(status_code=404, detail="corpus report not found")
    report = json.loads(valid[product].read_text(encoding="utf-8"))
    entry = report.get("strategies", {}).get(strategy)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"no {strategy} chunks for {product}")

    recs = entry.get("chunks", [])
    texts: dict[str, str] = {}
    ids = [r["chunk_id"] for r in recs]
    if ids:
        got = indexer.collection(strategy).get(ids=ids, include=["documents"])
        texts = dict(zip(got["ids"], got["documents"], strict=False))
    chunks = [{**r, "text": texts.get(r["chunk_id"], "")} for r in recs]
    return {
        "product": product,
        "strategy": strategy,
        "n_chunks": entry.get("n_chunks", len(chunks)),
        "clause_coverage": entry.get("clause_coverage"),
        "flag_counts": entry.get("flag_counts", {}),
        "chunks": chunks,
    }


@router.get("/eval_results")
def eval_results() -> dict:
    """评测历史列表，文件名倒序（最新在前）。"""
    d = get_settings().eval_results_dir
    files = sorted((p.name for p in d.glob("*.json")), reverse=True) if d.is_dir() else []
    return {"files": list(files)}


@router.get("/eval_results/{name}")
def eval_result(name: str) -> dict:
    d = get_settings().eval_results_dir
    valid = {p.name for p in d.glob("*.json")} if d.is_dir() else set()
    if name not in valid:  # 白名单校验，天然阻断路径穿越
        raise HTTPException(status_code=404, detail="result not found")
    return json.loads((d / name).read_text(encoding="utf-8"))


@router.get("/configs")
def configs() -> dict:
    """Playground 可选维度，由 RagConfig 模型推导，前端下拉自动同步。"""
    from typing import get_args

    fields = RagConfig.model_fields
    return {
        "chunking": list(get_args(fields["chunking"].annotation)),
        "retrieval": list(get_args(fields["retrieval"].annotation)),
        "rerank": [False, True],
        "production": _production_config().model_dump(),  # /ask 锁定的生产配置
    }


def _production_config() -> RagConfig:
    s = get_settings()
    return RagConfig(chunking=s.prod_chunking, retrieval=s.prod_retrieval, rerank=s.prod_rerank)


def _sse(pipeline: RagPipeline, question: str, config: RagConfig, request_id: str):
    """SSE 帧生成器。事件流必须有终结事件：中途异常发 error 事件而非裸断连——
    否则前端半截答案凭空消失、监控全盲（HTTP 已是 200，无任何错误信号）。"""
    try:
        for event, data in pipeline.ask_stream(question, config):
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    except Exception:
        logger.exception("ask_stream failed mid-stream")
        payload = {
            "message": "生成中断：上游服务暂时不可用，请稍后重试。",
            "request_id": request_id,
        }
        yield f"event: error\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _serve_ask(pipeline: RagPipeline, req: AskRequest, config: RagConfig):
    if req.stream:
        return StreamingResponse(
            _sse(pipeline, req.question, config, request_id_var.get()),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )
    result = pipeline.ask(req.question, config)
    return AskResponse(
        answer=result.answer,
        chunks=[ChunkOut(**vars(c)) for c in result.chunks],
        timings=result.timings,
        config=result.config,
        refused=result.refused,
        refuse_reason=result.refuse_reason,
        rerank_degraded=result.rerank_degraded,
        citations=[CitationOut(**vars(c)) for c in result.citations],
    )


@router.post("/retrieve", response_model=RetrieveResponse, dependencies=[Depends(abuse_guard)])
def retrieve(req: RetrieveRequest, pipeline: Annotated[RagPipeline, Depends(get_pipeline)]):
    """只检索不生成：Playground 调优检索时省去 LLM 延迟与费用。"""
    result = pipeline.retrieve_only(req.question, req.config)
    return RetrieveResponse(
        chunks=[ChunkOut(**vars(c)) for c in result.chunks],
        timings=result.timings,
        config=result.config,
        refused=result.refused,
        refuse_reason=result.refuse_reason,
        rerank_degraded=result.rerank_degraded,
        answer=result.answer,
    )


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(abuse_guard)])
def ask(req: AskRequest, pipeline: Annotated[RagPipeline, Depends(get_pipeline)]):
    """生产问答入口：config 服务端锁定（红线：安全相关开关不由调用方决定）。

    请求体中的 config 字段被忽略（保留以兼容旧客户端）；实验切换走 /playground/ask。
    """
    return _serve_ask(pipeline, req, _production_config())


@router.post(
    "/playground/ask", response_model=AskResponse, dependencies=[Depends(abuse_guard)]
)
def playground_ask(req: AskRequest, pipeline: Annotated[RagPipeline, Depends(get_pipeline)]):
    """Playground 实验入口：honor 调用方 config，用于策略对照与调参。"""
    return _serve_ask(pipeline, req, req.config)
