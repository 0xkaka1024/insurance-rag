import json
import threading
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.ingest.indexer import Indexer
from app.ingest.report import reports_dir
from app.rag.pipeline import RagConfig, RagPipeline, build_pipeline

router = APIRouter()


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


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
    }


def _sse(pipeline: RagPipeline, req: AskRequest):
    for event, data in pipeline.ask_stream(req.question, req.config):
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/retrieve", response_model=RetrieveResponse)
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


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Annotated[RagPipeline, Depends(get_pipeline)]):
    if req.stream:
        return StreamingResponse(_sse(pipeline, req), media_type="text/event-stream")
    result = pipeline.ask(req.question, req.config)
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
