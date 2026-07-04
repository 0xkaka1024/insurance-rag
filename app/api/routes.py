import json
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
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


@lru_cache
def get_pipeline() -> RagPipeline:
    return build_pipeline(get_settings())


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
