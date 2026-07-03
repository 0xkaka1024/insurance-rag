import json
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.embedding import EmbeddingClient
from app.core.llm import LLMClient
from app.ingest.indexer import Indexer
from app.rag.pipeline import RagConfig, RagPipeline
from app.rag.reranker import RerankClient
from app.rag.retriever import Retriever

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


class CitationOut(BaseModel):
    index: int
    label: str
    chunk_id: str


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
    settings = get_settings()
    indexer = Indexer(settings)
    embedder = EmbeddingClient(settings)
    return RagPipeline(
        retriever=Retriever(indexer, embedder, settings),
        llm=LLMClient(settings),
        reranker=RerankClient(settings),
        settings=settings,
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _sse(pipeline: RagPipeline, req: AskRequest):
    for event, data in pipeline.ask_stream(req.question, req.config):
        yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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
