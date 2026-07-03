from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.embedding import EmbeddingClient
from app.core.llm import LLMClient
from app.ingest.indexer import ChromaIndexer
from app.rag.pipeline import RagPipeline
from app.rag.retriever import VectorRetriever

router = APIRouter()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ChunkOut(BaseModel):
    chunk_id: str
    text: str
    product: str
    page_start: int
    page_end: int
    score: float


class AskResponse(BaseModel):
    answer: str
    chunks: list[ChunkOut]
    timings: dict[str, float]


@lru_cache
def get_pipeline() -> RagPipeline:
    settings = get_settings()
    indexer = ChromaIndexer(settings)
    embedder = EmbeddingClient(settings)
    return RagPipeline(
        retriever=VectorRetriever(indexer, embedder, settings),
        llm=LLMClient(settings),
    )


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipeline: Annotated[RagPipeline, Depends(get_pipeline)]) -> AskResponse:
    result = pipeline.ask(req.question)
    return AskResponse(
        answer=result.answer,
        chunks=[ChunkOut(**vars(c)) for c in result.chunks],
        timings=result.timings,
    )
