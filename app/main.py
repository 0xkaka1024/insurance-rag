import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes
from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import request_id_var, setup_logging

setup_logging()
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动即自检索引（顺带 eager 预热 Indexer，首个请求不再冷启动）。
    settings = get_settings()
    result = routes.probe_readiness(settings, routes.get_indexer())
    if result["index_ok"]:
        logger.info("index ready at startup", extra={"extra_fields": result})
    else:
        logger.critical("index NOT ready at startup", extra={"extra_fields": result})
        if settings.startup_require_index:
            raise RuntimeError(f"索引未就绪，拒绝启动（STARTUP_REQUIRE_INDEX=true）：{result}")
    if not result["keys_ok"]:
        logger.warning("api keys not fully configured", extra={"extra_fields": result["keys"]})
    yield


app = FastAPI(title="insurance-rag", lifespan=lifespan)
app.include_router(router)

_STATIC = Path(__file__).resolve().parent.parent / "static"
if _STATIC.is_dir():
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")


@app.middleware("http")
async def request_context(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = req_id  # 异常 handler 在 contextvar 重置后仍可取到
    token = request_id_var.set(req_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={"extra_fields": {"elapsed_ms": round(elapsed_ms, 1)}},
        )
        response.headers["x-request-id"] = req_id
        return response
    except Exception:
        # 最需要日志的失败请求以前恰好没有结构化日志（只有 uvicorn 裸 traceback）
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "%s %s -> EXC",
            request.method,
            request.url.path,
            extra={"extra_fields": {"elapsed_ms": round(elapsed_ms, 1)}},
        )
        raise
    finally:
        request_id_var.reset(token)


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """非流式路径的兜底：JSON 错误体带 request_id，用户可凭 ID 报障、日志可对上。

    以前是 Starlette 默认的纯文本 500，前端 resp.json() 直接解析爆炸。
    """
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务内部错误，请稍后重试。",
            "request_id": getattr(request.state, "request_id", "-"),
        },
    )
