import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
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
    finally:
        request_id_var.reset(token)
