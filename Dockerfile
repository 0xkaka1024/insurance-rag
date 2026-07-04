# 单容器部署：uvicorn + 静态文件（HF Spaces Docker 类型，端口约定 7860）
# 构建前提：本地已跑 `python scripts/ingest.py data/raw/`，data/index/ 随镜像打包
# （索引只含白名单公开产品简介的文本，无密钥、无内部材料、无个人信息）
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STARTUP_REQUIRE_INDEX=true \
    RATE_LIMIT_PER_MINUTE=20 \
    DAILY_REQUEST_BUDGET=300

# 生产硬化（本地/CI 默认关闭，仅镜像开启）：
# - STARTUP_REQUIRE_INDEX：索引未就绪 fail-fast，不允许静默拒答的假运行
# - RATE_LIMIT / DAILY_BUDGET：防公网脚本烧 API 账单；转 public 前另在
#   Space Secrets 配 API_AUTH_TOKEN 可再加一道 Bearer 鉴权

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY static/ static/
COPY eval/results/ eval/results/
COPY data/index/ data/index/

# HF Spaces 要求非 root（uid 1000）；chroma sqlite 需要写权限
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
# HEALTHCHECK 打 /ready（深检索引+key），比 /health 浅活更能反映真实可服务状态
HEALTHCHECK --interval=30s --timeout=5s CMD python -c \
    "import urllib.request;urllib.request.urlopen('http://127.0.0.1:7860/ready')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
