# 单容器部署：uvicorn + 静态文件（HF Spaces Docker 类型，端口约定 7860）
# 构建前提：本地已跑 `python scripts/ingest.py data/raw/`，data/index/ 随镜像打包
# （索引只含白名单公开产品简介的文本，无密钥、无内部材料、无个人信息）
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STARTUP_REQUIRE_INDEX=true

# 生产：索引未就绪直接 fail-fast（容器起不来，平台显性报错），
# 不允许「索引缺失却静默把所有问题拒答」的假运行状态。

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
