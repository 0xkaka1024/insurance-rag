---
title: Insurance RAG
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# insurance-rag：保险条款智能问答（引用溯源 + 拒答 + 策略 Playground）

[![CI](https://github.com/0xkaka1024/insurance-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/0xkaka1024/insurance-rag/actions/workflows/ci.yml)

面向保险代理人的条款知识助手：**每个事实论断带 [产品-条号/章节-页码] 引用，无依据必拒答**；
内置可插拔策略 Playground（切片 × 检索 × 重排 实时对照）与 RAGAS 量化评测。

> 🚧 线上 Demo：待部署（HF Spaces，D6）· 评测数字：待评测集定稿后跑分（D5）

## 为什么做这个

客户问"甲状腺结节能不能标体承保"，代理人要翻几十页条款 PDF 找依据；凭记忆回答有误导销售
的合规风险。本项目提供秒级、可溯源、拒绝编造的条款问答——第一用户是作者本人（保险副业日常使用）。

## 核心设计

```
question ──► 路由（保费类 → 检索前拦截拒答，零成本）
         ──► 术语归一（口语 ↔ 条款词 26 组）
         ──► 检索  vector | hybrid（Chroma 余弦 + BM25/jieba + RRF k=60）
         ──► 重排  bge-reranker-v2-m3（top20 → top5；失败自动降级不中断）
         ──► 拒答阈值（rerank 分数 < 0.3 → 拒答，不烧生成 token）
         ──► 生成  DeepSeek（LLM 只输出 [n] 编号，服务端回填完整引用标签）
         ──► SSE 流式返回 {answer, chunks, citations, timings, config}
```

**可插拔维度**（查询期实时切换，Playground 左右对照）：

| 维度 | 选项 | 说明 |
|---|---|---|
| chunking | `fixed` / `structural` | 512 字符重叠切片 vs 条款编号/章节标题层级切片（入库期预建双 collection） |
| retrieval | `vector` / `hybrid` | 纯向量 vs 向量+BM25+RRF 融合 |
| rerank | on / off | cross-encoder 精排 + 拒答阈值 |

## 评测（RAGAS + 拒答准确率）

`python eval/run_eval.py` 缺省跑 8 套配置全组合；结果持久化 `eval/results/{日期}_{git短hash}.json`
入 git，前端评测表读取最新一份并支持历史对比。

| 配置 | faithfulness | answer_relevancy | context_precision | context_recall | 拒答准确率 |
|---|---|---|---|---|---|
| （待评测集定稿后填入真实数字） | – | – | – | – | – |

## 工程实践

- **数据治理红线代码级强制**：内部培训材料 / 费率表 / 白名单外产品，ingest 直接拒绝；含个人信息的计划书永不入公开索引
- **容错降级**：外部 API 统一超时 + 指数退避重试；rerank 挂掉自动降级为粗排截断并打标 `rerank_degraded`，服务不中断
- **可观测**：JSON 结构化日志，request_id 贯穿全链路，每问记录各阶段耗时与 token 用量
- **测试密封**：120+ 单测全离线（外部调用全 mock），conftest 强制测试环境与 CI 一致，杜绝"本地绿 CI 红"
- **真实语料的脏活**：双栏 PDF 分栏提取（否则左右栏逐行交错）、繁→简归一、jieba 保险领域词典（默认词典把"等待期"切碎）
- **原子提交**：一个 feature = 一个 commit（代码+测试），Conventional Commits，每 commit 推送

## 快速开始

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # 填入 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY

python scripts/ingest.py data/raw/     # 白名单内 PDF 解析→双切片→双索引
uvicorn app.main:app --reload          # http://127.0.0.1:8000
ruff check . && pytest                 # 质量门禁（CI 同款）

pip install -r eval/requirements.txt
python eval/run_eval.py --metrics faithfulness answer_relevancy   # 手动评测
```

## 技术选型速记

Chroma（pip 即装、文件持久化，demo 规模够用；生产迁移 Milvus/pgvector 见 Roadmap）·
bge-m3 / bge-reranker-v2-m3 走 SiliconFlow API（容器保持轻量）· DeepSeek（便宜、国内直连、
OpenAI 兼容）· RAGAS judge 显式配 DeepSeek · 详细理由与被否掉的选项见
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## v2 Roadmap

1. 计划书上传 + VLM 结构化提取 + 多计划书对照
2. 保费查询：费率表结构化 + function calling 查表（数值零幻觉）
3. Next.js 前端 + 前后端分离部署
4. Chroma → Milvus/pgvector；多用户与文档权限
5. LLM tracing（Langfuse/OpenTelemetry）与监控告警

## 声明

本项目仅作条款信息检索辅助，不构成投保建议或销售依据；保费等数值一律拒答并引导以官方
计划书为准。条款原文版权归保险公司所有，仓库不含任何条款 PDF 与密钥。
