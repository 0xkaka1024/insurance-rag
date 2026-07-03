# 保险条款智能问答系统（Insurance RAG）架构设计

> v1.0 · 2026-07-03 · 作者：kaka
> 定位：面向保险代理人的条款知识助手，内置可插拔策略 Playground 与量化评测

---

## 1. 项目定位

**目标用户**：保险代理人/经纪人（第一用户是我自己，日常保险副业中真实使用）。

**核心问题**：客户提问（"甲状腺结节能不能标体承保""等待期内确诊怎么办"）时，代理人需要翻几十页条款 PDF 找依据；凭记忆回答有误导销售的合规风险。

**解决方案**：秒级条款检索问答，回答**必须附条款原文出处**，知识库中没有依据时**明确拒答**。

**项目的第二重身份**：一个可控变量的 RAG 实验平台（Playground）——每个环节的策略可勾选切换，配合评测集量化对比不同方案的效果差异。

---

## 2. 系统架构

```
┌─────────────────────── 前端 static/index.html ───────────────────────┐
│  聊天视图 │ Playground 对照视图（双 config 左右对比）│ 评测结果表        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ HTTP / SSE
┌──────────────────────────────▼───────────────────────────────────────┐
│                        FastAPI (app/main.py)                         │
│   POST /ask {question, config}   POST /upload   GET /eval_results    │
├───────────────────────────────────────────────────────────────────────┤
│ Query 预处理 (rag/preprocess.py)                                      │
│   ① 多轮改写(P1) ② 术语归一 ③ 意图路由: 条款问答→RAG                   │
│                          保费/数值→拒答并引导  闲聊/超范围→直接回复      │
├───────────────────────────────────────────────────────────────────────┤
│ 检索 (rag/retriever.py)          config 驱动                          │
│   向量检索(Chroma HNSW) ──┐                                           │
│   BM25(jieba 分词)     ──┴→ RRF 融合 → rerank(可选, top20→top5)       │
├───────────────────────────────────────────────────────────────────────┤
│ 生成 (rag/generator.py)                                               │
│   引用强制: 每个论断标注 [产品-条号-页码] │ 低分拒答阈值 │ SSE 流式      │
└───────────────────────────────────────────────────────────────────────┘

┌────────────────── 离线入库 pipeline (scripts/ingest.py) ──────────────┐
│ PDF → 解析(pdfplumber, 表格页 VLM fallback)                           │
│     → 切片(fixed / structural 两套并行)                               │
│     → 增强(元数据标注; HyQE 假设问题生成, P1)                          │
│     → 索引(Chroma 多 collection + BM25 索引持久化到磁盘)               │
└───────────────────────────────────────────────────────────────────────┘

┌────────────────── 评测 (eval/run_eval.py) ────────────────────────────┐
│ configs × 评测集(50题) → 逐题跑 pipeline → RAGAS 打分                  │
│ → results.json → 前端渲染分数对照表                                    │
└───────────────────────────────────────────────────────────────────────┘
```

## 3. Playground：config 驱动的可插拔设计

整个系统的核心机制是一个 config 对象贯穿 pipeline，每段实现可替换：

```python
config = {
    # ── 入库期维度（预建索引，查询时只能选用，不能实时改）──
    "chunking":  "fixed" | "structural",   # 决定读哪个 Chroma collection
    "enrich_hyqe": bool,                   # P1: 是否用了假设问题增强的索引
    # ── 查询期维度（实时切换）──
    "retrieval": "vector" | "hybrid",      # 纯向量 vs 向量+BM25+RRF
    "rerank":    bool,                     # 是否 cross-encoder 重排
    "llm":       "deepseek-chat" | "qwen-plus",  # 生成模型对比
}
```

**关键设计决策**：

- 入库期维度（分块、增强）在文档入库时**预建独立 collection**（`clauses_fixed`、`clauses_structural`、`clauses_structural_hyqe`），查询时按 config 选用。原因：embedding 重算成本高，不可能实时切换。
- 查询期维度实时生效，零额外成本。
- 组合爆炸控制：批量评测跑 2(分块)×2(检索)×2(重排)=8 套核心组合；LLM 维度**只在最优检索配置上**对比，不做全交叉。
- 每段实现遵循相同接口（如 `Chunker.split(doc) -> list[Chunk]`），新增策略 = 新增一个类 + 注册表登记，不改 pipeline 主干。

## 4. 各环节技术方案与理由

| 环节 | 方案 | 理由 / 被否掉的选项 |
|---|---|---|
| PDF 解析 | pdfplumber 文本提取为主；检测到表格或提取质量差的页 → Qwen-VL 转 Markdown（P1） | 条款 PDF 多为文字版，全量 VLM 贵且慢；纯文本表格必乱。注意 AIA 香港条款常为繁体中文/中英双语，解析后统一做繁简归一 |
| 切片 | ① fixed：递归固定长度 512 token、15% 重叠（baseline）② structural：正则匹配「第X章/第X条/条款编号」层级切分，过长条目二次切分 | structural 保住条款语义完整性（避免"核辐射免责被切碎"类问题）；两者对比正是 Playground 卖点 |
| 增强 | P0：元数据（产品名/章/条号/页码/繁简归一后文本）P1：HyQE——每 chunk 由 LLM 生成 2-3 个口语化假设问题一并 embedding | 元数据是引用溯源与过滤检索的基础；HyQE 解决"用户口语 vs 条款书面语"的语义鸿沟。关键词抽取不做（BM25 已覆盖），文档摘要放 P2 |
| Embedding | bge-m3（经 SiliconFlow API 调用；本地开发可选本地跑） | 多语言（应对繁中/英文混排）、中文效果好；API 化让部署容器保持轻量。DeepSeek 无 embedding 服务，故外配 |
| 索引/检索 | ChromaDB(HNSW) + BM25(jieba, rank_bm25) 双路，RRF 融合(k=60) | Chroma pip 即装、文件持久化、零运维，demo 规模完全够；Milvus 需 Docker 三容器，作为生产迁移路径写入 Roadmap。BM25 兜住精确术语匹配（产品名、条号、专有名词） |
| Rerank | bge-reranker-v2-m3，经 SiliconFlow API（本地 cross-encoder 作开发备选） | cross-encoder 精度显著高于 bi-encoder 粗排；API 化避免免费部署平台内存装不下模型 |
| 生成 | DeepSeek-chat 为主，Qwen-plus 作对比维度；prompt 强制引用格式；检索 top1 分数低于阈值 → 拒答模板 | DeepSeek 便宜且国内直连；两家均 OpenAI 兼容接口，`core/llm.py` 一个工厂函数切换 |
| Query 预处理 | 术语归一（同义词表）+ 规则意图路由（P0）；多轮改写、HyDE（P1） | 路由是防幻觉第一道闸门：保费类数值问题现阶段一律拒答并引导（v2 接计划书查表工具） |
| 评测 | RAGAS（faithfulness / answer_relevancy / context_precision / context_recall）+ 自定义拒答准确率 | RAGAS 是行业通用语言；拒答题 RAGAS 不覆盖，需自定义指标 |
| 后端 | FastAPI + SSE 流式 | 异步、类型安全（pydantic）、行业标配 |
| 前端 | 单页静态 HTML + 原生 JS | 两周工期内不引入前端框架；v2 可换 Next.js（部署 Vercel） |
| 部署 | 单容器 Dockerfile → Hugging Face Spaces（备选 Render） | Vercel 是 serverless，无常驻进程/持久磁盘，装不下 BM25 索引与 Chroma 文件，不适合本后端；HF Spaces 免费额度大、AI 社区认知度高 |

## 5. 目录结构

```
insurance-rag/
├── app/
│   ├── main.py               # FastAPI 入口, 静态文件挂载
│   ├── api/routes.py         # /ask /upload /configs /eval_results /health
│   ├── core/
│   │   ├── config.py         # pydantic Settings, .env 读取
│   │   └── llm.py            # LLM 工厂 (OpenAI 兼容: DeepSeek/Qwen)
│   ├── ingest/
│   │   ├── parser.py         # pdfplumber + VLM fallback + 繁简归一
│   │   ├── chunker.py        # FixedChunker / StructuralChunker (同接口)
│   │   ├── enricher.py       # 元数据标注; HyQE(P1)
│   │   └── indexer.py        # Chroma collections + BM25 持久化
│   └── rag/
│       ├── preprocess.py     # 术语归一 / 意图路由 / 改写(P1)
│       ├── retriever.py      # vector / hybrid + RRF
│       ├── reranker.py       # SiliconFlow rerank API
│       ├── generator.py      # 引用生成 + 拒答 + SSE
│       └── pipeline.py       # config 驱动组装, 注册表
├── scripts/ingest.py         # 批量入库 CLI: python scripts/ingest.py data/raw/
├── eval/
│   ├── seed_questions.jsonl  # 10 条 mock 种子(待人工核对)
│   ├── dataset.jsonl         # 正式评测集 50 条
│   └── run_eval.py           # configs × questions → RAGAS → results.json
├── tests/                    # pytest 单测(外部调用全 mock): chunker/RRF/路由/引用/幂等
├── .github/workflows/ci.yml  # push 触发 ruff + pytest
├── static/index.html         # 聊天 / Playground 对照 / 评测表 三视图
├── data/
│   ├── raw/                  # 条款 PDF(不入 git)
│   └── index/                # Chroma + BM25 持久化文件(不入 git)
├── docs/                     # 本文档 / SPEC / TODO
├── Dockerfile
├── requirements.txt
└── .env.example              # DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / DASHSCOPE_API_KEY
```

## 6. 关键数据流

**入库**：`PDF → parser(页级解析+表格检测) → 每种 chunking 策略各切一遍 → enricher 打元数据 → indexer 写入对应 collection + 更新 BM25 索引`

**查询**：`question + config → preprocess(归一/路由) → [路由=条款] retriever(按 config 选 collection 与检索模式) → rerank(可选) → generator(引用/拒答) → SSE 流式返回 {answer, chunks, citations, config, timings}`

**评测**：`for config in configs: for q in dataset: pipeline(q, config) → RAGAS 批量打分 + 拒答准确率 → results.json`

## 7. 非功能性要求

- 单问端到端 P95 延迟 < 5s（rerank 开启时）
- 回答中每个事实性论断有引用；无依据时拒答率 100%（评测集拒答题）
- API key 全部走环境变量，仓库不含任何密钥与条款原文（PDF 不入 git）
- 入库幂等：同一 PDF 重复入库不产生重复 chunk（按文件 hash 判断）

## 8. v2 Roadmap（写进 README 展示规划能力）

1. 计划书（Proposal/BI）上传 + VLM 结构化提取 + 多计划书对照表与差异摘要
2. 保费查询：费率表结构化入库 + function calling 查表路由（数值零幻觉方案）
3. Next.js 前端（Vercel）+ 后端分离部署
4. Chroma → Milvus/pgvector 迁移；多用户与文档权限
