# insurance-rag 项目约定

保险条款智能问答系统：面向保险代理人，带引用溯源与拒答，内置可插拔策略 Playground 与 RAGAS 评测。

## 必读文档（动手前先读）

- `docs/ARCHITECTURE.md` — 系统架构、模块划分、config schema、技术选型理由
- `docs/SPEC.md` — 需求 P0/P1/P2 与验收标准
- `docs/TODO.md` — 逐日执行计划（当前进度看勾选状态）、工程约定、降级预案

## 工程约定（严格执行）

- 一个 feature = 一个 commit，代码与对应 pytest 单测同一个 commit 提交；commit message 用英文 Conventional Commits（feat/fix/docs/test/refactor/chore + 祈使句），一个 commit 只含一个主题的改动
- 每次 commit 前必须本地通过：`ruff check . && pytest`
- 单测中所有 LLM / embedding / rerank 外部调用一律 mock，测试不联网不花钱
- RAGAS 评测（`python eval/run_eval.py`）只手动跑，不进 CI
- 每个 commit 完成后及时 `git push`，GitHub 与本地保持同步

## 红线

- API key 只放 `.env`（已 gitignore），任何代码、日志、commit 不得出现密钥
- `data/` 不入 git；文件名含 `training deck` 的内部培训材料**不得**入库到公开部署的索引，v1 入库白名单（按产品标识精确匹配，见 `app/ingest/service.py:INGEST_WHITELIST`）：危疾类条款 ×1（待选定）、newVHISmedical、GlobalFlexiSavingsInsurancePlan
- `premiumtable` / `premium-table` 费率表文件是 v2 查表功能原料，v1 不入库
- 保费/费率类问题一律路由拦截拒答，不允许 RAG 生成数字
- 回答中的事实论断必须带 [产品-条号-页码] 引用；检索低分必须拒答

## 技术栈速记

Python 3.11 · FastAPI + SSE · ChromaDB + rank_bm25(jieba) + RRF · rerank/embedding 走 SiliconFlow API（备选 DashScope，.env 切换）· LLM: DeepSeek（对比 qwen-plus）· 评测 RAGAS（judge 用 DeepSeek，需显式配置）· 部署 HF Spaces 单容器

## 语言注意

条款 PDF 为繁体中文（_tc 后缀），解析后统一 opencc 繁→简归一；embedding 用 bge-m3（多语言）。
