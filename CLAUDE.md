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
- RAGAS 评测（`python eval/run_eval.py`）只手动跑，不进 CI；评测结果持久化到 `eval/results/`（入 git）
- 每个 commit 完成后及时 `git push`，GitHub 与本地保持同步
- docs/TODO.md 勾选实时同步：每完成一项立即打勾，TODO 状态与 git log 始终一致

## 红线

- API key 只放 `.env`（已 gitignore），任何代码、日志、commit 不得出现密钥
- `data/` 不入 git；文件名含 `training deck` 的内部培训材料**不得**入库到公开部署的索引，v1 入库白名单（产品标识 + 内容 sha256 双因子，见 `app/ingest/governance.py` 的 INGEST_WHITELIST 与 FINGERPRINTS 登记表）：OnYourSideInsurancePlan2（危疾·爱伴航2）、newVHISmedical（医疗·VHIS）、GlobalFlexiSavingsInsurancePlan（储蓄）；新文件/新版本入库须先登记指纹
- `premiumtable` / `premium-table` 费率表文件是 v2 查表功能原料，v1 不入库
- 保费/费率类问题一律路由拦截拒答，不允许 RAG 生成数字
- 计划书/保单等含客户个人信息的文件一律不入公开部署的索引（v2 计划书功能为会话级上传，不落公共库）
- 回答中的事实论断必须带引用，格式双轨：语料有条款编号用 [产品-条号-页码]，简介类语料退化为 [产品-章节-页码]；检索低分必须拒答

## 技术栈速记

Python 3.11 · FastAPI + SSE · ChromaDB + rank_bm25(jieba) + RRF · rerank/embedding 走 SiliconFlow API（**国内站 bge-m3/bge-reranker；国际站 cloud.siliconflow.com 无 bge 系，用 Qwen3-Embedding/Qwen3-Reranker，key 互不通用，.env 切 base 与模型**；备选 DashScope）· LLM: DeepSeek（对比 qwen-plus）· 评测 RAGAS（judge 用 DeepSeek，需显式配置）· 部署 HF Spaces 单容器

## 语言注意

条款 PDF 为繁体中文（_tc 后缀），解析后统一 opencc 繁→简归一；embedding 用 bge-m3（多语言）。
