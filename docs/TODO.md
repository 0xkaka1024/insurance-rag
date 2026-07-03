# TODO：执行计划与准备清单

> 配合 SPEC.md 使用。每天结束对照「当日验收」打勾，落后一天以上按「降级预案」砍需求。

## Phase 0：开工前准备（半天，全部就绪再写代码）

- [ ] **DeepSeek API key**：platform.deepseek.com 充值 ¥10 起（评测 50 题 × 8 配置约消耗 ¥5-10）
- [ ] **SiliconFlow API key**：siliconflow.cn（bge-m3 embedding + bge-reranker，注册送额度）
- [ ] **DashScope API key**（P1 才需要）：Qwen-VL 表格解析 + qwen-plus 对比
- [ ] **条款 PDF ×3**：AIA 香港官网产品页下载，建议组合：重疾（如「爱伴航」系列）+ 自愿医保 + 储蓄/寿险各 1 款；存入 `data/raw/`，文件名规范：`产品名_版本.pdf`
- [ ] **GitHub 仓库**：新建 public repo `insurance-rag`，首个 commit 就是这三份 docs（commit 历史即工作证明）
- [ ] **Hugging Face 账号**：注册 + 新建 Docker 类型 Space（占位即可）
- [ ] **Python 3.11 venv**：`requirements.txt` 初版（fastapi/uvicorn/chromadb/rank_bm25/jieba/pdfplumber/openai/ragas/python-dotenv）

## 工程约定：Git 同步与测试（贯穿每一天）

- **一个 feature = 一个 commit（代码 + 对应单测一起提交）**，message 用英文 Conventional Commits（`feat:` `fix:` `docs:` `test:` `chore:`），祈使句；每个 commit 后及时 `git push`，GitHub 与本地保持同步
- 单测用 pytest；LLM / embedding / rerank 调用**全部 mock**（monkeypatch 或 fake client），测试不花钱、不联网、秒级跑完
- 重点单测对象：StructuralChunker 的条款边界切分、RRF 融合排序正确性、意图路由规则、引用格式解析、拒答阈值逻辑、入库幂等（同 hash 不重复）
- CI：GitHub Actions（`.github/workflows/ci.yml`），每次 push 自动跑 `ruff check` + `pytest`；徽章挂进 README
- RAGAS 评测**不进 CI**（调用真实 LLM，花钱且慢），保留为手动命令 `python eval/run_eval.py`
- D1 骨架即包含 `tests/` 目录、pytest 配置与 CI workflow，从第一个 commit 起 CI 常绿

## D1：端到端最小闭环（先跑通，再谈好）

- [ ] 项目骨架按 ARCHITECTURE.md 目录建立；.env.example、config.py
- [ ] parser.py：pdfplumber 逐页提取 + 繁简归一（opencc）
- [ ] chunker.py：先只做 FixedChunker（512 token / 15% overlap）
- [ ] indexer.py：SiliconFlow embedding → Chroma `clauses_fixed`
- [ ] /ask 最简版：纯向量检索 top5 → DeepSeek 生成（暂不流式）
- **当日验收**：对 1 份真实条款提问"等待期多少天"，返回含正确答案的回复

## D2：切片对比 + 混合检索

- [ ] StructuralChunker：正则匹配条款层级（第X章/第X条/编号体系），过长二次切分；元数据含 产品/章/条/页码
- [ ] 入库脚本支持双 collection 并行构建 + 文件 hash 幂等
- [ ] BM25 索引（jieba 分词）持久化；RRF 融合；/ask 支持 retrieval=vector|hybrid
- **当日验收**：同一问题 fixed vs structural、vector vs hybrid 的检索结果肉眼可见差异

## D3：重排 + 引用 + 拒答 + 路由

- [ ] reranker.py：SiliconFlow bge-reranker API，top20 → top5
- [ ] generator.py：prompt 强制 [产品-条号-页码] 引用；rerank 分数低于阈值走拒答模板
- [ ] preprocess.py：术语归一表（先手写 20 组常见口语↔条款词）；规则路由（保费类 → 拒答引导）
- **当日验收**：问"30岁买每年多少钱"被拦截；问文档里没有的内容得到拒答；正常问题回答带可溯源引用

## D4：Playground 前端 + 流式

- [ ] /ask 改 SSE 流式；响应附 chunks/citations/timings
- [ ] static/index.html 三视图：聊天（默认最优配置）、对照（双 config 勾选 + 左右渲染）、评测表（占位）
- **当日验收**：浏览器里勾选两套配置、同一问题左右对照，引用可点开原文

## D5：评测集 + RAGAS

- [ ] 以 eval/seed_questions.jsonl 为模板，LLM 从 chunk 批量生成候选题 → **人工逐条核对 ground_truth**（这步不能省，约半天）→ 定稿 dataset.jsonl（50 条：事实/跨段/表格/对比/拒答 五类 × 三档难度）
- [ ] run_eval.py：8 配置 × 50 题 → RAGAS 四指标 + 拒答准确率 → results.json；前端评测表渲染
- **当日验收**：评测表跑出完整数字，最优配置 faithfulness ≥ 0.85（低于则排查后再进 D6）

## D6：部署 + 展示物料

- [ ] Dockerfile（单容器：uvicorn + 静态文件；索引文件打进镜像或启动时构建）
- [ ] 部署 HF Spaces，验证公网可访问
- [ ] README：一句话定位 + 架构图 + Playground 截图 + **评测数字表** + 设计决策(为什么 Chroma/为什么拒答/为什么 API rerank) + v2 Roadmap
- [ ] 简历新增本项目条目（技术栈+架构+评测数字），联系方式行加 GitHub 与 demo 链接
- **当日验收**：手机打开 demo 链接能完整走一遍问答；开始投递

## D7-D8：Buffer + 消化

- [ ] P1 择优：HyQE > LLM 对比 > VLM 表格解析 > 多轮改写（按此优先级，做不完就砍）
- [ ] **逐行读懂全部代码**，对照自测清单模拟面试：RRF 的 k 为什么 60？rerank 为何用 cross-encoder？拒答阈值怎么定的？chunk 512 token 依据？RAGAS faithfulness 怎么算的？Chroma HNSW 参数含义？
- [ ] 答不上的问题回头补原理，补完记录进 docs/QA.md

## 降级预案（进度落后时按序砍）

1. 砍 P1 全部（HyQE/VLM/LLM 对比/多轮改写）
2. 评测集 50 → 30 题（五类保留、每类减量）
3. 评测配置 8 → 4 套（rerank 维度固定为开）
4. 前端评测表 → 直接展示 results.json 截图
5. **不可砍**：引用溯源、拒答、双切片对比、RAGAS 数字、部署链接

## 风险备忘

- AIA 条款若为扫描件/加密 PDF → 换可提取文本的产品文档，或提前启用 VLM 路径
- HF Spaces 免费实例休眠冷启动 ~30s → README 注明，面试演示前先唤醒
- RAGAS 依赖 LLM 判分，用 DeepSeek 作 judge 需在 run_eval.py 显式配置（默认走 OpenAI 会报错）
