# TODO：执行计划与准备清单

> 配合 SPEC.md 使用。**每完成一项立即打勾，勾选状态与 git log 始终同步**；每天结束对照「当日验收」核验，落后一天以上按「降级预案」砍需求。

## Phase 0：开工前准备（半天，全部就绪再写代码）

- [x] **DeepSeek API key**：platform.deepseek.com 充值 ¥10 起（评测 50 题 × 8 配置约消耗 ¥5-10）
- [x] **SiliconFlow API key**：实际使用国际站 cloud.siliconflow.com（Qwen3 系，见 .env.example）
- [x] **DashScope API key**（P1 才需要）：Qwen-VL 表格解析 + qwen-plus 对比
- [x] **条款 PDF ×3**：AIA 香港官网产品页下载，建议组合：重疾（如「爱伴航」系列）+ 自愿医保 + 储蓄/寿险各 1 款；存入 `data/raw/`，文件名规范：`产品名_版本.pdf`
- [x] **GitHub 仓库**：新建 public repo `insurance-rag`，首个 commit 就是这三份 docs（commit 历史即工作证明）
- [x] **Hugging Face 账号**：0xkaka，Space `0xkaka/insurance-rag`（Docker 类型，2026-07-04 建）
- [x] **Python 3.11 venv**：`requirements.txt` 初版（fastapi/uvicorn/chromadb/rank_bm25/jieba/pdfplumber/openai/ragas/python-dotenv）

## 工程约定：Git 同步与测试（贯穿每一天）

- **一个 feature = 一个 commit（代码 + 对应单测一起提交）**，message 用英文 Conventional Commits（`feat:` `fix:` `docs:` `test:` `chore:`），祈使句；每个 commit 后及时 `git push`，GitHub 与本地保持同步
- 单测用 pytest；LLM / embedding / rerank 调用**全部 mock**（monkeypatch 或 fake client），测试不花钱、不联网、秒级跑完
- 重点单测对象：StructuralChunker 的条款边界切分、RRF 融合排序正确性、意图路由规则、引用格式解析、拒答阈值逻辑、入库幂等（同 hash 不重复）
- CI：GitHub Actions（`.github/workflows/ci.yml`），每次 push 自动跑 `ruff check` + `pytest`；徽章挂进 README
- RAGAS 评测**不进 CI**（调用真实 LLM，花钱且慢），保留为手动命令 `python eval/run_eval.py`
- D1 骨架即包含 `tests/` 目录、pytest 配置与 CI workflow，从第一个 commit 起 CI 常绿

## D1：端到端最小闭环（先跑通，再谈好）

- [x] 项目骨架按 ARCHITECTURE.md 目录建立；.env.example、config.py
- [x] 可观测性基线：结构化 JSON 日志 + request_id 中间件 + `/health` 端点
- [x] parser.py：pdfplumber 逐页提取 + 繁简归一（opencc）
- [x] chunker.py：先只做 FixedChunker（512 token / 15% overlap）
- [x] indexer.py：SiliconFlow embedding → Chroma `clauses_fixed`
- [x] /ask 最简版：纯向量检索 top5 → DeepSeek 生成（暂不流式）
- **当日验收**：对 1 份真实条款提问"等待期多少天"，返回含正确答案的回复

## D2：切片对比 + 混合检索

- [x] parser 双栏检测与分栏提取：简介类 PDF 两栏交错修复，真实 PDF 抽页人工校验
- [x] StructuralChunker：层级感知（条款编号优先，简介类按章节标题），过长二次切分；元数据含 产品/层级/页码
- [x] 入库脚本支持双 collection 并行构建 + 文件 hash 幂等 + 白名单校验（training deck / 费率表文件代码级拒绝）
- [x] BM25 索引（jieba 分词）持久化；RRF 融合；/ask 支持 retrieval=vector|hybrid
- **当日验收**：同一问题 fixed vs structural、vector vs hybrid 的检索结果肉眼可见差异

## D3：重排 + 引用 + 拒答 + 路由

- [x] reranker.py：SiliconFlow bge-reranker API，top20 → top5
- [x] 外部 API 容错：统一超时 + 重试（指数退避）；rerank 失败自动跳过并记录告警，不中断问答
- [x] generator.py：prompt 强制 [产品-条号-页码] 引用；rerank 分数低于阈值走拒答模板
- [x] preprocess.py：术语归一表（先手写 20 组常见口语↔条款词）；规则路由（保费类 → 拒答引导）
- **当日验收**：问"30岁买每年多少钱"被拦截；问文档里没有的内容得到拒答；正常问题回答带可溯源引用

## D4：Playground 前端 + 流式

- [x] /ask 改 SSE 流式；响应附 chunks/citations/timings
- [x] static/index.html 三视图：聊天（默认最优配置）、对照（双 config 勾选 + 左右渲染）、评测表（占位）
- **当日验收**：浏览器里勾选两套配置、同一问题左右对照，引用可点开原文

## D5：评测集 + RAGAS

- [ ] 以 eval/seed_questions.jsonl 为模板，LLM 从 chunk 批量生成候选题 → **人工逐条核对 ground_truth**（这步不能省，约半天）→ 定稿 dataset.jsonl（50 条：事实/跨段/表格/对比/拒答 五类 × 三档难度）
- [ ] run_eval.py：缺省跑 8 配置 × 50 题 → RAGAS 四指标 + 拒答准确率 → 结果持久化；前端评测表渲染
- [x] 结果持久化（设计已定 2026-07-03）：每次运行输出 `eval/results/{YYYYMMDD}_{git短hash}.json`，内容含 config、RAGAS 四指标、拒答准确率、总成本与总耗时；`eval/results/` 目录**入 git**（.gitignore 已移除 eval/results.json 行）；前端评测表读取最新一份，并提供历史结果下拉对比
- [x] CLI 参数化（设计已定 2026-07-03）：`--chunking / --retrieval / --rerank / --llm` 指定单套配置（缺省 = 8 套全组合）；`--dataset` 指定题库文件（缺省 = eval/dataset.jsonl）；`--metrics` 支持只跑无需 ground_truth 的指标子集（faithfulness, answer_relevancy）
- **当日验收**：评测表跑出完整数字，最优配置 faithfulness ≥ 0.85（低于则排查后再进 D6）

## D6：部署 + 展示物料

- [ ] Dockerfile（单容器：uvicorn + 静态文件；索引文件打进镜像或启动时构建）
- [x] 部署 HF Spaces（2026-07-04 上线，**private** 先行；转 public 前须过 G3 门槛：限流/成本熔断/SSE error）；发版用 `bash scripts/deploy_hf.sh`（worktree + LFS 打包索引，勿手动 checkout 部署分支）
- [ ] README：一句话定位 + 架构图 + Playground 截图 + **评测数字表** + 设计决策(为什么 Chroma/为什么拒答/为什么 API rerank) + v2 Roadmap
- [ ] 简历新增本项目条目（技术栈+架构+评测数字），联系方式行加 GitHub 与 demo 链接
- **当日验收**：手机打开 demo 链接能完整走一遍问答；开始投递

## D7-D8：Buffer + 消化

- [ ] P1 择优：HyQE > LLM 对比 > VLM 表格解析 > 多轮改写（按此优先级，做不完就砍）
  - [x] VLM 表格解析框架先行（2026-07-03）：表格页/cid 低质量页检测打标、`parse_pdf` 可注入 vlm 接口缝、`parse_vlm_fallback` 开关（开而未注入客户端即报错）、报告记录 `vlm_fallback` 归因与 `table_pages_flat` 红旗清单；剩余：接 DashScope Qwen-VL 真实客户端 + 转写结果本地缓存 + 表格题子集 before/after 评测
- [ ] **逐行读懂全部代码**，对照自测清单模拟面试：RRF 的 k 为什么 60？rerank 为何用 cross-encoder？拒答阈值怎么定的？chunk 512 token 依据？RAGAS faithfulness 怎么算的？Chroma HNSW 参数含义？
- [ ] 答不上的问题回头补原理，补完记录进 docs/QA.md

## 生产化 Backlog（2026-07-03 全面 review 产出，背景与证据见 docs/REVIEW-2026-07.md）

### G1 红线加固（P0：可信机制必须不可绕过）

- [x] 空引用即拒发：非拒答且 citations==[] 时服务端替换为拒答话术；无效引用编号计数进日志指标（pipeline.py / citations.py）
- [x] 默认配置拒答兜底：vector_floor 向量余弦保守下限（rerank 关/降级时生效，初值 0.35 待评测校准）；生产入口锁定安全 config，自由切换移到 Playground 专用端点
- [x] 白名单加内容指纹：产品名 + 文件 sha256 双因子准入（governance.FINGERPRINTS 登记制）；deny 归一化匹配防重命名；`Indexer.index()` 入口二次断言 product ∈ 白名单
- [x] 清场式重入库：按 product 先删后写（Chroma delete + BM25 delete_by_product），杜绝旧版条款残留；补"块数收缩"单测
- [x] ingest 失败通道：单文件异常不炸批、0 页解析记 failed 不写 manifest、failed 列表 + 非零 exit code
- [x] 索引落盘原子化：BM25 pickle / manifest 走临时文件 + os.replace

### G2 评测闭环（P0：跑出第一份可信结果前不对外报数字）

- [x] 修 harness 幸存者偏差：误拒题显性入账（false_refusal_rate、n_scored/n_answerable、metrics_penalized 误拒计0）；逐题 RAGAS 分数回填 records 支持两次运行逐题 diff
- [x] records 落盘 retrieved/cited chunk_id；新增 retrieval_hit_rate / citation_hit_rate（金标按「产品+页码重叠」判定，跨切片策略公平、语料重建后仍有效）
- [x] judge 与被评模型解耦：judge_model/judge_base_url/judge_api_key 独立配置并记入结果；PRICE 表查不到 → 告警 + cost 记 None
- [x] 逐题容错（error 字段不进分母）+ 每配置立即落盘（partial 标记）+ `--limit`/`--dry-run` 冒烟；文件名加时分秒、git hash 带 -dirty、payload 记模型/阈值/数据集 sha/语料指纹
- [ ] **（人工，kaka 约半天）** 定稿 dataset.jsonl：`python eval/build_dataset.py` 出草稿（配额+跨产品对抗题+comparison 模板已内置）→ 逐条对照 PDF 原文核对 ground_truth → 删 needs_review 改名 dataset.jsonl（loader 强制门禁，未核对跑不了）
- [ ] 跑通 8 配置全量评测（约 ¥5-10），第一份结果入 git；顺手校准 refuse_threshold 与 vector_floor
- [ ] **评测前先做**查询侧 embedding 缓存（key=模型名+归一化问题，进程内 LRU）：8 配置×50 题只有 50 个独立问题，7/8 的查询 embedding 可省；与切片策略无关不会白做（2026-07-04 决策：推迟到评测前，入库侧缓存等策略定型）

### G3 公开部署前（P0/P1）

- [x] 鉴权（API_AUTH_TOKEN 可选 Bearer）+ 每 IP 限流 + LLM max_tokens + 每日额度熔断（app/api/guard.py 三道闸；默认关，Dockerfile 生产开启；Space 已 private）
- [x] SSE error 事件协议（带 request_id）+ 全局异常 handler（JSON 错误体）+ 异常请求结构化日志 + 前端错误/中断态与 readErr 统一解析 + SSE 反缓冲头
- [x] lifespan 启动校验索引 fail-fast（STARTUP_REQUIRE_INDEX，Dockerfile 置 true）；/ready 深检（collection>0、BM25 存在、key 已配）与 /health 浅活分离；HEALTHCHECK 改打 /ready
- [x] rerank 独立短超时（5s、重试 1 次）+ 连续 3 败熔断 60s + 响应越界/重复 index 过滤与显式重排序
- [x] 依赖全量 lock（uv pip compile --universal --generate-hashes，97 包入 git）；镜像 --require-hashes 安装；CI 四件套：coverage 门槛 90%（现值 98%）/ pip-audit / docker build（stub 索引）/ gitleaks 全历史

### G4 Playground 增强（实验平台体验，详见 REVIEW 第"Playground 优化方案"节）

- [x] 检索透明化：RetrievedChunk 携带各路 rank/score（vector/BM25/粗排/rerank），前端逐 chunk 展示与位次变化
- [x] 只检索模式：/retrieve 端点不调 LLM，Playground 加开关（调优迭代亚秒级、零生成成本）
- [x] chunk 浏览器 + ingest 静态报告：页级解析质量、边界质检 lint（截断/吞并/无条号/长度离群）、条号主轨覆盖率、语料视图与 Playground 双向跳转（v1 边界条形态；`--reports-only` 零成本补报告）
- [ ] 语料视图增强：页级明细（繁简对照，依赖 raw_text 修复）、双策略并排对照、真正的原文叠加色带（需 chunk 记字符偏移）
- [ ] 参数进 Playground config：top_k/recall_k/refuse_threshold/RRF k（生产端点锁定）；延迟瀑布 + token 成本展示
- [ ] 案例保存：对比结果一键存档 → gen_eval_candidates 导入通道；评测表下钻回放
- [ ] 解析多方案预览先行：pdfplumber(不分栏)/pymupdf/Qwen-VL 同页 diff，胜者才建 collection
- [ ] collection 命名/版本 manifest：入库配置+语料版本 → collection 显式映射（解析维度与评测语料指纹的共同前置）

### G5 上线后与企业化

- [ ] 审计日志（Q/A/引用/用户/模型版本持久化，采样与脱敏策略）+ /feedback 反馈端点
- [ ] async 化外呼 + 并发上限背压；/metrics Prometheus 指标（拒答率/降级率/p95/成本）
- [ ] raw_text 链路修复：引用展示繁体原文；opencc t2s → hk2s；UI 标注"以保单繁体原文为准"
- [ ] 检索 product 过滤与 query 产品消歧（多产品混淆防线）
- [ ] live 冒烟测试（pytest -m live，CI 排除）；中文端到端 PDF fixture；conftest 隔离 index_dir
- [ ] 条款版本/生效日期元数据模型；SSO/RBAC/多租户（企业落地前置）

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
- SiliconFlow 国内/国际双平台：key 互不通用、模型目录不同（国际站无 bge 系，用 Qwen3 系），.env 三行切换（已文档化，2026-07-03 实际踩坑）
- 真实案例（D1 验收发现）：VHIS 保障表页「(等候期：300日)」是个别保障项的标注，表格拍平后被引用为整体等待期 → R9 VLM 表格解析的直接依据；评测集 table 题型必须覆盖此页
- Qwen3-Reranker 分数分布与 bge 不同：分红实现率类无据问题 top 分 0.498，拒答阈值 0.3 偏低 → D5 在评测集上校准（LLM 有据拒答检测已兜底）
- 国际站 API RTT 较高：embedding 单查询 ~1.5s，P95<5s 目标偏紧 → 可选优化：查询 embedding 缓存、rerank 并行、或换国内站
- PYSEC-2026-311（chromadb 1.5.9）：官方暂无修复版本，CI pip-audit 显式豁免中——**每次升级依赖时复查**，有修复版即升并移除豁免（.github/workflows/ci.yml audit job）
