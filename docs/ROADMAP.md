# VertAI 重构路线图 → 1.0

> 基于对 v0.1.3 的实证审查 + 架构审计制定。

## 核心原则

**模块纵切，一次到位**：按子系统整体重构。每个阶段聚焦一个完整模块，从代码→测试→类型→安全→文档一次性改干净，交付自洽整体。不在多个模块上横切做零散小改动——那种横切片是不必要的中间态，干扰后续模块的修改。

**为什么这样**：LLM 适合在单一子系统内建立完整上下文后一次改透。横切（所有模块各改一点）会反复在文件间跳转，每次切换都丢失上下文，且留下"每个模块都半新半旧"的中间态。纵切（一个模块全改透）让每个交付点干净自洽，下一个模块从干净基线开始。

**其他原则**：
- **契约优先**：新增/重构核心抽象时，先定义 ABC/Protocol 接口契约，再改实现（contract-first）
- **自洽才推进**：模块未整体改完不进下一模块，不上 PyPI 发不自洽版
- **实测验证**：所有"已修复"由 mypy/ruff/pytest 输出支撑，不接受"应该可以了"
- **文档随代码 / 导出随代码**：改模块时同步该模块文档与 `__init__.py` 导出
- **不凑合发布**：不为快点发出去留已知缺陷
- **不追全家桶**：差异化是本地优先 + 轻量 + 质量极致，不是功能最多；不追 multi-agent/planner/tracing
- **测试分层最合适**：每处测试用最合适的方式——纯逻辑用真实断言，外部依赖用 fake/mock/stub，关键路径用端到端真实集成；不刷覆盖率，不用 mock 循环验证掩盖真实行为
- **阶段审查机制**：每阶段完成后做三重审查——代码真实实现审查（实现是否兑现契约/无摆设）、测试覆盖率与真实性审查（覆盖率数字是否可信、有无 except 掩盖/mock 循环）、文档与实现一致性审查。审查通过才进下一阶段

---

## 架构决策（基于事实）

**产品定位**：国际化、本地优先、垂直领域快速开发、质量极致的轻量 Agent SDK。

**现状事实**（架构审计确认，读全部19文件）：VertAI 当前是"RAG 工具包"，缺 Agent SDK 核心抽象——Tool/Agent/Retriever/Callbacks 全缺失，0 async，dashboard 与核心零耦合却在核心里。"垂直领域快速开发"没有 tool calling + agent loop 做不到。

**决策**：针对性架构级扩展，保留现有原语，不重写。

| 类别 | 处理 | 依据 |
|------|------|------|
| 保留（原语） | VectorStore ABC、Memory、Parser、Workflow、LocalModel | 设计合理，组合基础 |
| 重构 | `llm.py` → `LLMProvider` ABC + 每 provider 适配器 + tool calling + async | 当前单类伪装多 provider，OpenAI 路由坏 |
| 新增核心 | `Tool`+Registry、`Agent`(tool-calling loop)、`EmbeddingProvider` ABC、`Retriever` ABC、轻量 `Callbacks` | agent SDK vs RAG 工具包的分水岭 |
| 移出核心 | dashboard → 可选 `vertai[viz]` | 与 agent 循环零耦合 |
| async-first | Provider/Agent/Retriever async 接口，sync 兼容层 | 2024+ SDK 标准 |
| 国际化 | 英文为主，双语可选；错误消息英文 | "国际化"定位 |

**目标架构**：

```
vertai/
├── core/                  # 核心抽象（契约优先）
│   ├── provider.py        # LLMProvider ABC + 适配器（Ollama/Anthropic/DeepSeek/OpenAI）
│   ├── embedding.py       # EmbeddingProvider ABC（独立，真实默认）
│   ├── retriever.py       # Retriever ABC（reranking/query transform 可扩展）
│   ├── tool.py            # Tool ABC + Registry + @tool（新增）
│   ├── agent.py           # Agent + tool-calling loop（新增）
│   ├── callbacks.py       # 轻量可观测性钩子（新增）
│   ├── memory.py          # 会话记忆（重构：原子写/真实tokenizer/uuid）
│   └── vector.py          # VectorStore ABC + 后端（重构，移除hash随机默认）
├── scenarios/             # 场景（依赖 core 抽象而非具体实现）
│   ├── knowledge_qa.py    # 用 Retriever（重构）
│   └── reviewer.py        # 用 LLMProvider（重构）
├── workflow/              # 工作流（重构：并行锁/timeout）
├── data/parser.py         # 文档解析（重构）
├── output/                # 输出（docgen/structured 重构）
├── local/models.py        # 本地模型（重构：真实URL/镜像）
└── viz/                   # 移出核心 → 可选 vertai[viz]
```

**关键判断**：不追 LangChain 全家桶。最小可用 tool-calling agent 足够支撑"垂直领域快速开发"。差异化是本地优先 + 中文友好 + 轻量 + 质量极致。

---

## Context

v0.1.3 实证审查暴露：核心入口 `KnowledgeQA.ask()` 崩溃却被测试掩盖；94%覆盖失真（真实I/O路径0%覆盖）；mypy 66错/ruff 23错却声明 Typed；文档夸大（5MB实测60KB、完全离线、垂直领域营销、假模型名 deepseek-v4-flash）；间接注入防护摆设；0处async却宣称流式；并行竞态；空目录占位；缺 Tool/Agent/Retriever 核心抽象。

实际成熟度：**pre-alpha**（非声称的 Alpha）。目标：架构级扩展 + 模块纵切重构到生产级 1.0。

---

## 依赖关系

模块依赖方向（已实测）：`scenarios/*` → `core/*`。新增抽象依赖关系：`Agent` → `LLMProvider` + `Tool`；`Retriever` → `VectorStore` + `EmbeddingProvider`；`KnowledgeQA` → `Retriever`。阶段顺序遵循依赖，被依赖方先改；契约先于实现。

```
S1 架构决策与元数据基线（写架构设计文档定义核心契约 + 项目级元数据诚实化）
  ↓ 契约定义完成，后续模块实现契约
S2 LLMProvider 抽象（core/provider.py，被 S3/S5/S6 依赖）→ async + tool calling + 修C4/chat/死代码
  ↓
S3 Embedding/Vector/Retriever 抽象（依赖S2；含KnowledgeQA修C1/C2/C3 + 注入安全）
  ↓
S4 Tool 抽象（新增，依赖S2 LLMProvider的tool calling）
  ↓
S5 Agent 抽象（新增，依赖S2 Provider + S4 Tool）+ Callbacks 轻量可观测
  ↓
S6 工作流引擎（独立，并行锁/timeout/链式/mypy）
  ↓
S7 本地模型管理（独立，真实URL/镜像/GPU/mypy/覆盖率）
  ↓
S8 输出层 + 可视化移出（docgen/structured/parser 修透；dashboard→vertai[viz]）
  ↓
S9 会话记忆（原子写/tokenizer/uuid/session_id路径遍历）
  ↓
S10 公开API整合 + 国际化（__init__精简移除viz eager import；导出完整；英文文档为主）
  ↓
S11 工程化CI/CD
  ↓
S12 1.0 生产就绪
```

**跨阶段协调项**：ModelInfo 同名冲突（S2 core 侧命名，S7 local 侧，S10 校验）。

---

## 阶段总览

| 阶段 | 模块 | 交付版本 | 性质 |
|------|------|---------|------|
| S1 | 架构决策与元数据基线 | 0.2.0 | 契约定义 + 项目元数据诚实化 |
| S2 | LLMProvider 抽象与重构 | 0.3.0 | core 重构 + async + tool calling |
| S3 | Embedding/Vector/Retriever/TextSplitter 抽象 | 0.4.0 | RAG 核心重构 |
| S4 | Tool 抽象 + 内置工具（新增） | 0.5.0 | agent 核心能力 |
| S5 | Agent 抽象 + Callbacks（新增） | 0.6.0 | agent SDK 分水岭 |
| S6 | 工作流引擎 | 0.7.0 | 并发安全 |
| S7 | 本地模型管理 | 0.8.0 | 真实可用 |
| S8 | 输出层 + 可视化移出 | 0.9.0 | 纯化核心 |
| S9 | 会话记忆 | 0.9.5 | 记忆可靠 |
| S10 | 公开API整合 + 国际化 | 0.9.9 | API 一致 |
| S11 | 工程化CI/CD | 0.9.95 | 可持续交付 |
| S12 | 1.0 生产就绪 | 1.0.0 | 生产可上 |

---

## S1 — 架构决策与元数据基线（0.2.0）

**目的**：契约优先——先写架构设计文档定义所有核心抽象的接口契约（LLMProvider/EmbeddingProvider/Retriever/Tool/Agent/Callbacks），为后续模块实现提供契约；同时诚实化项目级元数据。不写实现，只定契约 + 改项目元数据。

**范围**：
- 写 `docs/ARCHITECTURE.md`：目标架构、模块边界、核心抽象接口契约（ABC/Protocol 签名）
- 契约定义（仅接口，不实现）：LLMProvider、EmbeddingProvider、Retriever、Tool、Agent、Callbacks 的方法签名与职责
- 项目元数据：版本单一来源 `dynamic = ["version"]`；CHANGELOG 回溯 v0.1.0–0.1.3 含缺陷声明；git tag v0.1.3 回填
- README 顶部"早期开发中，非生产就绪"警告；国际化方向（英文为主）
- pyproject classifier 对齐：暂移除 `Typing::Typed`（等 S2-S9 修完，S10 升回）；移除 dashboard 相关核心声明
- FUNCTION_DEPENDENCIES.md 项目级版本号同步

**不在 S1**：模块功能文档（随对应模块阶段）、假模型名（随 S2）、代码实现（随各模块）。

**Gate**：ARCHITECTURE.md 含全部核心抽象契约；版本单一来源；CHANGELOG 诚实；警告在位；classifier 不超前；项目级元数据自洽

---

## S2 — LLMProvider 抽象与重构（0.3.0）

**目的**：把 `core/llm.py` 重构为 `LLMProvider` ABC + 每 provider 适配器，async-first，支持 tool calling。这是被 S3/S4/S5 依赖的核心。

**范围（契约已在 S1 定义，本阶段实现）**：
- 代码：`core/provider.py` LLMProvider ABC + Ollama/Anthropic/DeepSeek/OpenAI 适配器；修复 C4（真实 OpenAI `/v1/chat/completions` + Bearer）；`chat()`/`achat()` 统一 dict/ChatMessage；tool calling payload 构建与 `tool_use` 解析；删除死代码 `_model_cache`/`current_block_type`；哨兵默认值改 None
- 异步：`agenerate`/`astream`/`achat`/`achat_stream`（httpx.AsyncClient）；流式真实 async iterator；sync 兼容层
- 类型：修复该模块全部 mypy 错误
- 测试：chat dict 测试；async 测试；用 VERTAI_API_KEY 真实 API 集成（非 skip）；tool calling 测试；移除 except 掩盖
- 文档：llm 相关同步（含 deepseek-chat 真实模型名）；明确 sync/async API；英文为主
- 导出：`__init__.py` 同步 llm 导出（LLMProvider 等）；与 S7 协调 ModelInfo core 侧命名

**Gate**：`mypy --strict` 该模块 0 错；async 真实可用；OpenAI 路由正确；chat 不崩溃；tool calling 工作；集成测试真实执行；docstring 与实现一致

---

## S3 — Embedding/Vector/Retriever/TextSplitter 抽象（0.4.0）

**目的**：独立 EmbeddingProvider ABC（真实默认）、Retriever ABC、TextSplitter ABC、VectorStore 重构、KnowledgeQA 用 Retriever。修 C1/C2/C3 + 注入安全。

**范围**：
- 代码：`core/embedding.py` EmbeddingProvider ABC（真实默认，移除 hash 随机）；`core/retriever.py` Retriever ABC（可扩展 reranking/query transform）；`core/text_splitter.py` TextSplitter ABC（递归/固定/语义分块策略，替换 knowledge_qa 硬编码 chunking）；`core/vector.py` 重构修 C2（无 embedding_fn 显式抛错不静默随机）、C3（FAISS.delete 真实删除）；删除 EmbeddingEngine.embed 死分支；定义 EmbeddingProvider Protocol
- 安全：`_sanitize_context` 真实检测注入或诚实移除注释；`_DANGEROUS_PATTERNS` 补中文模式；symlink 跟随防护（3.10-3.12）；C1 `knowledge_qa._get_llm` 修复用 S2 的 LLMProvider
- scenarios：knowledge_qa/reviewer 改依赖 core 抽象（Retriever/LLMProvider）而非具体实现；Reviewer 泛化为通用 Evaluation（LLM-as-judge 抽象）
- 类型：修复这些文件全部 mypy 错误
- 测试：重写 test_get_llm_default 移除 except；补 ask() 真实生成路径；Chroma/FAISS 真实集成（@integration）；间接/中文注入、symlink 安全测试；TextSplitter 各策略真实分块测试
- 文档：向量/RAG 相关同步（修正"语义搜索✅但默认随机"误导）
- 导出：同步 vector/embedding/retriever/text_splitter/qa/reviewer 导出

**Gate**：mypy 0 错；FAISS 删除一致；ask 不崩溃；无 hash 随机向量；安全测试通过；无 except 掩盖；TextSplitter 真实分块

---

## S4 — Tool 抽象 + 内置工具（0.5.0，新增）

**目的**：新增 `core/tool.py`——Tool ABC + Registry + @tool 装饰器 + 内置常用工具，集成 S2 LLMProvider 的 function calling。这是 agent 核心能力。

**范围（契约已在 S1 定义）**：
- 代码：`core/tool.py` Tool ABC（name/description/parameters schema/execute）、ToolRegistry、`@tool` 装饰器；JSON Schema 自动生成（从类型注解）；同步+异步 execute；与 LLMProvider tool calling 集成
- 内置工具（`core/tools/`）：web_search、file_read/file_write、http_request、calculator（开箱即用，覆盖垂直领域常见工具需求）
- 类型：严格类型（mypy strict）
- 测试：tool 注册/执行/Schema 生成真实测试；tool calling 端到端（fake LLM 返回 tool_use，验证执行+回传）；内置工具各自真实测试（file 用 tmp_path，http 用 fake server）
- 文档：Tool 抽象文档（英文为主）；内置工具清单与使用示例
- 导出：同步 tool 导出

**Gate**：mypy 0 错；@tool 装饰器工作；JSON Schema 正确生成；tool calling 端到端测试通过；内置工具真实可用

---

## S5 — Agent 抽象 + Callbacks（0.6.0，新增）

**目的**：新增 `core/agent.py`——最小可用 tool-calling agent loop + 终止条件；轻量 Callbacks 可观测性。这是 agent SDK 分水岭。依赖 S2 Provider + S4 Tool。

**范围（契约已在 S1 定义）**：
- 代码：`core/agent.py` Agent（LLMProvider + Tools + system prompt + 终止条件 + max_iterations）；tool-calling loop（generate→tool_use→execute→回传→再 generate）；`core/callbacks.py` 轻量钩子（on_llm_start/on_tool_start/on_tool_end/on_agent_end，可观测 token/延迟）；同步+异步
- 类型：严格类型
- 测试：agent loop 真实测试（mock LLM 多轮 tool_use）；终止条件测试；callbacks 触发测试；max_iterations 防失控测试
- 文档：Agent 抽象文档（英文为主）；ReAct 风格使用示例
- 导出：同步 agent/callbacks 导出

**Gate**：mypy 0 错；agent loop 正确执行 tool calling 序列；终止条件工作；callbacks 触发；无失控循环

---

## S6 — 工作流引擎（0.7.0）

**目的**：`workflow/workflow.py` 改透——并行竞态、timeout 死配置、链式 API、mypy、刷覆盖率测试。

**范围**：
- 代码：决定 step/branch/parallel/loop 返回 self（链式）或删链式文档；并行加锁（WorkflowContext 线程安全）；timeout 真实实现或删字段；删除死代码/冗余
- 类型：修复该文件 mypy 错误
- 测试：删除行号导向 Test*Exceptions 刷覆盖率类；补并行竞态测试、timeout 生效测试
- 文档：workflow 相关同步（链式真实性，英文为主）
- 导出：同步 workflow 导出

**Gate**：mypy 0 错；并行竞态测试通过；timeout 生效；无刷覆盖率测试；链式与文档一致

---

## S7 — 本地模型管理（0.8.0）

**目的**：`local/models.py` 改透——覆盖率仅58%、假URL、镜像死代码、GPU TypeError、伪造模型名。

**范围**：
- 代码：真实 whisper download_url 或移除该字段；镜像真实生效（HF_ENDPOINT）；check_hardware_requirements GPU None>=float 修复；`__init__` 副作用（mkdir）移除；删除死代码 model_url/丢弃赋值；与 S2 协调 ModelInfo local 侧命名
- 类型：修复该文件 mypy 错误
- 测试：whisper-tiny 真实下载/加载冒烟测试（非 mock 缓存命中）；该模块覆盖率 ≥85%
- 文档：local model 相关同步（真实 URL、镜像、硬件需求，英文为主）
- 导出：同步 local 导出

**Gate**：mypy 0 错；该模块覆盖率 ≥85%；GPU TypeError 修复；镜像真实生效；无假URL

---

## S8 — 输出层 + 可视化移出（0.9.0）

**目的**：`output/docgen.py`+`structured.py`+`data/parser.py` 修透；dashboard 移出核心 → 可选 `vertai[viz]`。

**范围**：
- 代码：C5 DocGen PDF 类型契约修正 save bytes/str 分支；StructuredOutput string 类型泛化（移除写死中文人名提取）、LLM 路径补 schema 验证；DocParser 返回键统一 text；删除 dir() 内省反模式
- 可视化移出：`viz/dashboard.py` 移至可选扩展 `vertai[viz]`；从根 `__init__.py` 移除 eager import；pyproject 加 `[viz]` extras
- 类型：修复这三个文件 mypy 错误
- 测试：移除 TestRemainingCoverage 刷覆盖率类；DocParser 各格式真实测试（需 doc-parser extras）；viz 独立测试
- 文档：输出层同步（返回类型、ExtractionResult，英文为主）；viz 移出说明
- 导出：同步 output/data 导出；移除 viz 核心导出

**Gate**：mypy 0 错；PDF 类型契约正确；string 类型通用；无刷覆盖率测试；dashboard 移出核心且 viz extras 可用

---

## S9 — 会话记忆（0.9.5）

**目的**：`core/memory.py` 改透——原子写、真实 tokenizer、uuid、session_id 路径遍历。

**范围**：
- 代码：save 原子写（tmp+os.replace）；load 损坏处理；真实 tokenizer（替换 len//4，中文严重低估）；`_generate_session_id` 用 uuid4；session_id 路径遍历清洗（白名单 `^[a-zA-Z0-9_-]+$`）
- 类型：修复该文件 mypy 错误、ruff F401
- 测试：原子写测试；tokenizer 精度测试；session id 唯一性测试；路径遍历安全测试
- 文档：memory 相关同步（英文为主）
- 导出：同步 memory 导出

**Gate**：mypy/ruff 0 错；原子写测试通过；tokenizer 中文精度合理；无同毫秒 id 冲突；路径遍历测试通过

---

## S10 — 公开API整合 + 国际化（0.9.9）

**目的**：所有模块改透后，整合公开 API——导出完整、命名一致、返回类型约定、`__init__.py` 精简、英文文档为主。这是纵切完成后的横向收口。

**范围**：
- 校验导出完整性：所有公开类型在 `__all__`（ExtractionResult/Template/OllamaDetector/Tool/Agent/Retriever/Callbacks 等）
- ModelInfo 同名冲突最终校验（S2/S7 已分别处理）
- `__init__.py` 精简：移除 viz eager import（S8 已移出）；lazy 可选导入重依赖
- 统一方法命名对称（generate/generate_stream/chat/chat_stream + async 对应）
- 返回类型约定文档化（dataclass/pydantic/dict 边界）
- 空目录 `vertai/security/`、`vertai/utils/` 删除（未实现承诺）+ 删设计文档对应承诺
- pyproject classifier 升级（Typing::Typed 此时如实声明）
- 国际化：README/文档英文为主，中文为辅；错误消息统一英文；双语可选
- 移除"非生产就绪"警告前置条件评估

**Gate**：`__all__` 完整无冲突；`__init__.py` 精简无 eager viz import；方法命名对称；无空目录；classifier 与实际一致；文档英文为主无虚假声称

---

## S11 — 工程化与 CI/CD（0.9.95）

**目的**：可持续交付，质量自动化。

**范围**：
- `.github/workflows/ci.yml`（mypy --strict + ruff + test + coverage 阈值 + build）
- CI 注入 VERTAI_API_KEY 跑集成测试
- `.github/workflows/release.yml`（tag→PyPI 自动发布）
- 性能基准测试套件（LLM 请求构建、向量搜索、流式解析、agent loop）
- 文档准确性自动化（CI 跑示例代码确保复制即用）
- API reference 自动生成（pdoc/sphinx）
- 移除 pytest.ini 废弃 python_paths

**Gate**：CI 全绿含集成测试；发布自动化；性能基准有基线；示例 CI 自动验证

---

## S12 — 1.0 生产就绪（1.0.0）

**目的**：API 冻结，生产可上。

**范围**：
- API 冻结审查（SemVer 边界，标注 deprecated）
- 安全审计通过（S3 防护 + 第三方 review）
- 性能基准达标
- 文档完整准确（API reference、迁移指南、部署指南，英文为主）
- 至少 2 个真实使用案例验证（垂直领域快速开发场景）
- Development Status 升级 `5 - Production/Stable`
- 移除"非生产就绪"警告

**Gate**：API 契约稳定；安全审计无 HIGH 遗留；≥2 真实案例；文档无虚假声称；可宣称生产就绪且经得起审查

---

## 跨阶段约束

1. **契约优先**：S1 定义核心抽象契约，S2-S9 实现契约；新增抽象先定接口再写实现
2. **模块纵切一次到位**：每阶段聚焦一个完整模块，代码+测试+类型+安全+文档同步改透
3. **遵循依赖**：被依赖模块先改（core 抽象先于 scenarios；Provider 先于 Agent；契约先于实现）
4. **自洽才推进**：模块未整体改完不进下一模块
5. **实测验证**：所有"已修复"由 mypy/ruff/pytest 输出支撑
6. **文档随代码 / 导出随代码**：改模块时同步该模块文档与 `__init__.py` 导出
7. **不追全家桶**：最小可用 agent，不追 multi-agent/planner/tracing
8. **不跳版本号**：PyPI 单调递增
9. **不凑合发布**：不为快发版留已知缺陷

## 测试策略（分层，最合适的方式）

每处测试用最合适的方式，不为刷覆盖率用错方法：

| 测试对象 | 方式 | 何时用 | 禁止 |
|---------|------|--------|------|
| 纯逻辑（解析/分块/校验/工作流控制流） | 真实断言 | 默认 | — |
| LLM/网络/磁盘 I/O 单元 | fake/stub | 不依赖真实外部行为时 | mock 循环验证（验证"调了 mock"无意义） |
| 外部服务协议（HTTP payload/SSE） | mock | 验证协议构造正确 | mock 掩盖真实 bug（except 吞异常） |
| 真实 API/真实模型/真实向量库 | 端到端 @integration | 关键路径，CI 注入凭据 | 全 skip 后宣称"有集成测试" |
| 并发/竞态 | 真实多线程+断言不变量 | 并行代码 | fake executor 替换真实并发 |

**测试真实性红线**：
- 禁止 `except Exception: pass` 掩盖被测行为
- 禁止以行号为目标的测试类（TestRemainingCoverage、Test*Exceptions 行号导向）
- 禁止测 mock 本身（断言 `mock.assert_called` 验证"代码调用了 mock"而无真实行为断言）
- mock 必须 stub 真实外部行为，不能 stub 出不可能的返回
- 集成测试必须在 CI 真实执行（注入凭据），不能全 skip

## 阶段审查机制（每阶段完成后）

每阶段完成后做三重审查，全部通过才进下一阶段：

1. **代码真实实现审查**：实现是否兑现 S1 契约？有无"声称实现实为摆设"（如 `_sanitize_context` 注释声称防护实际只剥控制字符）？有无死代码/死配置/伪造数据（假URL/假模型名）？跨模块是否遵循依赖方向（scenarios→core 抽象非具体实现）？
2. **测试覆盖率与真实性审查**：覆盖率数字是否可信（无刷行号、无 except 掩盖、无 mock 循环）？关键路径是否真实覆盖（非仅行覆盖）？集成测试是否真实执行？有无测试用异常吞没掩盖 bug？
3. **文档与实现一致性审查**：文档声称与实现一致（无 5MB/完全离线/垂直领域等不实）？示例代码复制即用（无 AttributeError/TypeError/KeyError）？导出与 `__all__` 一致？英文为主国际化到位？

审查产出：每阶段一份审查记录（写入 `docs/reviews/Sxx-review.md`），记录三重审查结果、实测命令输出、遗留项与决策。审查未过则返工，不进下一阶段。

---

## 1.x 后置增强（非 1.0 核心，诚实标注后续）

1.0 是"垂直领域快速开发核心闭环"自洽完整（对话+RAG+提取+工具+agent+评估+记忆+本地模型+工作流+输出）。以下为生产增强项，1.x 迭代：

| 增强项 | 场景 | 1.0 状态 |
|--------|------|---------|
| LLM 响应缓存 | 降本提速 | 后置 |
| 成本/限流控制 | 生产成本管控 | 后置 |
| Human-in-the-loop | 审批流/人工确认 | 后置 |
| 多模态输入 | 图片理解/OCR | 后置 |
| 数据清洗 | 脏数据预处理 | 后置 |
| 多 Agent 协作 | agents-as-tools/handoff | 后置 |
| 可观测性增强 | OTel/tracing 全家桶 | 后置（1.0 仅轻量 Callbacks） |
| 高级检索 | HyDE/multi-query/reranker 模型 | 后置（1.0 Retriever 可扩展） |

---

## 最终 1.0 验收

```bash
mypy --strict vertai/         # 0 errors
ruff check vertai/            # 0 errors
pytest tests/ -v              # 含集成测试，无 except 掩盖
pytest --cov=vertai           # 关键路径覆盖达标
python examples/quick_start_knowledge_base.py   # 复制即用
python examples/knowledge_qa_demo.py            # 复制即用
python examples/agent_demo.py                   # agent + tool calling 可用
gh workflow list              # ci.yml + release.yml
```

---

## 关键文件 → 阶段映射

| 阶段 | 主要文件 |
|------|---------|
| S1 | docs/ARCHITECTURE.md / README.md / pyproject.toml / CHANGELOG.md |
| S2 | vertai/core/provider.py（新）/ llm.py + tests/test_llm.py + test_deepseek_integration.py |
| S3 | vertai/core/embedding.py（新）/ retriever.py（新）/ text_splitter.py（新）/ vector.py + scenarios/* + tests |
| S4 | vertai/core/tool.py（新）+ core/tools/（内置）+ tests |
| S5 | vertai/core/agent.py（新）/ callbacks.py（新）+ tests |
| S6 | vertai/workflow/workflow.py + tests/test_workflow.py |
| S7 | vertai/local/models.py + tests/test_local_models.py |
| S8 | vertai/output/* + vertai/data/parser.py + viz/ 移出 + tests |
| S9 | vertai/core/memory.py + tests |
| S10 | vertai/__init__.py + pyproject.toml + 文档国际化 |
| S11 | .github/workflows/ + 性能测试 + API docs |
| S12 | 全项目稳定性收口 |
