# VertAI 功能依赖说明 | Function Dependencies

> ⚠️ Pre-Alpha. This document reflects current state; capabilities are being refactored
> per `docs/ROADMAP.md`. The 94% line-coverage figure below is misleading — real I/O
> paths have ~0% effective coverage. See `CHANGELOG.md` for known limitations.

## 版本信息 | Version Information

- 版本 | Version: 0.2.0 (pre-alpha)
- 测试 | Tests: 642 passed, 20 skipped (集成测试当前全 skip，有效覆盖远低于行覆盖率)
- 覆盖率 | Coverage: 94% line coverage (misleading — see note above)

## 架构设计 | Architecture Design

SDK 采用模块化设计，核心包轻量（仅 httpx + pydantic 硬依赖），按需扩展功能。

The SDK adopts a modular design with a lightweight core package (only httpx + pydantic as hard dependencies) and on-demand feature extensions.

```
vertai
├── 核心模块（无外部依赖）
│   ├── Workflow          # 工作流编排 | Workflow orchestration
│   ├── DocGen            # 文档生成 | Document generation
│   ├── DocParser (MD)    # Markdown解析 | Markdown parsing
│   ├── SessionMemory     # 会话管理 | Session management
│   └── VectorEngine      # 向量存储框架 | Vector storage framework
│
├── 可选扩展
│   ├── [embeddings]      # 语义向量搜索 | Semantic vector search
│   ├── [doc-parser]      # 文档解析扩展 | Document parsing extension
│   ├── [viz]             # 可视化（Dashboard，已从核心移出）| Visualization (Dashboard, moved out of core)
│   └── [production]      # 生产环境配置 | Production configuration
│
└── 外部服务（可选）
    ├── LLM API           # 云端LLM服务 | Cloud LLM service
    └── Ollama            # 本地LLM服务 | Local LLM service
```

## 功能模块详情 | Module Details

### 核心模块（完全离线）| Core Modules (Fully Offline)

| 模块 | 功能 | 离线 | 说明 |
|------|------|------|------|
| Workflow | 工作流编排 | ✅ | 顺序、分支、并行、循环 |
| DocGen | 文档生成 | ✅ | Markdown/HTML模板 |
| DocParser (MD) | Markdown解析 | ✅ | 纯Python实现 |
| SessionMemory | 会话管理 | ✅ | 本地文件持久化 |
| VectorEngine | 向量存储 | ✅ | 存储框架，需嵌入模型提供语义 |

| Module | Function | Offline | Description |
|--------|----------|---------|-------------|
| Workflow | Workflow orchestration | ✅ | Sequential, branching, parallel, loops |
| DocGen | Document generation | ✅ | Markdown/HTML templates |
| DocParser (MD) | Markdown parsing | ✅ | Pure Python implementation |
| SessionMemory | Session management | ✅ | Local file persistence |
| VectorEngine | Vector storage | ✅ | Storage framework, needs embedding model for semantics |

### 可视化扩展（已从核心移出）| Visualization Extension (Moved Out of Core)

**安装 | Installation**: `pip install vertai[viz]`

> Dashboard（数据可视化）已从核心包移出，不再随 `import vertai` 自动加载。
> Dashboard (data visualization) has been moved out of the core package and is
> no longer eagerly loaded by `import vertai`.

```python
# 显式从 viz 子包导入 | Import explicitly from the viz subpackage
from vertai.viz.dashboard import Dashboard, Metric, Chart

dash = Dashboard(title="Performance Dashboard")
dash.add_metric("任务完成率", 85, unit="%").export("report.html")
```

> Dashboard 当前无第三方运行时依赖（纯标准库 HTML/JS 生成）。`[viz]` extras
> 作为显式安装分组保留，便于将来添加图表渲染依赖。
>
> Dashboard currently has no third-party runtime dependencies (pure-stdlib
> HTML/JS generation). The `[viz]` extras group is kept as an explicit install
> path for future chart-rendering dependencies.

### 语义向量搜索 | Semantic Vector Search

**安装 | Installation**: `pip install vertai[embeddings]`

> ⚠️ **无默认随机向量 | No random-vector fallback.** `VectorEngine` 在未配置
> `EmbeddingProvider` 时会**显式抛错**，而非静默使用随机向量（随机向量不具备
> 语义相似性，曾误导为"语义搜索✅"）。请注入 `EmbeddingProvider` 或安装
> `vertai[embeddings]`。
>
> ⚠️ **No random-vector fallback.** `VectorEngine` **raises explicitly** when no
> `EmbeddingProvider` is configured, instead of silently falling back to random
> vectors (which have no semantic similarity and were previously mislabeled as
> "semantic search ✅"). Inject an `EmbeddingProvider` or install
> `vertai[embeddings]`.

| 配置 | 离线 | 语义能力 | 说明 |
|------|------|---------|------|
| 未配置 provider | — | — | 抛错（不静默随机） | Raises (no silent random) |
| 本地嵌入模型 | ✅ | ✅ | 完全离线语义搜索 | Fully offline semantic search |
| 云端嵌入API | ❌ | ✅ | 需网络连接 | Requires network connection |

| Configuration | Offline | Semantic | Description |
|---------------|---------|----------|-------------|
| No provider configured | — | — | Raises explicitly (no silent random) |
| Local embedding model | ✅ | ✅ | Fully offline semantic search |
| Cloud embedding API | ❌ | ✅ | Requires network connection |

```python
# 离线语义搜索 | Offline semantic search
from vertai.core.embedding import LocalSentenceTransformerProvider
from vertai.core.vector import VectorEngine, Document

provider = LocalSentenceTransformerProvider('BAAI/bge-small-zh-v1.5')  # 首次下载，之后离线 | First download, then offline
engine = VectorEngine(embedding_provider=provider)
engine.index_documents([Document(content="...")])
results = engine.search("语义查询 | semantic query")  # 离线语义搜索 | Offline semantic search

# 或注入自定义嵌入函数（不具备语义，仅接入用）| Or inject a custom embedding function (non-semantic, wiring only)
engine = VectorEngine(embedding_fn=lambda t: [0.0] * 384, store_type="memory")
```

### 结构化数据提取 | Structured Data Extraction

| 模式 | 离线 | 语义能力 | 适用场景 |
|------|------|---------|---------|
| 正则模式 | ✅ | ❌ | 固定格式、简单模式 |
| LLM模式 | ❌ | ✅ | 自然语言、复杂结构（schema 验证 + 重试）|

| Mode | Offline | Semantic | Use Case |
|------|---------|----------|----------|
| Regex mode | ✅ | ❌ | Fixed format, simple patterns |
| LLM mode | ❌ | ✅ | Natural language, complex structures (schema-validated + retried) |

`extract()` returns an `ExtractionResult` dataclass with fields:
- `data: dict[str, Any]` — extracted key-value mapping (empty on failure)
- `success: bool` — whether schema validation passed for all fields
- `retries: int` — number of correction retries performed
- `error: str | None` — optional human-readable error description

**Important contracts**:
- The local regex `string` parser is **generic** (returns the first
  non-whitespace token). A small, documented set of field names
  (`name` / `姓名` / `名字` / `buyer`) triggers a Chinese-name
  reimbursement/purchase heuristic — this is a domain-example hint, not part of
  the general contract.
- The **LLM path schema-validates every field** via `_validate_field` and
  re-prompts the LLM with the validation errors, up to `max_retries` times.
  In strict mode, persistent failure raises `SchemaValidationError`.

```python
from vertai import StructuredOutput

schema = {"name": "string", "amount": "number"}

# 正则模式（离线）| Regex mode (offline)
output = StructuredOutput(schema)
result = output.extract("张三报销500元 | Zhang San expense 500 yuan")  # 简单模式 | Simple pattern
print(result.data, result.success)  # {'name': '张三', 'amount': 500.0} True

# LLM模式（需API）| LLM mode (requires API)
output = StructuredOutput(schema, llm=llm_engine)
result = output.extract("张三消费了三百块 | Zhang San spent three hundred yuan")  # 语义理解 | Semantic understanding
```

### 文档解析扩展 | Document Parsing Extension

**安装 | Installation**: `pip install vertai[doc-parser]`

| 格式 | 依赖包 | 体积 |
|------|--------|------|
| PDF | PyMuPDF | ~20MB |
| Word | python-docx | ~5MB |
| Excel | openpyxl | ~5MB |
| PPT | python-pptx | ~10MB |

| Format | Dependency | Size |
|--------|------------|------|
| PDF | PyMuPDF | ~20MB |
| Word | python-docx | ~5MB |
| Excel | openpyxl | ~5MB |
| PPT | python-pptx | ~10MB |

### LLM 服务 | LLM Services

**选项一：本地 Ollama（离线）| Option 1: Local Ollama (Offline)**

```bash
ollama serve
ollama pull llama3.2
```

```python
from vertai import LLMEngine
llm = LLMEngine()  # 默认 localhost:11434 | Default localhost:11434
```

**选项二：云端 API | Option 2: Cloud API**

```python
from vertai import LLMEngine, LLMConfig, ModelProvider

config = LLMConfig(
    provider=ModelProvider.DEEPSEEK,
    base_url="https://api.deepseek.com/anthropic",
    api_key="sk-xxx",
    model="deepseek-chat",
)
llm = LLMEngine(config)
```

| 提供商 | base_url | 离线 |
|--------|----------|------|
| DeepSeek | api.deepseek.com/anthropic | ❌ |
| Anthropic | api.anthropic.com | ❌ |
| OpenAI | api.openai.com/v1 | ❌ |
| Ollama | localhost:11434 | ✅ |

| Provider | base_url | Offline |
|----------|----------|---------|
| DeepSeek | api.deepseek.com/anthropic | ❌ |
| Anthropic | api.anthropic.com | ❌ |
| OpenAI | api.openai.com/v1 | ❌ |
| Ollama | localhost:11434 | ✅ |

## 本地模型 | Local Models

### 嵌入模型（推荐）| Embedding Models (Recommended)

| 模型 | 体积 | 语言 | 内存需求 | 下载源 |
|------|------|------|---------|--------|
| bge-small-zh-v1.5 | 100MB | 中文 | 0.5GB | HuggingFace |
| bge-large-zh-v1.5 | 650MB | 中文 | 2GB | HuggingFace |
| all-MiniLM-L6-v2 | 80MB | 英文 | 0.5GB | HuggingFace |

| Model | Size | Language | Memory | Source |
|-------|------|----------|--------|--------|
| bge-small-zh-v1.5 | 100MB | Chinese | 0.5GB | HuggingFace |
| bge-large-zh-v1.5 | 650MB | Chinese | 2GB | HuggingFace |
| all-MiniLM-L6-v2 | 80MB | English | 0.5GB | HuggingFace |

**国内镜像 | China Mirror**: hf-mirror.com

> 镜像通过 `HF_ENDPOINT` 环境变量生效。`LocalModelManager` 在下载 sentence-transformers
> 模型时会自动把镜像地址（如 `https://hf-mirror.com/...`）写入 `HF_ENDPOINT`，
> `huggingface_hub` 会据此重定向下载。设置 `LocalModelConfig(use_mirror=False)` 可关闭。
>
> Mirrors take effect via the `HF_ENDPOINT` environment variable. `LocalModelManager`
> exports the selected mirror endpoint (e.g. `https://hf-mirror.com/...`) as
> `HF_ENDPOINT` while downloading sentence-transformers models, which
> `huggingface_hub` reads to redirect the download. Pass
> `LocalModelConfig(use_mirror=False)` to disable.

### 语音模型 | Speech Models

| 模型 | 体积 | 精度 | 内存需求 |
|------|------|------|---------|
| whisper-tiny | 75MB | 低 | 1GB |
| whisper-base | 142MB | 中 | 1.5GB |
| whisper-small | 466MB | 较高 | 2GB |
| whisper-medium | 1.5GB | 高 | 5GB |
| whisper-large-v3 | 2.9GB | 最高 | 10GB |

| Model | Size | Accuracy | Memory |
|-------|------|----------|--------|
| whisper-tiny | 75MB | Low | 1GB |
| whisper-base | 142MB | Medium | 1.5GB |
| whisper-small | 466MB | Good | 2GB |
| whisper-medium | 1.5GB | High | 5GB |
| whisper-large-v3 | 2.9GB | Best | 10GB |

## 安装选项 | Installation Options

| 命令 | 体积 | 包含 |
|------|------|------|
| `pip install vertai` | ~5MB | 核心模块 |
| `pip install vertai[embeddings]` | ~500MB | + 语义搜索 |
| `pip install vertai[doc-parser]` | ~50MB | + 文档解析 |
| `pip install vertai[production]` | ~600MB | 完整配置 |

| Command | Size | Includes |
|---------|------|----------|
| `pip install vertai` | ~5MB | Core modules |
| `pip install vertai[embeddings]` | ~500MB | + Semantic search |
| `pip install vertai[doc-parser]` | ~50MB | + Document parsing |
| `pip install vertai[production]` | ~600MB | Complete configuration |

## 环境变量 | Environment Variables

```bash
# LLM API Key（任选其一）
# LLM API Key (choose one)
export VERTAI_API_KEY="sk-xxx"
export ANTHROPIC_API_KEY="sk-xxx"
```

## 测试 | Testing

```bash
pip install vertai[dev]
python -m pytest tests/ -v --cov=vertai
```

| 测试类型 | 说明 |
|---------|------|
| 单元测试 | 模块功能验证 |
| 集成测试 | 组件协同验证 |
| API测试 | 需配置API Key（可选跳过）|

| Test Type | Description |
|-----------|-------------|
| Unit tests | Module functionality verification |
| Integration tests | Component coordination verification |
| API tests | Requires API Key (optional skip) |
