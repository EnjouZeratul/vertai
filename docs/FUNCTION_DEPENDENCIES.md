# AI Agent SDK 功能依赖说明 | Function Dependencies

## 版本信息 | Version Information

- 版本 | Version: 0.1.0
- 测试 | Tests: 642 passed, 20 skipped
- 覆盖率 | Coverage: 94%

## 架构设计 | Architecture Design

SDK 采用模块化设计，核心包轻量（~5MB），按需扩展功能。

The SDK adopts a modular design with a lightweight core package (~5MB) and on-demand feature extensions.

```
vertai
├── 核心模块（无外部依赖）
│   ├── Workflow          # 工作流编排 | Workflow orchestration
│   ├── Dashboard         # 数据可视化 | Data visualization
│   ├── DocGen            # 文档生成 | Document generation
│   ├── DocParser (MD)    # Markdown解析 | Markdown parsing
│   ├── SessionMemory     # 会话管理 | Session management
│   └── VectorEngine      # 向量存储框架 | Vector storage framework
│
├── 可选扩展
│   ├── [embeddings]      # 语义向量搜索 | Semantic vector search
│   ├── [doc-parser]      # 文档解析扩展 | Document parsing extension
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
| Dashboard | 数据可视化 | ✅ | HTML导出、JSON序列化 |
| DocGen | 文档生成 | ✅ | Markdown/HTML模板 |
| DocParser (MD) | Markdown解析 | ✅ | 纯Python实现 |
| SessionMemory | 会话管理 | ✅ | 本地文件持久化 |
| VectorEngine | 向量存储 | ✅ | 存储框架，需嵌入模型提供语义 |

| Module | Function | Offline | Description |
|--------|----------|---------|-------------|
| Workflow | Workflow orchestration | ✅ | Sequential, branching, parallel, loops |
| Dashboard | Data visualization | ✅ | HTML export, JSON serialization |
| DocGen | Document generation | ✅ | Markdown/HTML templates |
| DocParser (MD) | Markdown parsing | ✅ | Pure Python implementation |
| SessionMemory | Session management | ✅ | Local file persistence |
| VectorEngine | Vector storage | ✅ | Storage framework, needs embedding model for semantics |

### 语义向量搜索 | Semantic Vector Search

**安装 | Installation**: `pip install vertai[embeddings]`

| 配置 | 离线 | 语义能力 | 说明 |
|------|------|---------|------|
| 默认（随机向量） | ✅ | ❌ | 仅存储框架，无语义匹配 |
| 本地嵌入模型 | ✅ | ✅ | 完全离线语义搜索 |
| 云端嵌入API | ❌ | ✅ | 需网络连接 |

| Configuration | Offline | Semantic | Description |
|---------------|---------|----------|-------------|
| Default (random vectors) | ✅ | ❌ | Storage framework only, no semantic matching |
| Local embedding model | ✅ | ✅ | Fully offline semantic search |
| Cloud embedding API | ❌ | ✅ | Requires network connection |

```python
# 离线语义搜索 | Offline semantic search
from sentence_transformers import SentenceTransformer
from vertai import VectorEngine, Document

model = SentenceTransformer('bge-small-zh-v1.5')  # 首次下载，之后离线 | First download, then offline

def embedding_fn(text):
    return model.encode(text).tolist()

engine = VectorEngine(embedding_fn=embedding_fn)
engine.index_documents([...])
results = engine.search("语义查询 | semantic query")  # 离线语义搜索 | Offline semantic search
```

### 结构化数据提取 | Structured Data Extraction

| 模式 | 离线 | 语义能力 | 适用场景 |
|------|------|---------|---------|
| 正则模式 | ✅ | ❌ | 固定格式、简单模式 |
| LLM模式 | ❌ | ✅ | 自然语言、复杂结构 |

| Mode | Offline | Semantic | Use Case |
|------|---------|----------|----------|
| Regex mode | ✅ | ❌ | Fixed format, simple patterns |
| LLM mode | ❌ | ✅ | Natural language, complex structures |

```python
from vertai import StructuredOutput

schema = {"name": "string", "amount": "number"}

# 正则模式（离线）| Regex mode (offline)
output = StructuredOutput(schema)
result = output.extract("张三报销500元 | Zhang San expense 500 yuan")  # 简单模式 | Simple pattern

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
    model="deepseek-v4-flash",
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
