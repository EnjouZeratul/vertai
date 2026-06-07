# VertAI | 垂直AI智能体SDK

垂直领域 AI 智能体开发 SDK，支持完全离线运行。

A vertical-domain AI agent development SDK designed for fully offline operation.

## 设计理念 | Design Philosophy

**模块化架构**：核心功能轻量安装，按需扩展语义能力。

**Modular Architecture**: Lightweight core installation with optional semantic capabilities.

```
vertai (核心 ~5MB | Core ~5MB)
├── Workflow        # 工作流编排 | Workflow orchestration
├── Dashboard       # 数据可视化 | Data visualization
├── DocGen          # 文档生成 | Document generation
├── DocParser       # 文档解析 (Markdown) | Document parsing (Markdown)
├── SessionMemory   # 会话管理 | Session management
└── VectorEngine    # 向量存储 (需嵌入模型提供语义能力) | Vector storage (requires embedding model for semantic capabilities)

可选扩展 | Optional Extensions
├── [embeddings]    # 离线语义搜索 | Offline semantic search
├── [doc-parser]    # 文档解析 (PDF/Word/Excel) | Document parsing (PDF/Word/Excel)
└── [production]    # 生产环境完整配置 | Complete production configuration
```

## 安装 | Installation

### 核心安装 | Core Installation

```bash
pip install vertai
```

### 扩展安装 | Optional Extensions

```bash
# 离线语义搜索支持
# Offline semantic search support
pip install vertai[embeddings]

# 文档解析支持 (PDF/Word/Excel/PPT)
# Document parsing support (PDF/Word/Excel/PPT)
pip install vertai[doc-parser]

# 完整生产配置
# Complete production configuration
pip install vertai[production]
```

| 安装选项 | 体积 | 功能 |
|---------|------|------|
| 核心 | ~5MB | Workflow, Dashboard, DocGen, Markdown解析 |
| [embeddings] | ~500MB | 离线语义向量搜索 |
| [doc-parser] | ~50MB | PDF/Word/Excel/PPT解析 |
| [production] | ~600MB | 完整生产配置 |

| Installation Option | Size | Features |
|---------------------|------|----------|
| Core | ~5MB | Workflow, Dashboard, DocGen, Markdown parsing |
| [embeddings] | ~500MB | Offline semantic vector search |
| [doc-parser] | ~50MB | PDF/Word/Excel/PPT parsing |
| [production] | ~600MB | Complete production configuration |

## 快速开始 | Quick Start

### 工作流编排（完全离线）| Workflow Orchestration (Fully Offline)

```python
from vertai import Workflow

wf = Workflow()
wf.step("load", lambda ctx: ctx.set("data", [1, 2, 3, 4, 5]))
wf.step("process", lambda ctx: ctx.set("sum", sum(ctx.get("data"))))
wf.step("output", lambda ctx: print(f"总和 | Sum: {ctx.get('sum')}"))
wf.run()
```

### 语义向量搜索（需安装 embeddings）| Semantic Vector Search (requires embeddings)

```python
from vertai import VectorEngine, Document
from sentence_transformers import SentenceTransformer

# 加载嵌入模型（首次下载约100MB，之后离线可用）
# Load embedding model (~100MB first download, then works offline)
model = SentenceTransformer('bge-small-zh-v1.5')

def embedding_fn(text):
    return model.encode(text).tolist()

# 创建向量引擎
# Create vector engine
engine = VectorEngine(store_type="memory", embedding_fn=embedding_fn)

# 索引文档
# Index documents
engine.index_documents([
    Document(content="Python是一种编程语言，由Guido van Rossum创建 | Python is a programming language created by Guido van Rossum"),
    Document(content="机器学习是人工智能的子领域 | Machine learning is a subfield of artificial intelligence"),
    Document(content="深度学习使用多层神经网络 | Deep learning uses multi-layer neural networks"),
])

# 语义搜索
# Semantic search
results = engine.search("编程语言 | programming language")
# 返回：Python是一种编程语言...（语义匹配，非关键词匹配）
# Returns: Python是一种编程语言... (semantic match, not keyword match)
```

### LLM 对话（需配置 API）| LLM Chat (requires API configuration)

```python
from vertai import LLMEngine, LLMConfig, ModelProvider

config = LLMConfig(
    provider=ModelProvider.DEEPSEEK,
    base_url="https://api.deepseek.com/anthropic",
    api_key="sk-xxx",  # 或设置环境变量 VERTAI_API_KEY | Or set environment variable VERTAI_API_KEY
    model="deepseek-v4-flash",
)

llm = LLMEngine(config)

# 单次生成
# Single generation
result = llm.generate("你好 | Hello")

# 流式输出
# Streaming output
for chunk in llm.stream("讲个故事 | Tell a story"):
    print(chunk, end="", flush=True)

# 多轮对话
# Multi-turn conversation
messages = [
    {"role": "user", "content": "我叫小明 | My name is Xiao Ming"},
    {"role": "assistant", "content": "你好小明！| Hello Xiao Ming!"},
    {"role": "user", "content": "我叫什么名字？| What's my name?"},
]
result = llm.chat(messages)
```

### 结构化数据提取 | Structured Data Extraction

```python
from vertai import StructuredOutput

schema = {"name": "string", "amount": "number"}

# 正则模式（完全离线，简单模式）
# Regex mode (fully offline, simple patterns)
output = StructuredOutput(schema)
result = output.extract("张三报销500元 | Zhang San expense 500 yuan")
# {'name': '张三', 'amount': 500.0} | Result shows extracted name and amount

# LLM模式（需配置API，语义理解）
# LLM mode (requires API, semantic understanding)
from vertai import LLMEngine, LLMConfig, ModelProvider
llm = LLMEngine(LLMConfig(
    provider=ModelProvider.DEEPSEEK,
    base_url="https://api.deepseek.com/anthropic",
    api_key="sk-xxx",
))
output = StructuredOutput(schema, llm=llm)
result = output.extract("李四消费了三百块 | Li Si spent three hundred yuan")
# {'name': '李四', 'amount': 300.0}（语义理解中文数字）
# {'name': 'Li Si', 'amount': 300.0} - semantic understanding of Chinese numbers
```

## 功能模块 | Feature Modules

| 模块 | 离线可用 | 依赖 |
|------|---------|------|
| Workflow | ✅ | 无 |
| Dashboard | ✅ | 无 |
| DocGen (Markdown/HTML) | ✅ | 无 |
| DocParser (Markdown) | ✅ | 无 |
| SessionMemory | ✅ | 无 |
| VectorEngine (存储) | ✅ | 无 |
| VectorEngine (语义搜索) | ✅ | sentence-transformers |
| StructuredOutput (正则) | ✅ | 无 |
| StructuredOutput (语义) | ❌ | LLM API |
| LLMEngine | ❌ | LLM API 或 Ollama |
| KnowledgeQA | ✅ | 向量搜索离线，生成需LLM |
| LocalModelManager | ✅ | 模型文件本地存储 |

| Module | Offline | Dependencies |
|--------|---------|--------------|
| Workflow | ✅ | None |
| Dashboard | ✅ | None |
| DocGen (Markdown/HTML) | ✅ | None |
| DocParser (Markdown) | ✅ | None |
| SessionMemory | ✅ | None |
| VectorEngine (Storage) | ✅ | None |
| VectorEngine (Semantic Search) | ✅ | sentence-transformers |
| StructuredOutput (Regex) | ✅ | None |
| StructuredOutput (Semantic) | ❌ | LLM API |
| LLMEngine | ❌ | LLM API or Ollama |
| KnowledgeQA | ✅ | Vector search offline, generation needs LLM |
| LocalModelManager | ✅ | Local model file storage |

## 本地模型 | Local Models

### 嵌入模型 | Embedding Models

| 模型 | 体积 | 语言 | 离线 |
|------|------|------|------|
| bge-small-zh-v1.5 | 100MB | 中文 | ✅ |
| bge-large-zh-v1.5 | 650MB | 中文 | ✅ |
| all-MiniLM-L6-v2 | 80MB | 英文 | ✅ |

| Model | Size | Language | Offline |
|-------|------|----------|---------|
| bge-small-zh-v1.5 | 100MB | Chinese | ✅ |
| bge-large-zh-v1.5 | 650MB | Chinese | ✅ |
| all-MiniLM-L6-v2 | 80MB | English | ✅ |

### 语音模型 | Speech Models

| 模型 | 体积 | 最低配置 | 离线 |
|------|------|---------|------|
| whisper-tiny | 75MB | 1GB RAM | ✅ |
| whisper-small | 466MB | 2GB RAM | ✅ |
| whisper-large-v3 | 2.9GB | 10GB RAM | ✅ |

| Model | Size | Min Requirements | Offline |
|-------|------|------------------|---------|
| whisper-tiny | 75MB | 1GB RAM | ✅ |
| whisper-small | 466MB | 2GB RAM | ✅ |
| whisper-large-v3 | 2.9GB | 10GB RAM | ✅ |

```python
from vertai import LocalModelManager

manager = LocalModelManager()
manager.download("bge-small-zh-v1.5")  # 首次下载 | First download
model = manager.load("bge-small-zh-v1.5")  # 之后离线加载 | Then load offline
```

## 测试 | Testing

```bash
pip install vertai[dev]
python -m pytest tests/ -v --cov=vertai

# 642 passed, 20 skipped, 94% coverage | 642 通过, 20 跳过, 94% 覆盖率
```

## 许可证 | License

MIT License - Copyright (c) 2026 VertAI Team

详见 [LICENSE](LICENSE) 文件。
