# VertAI 架构设计

> 本文档定义 VertAI 的目标架构与核心抽象接口契约。S1 仅定义契约（接口签名与职责），S2-S9 实现契约。
> 契约优先：实现必须兑现此处的接口契约，不得"声称实现实为摆设"。

## 1. 产品定位

**国际化、本地优先、垂直领域快速开发、质量极致的轻量 Agent SDK。**

- **本地优先**：核心抽象支持本地部署（Ollama、本地嵌入模型），不强制云依赖
- **垂直领域快速开发**：提供 tool calling + agent loop + RAG + 提取 + 输出的核心闭环，让垂直领域应用快速搭建
- **轻量**：核心小依赖（httpx + pydantic），重依赖可选 extras
- **质量极致**：mypy strict、真实测试、文档与实现一致
- **不追全家桶**：差异化是本地优先 + 轻量 + 质量，非功能最多；不追 multi-agent/planner/tracing

## 2. 模块边界

```
vertai/
├── core/                  # 核心抽象（契约层，被依赖方）
│   ├── provider.py        # LLMProvider ABC + 适配器（S2）
│   ├── embedding.py       # EmbeddingProvider ABC（S3）
│   ├── retriever.py       # Retriever ABC（S3）
│   ├── text_splitter.py   # TextSplitter ABC（S3）
│   ├── vector.py          # VectorStore ABC + 后端（S3 重构）
│   ├── tool.py            # Tool ABC + Registry + @tool（S4）
│   ├── tools/             # 内置工具（web/file/http/calc）（S4）
│   ├── agent.py           # Agent + tool-calling loop（S5）
│   ├── callbacks.py       # 轻量可观测性钩子（S5）
│   └── memory.py          # 会话记忆（S9 重构）
├── scenarios/             # 场景，依赖 core 抽象非具体实现
│   ├── knowledge_qa.py    # 用 Retriever + LLMProvider（S3）
│   └── reviewer.py        # 泛化为通用 Evaluation（S3）
├── workflow/workflow.py   # 工作流（S6 重构）
├── data/parser.py         # 文档解析（S8 重构）
├── output/                # docgen/structured（S8 重构）
├── local/models.py        # 本地模型管理（S7 重构）
└── viz/                   # 移出核心 → vertai[viz]（S8）
```

**依赖方向规则**（强制）：
- `core/` 不依赖 `scenarios/`/`workflow/`/`viz/`
- `scenarios/` 依赖 `core/` 抽象（ABC/Protocol），不依赖具体适配器实现
- `agent` 依赖 `provider` + `tool`
- `retriever` 依赖 `vector` + `embedding`
- `knowledge_qa` 依赖 `retriever` + `provider`

## 3. 核心抽象契约

### 3.1 LLMProvider（S2 实现）

LLM 调用的统一抽象。每 provider 一个适配器，支持 sync + async + tool calling + 流式。

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Protocol

class LLMProvider(ABC):
    """LLM 提供者抽象。每个 provider（Ollama/Anthropic/DeepSeek/OpenAI）一个实现。"""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]: ...

    @abstractmethod
    async def agenerate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> GenerateResult: ...

    @abstractmethod
    async def astream(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]: ...
```

**契约要点**：
- `messages` 接受 `ChatMessage`（统一类型，不混用 dict）
- `tools` 参数原生支持 function calling；返回 `GenerateResult` 含 `tool_calls`
- sync 与 async 双 API，async 是真实 `httpx.AsyncClient`（非同步包装）
- 流式返回 `StreamEvent`（text_delta / tool_use / done），非裸 str

### 3.2 EmbeddingProvider（S3 实现）

嵌入向量的独立抽象。真实默认，移除 hash 随机。

```python
class EmbeddingProvider(ABC):
    """嵌入提供者抽象。独立于 VectorStore，可替换/注入。"""

    @abstractmethod
    def embed(self, texts: str | list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def aembed(self, texts: str | list[str]) -> list[list[float]]: ...

    @property
    @abstractmethod
    def dimension(self) -> int: ...
```

**契约要点**：
- 无 `embedding_fn` 注入时，`VectorEngine` 显式抛错，**不静默用随机向量**
- 真实默认：本地 `SentenceTransformerProvider`（需 `[embeddings]`），云端 `OpenAIEmbeddingProvider`
- `dimension` 暴露维度，供 VectorStore 初始化

### 3.3 VectorStore（S3 重构）

向量存储抽象。修 C2/C3，删除 hash 随机默认与 FAISS 删除不一致。

```python
class VectorStore(ABC):
    """向量存储抽象。InMemory/Chroma/FAISS 后端。"""

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None: ...

    @abstractmethod
    def search(
        self, query_embedding: list[float], *, top_k: int = 5
    ) -> list[SearchResult]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...  # 真实删除，count/search 一致

    @abstractmethod
    def count(self) -> int: ...
```

**契约要点**：
- `delete` 真实删除，`count`/`search` 必须一致（修 C3）
- 后端选择 `auto` 必须诚实（FAISS 不能从不被自动选中）

### 3.4 Retriever（S3 实现）

检索抽象，可扩展 reranking/query transform。

```python
class Retriever(ABC):
    """检索抽象。组合 EmbeddingProvider + VectorStore，可扩展 reranking。"""

    @abstractmethod
    def retrieve(
        self, query: str, *, top_k: int = 5
    ) -> list[SearchResult]: ...

    @abstractmethod
    async def aretrieve(
        self, query: str, *, top_k: int = 5
    ) -> list[SearchResult]: ...
```

**契约要点**：
- `KnowledgeQA` 依赖 `Retriever` 抽象，不再直接调 `vector.search`
- reranking/query transform 作为 Retriever 装饰器或子类扩展（1.0 提供可扩展点，高级实现在 1.x）

### 3.5 TextSplitter（S3 实现）

文本分块抽象。替换 knowledge_qa 硬编码 chunking。

```python
class TextSplitter(ABC):
    """文本分块抽象。递归/固定/语义策略。"""

    @abstractmethod
    def split(self, text: str) -> list[str]: ...
```

**内置实现**：`RecursiveTextSplitter`（默认，按分隔符层级递归）、`FixedLengthSplitter`（固定长度+overlap）、`SemanticTextSplitter`（语义边界，1.x）。

### 3.6 Tool（S4 实现）

工具调用抽象。Agent 的核心能力。**对标 OpenAI Agents SDK `@function_tool` 行业标准（2025-2026）。**

```python
class Tool(ABC):
    """工具抽象。name/description/parameters schema/execute。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:  # JSON Schema
        ...

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any: ...

    @abstractmethod
    async def aexecute(self, **kwargs: Any) -> Any: ...


class FunctionTool(Tool):
    """从 Python 函数自动生成的工具（@tool 装饰器产物）。"""


class ToolRegistry:
    """工具注册表。@tool 装饰器自动注册。"""
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def to_specs(self) -> list[ToolSpec]: ...  # 传给 LLMProvider


def tool(
    name: str | None = None,
    description: str | None = None,
    *,
    timeout: float | None = None,
    failure_error_function: Callable[[Exception], str] | None = ...,
) -> Callable: ...  # 装饰器
```

**契约要点（对标 OpenAI Agents SDK）：**
- `@tool` 装饰器从函数签名（inspect）+ docstring（解析，支持 google/numpy/sphinx，best-effort）+ pydantic 自动生成 JSON Schema（含类型/约束/描述）
- **Pydantic Field 约束支持**：`arg: Annotated[int, Field(ge=0, le=100, description="...")]` 和默认式 `arg: int = Field(..., ge=0)`，生成 schema 含约束
- `execute`/`aexecute` sync + async
- **timeout（per-tool）**：超时行为 error_as_result（默认，给 LLM 超时消息恢复）/ raise_exception
- **failure_error_function**：tool 失败给 LLM 友好消息（默认）/ 自定义 / None（re-raise）
- `to_specs()` 输出 `ToolSpec` 传给 `LLMProvider.generate(tools=...)`
- 内置工具：`web_search`/`file_read`/`file_write`/`http_request`/`calculator`（开箱即用，覆盖垂直领域常见集成需求）

**1.0 不实现（诚实标注 1.x）**：`tool_namespace` 分组、`defer_loading`/ToolSearch（大量工具）、`is_enabled` 条件启用、`agents-as-tools`、`needs_approval`（Human-in-the-loop）、`custom_output_extractor`。1.0 Tool 抽象预留这些扩展点但不实现。

### 3.7 Agent（S5 实现）

tool-calling agent loop。agent SDK 分水岭。

```python
class Agent:
    """最小可用 tool-calling agent。LLMProvider + Tools + 终止条件。"""

    def __init__(
        self,
        provider: LLMProvider,
        tools: list[Tool] | None = None,
        *,
        system_prompt: str | None = None,
        max_iterations: int = 10,
        callbacks: list[Callback] | None = None,
    ) -> None: ...

    def run(self, input: str) -> AgentResult: ...

    async def arun(self, input: str) -> AgentResult: ...
```

**契约要点**：
- loop：generate(tools)→ 若 tool_calls 则 execute→ 结果回传→ 再 generate→ 直至无 tool_call 或 max_iterations
- `max_iterations` 防失控（必填，有界）
- `AgentResult` 含最终输出、tool 调用历史、token/延迟统计
- 1.0 最小可用（单 agent + tool calling），不追 multi-agent/handoff（1.x）

### 3.8 Callbacks（S5 实现）

轻量可观测性钩子。1.0 仅事件钩子，OTel/tracing 全家桶在 1.x。

```python
class Callback(Protocol):
    """可观测性钩子协议。"""
    def on_agent_start(self, input: str) -> None: ...
    def on_llm_start(self, messages: list[ChatMessage]) -> None: ...
    def on_llm_end(self, result: GenerateResult) -> None: ...
    def on_tool_start(self, tool_name: str, args: dict) -> None: ...
    def on_tool_end(self, tool_name: str, result: Any) -> None: ...
    def on_agent_end(self, result: AgentResult) -> None: ...
```

**契约要点**：Protocol（鸭子类型，非强制继承），用户实现需要的钩子即可。

### 3.9 Memory（S9 重构）

会话记忆。修原子写/真实 tokenizer/uuid/路径遍历。

```python
class SessionCorruptedError(ValueError):
    """Raised when a persisted session file cannot be parsed. Carries the path."""

class SessionMemory:
    """会话记忆。原子持久化，真实 token 估算。"""
    def add_message(self, role: str, content: str) -> None: ...
    def get_history(self) -> list[Message]: ...
    def save(self, path: str | Path | None = None) -> Path: ...  # 原子写 tmp+os.replace
    def load(self, path: str | Path | None = None) -> None: ...  # in-place; 损坏抛 SessionCorruptedError
    @classmethod
    def from_file(cls, path: str | Path) -> SessionMemory: ...  # 构造 + load
```

**契约要点**：
- `session_id` 白名单校验 `^[a-zA-Z0-9_-]+$`（构造时 + save 时双重校验，防路径遍历；空字符串显式拒绝而非静默自动生成）
- `save` 原子写（同目录 tmp 文件 + `os.fsync` + `os.replace`；失败清理 tmp，不污染原文件）
- token 估算：`tiktoken` 可选（cl100k_base），否则语言感知启发式（CJK/非 ASCII 字符按 1 token，ASCII 按 ~4 字符 1 token）——替换 `len//4`（对中文低估 3-4x）
- `_generate_session_id` 用 `uuid4`（消除同毫秒冲突）
- `load` 损坏文件抛 `SessionCorruptedError`（携带路径），不抛裸 `json.JSONDecodeError`/`KeyError`
- `_trim_if_needed` 保留首条 system prompt（避免裁剪改变 agent 行为），保证至少 1 条消息

## 4. 异步模型

- **async-first**：Provider/Agent/Retriever/Tool 的主接口含 async 版本
- async 是真实 `httpx.AsyncClient`，非同步包装
- sync 接口保留为兼容层（内部独立 `httpx.Client`，非 `asyncio.run` 包装以免嵌套事件循环问题）
- 流式：sync `Iterator[StreamEvent]` + async `AsyncIterator[StreamEvent]`

## 5. 配置与依赖

- **硬依赖**：`httpx` + `pydantic`（核心轻量）
- **可选 extras**：`[embeddings]`(sentence-transformers)、`[doc-parser]`(PyMuPDF/docx/openpyxl/pptx)、`[viz]`(dashboard 移出后)、`[production]`
- **Pydantic v2 配置**：所有 Config 用 `pydantic.BaseModel` + `extra="forbid"` + `model_validator(mode="before")` 注入环境变量（非 `__init__` 重写反模式）
- **Provider 注册**：工厂函数 + 注册表，不硬编码 provider 路由

## 6. 类型系统

- **mypy --strict 零错误**：所有公开 API 严格类型
- **Protocol** 用于鸭子类型抽象（Callback、EmbeddingLike）
- **ABC** 用于需要共享实现的抽象（LLMProvider/VectorStore/Tool）
- **泛型 TypeVar** 用于可复用容器
- **不声明 `Typing::Typed` 直到 mypy strict 真实零错误**（S10 才升回）

## 7. 国际化

- 文档英文为主，中文为辅（双语可选）
- 错误消息统一英文（当前 workflow 中文 vs dashboard 英文不一致）
- 代码 docstring 英文为主
- 示例数据可双语

## 8. 不在 1.0 范围（诚实标注）

以下为 1.x 后置增强，1.0 不实现：
- LLM 响应缓存、成本/限流控制、Human-in-the-loop、多模态输入、数据清洗
- 多 Agent 协作（agents-as-tools/handoff）、可观测性增强（OTel/tracing 全家桶）
- 高级检索（HyDE/multi-query/reranker 模型）——1.0 Retriever 提供可扩展点

1.0 的 Retriever/Callbacks 等抽象预留扩展点，但具体高级实现在 1.x。
