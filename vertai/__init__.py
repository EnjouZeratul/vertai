"""VertAI - A local-first lightweight AI agent SDK for vertical-domain rapid development.

NOTE: Alpha. Core abstractions are implemented and the public API is consolidating,
but it is not yet certified production-ready. See docs/ARCHITECTURE.md for the design
and docs/ROADMAP.md for the path to 1.0.
"""

__version__ = "0.9.9"

from vertai.core.agent import Agent, AgentResult
from vertai.core.callbacks import (
    Callback,
    LoggingCallback,
    TokenCountCallback,
)
from vertai.core.llm import (
    LLMEngine,
    LLMModelInfo,
)
from vertai.core.provider import (
    ANTHROPIC_API_VERSION,
    AnthropicProvider,
    ChatMessage,
    DeepSeekProvider,
    DoneEvent,
    GenerateResult,
    LLMConfig,
    LLMProvider,
    ModelProvider,
    ModelStatus,
    OllamaProvider,
    OpenAIProvider,
    StreamEvent,
    TextDeltaEvent,
    ToolCall,
    ToolSpec,
    ToolUseEvent,
    create_provider,
)
from vertai.core.memory import (
    Message,
    SessionConfig,
    SessionCorruptedError,
    SessionMemory,
)
from vertai.core.embedding import (
    EmbeddingProvider,
    FunctionEmbeddingProvider,
    LocalSentenceTransformerProvider,
)
from vertai.core.text_splitter import (
    FixedLengthSplitter,
    RecursiveTextSplitter,
    TextSplitter,
)
from vertai.core.retriever import (
    Retriever,
    VectorRetriever,
)
from vertai.core.tool import (
    FunctionTool,
    Tool,
    ToolError,
    ToolRegistry,
    ToolTimeoutError,
    tool,
)
from vertai.core.tools import (
    calculator,
    file_read,
    file_write,
    http_request,
    web_search,
)
from vertai.core.vector import (
    Document,
    VectorEngine,
    VectorConfig,
    VectorStore,
    InMemoryVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    SearchResult,
)
from vertai.data.parser import DocParser
from vertai.output.structured import StructuredOutput
from vertai.output.docgen import DocGen
from vertai.scenarios.reviewer import (
    Evaluation,
    Reviewer,
    ReviewerConfig,
    ReviewResult,
)
from vertai.scenarios.knowledge_qa import (
    KnowledgeQA,
    KnowledgeQAConfig,
    AnswerResult,
    SourceReference,
)
# Dashboard visualization lives in the optional ``vertai[viz]`` extra and is
# NOT eagerly imported here. Users who want it should either install the extra
# and import directly:
#     from vertai.viz.dashboard import Dashboard, Metric, Chart
# This keeps the core package lightweight (per ARCHITECTURE.md §2 / ROADMAP S8).
from vertai.workflow import (
    Workflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowResult,
    StepResult,
    StepStatus,
    Step,
    StepType,
    ParallelConfig,
    LoopConfig,
    LoopType,
)
from vertai.local import (
    LocalModelManager,
    LocalModelConfig,
    LocalModelInfo,
    ModelCategory,
    ModelInfo,
    WhisperModel,
    EmbeddingModel,
)
from vertai.local.models import check_hardware_requirements

__all__ = [
    # Agent (S5)
    "Agent",
    "AgentResult",
    # Callbacks (S5)
    "Callback",
    "LoggingCallback",
    "TokenCountCallback",
    # LLM provider abstraction (S2)
    "ANTHROPIC_API_VERSION",
    "AnthropicProvider",
    "ChatMessage",
    "DeepSeekProvider",
    "DoneEvent",
    "GenerateResult",
    "LLMConfig",
    "LLMEngine",
    "LLMModelInfo",
    "LLMProvider",
    "ModelProvider",
    "ModelStatus",
    "OllamaProvider",
    "OpenAIProvider",
    "StreamEvent",
    "TextDeltaEvent",
    "ToolCall",
    "ToolSpec",
    "ToolUseEvent",
    "create_provider",
    # Memory
    "Message",
    "SessionConfig",
    "SessionCorruptedError",
    "SessionMemory",
    # Embedding (S3)
    "EmbeddingProvider",
    "FunctionEmbeddingProvider",
    "LocalSentenceTransformerProvider",
    # Text splitter (S3)
    "TextSplitter",
    "RecursiveTextSplitter",
    "FixedLengthSplitter",
    # Retriever (S3)
    "Retriever",
    "VectorRetriever",
    # Tool (S4)
    "FunctionTool",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolTimeoutError",
    "calculator",
    "file_read",
    "file_write",
    "http_request",
    "tool",
    "web_search",
    # Vector (S3)
    "Document",
    "VectorEngine",
    "VectorConfig",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
    "SearchResult",
    # Data
    "DocParser",
    # Output
    "StructuredOutput",
    "DocGen",
    # Scenarios
    "Evaluation",
    "Reviewer",
    "ReviewerConfig",
    "ReviewResult",
    "KnowledgeQA",
    "KnowledgeQAConfig",
    "AnswerResult",
    "SourceReference",
    # Viz: optional extras (vertai[viz]). Import from vertai.viz.dashboard.
    # Workflow
    "Workflow",
    "WorkflowConfig",
    "WorkflowContext",
    "WorkflowResult",
    "StepResult",
    "StepStatus",
    "Step",
    "StepType",
    "ParallelConfig",
    "LoopConfig",
    "LoopType",
    # Local Models
    "LocalModelManager",
    "LocalModelConfig",
    "LocalModelInfo",
    "ModelCategory",
    "ModelInfo",
    "WhisperModel",
    "EmbeddingModel",
    "check_hardware_requirements",
]