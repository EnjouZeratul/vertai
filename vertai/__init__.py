"""VertAI - 垂直领域 AI 智能体开发 SDK | A vertical-domain AI agent development SDK"""

__version__ = "0.1.3"

from vertai.core.llm import (
    LLMEngine,
    LLMConfig,
    ModelProvider,
    ChatMessage,
    GenerateResult,
)
from vertai.core.memory import (
    Message,
    SessionConfig,
    SessionMemory,
)
from vertai.core.vector import (
    Document,
    VectorEngine,
    VectorConfig,
    SearchResult,
)
from vertai.data.parser import DocParser
from vertai.output.structured import StructuredOutput
from vertai.output.docgen import DocGen
from vertai.scenarios.reviewer import Reviewer, ReviewerConfig, ReviewResult
from vertai.scenarios.knowledge_qa import (
    KnowledgeQA,
    KnowledgeQAConfig,
    AnswerResult,
    SourceReference,
)
from vertai.viz.dashboard import (
    Dashboard,
    DashboardTheme,
    Metric,
    Chart,
    ChartType,
    ChartConfig,
)
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
    ModelCategory,
    ModelInfo,
    WhisperModel,
    EmbeddingModel,
)
from vertai.local.models import check_hardware_requirements

__all__ = [
    # LLM
    "LLMEngine",
    "LLMConfig",
    "ModelProvider",
    "ChatMessage",
    "GenerateResult",
    # Memory
    "Message",
    "SessionConfig",
    "SessionMemory",
    # Vector
    "Document",
    "VectorEngine",
    "VectorConfig",
    "SearchResult",
    # Data
    "DocParser",
    # Output
    "StructuredOutput",
    "DocGen",
    # Scenarios
    "Reviewer",
    "ReviewerConfig",
    "ReviewResult",
    "KnowledgeQA",
    "KnowledgeQAConfig",
    "AnswerResult",
    "SourceReference",
    # Viz
    "Dashboard",
    "DashboardTheme",
    "Metric",
    "Chart",
    "ChartType",
    "ChartConfig",
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
    "ModelCategory",
    "ModelInfo",
    "WhisperModel",
    "EmbeddingModel",
    "check_hardware_requirements",
]