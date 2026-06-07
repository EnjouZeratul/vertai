"""本地模型模块

支持本地化小模型的下载和调用，无需云端 API。
"""

from vertai.local.models import (
    LocalModelManager,
    LocalModelConfig,
    ModelCategory,
    ModelInfo,
    WhisperModel,
    EmbeddingModel,
)

__all__ = [
    "LocalModelManager",
    "LocalModelConfig",
    "ModelCategory",
    "ModelInfo",
    "WhisperModel",
    "EmbeddingModel",
]