"""Local model module.

Download and run small local models (Whisper, sentence-transformers) without a
cloud API.
"""

from vertai.local.models import (
    EmbeddingModel,
    LocalModelConfig,
    LocalModelInfo,
    LocalModelManager,
    ModelCategory,
    ModelInfo,
    WhisperModel,
)

__all__ = [
    "LocalModelManager",
    "LocalModelConfig",
    "LocalModelInfo",
    "ModelCategory",
    "ModelInfo",
    "WhisperModel",
    "EmbeddingModel",
]