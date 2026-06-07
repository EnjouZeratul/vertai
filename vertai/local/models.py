"""本地模型管理

支持本地化小模型的下载、管理和调用。

支持的模型类型：
1. 语音转文字 (Whisper) - 需要至少 2GB 内存
2. 向量嵌入模型 (sentence-transformers) - 需要至少 1GB 内存
3. 图像识别模型 - 需要至少 4GB 内存

网络需求：
- Whisper: 首次下载约 150MB-1.5GB (取决于模型大小)
- sentence-transformers: 首次下载约 100MB-500MB
- 图像模型: 首次下载约 300MB-2GB
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# 默认模型缓存目录
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vertai" / "models"


class ModelCategory(Enum):
    """模型类别"""
    SPEECH_TO_TEXT = "speech_to_text"
    EMBEDDING = "embedding"
    IMAGE = "image"
    TEXT_GENERATION = "text_generation"


@dataclass
class HardwareRequirements:
    """硬件需求"""
    min_ram_gb: float
    recommended_ram_gb: float
    min_gpu_vram_gb: Optional[float] = None
    recommended_gpu_vram_gb: Optional[float] = None
    supports_cpu: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_ram_gb": self.min_ram_gb,
            "recommended_ram_gb": self.recommended_ram_gb,
            "min_gpu_vram_gb": self.min_gpu_vram_gb,
            "recommended_gpu_vram_gb": self.recommended_gpu_vram_gb,
            "supports_cpu": self.supports_cpu,
        }


@dataclass
class NetworkRequirements:
    """网络需求"""
    download_size_mb: float
    download_url: str
    mirrors: list[str] = field(default_factory=list)
    estimated_download_time_minutes: Optional[float] = None  # 基于 10Mbps

    def to_dict(self) -> dict[str, Any]:
        return {
            "download_size_mb": self.download_size_mb,
            "download_url": self.download_url,
            "mirrors": self.mirrors,
            "estimated_download_time_minutes": self.estimated_download_time_minutes,
        }


@dataclass
class ModelInfo:
    """模型信息"""
    name: str
    category: ModelCategory
    description: str
    version: str
    hardware: HardwareRequirements
    network: NetworkRequirements
    languages: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    license: str = "MIT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "hardware": self.hardware.to_dict(),
            "network": self.network.to_dict(),
            "languages": self.languages,
            "tags": self.tags,
            "license": self.license,
        }


# 预定义的模型信息
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    # Whisper 语音转文字模型
    "whisper-tiny": ModelInfo(
        name="whisper-tiny",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper 微型模型，快速但精度较低",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=1.0,
            recommended_ram_gb=2.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=75,
            download_url="https://openaipublic.azureedge.net/whisper/models/65147644a518d12f04e7d6098a2b6a45a7c2e0e0",
            estimated_download_time_minutes=1,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "fast", "cpu-friendly"],
        license="MIT",
    ),
    "whisper-base": ModelInfo(
        name="whisper-base",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper 基础模型，平衡速度和精度",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=1.5,
            recommended_ram_gb=3.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=142,
            download_url="https://openaipublic.azureedge.net/whisper/models/ed3d97b362a9530e7b8d29b1d1e1e1e1e1e1e1e1",
            estimated_download_time_minutes=2,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "balanced"],
        license="MIT",
    ),
    "whisper-small": ModelInfo(
        name="whisper-small",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper 小型模型，较高精度",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=2.0,
            recommended_ram_gb=4.0,
            min_gpu_vram_gb=2.0,
            recommended_gpu_vram_gb=4.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=466,
            download_url="https://openaipublic.azureedge.net/whisper/models/ed3d97b362a9530e7b8d29b1d1e1e1e1e1e1e1e",
            estimated_download_time_minutes=8,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "accurate"],
        license="MIT",
    ),
    "whisper-medium": ModelInfo(
        name="whisper-medium",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper 中型模型，高精度",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=5.0,
            recommended_ram_gb=8.0,
            min_gpu_vram_gb=5.0,
            recommended_gpu_vram_gb=8.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=1500,
            download_url="https://openaipublic.azureedge.net/whisper/models/ed3d97b362a9530e7b8d29b1d1e1e1e1e1e1e1e",
            estimated_download_time_minutes=25,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "high-accuracy"],
        license="MIT",
    ),
    "whisper-large-v3": ModelInfo(
        name="whisper-large-v3",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper 大型模型 v3，最高精度",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=10.0,
            recommended_ram_gb=16.0,
            min_gpu_vram_gb=10.0,
            recommended_gpu_vram_gb=16.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=2900,
            download_url="https://openaipublic.azureedge.net/whisper/models/ed3d97b362a9530e7b8d29b1d1e1e1e1e1e1e1e",
            estimated_download_time_minutes=48,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "best-accuracy", "gpu-recommended"],
        license="MIT",
    ),

    # 向量嵌入模型
    "all-MiniLM-L6-v2": ModelInfo(
        name="all-MiniLM-L6-v2",
        category=ModelCategory.EMBEDDING,
        description="轻量级句子嵌入模型，适合通用语义搜索",
        version="1.0",
        hardware=HardwareRequirements(
            min_ram_gb=0.5,
            recommended_ram_gb=1.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=80,
            download_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
            mirrors=["https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2"],
            estimated_download_time_minutes=1,
        ),
        languages=["en"],
        tags=["embedding", "semantic-search", "lightweight", "cpu-friendly"],
        license="Apache-2.0",
    ),
    "paraphrase-multilingual-MiniLM-L12-v2": ModelInfo(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        category=ModelCategory.EMBEDDING,
        description="多语言句子嵌入模型，支持中文",
        version="1.0",
        hardware=HardwareRequirements(
            min_ram_gb=1.0,
            recommended_ram_gb=2.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=420,
            download_url="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            mirrors=["https://hf-mirror.com/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
            estimated_download_time_minutes=7,
        ),
        languages=["zh", "en", "es", "fr", "de", "it", "nl", "pt", "pl", "ru"],
        tags=["embedding", "semantic-search", "multilingual", "chinese"],
        license="Apache-2.0",
    ),
    "text-embedding-3-small": ModelInfo(
        name="text-embedding-3-small",
        category=ModelCategory.EMBEDDING,
        description="中文优化的嵌入模型",
        version="1.0",
        hardware=HardwareRequirements(
            min_ram_gb=1.0,
            recommended_ram_gb=2.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=200,
            download_url="https://huggingface.co/shibing624/text2vec-base-chinese",
            mirrors=["https://hf-mirror.com/shibing624/text2vec-base-chinese"],
            estimated_download_time_minutes=3,
        ),
        languages=["zh"],
        tags=["embedding", "chinese", "semantic-search"],
        license="Apache-2.0",
    ),
    "bge-small-zh-v1.5": ModelInfo(
        name="bge-small-zh-v1.5",
        category=ModelCategory.EMBEDDING,
        description="BGE 中文小型嵌入模型，高性能",
        version="1.5",
        hardware=HardwareRequirements(
            min_ram_gb=0.5,
            recommended_ram_gb=1.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=100,
            download_url="https://huggingface.co/BAAI/bge-small-zh-v1.5",
            mirrors=["https://hf-mirror.com/BAAI/bge-small-zh-v1.5"],
            estimated_download_time_minutes=2,
        ),
        languages=["zh"],
        tags=["embedding", "chinese", "bge", "high-performance"],
        license="Apache-2.0",
    ),
    "bge-large-zh-v1.5": ModelInfo(
        name="bge-large-zh-v1.5",
        category=ModelCategory.EMBEDDING,
        description="BGE 中文大型嵌入模型，最高性能",
        version="1.5",
        hardware=HardwareRequirements(
            min_ram_gb=2.0,
            recommended_ram_gb=4.0,
            min_gpu_vram_gb=2.0,
            recommended_gpu_vram_gb=4.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=650,
            download_url="https://huggingface.co/BAAI/bge-large-zh-v1.5",
            mirrors=["https://hf-mirror.com/BAAI/bge-large-zh-v1.5"],
            estimated_download_time_minutes=11,
        ),
        languages=["zh"],
        tags=["embedding", "chinese", "bge", "best-performance"],
        license="Apache-2.0",
    ),
}


@dataclass
class LocalModelConfig:
    """本地模型配置"""
    model_name: str
    cache_dir: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    download_timeout: int = 3600  # 秒
    use_mirror: bool = True  # 使用国内镜像

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = str(DEFAULT_CACHE_DIR)


class LocalModelManager:
    """本地模型管理器

    管理本地模型的下载、加载和调用。

    示例:
        >>> manager = LocalModelManager()
        >>>
        >>> # 列出可用模型
        >>> manager.list_models(ModelCategory.SPEECH_TO_TEXT)
        >>>
        >>> # 下载模型
        >>> manager.download("whisper-small")
        >>>
        >>> # 加载模型
        >>> model = manager.load("whisper-small")
        >>> result = model.transcribe("audio.mp3")
    """

    def __init__(self, config: Optional[LocalModelConfig] = None):
        self.config = config or LocalModelConfig(model_name="")
        self._loaded_models: dict[str, Any] = {}
        self._cache_dir = Path(self.config.cache_dir) if self.config else DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        language: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[ModelInfo]:
        """列出可用模型

        Args:
            category: 按类别过滤
            language: 按支持语言过滤
            tag: 按标签过滤

        Returns:
            模型信息列表
        """
        models = list(AVAILABLE_MODELS.values())

        if category:
            models = [m for m in models if m.category == category]

        if language:
            models = [m for m in models if language in m.languages or "auto" in m.languages]

        if tag:
            models = [m for m in models if tag in m.tags]

        return models

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        return AVAILABLE_MODELS.get(model_name)

    def is_downloaded(self, model_name: str) -> bool:
        """检查模型是否已下载"""
        model_info = self.get_model_info(model_name)
        if not model_info:
            return False

        model_dir = self._cache_dir / model_name
        return model_dir.exists() and any(model_dir.iterdir())

    def download(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """下载模型

        Args:
            model_name: 模型名称
            progress_callback: 进度回调函数 (0.0 - 1.0)

        Returns:
            是否成功
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"未知模型: {model_name}")

        model_dir = self._cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始下载模型 {model_name}...")
        logger.info(f"  大小: {model_info.network.download_size_mb:.0f} MB")
        logger.info(f"  类别: {model_info.category.value}")
        logger.info(f"  语言: {', '.join(model_info.languages)}")

        try:
            if model_info.category == ModelCategory.SPEECH_TO_TEXT:
                return self._download_whisper(model_name, model_info, progress_callback)
            elif model_info.category == ModelCategory.EMBEDDING:
                return self._download_embedding(model_name, model_info, progress_callback)
            else:
                logger.warning(f"暂不支持下载 {model_info.category.value} 类型模型")
                return False
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return False

    def _download_whisper(
        self,
        model_name: str,
        model_info: ModelInfo,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """下载 Whisper 模型"""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "需要安装 openai-whisper: pip install openai-whisper\n"
                "或使用 faster-whisper: pip install faster-whisper"
            )

        whisper_name = model_name.replace("whisper-", "")

        try:
            # whisper.load_model 会自动下载
            model = whisper.load_model(
                whisper_name,
                download_root=str(self._cache_dir),
            )

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"模型 {model_name} 下载完成")
            return True
        except Exception as e:
            logger.error(f"Whisper 模型下载失败: {e}")
            return False

    def _download_embedding(
        self,
        model_name: str,
        model_info: ModelInfo,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """下载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "需要安装 sentence-transformers: pip install sentence-transformers"
            )

        try:
            # 使用镜像
            model_url = model_info.network.download_url
            if self.config.use_mirror and model_info.network.mirrors:
                model_url = model_info.network.mirrors[0]

            model = SentenceTransformer(
                model_name,
                cache_folder=str(self._cache_dir),
            )

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"模型 {model_name} 下载完成")
            return True
        except Exception as e:
            logger.error(f"嵌入模型下载失败: {e}")
            return False

    def load(self, model_name: str) -> Any:
        """加载模型

        Args:
            model_name: 模型名称

        Returns:
            模型实例
        """
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"未知模型: {model_name}")

        if not self.is_downloaded(model_name):
            logger.info(f"模型 {model_name} 未下载，开始下载...")
            self.download(model_name)

        logger.info(f"加载模型 {model_name}...")

        if model_info.category == ModelCategory.SPEECH_TO_TEXT:
            model = self._load_whisper(model_name)
        elif model_info.category == ModelCategory.EMBEDDING:
            model = self._load_embedding(model_name)
        else:
            raise ValueError(f"不支持的模型类别: {model_info.category}")

        self._loaded_models[model_name] = model
        return model

    def _load_whisper(self, model_name: str) -> "WhisperModel":
        """加载 Whisper 模型"""
        try:
            import whisper
        except ImportError:
            raise ImportError("需要安装 openai-whisper: pip install openai-whisper")

        whisper_name = model_name.replace("whisper-", "")
        device = self._get_device()

        model = whisper.load_model(
            whisper_name,
            download_root=str(self._cache_dir),
            device=device,
        )

        return WhisperModel(model, model_name, device)

    def _load_embedding(self, model_name: str) -> "EmbeddingModel":
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("需要安装 sentence-transformers: pip install sentence-transformers")

        device = self._get_device()

        model = SentenceTransformer(
            model_name,
            cache_folder=str(self._cache_dir),
            device=device,
        )

        return EmbeddingModel(model, model_name, device)

    def _get_device(self) -> str:
        """获取计算设备"""
        if self.config.device != "auto":
            return self.config.device

        # 自动检测
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def unload(self, model_name: str) -> None:
        """卸载模型，释放内存"""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info(f"已卸载模型 {model_name}")

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """清除缓存

        Args:
            model_name: 指定模型，None 表示清除所有
        """
        import shutil

        if model_name:
            model_dir = self._cache_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info(f"已清除模型 {model_name} 缓存")
        else:
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("已清除所有模型缓存")

    def get_cache_size(self) -> float:
        """获取缓存大小 (MB)"""
        total_size = 0
        if self._cache_dir.exists():
            for path in self._cache_dir.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
        return total_size / (1024 * 1024)


class WhisperModel:
    """Whisper 语音转文字模型"""

    def __init__(self, model: Any, model_name: str, device: str):
        self._model = model
        self._model_name = model_name
        self._device = device

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",  # transcribe, translate
        verbose: bool = False,
    ) -> dict[str, Any]:
        """转录音频

        Args:
            audio_path: 音频文件路径 (支持 mp3, wav, m4a 等)
            language: 语言代码 (zh, en, ja 等)，None 表示自动检测
            task: transcribe 转录为原语言，translate 翻译为英文
            verbose: 是否输出详细信息

        Returns:
            包含 text, segments, language 等字段的字典
        """
        logger.info(f"开始转录: {audio_path}")

        result = self._model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=verbose,
        )

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown"),
        }

    def detect_language(self, audio_path: str) -> str:
        """检测音频语言"""
        import whisper
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self._model.device)
        _, probs = self._model.detect_language(mel)
        return max(probs, key=probs.get)


class EmbeddingModel:
    """向量嵌入模型"""

    def __init__(self, model: Any, model_name: str, device: str):
        self._model = model
        self._model_name = model_name
        self._device = device

    def embed(self, text: str) -> list[float]:
        """生成文本嵌入向量"""
        embedding = self._model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成嵌入向量"""
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    def similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        import numpy as np
        embeddings = self._model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """在文档中搜索与查询最相似的

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前 k 个结果

        Returns:
            (文档索引, 相似度分数) 的列表
        """
        import numpy as np

        query_embedding = self._model.encode(query)
        doc_embeddings = self._model.encode(documents)

        # 计算余弦相似度
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取 top_k 索引
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(i), float(similarities[i])) for i in top_indices]


def check_hardware_requirements(model_name: str) -> dict[str, Any]:
    """检查硬件是否满足模型需求

    Returns:
        {
            "satisfied": bool,
            "current_ram_gb": float,
            "required_ram_gb": float,
            "gpu_available": bool,
            "gpu_name": str | None,
            "gpu_vram_gb": float | None,
        }
    """
    model_info = AVAILABLE_MODELS.get(model_name)
    if not model_info:
        return {"satisfied": False, "error": f"未知模型: {model_name}"}

    # 获取当前内存
    try:
        import psutil
        current_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # 如果 psutil 未安装，返回基本信息
        return {
            "satisfied": None,
            "error": "需要安装 psutil: pip install psutil",
            "required_ram_gb": model_info.hardware.min_ram_gb,
            "supports_cpu": model_info.hardware.supports_cpu,
        }

    # 检查 GPU
    gpu_available = False
    gpu_name = None
    gpu_vram_gb = None

    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    # 判断是否满足需求
    ram_satisfied = current_ram_gb >= model_info.hardware.min_ram_gb
    gpu_satisfied = True

    if model_info.hardware.min_gpu_vram_gb and not model_info.hardware.supports_cpu:
        gpu_satisfied = gpu_available and gpu_vram_gb >= model_info.hardware.min_gpu_vram_gb

    return {
        "satisfied": ram_satisfied and gpu_satisfied,
        "current_ram_gb": round(current_ram_gb, 1),
        "required_ram_gb": model_info.hardware.min_ram_gb,
        "recommended_ram_gb": model_info.hardware.recommended_ram_gb,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_vram_gb, 1) if gpu_vram_gb else None,
        "required_gpu_vram_gb": model_info.hardware.min_gpu_vram_gb,
        "supports_cpu": model_info.hardware.supports_cpu,
    }
