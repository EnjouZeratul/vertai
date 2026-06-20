"""Local model management.

Download, cache and load small local models (Whisper speech-to-text and
sentence-transformers embeddings). No cloud API is required.

Supported model types:
  1. Speech-to-text (Whisper) - needs >= 1GB RAM.
  2. Sentence embedding (sentence-transformers) - needs >= 0.5GB RAM.

Network: the first load of a model downloads weights through the underlying
library (``openai-whisper`` / ``sentence-transformers``), which reads weights
from Hugging Face / OpenAI's public blob storage. The ``download_url`` field on
each :class:`NetworkRequirements` is informational metadata (the canonical
source of the weights); it is not used to fetch bytes directly.

Mirror support: when ``use_mirror`` is enabled and a mirror is configured, the
selected mirror endpoint is exported via the ``HF_ENDPOINT`` environment
variable, which ``huggingface_hub`` and ``sentence-transformers`` read at
download time. This is the documented way to redirect Hugging Face downloads
(e.g. to ``https://hf-mirror.com`` for users behind the GFW).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Default model cache directory. The directory is created lazily on first
# download, never at construction time (the constructor must be side-effect
# free so that merely inspecting model metadata does not touch the filesystem).
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "vertai" / "models"


class ModelCategory(Enum):
    """Model category."""

    SPEECH_TO_TEXT = "speech_to_text"
    EMBEDDING = "embedding"
    IMAGE = "image"
    TEXT_GENERATION = "text_generation"


@dataclass
class HardwareRequirements:
    """Hardware requirements for a model."""

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
    """Network requirements for downloading a model.

    ``download_url`` is informational metadata pointing at the canonical source
    of the weights (Hugging Face model repo or OpenAI's public mirror). The
    actual bytes are fetched by the underlying library (``openai-whisper`` /
    ``sentence-transformers``) using their own resolution logic; this URL is not
    used to fetch bytes directly. Mirrors are applied by exporting
    ``HF_ENDPOINT`` for ``huggingface_hub``-based downloads.
    """

    download_size_mb: float
    download_url: str
    mirrors: list[str] = field(default_factory=list)
    estimated_download_time_minutes: Optional[float] = None  # at ~10Mbps

    def to_dict(self) -> dict[str, Any]:
        return {
            "download_size_mb": self.download_size_mb,
            "download_url": self.download_url,
            "mirrors": self.mirrors,
            "estimated_download_time_minutes": self.estimated_download_time_minutes,
        }


@dataclass
class LocalModelInfo:
    """Metadata for a locally-downloadable model.

    Renamed from ``ModelInfo`` (kept in :mod:`vertai.local` as a deprecated
    alias) to distinguish it from :class:`vertai.core.llm.LLMModelInfo` which
    describes cloud LLM models.
    """

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


# Backwards-compatible alias. New code should use :class:`LocalModelInfo` to
# avoid any confusion with :class:`vertai.core.llm.LLMModelInfo`.
ModelInfo = LocalModelInfo


# Predefined model metadata.
#
# ``download_url`` values point at the canonical public source of the weights
# (informational). They are NOT used by this module to fetch bytes; the
# underlying library performs the download.
AVAILABLE_MODELS: dict[str, LocalModelInfo] = {
    # --- Whisper speech-to-text models ---
    "whisper-tiny": LocalModelInfo(
        name="whisper-tiny",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper tiny model. Fast, lower accuracy.",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=1.0,
            recommended_ram_gb=2.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=75,
            download_url="https://huggingface.co/openai/whisper-tiny",
            estimated_download_time_minutes=1,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "fast", "cpu-friendly"],
        license="MIT",
    ),
    "whisper-base": LocalModelInfo(
        name="whisper-base",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper base model. Balanced speed and accuracy.",
        version="20231117",
        hardware=HardwareRequirements(
            min_ram_gb=1.5,
            recommended_ram_gb=3.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=142,
            download_url="https://huggingface.co/openai/whisper-base",
            estimated_download_time_minutes=2,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "balanced"],
        license="MIT",
    ),
    "whisper-small": LocalModelInfo(
        name="whisper-small",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper small model. Higher accuracy.",
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
            download_url="https://huggingface.co/openai/whisper-small",
            estimated_download_time_minutes=8,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "accurate"],
        license="MIT",
    ),
    "whisper-medium": LocalModelInfo(
        name="whisper-medium",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper medium model. High accuracy.",
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
            download_url="https://huggingface.co/openai/whisper-medium",
            estimated_download_time_minutes=25,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "high-accuracy"],
        license="MIT",
    ),
    "whisper-large-v3": LocalModelInfo(
        name="whisper-large-v3",
        category=ModelCategory.SPEECH_TO_TEXT,
        description="OpenAI Whisper large-v3 model. Best accuracy.",
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
            download_url="https://huggingface.co/openai/whisper-large-v3",
            estimated_download_time_minutes=48,
        ),
        languages=["zh", "en", "ja", "ko", "es", "fr", "de", "ru", "auto"],
        tags=["speech", "transcription", "best-accuracy", "gpu-recommended"],
        license="MIT",
    ),

    # --- Sentence embedding models ---
    "all-MiniLM-L6-v2": LocalModelInfo(
        name="all-MiniLM-L6-v2",
        category=ModelCategory.EMBEDDING,
        description="Lightweight sentence embedding model for general semantic search.",
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
    "paraphrase-multilingual-MiniLM-L12-v2": LocalModelInfo(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        category=ModelCategory.EMBEDDING,
        description="Multilingual sentence embedding model with Chinese support.",
        version="1.0",
        hardware=HardwareRequirements(
            min_ram_gb=1.0,
            recommended_ram_gb=2.0,
            supports_cpu=True,
        ),
        network=NetworkRequirements(
            download_size_mb=420,
            download_url=(
                "https://huggingface.co/"
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ),
            mirrors=[
                "https://hf-mirror.com/"
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            ],
            estimated_download_time_minutes=7,
        ),
        languages=["zh", "en", "es", "fr", "de", "it", "nl", "pt", "pl", "ru"],
        tags=["embedding", "semantic-search", "multilingual", "chinese"],
        license="Apache-2.0",
    ),
    "text-embedding-3-small": LocalModelInfo(
        name="text-embedding-3-small",
        category=ModelCategory.EMBEDDING,
        description="Chinese-optimized embedding model (text2vec-base-chinese).",
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
    "bge-small-zh-v1.5": LocalModelInfo(
        name="bge-small-zh-v1.5",
        category=ModelCategory.EMBEDDING,
        description="BGE Chinese small embedding model. High performance.",
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
    "bge-large-zh-v1.5": LocalModelInfo(
        name="bge-large-zh-v1.5",
        category=ModelCategory.EMBEDDING,
        description="BGE Chinese large embedding model. Best performance.",
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
    """Local model manager configuration."""

    model_name: str
    cache_dir: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    download_timeout: int = 3600  # seconds
    use_mirror: bool = True  # export HF_ENDPOINT for the configured mirror

    def __post_init__(self) -> None:
        if self.cache_dir is None:
            self.cache_dir = str(DEFAULT_CACHE_DIR)


class LocalModelManager:
    """Local model manager.

    Downloads, loads, caches and unloads local models. The cache directory is
    created lazily on first download; the constructor performs no filesystem
    I/O so merely instantiating the manager (e.g. to inspect metadata) is
    side-effect free.

    Example:
        >>> manager = LocalModelManager()
        >>> manager.list_models(ModelCategory.SPEECH_TO_TEXT)
        >>> manager.download("whisper-small")
        >>> model = manager.load("whisper-small")
        >>> result = model.transcribe("audio.mp3")
    """

    def __init__(self, config: Optional[LocalModelConfig] = None) -> None:
        self.config = config or LocalModelConfig(model_name="")
        self._loaded_models: dict[str, Any] = {}
        self._cache_dir = (
            Path(self.config.cache_dir)
            if self.config.cache_dir is not None
            else DEFAULT_CACHE_DIR
        )
        # NOTE: deliberately no mkdir here — construction is side-effect free.

    def list_models(
        self,
        category: Optional[ModelCategory] = None,
        language: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> list[LocalModelInfo]:
        """List available models, optionally filtered.

        Args:
            category: Filter by model category.
            language: Filter by supported language.
            tag: Filter by tag.

        Returns:
            List of matching model metadata.
        """
        models = list(AVAILABLE_MODELS.values())

        if category:
            models = [m for m in models if m.category == category]

        if language:
            models = [m for m in models if language in m.languages or "auto" in m.languages]

        if tag:
            models = [m for m in models if tag in m.tags]

        return models

    def get_model_info(self, model_name: str) -> Optional[LocalModelInfo]:
        """Get metadata for a named model, or ``None`` if unknown."""
        return AVAILABLE_MODELS.get(model_name)

    def is_downloaded(self, model_name: str) -> bool:
        """Return True if a model appears to be present in the cache."""
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
        """Download a model into the cache.

        Args:
            model_name: Model name (must be in :data:`AVAILABLE_MODELS`).
            progress_callback: Called with progress in ``[0.0, 1.0]``.

        Returns:
            True on success, False on failure.
        """
        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")

        # Lazily create the cache directory on first download only.
        model_dir = self._cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Downloading model %s...", model_name)
        logger.info("  size: %.0f MB", model_info.network.download_size_mb)
        logger.info("  category: %s", model_info.category.value)
        logger.info("  languages: %s", ", ".join(model_info.languages))

        # Dispatch to the category-specific downloader. Each downloader is
        # responsible for its own transient-failure handling (returns False on
        # runtime errors, re-raises ImportError for missing optional deps so
        # the caller gets an actionable configuration error instead of a
        # misleading "download failed" message).
        if model_info.category == ModelCategory.SPEECH_TO_TEXT:
            return self._download_whisper(model_name, progress_callback)
        elif model_info.category == ModelCategory.EMBEDDING:
            return self._download_embedding(model_name, model_info, progress_callback)
        else:
            logger.warning(
                "Download for category %s is not supported.",
                model_info.category.value,
            )
            return False

    def _download_whisper(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download a Whisper model via ``openai-whisper``'s own loader."""
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required: pip install openai-whisper "
                "(or faster-whisper: pip install faster-whisper)"
            ) from exc

        whisper_name = model_name.replace("whisper-", "")

        try:
            # whisper.load_model performs the download using its own URL
            # resolution; the bytes are cached under download_root.
            whisper.load_model(
                whisper_name,
                download_root=str(self._cache_dir),
            )

            if progress_callback:
                progress_callback(1.0)

            logger.info("Model %s downloaded.", model_name)
            return True
        except Exception as e:
            logger.error("Whisper download failed: %s", e)
            return False

    def _download_embedding(
        self,
        model_name: str,
        model_info: LocalModelInfo,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download an embedding model via ``sentence-transformers``.

        When ``use_mirror`` is enabled and the model declares mirrors, the
        first mirror is exported as ``HF_ENDPOINT`` so ``huggingface_hub``
        (used internally by sentence-transformers) actually fetches from the
        mirror instead of the default endpoint.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        mirror_endpoint = self._select_mirror(model_info)

        try:
            with _export_hf_endpoint(mirror_endpoint):
                SentenceTransformer(
                    model_name,
                    cache_folder=str(self._cache_dir),
                )

            if progress_callback:
                progress_callback(1.0)

            logger.info("Model %s downloaded.", model_name)
            return True
        except Exception as e:
            logger.error("Embedding download failed: %s", e)
            return False

    def _select_mirror(self, model_info: LocalModelInfo) -> Optional[str]:
        """Pick the HF mirror endpoint to use, or ``None`` for the default."""
        if not self.config.use_mirror:
            return None
        if not model_info.network.mirrors:
            return None
        return model_info.network.mirrors[0]

    def load(self, model_name: str) -> Any:
        """Load (downloading first if needed) and cache a model instance."""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        model_info = self.get_model_info(model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {model_name}")

        if not self.is_downloaded(model_name):
            logger.info("Model %s not cached, downloading...", model_name)
            self.download(model_name)

        logger.info("Loading model %s...", model_name)

        if model_info.category == ModelCategory.SPEECH_TO_TEXT:
            model: Any = self._load_whisper(model_name)
        elif model_info.category == ModelCategory.EMBEDDING:
            model = self._load_embedding(model_name, model_info)
        else:
            raise ValueError(f"Unsupported model category: {model_info.category}")

        self._loaded_models[model_name] = model
        return model

    def _load_whisper(self, model_name: str) -> WhisperModel:
        """Load a Whisper model."""
        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required: pip install openai-whisper"
            ) from exc

        whisper_name = model_name.replace("whisper-", "")
        device = self._get_device()

        model = whisper.load_model(
            whisper_name,
            download_root=str(self._cache_dir),
            device=device,
        )

        return WhisperModel(model, model_name, device)

    def _load_embedding(self, model_name: str, model_info: LocalModelInfo) -> EmbeddingModel:
        """Load an embedding model (applying the configured mirror)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        device = self._get_device()
        mirror_endpoint = self._select_mirror(model_info)

        with _export_hf_endpoint(mirror_endpoint):
            model = SentenceTransformer(
                model_name,
                cache_folder=str(self._cache_dir),
                device=device,
            )

        return EmbeddingModel(model, model_name, device)

    def _get_device(self) -> str:
        """Resolve the compute device. Falls back to ``cpu`` when torch is absent."""
        if self.config.device != "auto":
            return self.config.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    def unload(self, model_name: str) -> None:
        """Unload a model from the in-memory cache."""
        if model_name in self._loaded_models:
            del self._loaded_models[model_name]
            logger.info("Unloaded model %s.", model_name)

    def clear_cache(self, model_name: Optional[str] = None) -> None:
        """Clear the on-disk cache for one model, or all models if ``None``."""
        import shutil

        if model_name:
            model_dir = self._cache_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                logger.info("Cleared cache for model %s.", model_name)
        else:
            if self._cache_dir.exists():
                shutil.rmtree(self._cache_dir)
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model caches.")

    def get_cache_size(self) -> float:
        """Return the cache size in MB (0 if the cache does not exist)."""
        total_size = 0
        if self._cache_dir.exists():
            for path in self._cache_dir.rglob("*"):
                if path.is_file():
                    total_size += path.stat().st_size
        return total_size / (1024 * 1024)


class _export_hf_endpoint:
    """Context manager that temporarily exports ``HF_ENDPOINT``.

    ``huggingface_hub`` reads ``HF_ENDPOINT`` at download time, so setting it
    around the ``SentenceTransformer`` constructor causes the actual download
    to go through the mirror. When ``endpoint`` is ``None`` the environment is
    left untouched.
    """

    def __init__(self, endpoint: Optional[str]) -> None:
        self._endpoint = endpoint
        self._previous: Optional[str] = None
        self._had_previous = False

    def __enter__(self) -> None:
        if self._endpoint is None:
            return
        self._had_previous = "HF_ENDPOINT" in os.environ
        self._previous = os.environ.get("HF_ENDPOINT")
        os.environ["HF_ENDPOINT"] = self._endpoint

    def __exit__(self, *exc: object) -> None:
        if self._endpoint is None:
            return
        if self._had_previous:
            # previous was a real string (not None) since the key existed
            os.environ["HF_ENDPOINT"] = self._previous or ""
        else:
            os.environ.pop("HF_ENDPOINT", None)


class WhisperModel:
    """Whisper speech-to-text model wrapper."""

    def __init__(self, model: Any, model_name: str, device: str) -> None:
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
        """Transcribe an audio file.

        Args:
            audio_path: Path to an audio file (mp3, wav, m4a, ...).
            language: Language code (zh, en, ja, ...). ``None`` auto-detects.
            task: ``transcribe`` keeps the source language; ``translate``
                translates to English.
            verbose: If True, the underlying library prints progress.

        Returns:
            Dict with ``text``, ``segments`` and ``language``.
        """
        logger.info("Transcribing: %s", audio_path)

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
        """Detect the dominant language of an audio file."""
        import whisper

        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self._model.device)
        _, probs = self._model.detect_language(mel)
        detected = max(probs, key=probs.get)
        return str(detected)


class EmbeddingModel:
    """Sentence embedding model wrapper."""

    def __init__(self, model: Any, model_name: str, device: str) -> None:
        self._model = model
        self._model_name = model_name
        self._device = device

    def embed(self, text: str) -> list[float]:
        """Embed a single text into a vector."""
        embedding = self._model.encode(text)
        return embedding.tolist()  # type: ignore[no-any-return]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        embeddings = self._model.encode(texts)
        return [e.tolist() for e in embeddings]

    def similarity(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
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
        """Return ``(doc_index, score)`` pairs for the top-k most similar docs."""
        import numpy as np

        query_embedding = self._model.encode(query)
        doc_embeddings = self._model.encode(documents)

        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(int(i), float(similarities[i])) for i in top_indices]


def check_hardware_requirements(model_name: str) -> dict[str, Any]:
    """Check whether the host hardware satisfies a model's requirements.

    Returns a dict describing current vs required RAM and GPU state. Without
    ``psutil`` installed the function returns a best-effort dict (``satisfied``
    set to ``None``) rather than raising, and without ``torch`` the GPU branch
    is skipped gracefully (no ``None >= float`` TypeError).
    """
    model_info = AVAILABLE_MODELS.get(model_name)
    if not model_info:
        return {"satisfied": False, "error": f"Unknown model: {model_name}"}

    # Current RAM.
    try:
        import psutil

        current_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        return {
            "satisfied": None,
            "error": "psutil is required: pip install psutil",
            "required_ram_gb": model_info.hardware.min_ram_gb,
            "supports_cpu": model_info.hardware.supports_cpu,
        }

    # GPU detection (optional).
    gpu_available = False
    gpu_name: Optional[str] = None
    gpu_vram_gb: Optional[float] = None

    try:
        import torch

        if torch.cuda.is_available():
            gpu_available = True
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except ImportError:
        pass

    ram_satisfied = current_ram_gb >= model_info.hardware.min_ram_gb

    # GPU check only applies when the model requires GPU and cannot run on CPU.
    # Guard against gpu_vram_gb being None (no torch / no CUDA) to avoid
    # ``None >= float`` TypeError.
    gpu_satisfied = True
    if (
        model_info.hardware.min_gpu_vram_gb is not None
        and not model_info.hardware.supports_cpu
    ):
        if not gpu_available or gpu_vram_gb is None:
            gpu_satisfied = False
        else:
            gpu_satisfied = gpu_vram_gb >= model_info.hardware.min_gpu_vram_gb

    return {
        "satisfied": ram_satisfied and gpu_satisfied,
        "current_ram_gb": round(current_ram_gb, 1),
        "required_ram_gb": model_info.hardware.min_ram_gb,
        "recommended_ram_gb": model_info.hardware.recommended_ram_gb,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_vram_gb": round(gpu_vram_gb, 1) if gpu_vram_gb is not None else None,
        "required_gpu_vram_gb": model_info.hardware.min_gpu_vram_gb,
        "supports_cpu": model_info.hardware.supports_cpu,
    }
