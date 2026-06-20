"""Tests for the local model manager.

The heavy optional dependencies (``openai-whisper``, ``sentence-transformers``,
``torch``) are typically NOT installed in CI's lightweight environment, so most
of these tests exercise the pure logic (metadata, filtering, cache management,
device fallback, hardware-check guards, mirror endpoint export) without them.

Real download/load integration tests are marked ``@pytest.mark.integration``
and only run when the relevant dependency is importable; otherwise they are
honestly skipped (never faked via a mock that short-circuits the load path).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from vertai.local.models import (
    AVAILABLE_MODELS,
    EmbeddingModel,
    HardwareRequirements,
    LocalModelConfig,
    LocalModelInfo,
    LocalModelManager,
    ModelCategory,
    ModelInfo,
    NetworkRequirements,
    WhisperModel,
    _export_hf_endpoint,
    check_hardware_requirements,
)

integration = pytest.mark.integration


def _has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


HAS_WHISPER = _has_module("whisper")
HAS_ST = _has_module("sentence_transformers")
HAS_TORCH = _has_module("torch")
HAS_PSUTIL = _has_module("psutil")


# ---------------------------------------------------------------------------
# Metadata dataclasses
# ---------------------------------------------------------------------------


class TestModelCategory:
    def test_categories_exist(self) -> None:
        assert ModelCategory.SPEECH_TO_TEXT.value == "speech_to_text"
        assert ModelCategory.EMBEDDING.value == "embedding"
        assert ModelCategory.IMAGE.value == "image"
        assert ModelCategory.TEXT_GENERATION.value == "text_generation"


class TestHardwareRequirements:
    def test_basic_requirements(self) -> None:
        hw = HardwareRequirements(min_ram_gb=2.0, recommended_ram_gb=4.0, supports_cpu=True)
        assert hw.min_ram_gb == 2.0
        assert hw.recommended_ram_gb == 4.0
        assert hw.supports_cpu is True
        assert hw.min_gpu_vram_gb is None

    def test_with_gpu_requirements(self) -> None:
        hw = HardwareRequirements(
            min_ram_gb=8.0,
            recommended_ram_gb=16.0,
            min_gpu_vram_gb=6.0,
            recommended_gpu_vram_gb=12.0,
            supports_cpu=False,
        )
        assert hw.min_gpu_vram_gb == 6.0
        assert hw.recommended_gpu_vram_gb == 12.0
        assert hw.supports_cpu is False

    def test_to_dict(self) -> None:
        hw = HardwareRequirements(min_ram_gb=2.0, recommended_ram_gb=4.0)
        d = hw.to_dict()
        assert d["min_ram_gb"] == 2.0
        assert d["recommended_ram_gb"] == 4.0
        assert d["supports_cpu"] is True


class TestNetworkRequirements:
    def test_basic_network(self) -> None:
        net = NetworkRequirements(download_size_mb=150, download_url="https://example.com/m")
        assert net.download_size_mb == 150
        assert net.download_url == "https://example.com/m"
        assert net.mirrors == []

    def test_with_mirrors(self) -> None:
        net = NetworkRequirements(
            download_size_mb=500,
            download_url="https://huggingface.co/model",
            mirrors=["https://hf-mirror.com/model"],
            estimated_download_time_minutes=10,
        )
        assert len(net.mirrors) == 1
        assert net.estimated_download_time_minutes == 10

    def test_to_dict(self) -> None:
        net = NetworkRequirements(download_size_mb=100, download_url="https://example.com/m")
        d = net.to_dict()
        assert d["download_size_mb"] == 100
        assert d["download_url"] == "https://example.com/m"
        assert d["mirrors"] == []


class TestLocalModelInfo:
    def test_whisper_tiny_info(self) -> None:
        info = AVAILABLE_MODELS["whisper-tiny"]
        assert info.name == "whisper-tiny"
        assert info.category == ModelCategory.SPEECH_TO_TEXT
        assert info.hardware.min_ram_gb == 1.0
        assert info.network.download_size_mb == 75
        assert "zh" in info.languages
        assert "cpu-friendly" in info.tags

    def test_embedding_model_info(self) -> None:
        info = AVAILABLE_MODELS["bge-small-zh-v1.5"]
        assert info.category == ModelCategory.EMBEDDING
        assert info.hardware.min_ram_gb == 0.5
        assert "chinese" in info.tags

    def test_to_dict(self) -> None:
        info = AVAILABLE_MODELS["whisper-base"]
        d = info.to_dict()
        assert d["name"] == "whisper-base"
        assert d["category"] == "speech_to_text"
        assert "hardware" in d
        assert "network" in d

    def test_modelinfo_alias_is_localmodelinfo(self) -> None:
        # ModelInfo is the backwards-compatible alias for LocalModelInfo.
        assert ModelInfo is LocalModelInfo


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLocalModelConfig:
    def test_default_config(self) -> None:
        config = LocalModelConfig(model_name="test")
        assert config.model_name == "test"
        assert config.device == "auto"
        assert config.download_timeout == 3600
        assert config.use_mirror is True
        assert config.cache_dir is not None

    def test_custom_config(self) -> None:
        config = LocalModelConfig(
            model_name="whisper-small",
            cache_dir="/custom/cache",
            device="cuda",
            download_timeout=7200,
            use_mirror=False,
        )
        assert config.cache_dir == "/custom/cache"
        assert config.device == "cuda"
        assert config.use_mirror is False


# ---------------------------------------------------------------------------
# Manager: pure logic
# ---------------------------------------------------------------------------


class TestLocalModelManager:
    def test_list_all_models(self) -> None:
        manager = LocalModelManager()
        models = manager.list_models()
        assert len(models) == len(AVAILABLE_MODELS)

    def test_list_by_category(self) -> None:
        manager = LocalModelManager()
        speech_models = manager.list_models(category=ModelCategory.SPEECH_TO_TEXT)
        assert all(m.category == ModelCategory.SPEECH_TO_TEXT for m in speech_models)
        assert len(speech_models) == 5  # tiny, base, small, medium, large-v3

    def test_list_by_language(self) -> None:
        manager = LocalModelManager()
        chinese_models = manager.list_models(language="zh")
        assert all("zh" in m.languages or "auto" in m.languages for m in chinese_models)

    def test_list_by_tag(self) -> None:
        manager = LocalModelManager()
        cpu_models = manager.list_models(tag="cpu-friendly")
        assert all("cpu-friendly" in m.tags for m in cpu_models)

    def test_get_model_info(self) -> None:
        manager = LocalModelManager()
        info = manager.get_model_info("whisper-small")
        assert info is not None
        assert info.name == "whisper-small"

    def test_get_unknown_model_info(self) -> None:
        manager = LocalModelManager()
        assert manager.get_model_info("unknown-model") is None

    def test_is_downloaded_false(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        assert manager.is_downloaded("whisper-tiny") is False

    def test_is_downloaded_true(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        model_dir = tmp_path / "whisper-tiny"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        assert manager.is_downloaded("whisper-tiny") is True

    def test_is_downloaded_unknown_model(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        assert manager.is_downloaded("no-such-model") is False

    def test_get_cache_size(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"x" * 1024 * 1024)  # 1MB
        size = manager.get_cache_size()
        assert size >= 1.0

    def test_get_cache_size_no_dir(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path / "absent"))
        manager = LocalModelManager(config)
        assert manager.get_cache_size() == 0.0

    def test_clear_cache_specific(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        manager.clear_cache("test-model")
        assert not model_dir.exists()

    def test_clear_cache_all(self, tmp_path: Path) -> None:
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        for name in ["model1", "model2"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            (model_dir / "model.bin").touch()
        manager.clear_cache()
        assert len(list(tmp_path.iterdir())) == 0

    def test_unload_model(self) -> None:
        manager = LocalModelManager()
        manager._loaded_models["test-model"] = Mock()
        manager.unload("test-model")
        assert "test-model" not in manager._loaded_models

    def test_unload_not_loaded_is_noop(self) -> None:
        manager = LocalModelManager()
        # Should not raise.
        manager.unload("never-loaded")

    def test_download_unknown_raises(self) -> None:
        manager = LocalModelManager()
        with pytest.raises(ValueError, match="Unknown model"):
            manager.download("no-such-model")


class TestManagerConstructorNoSideEffects:
    """The constructor must NOT touch the filesystem."""

    def test_construction_does_not_create_cache_dir(self, tmp_path: Path) -> None:
        cache = tmp_path / "nested" / "does-not-exist-yet"
        config = LocalModelConfig(model_name="x", cache_dir=str(cache))
        # Construction alone must not create the directory.
        LocalModelManager(config)
        assert not cache.exists()

    def test_default_construction_does_not_create_default_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Redirect HOME so we can observe whether DEFAULT_CACHE_DIR is created.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        LocalModelManager()
        cache = tmp_path / ".cache" / "vertai" / "models"
        assert not cache.exists()


class TestDeviceSelection:
    def test_explicit_device_cpu(self) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x", device="cpu"))
        assert manager._get_device() == "cpu"

    def test_explicit_device_cuda(self) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x", device="cuda"))
        assert manager._get_device() == "cuda"

    def test_auto_falls_back_to_cpu_without_torch(self) -> None:
        # If torch is installed this still returns a real device string; the
        # important assertion is "no exception, returns a str".
        manager = LocalModelManager(LocalModelConfig(model_name="x", device="auto"))
        device = manager._get_device()
        assert isinstance(device, str)
        assert device in {"cpu", "cuda", "mps"}


# ---------------------------------------------------------------------------
# Mirror endpoint export (HF_ENDPOINT)
# ---------------------------------------------------------------------------


class TestMirrorExport:
    def test_select_mirror_disabled(self) -> None:
        manager = LocalModelManager(
            LocalModelConfig(model_name="x", use_mirror=False)
        )
        info = AVAILABLE_MODELS["all-MiniLM-L6-v2"]
        assert manager._select_mirror(info) is None

    def test_select_mirror_no_mirrors(self) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x", use_mirror=True))
        info = AVAILABLE_MODELS["whisper-tiny"]  # no mirrors
        assert manager._select_mirror(info) is None

    def test_select_mirror_picks_first(self) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x", use_mirror=True))
        info = AVAILABLE_MODELS["all-MiniLM-L6-v2"]
        selected = manager._select_mirror(info)
        assert selected == "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2"

    def test_export_hf_endpoint_sets_and_restores(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        endpoint = "https://hf-mirror.com"
        with _export_hf_endpoint(endpoint):
            assert os.environ["HF_ENDPOINT"] == endpoint
        assert "HF_ENDPOINT" not in os.environ

    def test_export_hf_endpoint_restores_previous_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HF_ENDPOINT", "https://pre-existing.example")
        endpoint = "https://hf-mirror.com"
        with _export_hf_endpoint(endpoint):
            assert os.environ["HF_ENDPOINT"] == endpoint
        assert os.environ["HF_ENDPOINT"] == "https://pre-existing.example"

    def test_export_hf_endpoint_none_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
        with _export_hf_endpoint(None):
            assert "HF_ENDPOINT" not in os.environ
        assert "HF_ENDPOINT" not in os.environ


# ---------------------------------------------------------------------------
# Download / load: error paths and real integration
# ---------------------------------------------------------------------------


class TestDownloadErrorPaths:
    def test_download_whisper_missing_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without openai-whisper, download() raises ImportError surfaced up."""
        if HAS_WHISPER:
            pytest.skip("openai-whisper is installed; cannot test ImportError path")

        config = LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        # download() swallows generic Exception and returns False, but ImportError
        # from the missing dependency is re-raised.
        with pytest.raises(ImportError):
            manager.download("whisper-tiny")

    def test_download_embedding_missing_dependency(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        if HAS_ST:
            pytest.skip("sentence-transformers is installed; cannot test ImportError path")

        config = LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        with pytest.raises(ImportError):
            manager.download("all-MiniLM-L6-v2")

    def test_download_creates_cache_dir_lazily(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """download() must create the cache dir (lazy), even if import fails."""
        if HAS_WHISPER:
            pytest.skip("openai-whisper is installed; cannot test ImportError path")

        cache = tmp_path / "will-be-created"
        config = LocalModelConfig(model_name="x", cache_dir=str(cache))
        manager = LocalModelManager(config)
        assert not cache.exists()
        with pytest.raises(ImportError):
            manager.download("whisper-tiny")
        # mkdir happens before the import is attempted.
        assert cache.exists()


class TestLoadErrorPaths:
    def test_load_unknown_model_raises(self) -> None:
        manager = LocalModelManager()
        with pytest.raises(ValueError, match="Unknown model"):
            manager.load("unknown-model")

    def test_load_whisper_missing_dependency(self, tmp_path: Path) -> None:
        if HAS_WHISPER:
            pytest.skip("openai-whisper is installed")

        config = LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        # Pretend the model is already on disk so load() reaches _load_whisper.
        model_dir = tmp_path / "whisper-tiny"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        with pytest.raises(ImportError):
            manager.load("whisper-tiny")

    def test_load_embedding_missing_dependency(self, tmp_path: Path) -> None:
        if HAS_ST:
            pytest.skip("sentence-transformers is installed")

        config = LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        model_dir = tmp_path / "all-MiniLM-L6-v2"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        with pytest.raises(ImportError):
            manager.load("all-MiniLM-L6-v2")

    def test_load_unsupported_category_raises(self, tmp_path: Path) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x", cache_dir=str(tmp_path)))
        # Inject a fake model entry with an unsupported category to reach the
        # else-branch in load(). This stubs the *metadata lookup*, not the
        # load path itself.
        fake_info = LocalModelInfo(
            name="fake-image",
            category=ModelCategory.IMAGE,
            description="fake",
            version="1",
            hardware=HardwareRequirements(min_ram_gb=1.0, recommended_ram_gb=2.0),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        monkeypatch_setitem(manager, fake_info)
        # Pretend it's downloaded so load() skips download and hits the
        # category dispatch.
        model_dir = tmp_path / "fake-image"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()
        with pytest.raises(ValueError, match="Unsupported model category"):
            manager.load("fake-image")


def monkeypatch_setitem(manager: LocalModelManager, info: LocalModelInfo) -> None:
    """Helper: register a fake model in the global AVAILABLE_MODELS table."""
    AVAILABLE_MODELS[info.name] = info


def _install_fake_whisper(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``whisper`` module on sys.modules.

    Returns the fake module so individual tests can configure return values.
    Stubbing the *external library* (not the manager) is the legitimate way to
    exercise the manager's real download/load path without requiring the heavy
    openai-whisper dependency.
    """
    fake = MagicMock(name="whisper")
    fake.load_model = MagicMock(return_value=MagicMock(name="whisper_model"))
    fake.load_audio = MagicMock(return_value=np.zeros(16000, dtype=np.float32))
    fake.pad_or_trim = MagicMock(return_value=np.zeros(16000, dtype=np.float32))
    # log_mel_spectrogram returns a tensor-like with a .to(device) method.
    mel = MagicMock(name="mel")
    mel.to = MagicMock(return_value=mel)
    fake.log_mel_spectrogram = MagicMock(return_value=mel)
    monkeypatch.setitem(__import__("sys").modules, "whisper", fake)
    return fake


def _install_fake_st(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``sentence_transformers`` module on sys.modules."""
    fake_pkg = MagicMock(name="sentence_transformers")

    class _FakeST:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def encode(self, x: Any) -> Any:
            if isinstance(x, str):
                return np.array([0.1, 0.2, 0.3])
            return np.array([[0.1, 0.2, 0.3] for _ in x])

    fake_pkg.SentenceTransformer = _FakeST
    monkeypatch.setitem(
        __import__("sys").modules, "sentence_transformers", fake_pkg
    )
    return fake_pkg


class TestManagerWithFakeLibraries:
    """Exercise the manager's real download/load code paths by stubbing only
    the *external* libraries (whisper / sentence-transformers), never the
    manager's own logic."""

    def test_download_whisper_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_whisper = _install_fake_whisper(monkeypatch)
        progress: list[float] = []
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        ok = manager.download("whisper-tiny", progress_callback=progress.append)
        assert ok is True
        assert progress == [1.0]
        fake_whisper.load_model.assert_called_once()
        call_kwargs = fake_whisper.load_model.call_args
        assert call_kwargs.args[0] == "tiny"
        assert call_kwargs.kwargs["download_root"] == str(tmp_path)

    def test_download_embedding_success_sets_hf_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_st(monkeypatch)
        # Capture HF_ENDPOINT observed during the SentenceTransformer call.
        seen: dict[str, str | None] = {}

        import sentence_transformers as _st  # the fake module

        original = _st.SentenceTransformer

        class _Capturing(original):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
                super().__init__(*args, **kwargs)

        _st.SentenceTransformer = _Capturing  # type: ignore[assignment]

        config = LocalModelConfig(
            model_name="all-MiniLM-L6-v2",
            cache_dir=str(tmp_path),
            use_mirror=True,
        )
        manager = LocalModelManager(config)
        monkeypatch.delenv("HF_ENDPOINT", raising=False)

        ok = manager.download("all-MiniLM-L6-v2")
        assert ok is True
        # The mirror endpoint must have been exported during the call.
        assert seen["HF_ENDPOINT"] == (
            "https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2"
        )
        # And restored afterwards.
        assert "HF_ENDPOINT" not in os.environ

    def test_download_embedding_progress_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_st(monkeypatch)
        config = LocalModelConfig(
            model_name="all-MiniLM-L6-v2", cache_dir=str(tmp_path)
        )
        manager = LocalModelManager(config)
        progress: list[float] = []
        assert manager.download("all-MiniLM-L6-v2", progress_callback=progress.append) is True
        assert progress == [1.0]

    def test_download_embedding_mirror_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_st(monkeypatch)
        seen: dict[str, str | None] = {}

        import sentence_transformers as _st

        class _Capturing(_st.SentenceTransformer):  # type: ignore[misc, valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                seen["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT")
                super().__init__(*args, **kwargs)

        _st.SentenceTransformer = _Capturing  # type: ignore[assignment]

        config = LocalModelConfig(
            model_name="all-MiniLM-L6-v2",
            cache_dir=str(tmp_path),
            use_mirror=False,
        )
        manager = LocalModelManager(config)
        monkeypatch.delenv("HF_ENDPOINT", raising=False)

        ok = manager.download("all-MiniLM-L6-v2")
        assert ok is True
        # Mirror disabled -> HF_ENDPOINT must not be set during the call.
        assert seen["HF_ENDPOINT"] is None

    def test_download_embedding_no_mirrors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_st(monkeypatch)
        config = LocalModelConfig(
            model_name="text-embedding-3-small",  # has mirrors, but...
            cache_dir=str(tmp_path),
            use_mirror=True,
        )
        # Replace with a model that has no mirrors.
        no_mirror = LocalModelInfo(
            name="__no_mirror__",
            category=ModelCategory.EMBEDDING,
            description="x",
            version="1",
            hardware=HardwareRequirements(min_ram_gb=1.0, recommended_ram_gb=2.0),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        AVAILABLE_MODELS["__no_mirror__"] = no_mirror
        try:
            manager = LocalModelManager(config)
            assert manager.download("__no_mirror__") is True
        finally:
            AVAILABLE_MODELS.pop("__no_mirror__", None)

    def test_load_whisper_full_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_whisper(monkeypatch)
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        model = manager.load("whisper-tiny")
        assert isinstance(model, WhisperModel)
        # Cached on second load.
        assert manager.load("whisper-tiny") is model

    def test_load_embedding_full_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_fake_st(monkeypatch)
        config = LocalModelConfig(
            model_name="all-MiniLM-L6-v2", cache_dir=str(tmp_path)
        )
        manager = LocalModelManager(config)
        model = manager.load("all-MiniLM-L6-v2")
        assert isinstance(model, EmbeddingModel)
        assert manager.load("all-MiniLM-L6-v2") is model

    def test_load_triggers_download_when_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_whisper = _install_fake_whisper(monkeypatch)
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        assert not manager.is_downloaded("whisper-tiny")
        manager.load("whisper-tiny")
        # load() must have invoked whisper.load_model (the download path).
        assert fake_whisper.load_model.called

    def test_get_device_with_fake_torch_cuda(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        fake_torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

        class _Backends:
            mps = MagicMock()

        fake_torch.cuda = _Cuda()
        fake_torch.backends = _Backends()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        manager = LocalModelManager(LocalModelConfig(model_name="x", device="auto"))
        assert manager._get_device() == "cuda"

    def test_get_device_with_fake_torch_mps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        fake_torch = types.ModuleType("torch")

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _Mps:
            @staticmethod
            def is_available() -> bool:
                return True

        class _Backends:
            mps = _Mps()

        fake_torch.cuda = _Cuda()
        fake_torch.backends = _Backends()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        manager = LocalModelManager(LocalModelConfig(model_name="x", device="auto"))
        assert manager._get_device() == "mps"

    def test_unload_logs_when_present(self, tmp_path: Path) -> None:
        manager = LocalModelManager(LocalModelConfig(model_name="x"))
        manager._loaded_models["m"] = Mock()
        manager.unload("m")
        assert "m" not in manager._loaded_models

    def test_whisper_detect_language(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_whisper = _install_fake_whisper(monkeypatch)
        inner = MagicMock()
        inner.device = "cpu"
        # detect_language returns (_, probs); max picks the highest-prob lang.
        inner.detect_language.return_value = (None, {"en": 0.9, "zh": 0.1})
        w = WhisperModel(inner, "whisper-tiny", "cpu")

        lang = w.detect_language("audio.mp3")
        assert lang == "en"
        fake_whisper.load_audio.assert_called_once_with("audio.mp3")
        fake_whisper.pad_or_trim.assert_called_once()
        fake_whisper.log_mel_spectrogram.assert_called_once()

    def test_download_unsupported_category_returns_false(
        self, tmp_path: Path
    ) -> None:
        """An IMAGE-category model reaches the unsupported-branch in download()."""
        fake = LocalModelInfo(
            name="__img__",
            category=ModelCategory.IMAGE,
            description="x",
            version="1",
            hardware=HardwareRequirements(min_ram_gb=1.0, recommended_ram_gb=2.0),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        AVAILABLE_MODELS["__img__"] = fake
        try:
            manager = LocalModelManager(
                LocalModelConfig(model_name="x", cache_dir=str(tmp_path))
            )
            assert manager.download("__img__") is False
        finally:
            AVAILABLE_MODELS.pop("__img__", None)

    def test_download_whisper_runtime_failure_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_whisper = _install_fake_whisper(monkeypatch)
        fake_whisper.load_model.side_effect = RuntimeError("network down")
        manager = LocalModelManager(
            LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        )
        # RuntimeError is a transient failure -> swallowed, returns False.
        assert manager.download("whisper-tiny") is False

    def test_download_embedding_runtime_failure_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_pkg = _install_fake_st(monkeypatch)

        class _Boom:
            def __init__(self, *a: Any, **k: Any) -> None:
                raise RuntimeError("disk full")

        fake_pkg.SentenceTransformer = _Boom  # type: ignore[assignment]
        manager = LocalModelManager(
            LocalModelConfig(
                model_name="all-MiniLM-L6-v2", cache_dir=str(tmp_path)
            )
        )
        assert manager.download("all-MiniLM-L6-v2") is False


class TestCheckHardwareWithFakePsutil:
    """Cover check_hardware_requirements' psutil + GPU branches by stubbing
    psutil (and torch) — the *external* introspection libraries, never the
    function under test."""

    def _install_fake_psutil(
        self, monkeypatch: pytest.MonkeyPatch, ram_gb: float
    ) -> None:
        import sys
        import types

        fake = types.ModuleType("psutil")

        class _VM:
            def __init__(self, total: int) -> None:
                self.total = total

        fake.virtual_memory = lambda: _VM(int(ram_gb * 1024**3))
        monkeypatch.setitem(sys.modules, "psutil", fake)

    def test_cpu_only_model_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._install_fake_psutil(monkeypatch, ram_gb=32.0)
        # No torch -> GPU branch skipped, supports_cpu True -> satisfied by RAM.
        result = check_hardware_requirements("whisper-tiny")
        assert result["satisfied"] is True
        assert result["current_ram_gb"] == 32.0
        assert result["gpu_available"] is False
        assert result["gpu_vram_gb"] is None

    def test_gpu_required_model_without_torch_unsatisfied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_psutil(monkeypatch, ram_gb=64.0)
        gpu_only = LocalModelInfo(
            name="__gpu_only__",
            category=ModelCategory.IMAGE,
            description="gpu only",
            version="1",
            hardware=HardwareRequirements(
                min_ram_gb=1.0,
                recommended_ram_gb=2.0,
                min_gpu_vram_gb=8.0,
                supports_cpu=False,
            ),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        AVAILABLE_MODELS["__gpu_only__"] = gpu_only
        try:
            result = check_hardware_requirements("__gpu_only__")
            # No torch -> GPU not available -> gpu_satisfied False.
            assert result["satisfied"] is False
            assert result["gpu_available"] is False
        finally:
            AVAILABLE_MODELS.pop("__gpu_only__", None)

    def test_gpu_required_model_with_sufficient_gpu(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._install_fake_psutil(monkeypatch, ram_gb=64.0)
        import sys
        import types

        fake_torch = types.ModuleType("torch")

        class _Props:
            total_memory = int(16 * 1024**3)

        class _Cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def get_device_name(idx: int) -> str:
                return "Fake RTX"

            @staticmethod
            def get_device_properties(idx: int) -> _Props:
                return _Props()

        fake_torch.cuda = _Cuda()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        gpu_only = LocalModelInfo(
            name="__gpu_ok__",
            category=ModelCategory.IMAGE,
            description="gpu only",
            version="1",
            hardware=HardwareRequirements(
                min_ram_gb=1.0,
                recommended_ram_gb=2.0,
                min_gpu_vram_gb=8.0,
                supports_cpu=False,
            ),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        AVAILABLE_MODELS["__gpu_ok__"] = gpu_only
        try:
            result = check_hardware_requirements("__gpu_ok__")
            assert result["satisfied"] is True
            assert result["gpu_available"] is True
            assert result["gpu_name"] == "Fake RTX"
            assert result["gpu_vram_gb"] == 16.0
        finally:
            AVAILABLE_MODELS.pop("__gpu_ok__", None)


# ---------------------------------------------------------------------------
# Whisper / Embedding model wrappers (pure logic with injected fakes)
# ---------------------------------------------------------------------------


class TestWhisperModel:
    def test_creation(self) -> None:
        mock_model = Mock()
        mock_model.device = "cpu"
        w = WhisperModel(mock_model, "whisper-tiny", "cpu")
        assert w._model_name == "whisper-tiny"
        assert w._device == "cpu"

    def test_transcribe(self) -> None:
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": [],
            "language": "en",
        }
        w = WhisperModel(mock_model, "whisper-tiny", "cpu")
        result = w.transcribe("test.mp3")
        assert result["text"] == "Hello world"
        assert result["language"] == "en"

    def test_transcribe_with_options(self) -> None:
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "你好世界",
            "segments": [],
            "language": "zh",
        }
        w = WhisperModel(mock_model, "whisper-small", "cpu")
        w.transcribe("test.mp3", language="zh", task="transcribe")
        mock_model.transcribe.assert_called_with(
            "test.mp3", language="zh", task="transcribe", verbose=False
        )

    def test_transcribe_missing_keys_default(self) -> None:
        mock_model = Mock()
        mock_model.transcribe.return_value = {}  # missing keys
        w = WhisperModel(mock_model, "whisper-tiny", "cpu")
        result = w.transcribe("test.mp3")
        assert result["text"] == ""
        assert result["language"] == "unknown"
        assert result["segments"] == []


class TestEmbeddingModel:
    def test_creation(self) -> None:
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        assert emb._model_name == "test-model"
        assert emb._device == "cpu"

    def test_embed_single(self) -> None:
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        result = emb.embed("test text")
        assert len(result) == 3
        mock_model.encode.assert_called_with("test text")

    def test_embed_batch(self) -> None:
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
            ]
        )
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        result = emb.embed_batch(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 3

    def test_similarity_orthogonal_is_zero(self) -> None:
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        sim = emb.similarity("a", "b")
        assert abs(sim) < 0.01

    def test_similarity_identical_is_one(self) -> None:
        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [
                [1.0, 2.0],
                [1.0, 2.0],
            ]
        )
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        sim = emb.similarity("a", "b")
        assert abs(sim - 1.0) < 0.01

    def test_search_top_k(self) -> None:
        mock_model = Mock()

        def encode_side_effect(x: Any) -> Any:
            if isinstance(x, str):
                return np.array([1.0, 1.0]) / np.sqrt(2)  # query
            return np.array(
                [
                    [1.0, 0.0],
                    [0.707, 0.707],  # most similar to query
                    [0.0, 1.0],
                ]
            )

        mock_model.encode.side_effect = encode_side_effect
        emb = EmbeddingModel(mock_model, "test-model", "cpu")
        results = emb.search("query", ["doc1", "doc2", "doc3"], top_k=2)
        assert len(results) == 2
        # doc2 (index 1) should rank first.
        assert results[0][0] == 1


# ---------------------------------------------------------------------------
# Hardware requirements check
# ---------------------------------------------------------------------------


class TestCheckHardwareRequirements:
    def test_unknown_model(self) -> None:
        result = check_hardware_requirements("unknown-model")
        assert result["satisfied"] is False
        assert "error" in result

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_whisper_tiny_basic_fields(self) -> None:
        result = check_hardware_requirements("whisper-tiny")
        assert "current_ram_gb" in result
        assert result["required_ram_gb"] == 1.0
        assert result["supports_cpu"] is True
        # Most modern hosts have >= 1GB.
        if result["satisfied"] is not None:
            assert result["satisfied"] is True

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_whisper_large_fields(self) -> None:
        result = check_hardware_requirements("whisper-large-v3")
        assert result["supports_cpu"] is True
        assert result["required_ram_gb"] == 10.0
        assert result["required_gpu_vram_gb"] == 10.0

    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not installed")
    def test_gpu_required_non_cpu_model_without_torch_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: GPU-required model with no torch must not raise
        ``None >= float`` TypeError. It should report gpu_satisfied=False."""
        # Build a synthetic GPU-only model and inject it.
        gpu_only = LocalModelInfo(
            name="__gpu_only__",
            category=ModelCategory.IMAGE,
            description="gpu only",
            version="1",
            hardware=HardwareRequirements(
                min_ram_gb=1.0,
                recommended_ram_gb=2.0,
                min_gpu_vram_gb=8.0,
                supports_cpu=False,
            ),
            network=NetworkRequirements(download_size_mb=1, download_url="https://x"),
        )
        AVAILABLE_MODELS["__gpu_only__"] = gpu_only
        try:
            # If torch is installed, force the GPU branch to look unavailable
            # by hiding torch.cuda. If torch is absent, the import guard
            # already leaves gpu_vram_gb=None.
            if HAS_TORCH:
                import torch

                class _FakeCuda:
                    @staticmethod
                    def is_available() -> bool:
                        return False

                monkeypatch.setattr(torch, "cuda", _FakeCuda())

            result = check_hardware_requirements("__gpu_only__")
            # Must not raise; gpu-not-satisfied lowers satisfied to False.
            assert "satisfied" in result
            assert result["gpu_available"] is False
            assert result["gpu_vram_gb"] is None
            assert result["satisfied"] is False
        finally:
            AVAILABLE_MODELS.pop("__gpu_only__", None)

    def test_without_psutil(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When psutil cannot be imported, return best-effort (no crash)."""
        if HAS_PSUTIL:
            # Force the import inside the function to fail by blocking the
            # module on sys.meta_path. This is robust against psutil being
            # installed in the test environment.
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "psutil":
                    raise ImportError("blocked for test")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fake_import)

        result = check_hardware_requirements("whisper-tiny")
        assert result["satisfied"] is None
        assert "error" in result
        assert "psutil" in result["error"]


# ---------------------------------------------------------------------------
# Model availability / metadata integrity (no fake placeholder URLs)
# ---------------------------------------------------------------------------


FAKE_URL_PATTERNS = ["e1e1e1", "ed3d97b362a9530e7b8d29b1d1e1e1"]


class TestModelAvailability:
    def test_all_models_have_required_fields(self) -> None:
        for name, info in AVAILABLE_MODELS.items():
            assert info.name == name
            assert info.category in ModelCategory
            assert info.hardware.min_ram_gb > 0
            assert info.network.download_size_mb > 0
            assert info.network.download_url

    def test_no_fake_placeholder_urls(self) -> None:
        """No model exposes a placeholder/garbage download_url."""
        for name, info in AVAILABLE_MODELS.items():
            url = info.network.download_url
            for pattern in FAKE_URL_PATTERNS:
                assert pattern not in url, (
                    f"model {name} still carries placeholder URL fragment "
                    f"{pattern!r}: {url}"
                )
            # URLs must be well-formed http(s).
            assert url.startswith("https://") or url.startswith("http://"), url

    def test_whisper_models_exist(self) -> None:
        for model_name in [
            "whisper-tiny",
            "whisper-base",
            "whisper-small",
            "whisper-medium",
            "whisper-large-v3",
        ]:
            assert model_name in AVAILABLE_MODELS
            assert AVAILABLE_MODELS[model_name].category == ModelCategory.SPEECH_TO_TEXT

    def test_embedding_models_exist(self) -> None:
        for model_name in [
            "all-MiniLM-L6-v2",
            "bge-small-zh-v1.5",
            "bge-large-zh-v1.5",
        ]:
            assert model_name in AVAILABLE_MODELS
            assert AVAILABLE_MODELS[model_name].category == ModelCategory.EMBEDDING

    def test_embedding_models_have_mirrors(self) -> None:
        """All embedding models should declare at least one HF mirror."""
        for name, info in AVAILABLE_MODELS.items():
            if info.category == ModelCategory.EMBEDDING:
                assert info.network.mirrors, f"{name} declares no mirror"

    def test_chinese_models_support_chinese(self) -> None:
        chinese_models = [
            "bge-small-zh-v1.5",
            "bge-large-zh-v1.5",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ]
        for model_name in chinese_models:
            assert "zh" in AVAILABLE_MODELS[model_name].languages


# ---------------------------------------------------------------------------
# Real integration tests (honestly skipped when deps absent)
# ---------------------------------------------------------------------------


@integration
class TestRealIntegration:
    """Real download/load tests. Require the optional deps and network access.

    These are skipped when the dependency is missing — never faked. In CI they
    should be run with the `[embeddings]` extra and `openai-whisper` installed.
    """

    @pytest.mark.skipif(
        not (HAS_WHISPER and HAS_TORCH),
        reason="requires openai-whisper + torch (not installed)",
    )
    def test_whisper_tiny_real_download_and_load(self, tmp_path: Path) -> None:
        config = LocalModelConfig(
            model_name="whisper-tiny", cache_dir=str(tmp_path), device="cpu"
        )
        manager = LocalModelManager(config)
        assert manager.download("whisper-tiny") is True
        assert manager.is_downloaded("whisper-tiny") is True
        model = manager.load("whisper-tiny")
        assert isinstance(model, WhisperModel)
        # Second load returns cached instance.
        assert manager.load("whisper-tiny") is model

    @pytest.mark.skipif(
        not HAS_ST, reason="requires sentence-transformers (not installed)"
    )
    def test_embedding_real_download_and_load(self, tmp_path: Path) -> None:
        config = LocalModelConfig(
            model_name="all-MiniLM-L6-v2", cache_dir=str(tmp_path), device="cpu"
        )
        manager = LocalModelManager(config)
        assert manager.download("all-MiniLM-L6-v2") is True
        model = manager.load("all-MiniLM-L6-v2")
        assert isinstance(model, EmbeddingModel)
        vec = model.embed("hello world")
        assert len(vec) > 0
