"""本地模型管理测试"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from vertai.local.models import (
    LocalModelManager,
    LocalModelConfig,
    ModelCategory,
    ModelInfo,
    HardwareRequirements,
    NetworkRequirements,
    WhisperModel,
    EmbeddingModel,
    AVAILABLE_MODELS,
    check_hardware_requirements,
)


class TestModelCategory:
    """模型类别测试"""

    def test_categories_exist(self):
        assert ModelCategory.SPEECH_TO_TEXT.value == "speech_to_text"
        assert ModelCategory.EMBEDDING.value == "embedding"
        assert ModelCategory.IMAGE.value == "image"
        assert ModelCategory.TEXT_GENERATION.value == "text_generation"


class TestHardwareRequirements:
    """硬件需求测试"""

    def test_basic_requirements(self):
        hw = HardwareRequirements(
            min_ram_gb=2.0,
            recommended_ram_gb=4.0,
            supports_cpu=True,
        )
        assert hw.min_ram_gb == 2.0
        assert hw.recommended_ram_gb == 4.0
        assert hw.supports_cpu is True
        assert hw.min_gpu_vram_gb is None

    def test_with_gpu_requirements(self):
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

    def test_to_dict(self):
        hw = HardwareRequirements(
            min_ram_gb=2.0,
            recommended_ram_gb=4.0,
        )
        d = hw.to_dict()
        assert d["min_ram_gb"] == 2.0
        assert d["recommended_ram_gb"] == 4.0


class TestNetworkRequirements:
    """网络需求测试"""

    def test_basic_network(self):
        net = NetworkRequirements(
            download_size_mb=150,
            download_url="https://example.com/model.bin",
        )
        assert net.download_size_mb == 150
        assert net.download_url == "https://example.com/model.bin"
        assert net.mirrors == []

    def test_with_mirrors(self):
        net = NetworkRequirements(
            download_size_mb=500,
            download_url="https://huggingface.co/model",
            mirrors=["https://hf-mirror.com/model"],
            estimated_download_time_minutes=10,
        )
        assert len(net.mirrors) == 1
        assert net.estimated_download_time_minutes == 10

    def test_to_dict(self):
        net = NetworkRequirements(
            download_size_mb=100,
            download_url="https://example.com/model",
        )
        d = net.to_dict()
        assert d["download_size_mb"] == 100
        assert d["download_url"] == "https://example.com/model"


class TestModelInfo:
    """模型信息测试"""

    def test_whisper_tiny_info(self):
        info = AVAILABLE_MODELS["whisper-tiny"]
        assert info.name == "whisper-tiny"
        assert info.category == ModelCategory.SPEECH_TO_TEXT
        assert info.hardware.min_ram_gb == 1.0
        assert info.network.download_size_mb == 75
        assert "zh" in info.languages
        assert "cpu-friendly" in info.tags

    def test_embedding_model_info(self):
        info = AVAILABLE_MODELS["bge-small-zh-v1.5"]
        assert info.category == ModelCategory.EMBEDDING
        assert info.hardware.min_ram_gb == 0.5
        assert "chinese" in info.tags

    def test_to_dict(self):
        info = AVAILABLE_MODELS["whisper-base"]
        d = info.to_dict()
        assert d["name"] == "whisper-base"
        assert d["category"] == "speech_to_text"
        assert "hardware" in d
        assert "network" in d


class TestLocalModelConfig:
    """本地模型配置测试"""

    def test_default_config(self):
        config = LocalModelConfig(model_name="test")
        assert config.model_name == "test"
        assert config.device == "auto"
        assert config.download_timeout == 3600
        assert config.use_mirror is True

    def test_custom_config(self):
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


class TestLocalModelManager:
    """本地模型管理器测试"""

    def test_list_all_models(self):
        manager = LocalModelManager()
        models = manager.list_models()
        assert len(models) == len(AVAILABLE_MODELS)

    def test_list_by_category(self):
        manager = LocalModelManager()
        speech_models = manager.list_models(category=ModelCategory.SPEECH_TO_TEXT)
        assert all(m.category == ModelCategory.SPEECH_TO_TEXT for m in speech_models)

    def test_list_by_language(self):
        manager = LocalModelManager()
        chinese_models = manager.list_models(language="zh")
        assert all("zh" in m.languages or "auto" in m.languages for m in chinese_models)

    def test_list_by_tag(self):
        manager = LocalModelManager()
        cpu_models = manager.list_models(tag="cpu-friendly")
        assert all("cpu-friendly" in m.tags for m in cpu_models)

    def test_get_model_info(self):
        manager = LocalModelManager()
        info = manager.get_model_info("whisper-small")
        assert info is not None
        assert info.name == "whisper-small"

    def test_get_unknown_model_info(self):
        manager = LocalModelManager()
        info = manager.get_model_info("unknown-model")
        assert info is None

    def test_is_downloaded_false(self, tmp_path):
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)
        assert manager.is_downloaded("whisper-tiny") is False

    def test_is_downloaded_true(self, tmp_path):
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 模拟已下载的模型目录
        model_dir = tmp_path / "whisper-tiny"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        assert manager.is_downloaded("whisper-tiny") is True

    def test_get_cache_size(self, tmp_path):
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 创建测试文件
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        test_file = model_dir / "model.bin"
        test_file.write_bytes(b"x" * 1024 * 1024)  # 1MB

        size = manager.get_cache_size()
        assert size >= 1.0

    def test_clear_cache_specific(self, tmp_path):
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 创建模型目录
        model_dir = tmp_path / "test-model"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        manager.clear_cache("test-model")
        assert not model_dir.exists()

    def test_clear_cache_all(self, tmp_path):
        config = LocalModelConfig(model_name="test", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 创建多个模型目录
        for name in ["model1", "model2"]:
            model_dir = tmp_path / name
            model_dir.mkdir()
            (model_dir / "model.bin").touch()

        manager.clear_cache()
        assert len(list(tmp_path.iterdir())) == 0

    def test_unload_model(self):
        manager = LocalModelManager()

        # 模拟已加载的模型
        manager._loaded_models["test-model"] = Mock()

        manager.unload("test-model")
        assert "test-model" not in manager._loaded_models


class TestWhisperModel:
    """Whisper 模型测试"""

    def test_whisper_model_creation(self):
        mock_model = Mock()
        mock_model.device = "cpu"

        whisper = WhisperModel(mock_model, "whisper-tiny", "cpu")

        assert whisper._model_name == "whisper-tiny"
        assert whisper._device == "cpu"

    def test_transcribe(self):
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "Hello world",
            "segments": [],
            "language": "en",
        }

        whisper = WhisperModel(mock_model, "whisper-tiny", "cpu")
        result = whisper.transcribe("test.mp3")

        assert result["text"] == "Hello world"
        assert result["language"] == "en"

    def test_transcribe_with_options(self):
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "你好世界",
            "segments": [],
            "language": "zh",
        }

        whisper = WhisperModel(mock_model, "whisper-small", "cpu")
        result = whisper.transcribe("test.mp3", language="zh", task="transcribe")

        mock_model.transcribe.assert_called_with(
            "test.mp3",
            language="zh",
            task="transcribe",
            verbose=False,
        )


class TestEmbeddingModel:
    """嵌入模型测试"""

    def test_embedding_model_creation(self):
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        embedding = EmbeddingModel(mock_model, "test-model", "cpu")

        assert embedding._model_name == "test-model"
        assert embedding._device == "cpu"

    def test_embed_single(self):
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

        embedding = EmbeddingModel(mock_model, "test-model", "cpu")
        result = embedding.embed("test text")

        assert len(result) == 3
        mock_model.encode.assert_called_with("test text")

    def test_embed_batch(self):
        import numpy as np

        mock_model = Mock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])

        embedding = EmbeddingModel(mock_model, "test-model", "cpu")
        result = embedding.embed_batch(["text1", "text2"])

        assert len(result) == 2
        assert len(result[0]) == 3

    def test_similarity(self):
        import numpy as np

        mock_model = Mock()
        # 正交向量的相似度为0
        mock_model.encode.return_value = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        embedding = EmbeddingModel(mock_model, "test-model", "cpu")
        sim = embedding.similarity("text1", "text2")

        # 正交向量相似度应为0
        assert abs(sim) < 0.01

    def test_search(self):
        import numpy as np

        mock_model = Mock()
        mock_model.encode.side_effect = lambda x: np.array([0.1, 0.2]) if isinstance(x, str) else np.array([
            [1.0, 0.0],  # doc1
            [0.707, 0.707],  # doc2 - more similar to query
            [0.0, 1.0],  # doc3
        ]) if len(x) == 3 else np.array([[0.1, 0.2]])

        embedding = EmbeddingModel(mock_model, "test-model", "cpu")

        # 重新设置 side_effect
        call_count = [0]

        def encode_side_effect(x):
            call_count[0] += 1
            if isinstance(x, str):
                return np.array([1.0, 1.0]) / np.sqrt(2)  # query
            return np.array([
                [1.0, 0.0],
                [0.707, 0.707],  # most similar
                [0.0, 1.0],
            ])

        mock_model.encode.side_effect = encode_side_effect

        results = embedding.search("query", ["doc1", "doc2", "doc3"], top_k=2)

        assert len(results) == 2
        # doc2 应该是最相似的
        assert results[0][0] == 1


class TestCheckHardwareRequirements:
    """硬件需求检查测试"""

    def test_check_unknown_model(self):
        result = check_hardware_requirements("unknown-model")
        assert result["satisfied"] is False
        assert "error" in result

    def test_check_whisper_tiny_with_psutil(self):
        """测试检查硬件需求（需要 psutil）"""
        try:
            import psutil
        except ImportError:
            pytest.skip("需要安装 psutil: pip install psutil")

        result = check_hardware_requirements("whisper-tiny")

        # whisper-tiny 需要 1GB RAM
        assert "current_ram_gb" in result
        assert result["required_ram_gb"] == 1.0
        # 大多数现代电脑应该满足 1GB RAM
        if result["satisfied"] is not None:
            assert result["satisfied"] is True

    def test_check_returns_required_info(self):
        """测试返回的信息包含必需字段"""
        try:
            import psutil
        except ImportError:
            pytest.skip("需要安装 psutil: pip install psutil")

        result = check_hardware_requirements("whisper-large-v3")

        assert "supports_cpu" in result
        assert result["supports_cpu"] is True
        assert result["required_ram_gb"] == 10.0

    def test_check_without_psutil(self):
        """测试 psutil 未安装时的处理"""
        import sys
        import importlib
        import vertai.local.models as models_module

        # 临时移除 psutil
        psutil_backup = sys.modules.get("psutil")
        if "psutil" in sys.modules:
            del sys.modules["psutil"]

        try:
            # 重新导入以获取新的结果
            importlib.reload(models_module)
            result = models_module.check_hardware_requirements("whisper-tiny")

            assert result["satisfied"] is None
            assert "error" in result
            assert "psutil" in result["error"]
        finally:
            # 恢复 psutil
            if psutil_backup:
                sys.modules["psutil"] = psutil_backup
            importlib.reload(models_module)


class TestModelAvailability:
    """模型可用性测试"""

    def test_all_models_have_required_fields(self):
        for name, info in AVAILABLE_MODELS.items():
            assert info.name == name
            assert info.category in ModelCategory
            assert info.hardware.min_ram_gb > 0
            assert info.network.download_size_mb > 0
            assert info.network.download_url

    def test_whisper_models_exist(self):
        for model_name in ["whisper-tiny", "whisper-base", "whisper-small", "whisper-medium", "whisper-large-v3"]:
            assert model_name in AVAILABLE_MODELS
            assert AVAILABLE_MODELS[model_name].category == ModelCategory.SPEECH_TO_TEXT

    def test_embedding_models_exist(self):
        for model_name in ["all-MiniLM-L6-v2", "bge-small-zh-v1.5", "bge-large-zh-v1.5"]:
            assert model_name in AVAILABLE_MODELS
            assert AVAILABLE_MODELS[model_name].category == ModelCategory.EMBEDDING

    def test_chinese_models_support_chinese(self):
        chinese_models = ["bge-small-zh-v1.5", "bge-large-zh-v1.5", "paraphrase-multilingual-MiniLM-L12-v2"]
        for model_name in chinese_models:
            assert "zh" in AVAILABLE_MODELS[model_name].languages


class TestManagerIntegration:
    """管理器集成测试（不实际下载）"""

    def test_manager_initialization(self):
        manager = LocalModelManager()
        assert manager._cache_dir.exists()
        assert manager._loaded_models == {}

    def test_manager_with_config(self, tmp_path):
        config = LocalModelConfig(
            model_name="test",
            cache_dir=str(tmp_path),
            device="cpu",
        )
        manager = LocalModelManager(config)

        assert manager.config.device == "cpu"
        assert manager._cache_dir == tmp_path

    def test_load_returns_whisper_model(self, tmp_path):
        """测试加载 Whisper 模型（Mock）"""
        config = LocalModelConfig(model_name="whisper-tiny", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 模拟已下载
        model_dir = tmp_path / "whisper-tiny"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        # 直接模拟已加载的模型
        mock_whisper = Mock()
        manager._loaded_models["whisper-tiny"] = mock_whisper

        # 第二次加载应返回已缓存的模型
        result = manager.load("whisper-tiny")

        assert result == mock_whisper

    def test_load_returns_embedding_model(self, tmp_path):
        """测试加载嵌入模型（Mock）"""
        config = LocalModelConfig(model_name="bge-small-zh-v1.5", cache_dir=str(tmp_path))
        manager = LocalModelManager(config)

        # 模拟已下载
        model_dir = tmp_path / "bge-small-zh-v1.5"
        model_dir.mkdir()
        (model_dir / "model.bin").touch()

        # 直接模拟已加载的模型
        mock_embedding = Mock()
        manager._loaded_models["bge-small-zh-v1.5"] = mock_embedding

        result = manager.load("bge-small-zh-v1.5")

        assert result == mock_embedding

    def test_load_unknown_model_raises(self):
        """测试加载未知模型抛出错误"""
        manager = LocalModelManager()

        with pytest.raises(ValueError, match="未知模型"):
            manager.load("unknown-model")
