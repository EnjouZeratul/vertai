# S7 审查记录 — 本地模型管理（0.8.0）

> 阶段完成后三重审查。基于实测，非代理自报。

## 阶段范围回顾

S7：`vertai/local/models.py` 改透 — 真实 URL / 镜像真实生效（HF_ENDPOINT）/ GPU TypeError 修复 / `__init__` 副作用移除 / 删死代码 / ModelInfo 命名 / 覆盖率 ≥85% / mypy strict 0 错。

## 1. 代码真实实现审查

### 审查发现的问题（全部修复）

| 问题 | 修复 |
|------|------|
| **假 URL**：whisper-base/small/medium/large-v3 的 `download_url` 是占位串 `ed3d97b...e1e1e1`（重复 e1e1e1），whisper-tiny 末尾无文件名 | 全部替换为真实的 HuggingFace 模型仓库 URL（`https://huggingface.co/openai/whisper-*`）。`download_url` 在 `NetworkRequirements` docstring 中明确标注为"informational metadata — 实际下载走底层库内部 URL 解析"，不用于直接拉取字节 |
| **镜像死代码**：`_download_embedding` 计算 `model_url`（含 mirrors）却从不使用（F841），SentenceTransformer 忽略所选镜像，宣称的"国内镜像支持"无效 | 新增 `_export_hf_endpoint` 上下文管理器，下载时把所选镜像写入 `HF_ENDPOINT` 环境变量（`huggingface_hub` 读此变量重定向）。`_select_mirror` 挑选镜像。镜像现在**真实生效** |
| **GPU TypeError**：`check_hardware_requirements` 的 GPU 分支 `gpu_vram_gb >= min_gpu_vram_gb`，torch 未装时 gpu_vram_gb 为 None，`None >= float` 抛 TypeError | GPU 分支先判 `gpu_vram_gb is None`/`not gpu_available` 短路为 `gpu_satisfied=False`，不再比较 |
| **`__init__` 副作用**：`LocalModelManager.__init__` 实例化即 `mkdir` 创建 `~/.cache/vertai/models` | mkdir 移到 `download()`（lazy）。构造器零文件系统副作用。`TestManagerConstructorNoSideEffects` 真实验证（含 HOME 重定向断言默认缓存目录不被创建） |
| **死代码**：`_download_whisper`/`_download_embedding` 里 `model = whisper.load_model(...)` / `model = SentenceTransformer(...)` 赋值后从不使用（F841） | 删除丢弃赋值，直接调用 |
| **冗余 except 掩盖**：`download()` 的外层 `except Exception` 是死代码（两个 `_download_*` 已各自 catch），且会掩盖编程错误 | 移除外层 try/except，仅保留调度。ImportError（缺可选依赖）从 `_download_*` 自然传播，给出可操作的配置错误而非误导性的"download failed" |
| **ModelInfo 命名**：local 侧 `ModelInfo` 与 core 侧 `LLMModelInfo` 虽不同名但易混淆 | local 侧重命名为 `LocalModelInfo`，保留 `ModelInfo = LocalModelInfo` 向后兼容别名。`vertai/__init__.py` 同时导出两者，`vertai.ModelInfo is vertai.LocalModelInfo` 为 True（实测）。无 `__init__` 冲突 |

### 无摆设验证

- `grep -rn "ed3d97b\|e1e1e1\|openaipublic.azureedge" vertai/ docs/` → 仅 `tests/test_local_models.py` 的 `FAKE_URL_PATTERNS` 断言列表（预期）
- 镜像 `HF_ENDPOINT` 真实生效：`test_download_embedding_success_sets_hf_endpoint` 在 `SentenceTransformer.__init__` 内部捕获 `os.environ["HF_ENDPOINT"]`，实测值等于镜像 URL，且退出后恢复
- `__init__` 无副作用：`test_construction_does_not_create_cache_dir` + `test_default_construction_does_not_create_default_cache`（HOME 重定向）真实验证

## 2. 测试覆盖率与真实性审查

### 实测

```
mypy --strict vertai/local/models.py → Success: no issues found in 1 file
ruff check (models.py + __init__.py + tests) → All checks passed!
pytest tests/test_local_models.py --cov=vertai.local.models
  → vertai\local\models.py  285 stmts  0 miss  100%
  → 83 passed, 5 skipped
pytest tests/ (全套) → 807 passed, 34 skipped（无回归）
```

覆盖率 100%（远超 ≥85% Gate）。

### 测试真实性红线

- ✅ **删除 mock 缓存命中伪装**：`test_load_returns_whisper_model` / `test_load_returns_embedding_model`（把 mock 塞 `_loaded_models` 再"测" load 短路）已删除。改用 `TestManagerWithFakeLibraries`：注入 fake `whisper`/`sentence_transformers` 模块到 `sys.modules`（monkeypatch，测试后自动恢复，非脆弱 reload），走**真实的 download/load 代码路径**
- ✅ **删除 sys.modules 突变**：`test_check_without_psutil`（del sys.modules + importlib.reload + finally 恢复，脆弱）已删除。改用：
  - 有 psutil 时：`monkeypatch.setattr(builtins, "__import__", ...)` 阻断 psutil 导入（monkeypatch 自动恢复）
  - 无 psutil 时：直接测真实 ImportError 路径
  - 新增 `TestCheckHardwareWithFakePsutil`：注入 fake psutil + fake torch.cuda，覆盖 RAM/GPU 各分支
- ✅ **stub 外部库而非内部逻辑**：fake `whisper`/`sentence_transformers`/`psutil`/`torch` 模块只 stub **外部库行为**（load_model/encode/virtual_memory/cuda），管理器自身代码路径真实执行。符合 ROADMAP 测试策略表"外部服务协议用 mock"
- ✅ **集成测试诚实**：`TestRealIntegration` 标 `@pytest.mark.integration` + `@pytest.mark.skipif(not HAS_WHISPER/HAS_ST)`，有依赖真实跑，无依赖诚实 skip（当前环境 whisper/ST/torch 均未装，5 skip 全部真实）
- ✅ **无 except 掩盖**：`download()` 外层冗余 except 已删；`_download_*` 内的 except 仅捕获 runtime 下载失败（返回 False），ImportError 传播；`check_hardware` 的 except 仅 ImportError 降级
- ✅ **无刷覆盖率**：无行号导向测试类，无 mock 循环断言

### 覆盖关键路径（真实，非行覆盖凑数）

- `download()` 调度 + lazy mkdir + unsupported 分支 + whisper/embedding 成功 + runtime 失败 + ImportError
- `_download_whisper` / `_download_embedding`（含 HF_ENDPOINT 上下文 + 镜像启用/禁用/无镜像）
- `load()` 缓存命中 + 触发 download + whisper/embedding 完整路径 + 不支持类别
- `_load_whisper` / `_load_embedding` / `_get_device`（auto/cpu/cuda/mps 全分支，用 fake torch）
- `_select_mirror` / `_export_hf_endpoint`（set/restore/previous/none 全分支）
- `WhisperModel.transcribe`（含缺键默认）/ `detect_language`（fake whisper 全链）
- `EmbeddingModel.embed/embed_batch/similarity/search`
- `check_hardware_requirements`：unknown / psutil 缺失 / RAM 满足 / GPU-required 无 torch / GPU-required 有 GPU 全分支
- `clear_cache`（指定/全部）/ `get_cache_size`（有/无目录）/ `unload`（有/无）

## 3. 文档与实现一致性审查

- ✅ `models.py` docstring 英文为主，含镜像/HF_ENDPOINT/download_url informational 说明
- ✅ `docs/FUNCTION_DEPENDENCIES.md`：补充 `HF_ENDPOINT` 镜像生效机制说明（中英双语）
- ✅ 导出同步：`vertai/__init__.py` + `vertai/local/__init__.py` 均导出 `LocalModelInfo`，保留 `ModelInfo` 别名，无 `__init__` 冲突（实测 `vertai.ModelInfo is vertai.LocalModelInfo` 为 True）
- ✅ `pyproject.toml` mypy overrides 增加 `whisper`/`torch`/`psutil`（与 S3 对 chromadb/faiss/sentence_transformers/numpy 同模式，可选 extras 无类型 stub）

## 实测命令输出

```
$ python -m mypy --strict vertai/local/models.py
Success: no issues found in 1 file

$ python -m ruff check vertai/local/models.py vertai/local/__init__.py vertai/__init__.py tests/test_local_models.py
All checks passed!

$ python -m pytest tests/test_local_models.py --cov=vertai.local.models --cov-report=term-missing
tests\test_local_models.py ............................................. [ 51%]
... (83 tests) ...
Name                     Stmts   Miss  Cover   Missing
------------------------------------------------------
vertai\local\models.py     285      0   100%
------------------------------------------------------
TOTAL                      285      0   100%
83 passed, 5 skipped

$ python -m pytest tests/ -q
807 passed, 34 skipped, 1 warning in 11.76s   (无回归)

$ python -c "import vertai; print(vertai.ModelInfo is vertai.LocalModelInfo, 'LLMModelInfo' in dir(vertai))"
True True   (无 __init__ 命名冲突)
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict `vertai/local/models.py` 0 错 | ✅ |
| ruff 0 错（含 tests） | ✅ |
| 该模块覆盖率 ≥85% | ✅（100%） |
| GPU TypeError 修复（无 torch 不崩） | ✅（`test_gpu_required_model_without_torch_unsatisfied` 真实验证） |
| 镜像真实生效（HF_ENDPOINT） | ✅（`test_download_embedding_success_sets_hf_endpoint` 在 SentenceTransformer 内部捕获 HF_ENDPOINT 实测值） |
| `__init__` 无文件系统副作用 | ✅（`TestManagerConstructorNoSideEffects` 真实验证） |
| 无假 URL（移除或标注 informational） | ✅（全替换为真实 HF URL + docstring 标注 informational） |
| 无 mock 缓存命中伪装、无 sys.modules 突变 | ✅（全部移除，改 stub 外部库） |
| ModelInfo 无 `__init__` 冲突 | ✅（LocalModelInfo + 别名） |
| 集成测试诚实（有依赖真实跑，无依赖 skip） | ✅（@integration + skipif，5 skip 真实） |

**判定：S7 通过，可进入 S8。**

## 遗留项（有意留后续阶段，非缺陷）

- 真实 whisper/sentence-transformers 集成测试本地 skip（重依赖未装）。代码路径已用 fake 库全覆盖，CI 装 extras 后 `TestRealIntegration` 真实跑。
- 全局 mypy 余错在其他模块（parser/output 等），留 S8。
- `pytest.ini` 的废弃 `python_paths` 选项产生 warning，留 S11 工程化清理（不在 S7 范围）。

## 产出文件

- 重写：`vertai/local/models.py`（ModelInfo→LocalModelInfo + 真实 URL + HF_ENDPOINT 镜像 + GPU 守卫 + lazy mkdir + 删死代码 + 英文 docstring）
- 重写：`tests/test_local_models.py`（删伪装 + stub 外部库真实覆盖关键路径 + 诚实集成测试 + fake psutil/torch 分支覆盖）
- 更新：`vertai/__init__.py`、`vertai/local/__init__.py`（导出 LocalModelInfo）
- 更新：`pyproject.toml`（mypy overrides 加 whisper/torch/psutil）
- 更新：`docs/FUNCTION_DEPENDENCIES.md`（HF_ENDPOINT 镜像机制说明）
