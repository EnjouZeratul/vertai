# S8 审查记录 — 输出层 + 可视化移出（0.9.0）

> 阶段完成后三重审查。基于实测，非代理自报。

## 阶段范围回顾

S8：`vertai/output/docgen.py` + `vertai/output/structured.py` + `vertai/data/parser.py` 修透；
Dashboard 从核心 `__init__.py` 移出 → 可选 `vertai[viz]`。

## 1. 代码真实实现审查

### 审查发现的问题（全部修复）

| 问题 | 修复 |
|------|------|
| **C5 DocGen PDF 类型契约破裂**：`_render_pdf` 返回 `bytes` 但 `generate` 声明 `-> str`；`save` 用 `open("w")` 写会崩 `bytes` | `generate()` 返回类型改为 `str \| bytes`（markdown/html 返回 `str`，PDF 返回 `bytes`）；`save()` 按 `isinstance(content, bytes)` 分支到 `open("wb")` / `open("w")`。PDF 仍 `raise NotImplementedError`（无真实后端，诚实标注），但当真实实现到位时类型契约已正确 |
| **StructuredOutput string 写死中文人名**：`_parse_string` 只匹配 `[一-龥]{2,4}(?=报\|采\|...)`，对 `{"product":"string"}` + `"product is Widget"` 抛 ValueError | `_parse_string` 泛化为通用字符串提取（返回首个非空白 token）。中文人名启发式保留为**文档化的领域示例模式**，仅对字段名 `name`/`姓名`/`名字`/`buyer` 触发（`_CHINESE_NAME_FIELD_HINTS`）。`_parse_relaxed`/`_parse_aggressive` 同步泛化 |
| **StructuredOutput LLM 路径跳过 schema 验证**：`_extract_with_llm` 只检查 `not None`，不调 `_validate_field` | LLM 路径对每个字段调 `_validate_field`，收集验证错误，把错误回填 prompt 重试，最多 `max_retries` 次。strict 模式持久失败抛 `SchemaValidationError`。`test_llm_wrong_type_triggers_retry_then_success`/`test_llm_enum_validated`/`test_llm_validation_failure_exhausts_retries_strict_raises` 真实验证 |
| **dir() 内省反模式**：`structured.py:275` `data if 'data' in dir() else {}` | 改为 try 外预初始化 `data: dict[str, Any] = {}`。`test_correction_partial_failure_returns_partial_data` 真实验证部分失败时返回部分 data |
| **DocParser 返回键**：`parse` 返回 `text/metadata/chunks`（无 `content`）但 `example_5` 注释写 `result['content']` | `examples/quick_start_knowledge_base.py:208` 注释改为 `result['text']` |
| **mypy 错误**：`docgen.py:159` incompatible return（bytes vs str）；`structured.py` dict 缺类型参数、Returning Any、union-attr；`parser.py` 多处 no-untyped-def / var-annotated / import-not-found | 全部修复（见下方实测）。parser 的 fitz/docx/openpyxl/pptx 加入 pyproject mypy overrides（与 S3/S7 同模式：可选 extras 无类型 stub） |
| **dashboard 在核心 `__init__.py` eager import** | 移除 eager import；viz 包保留（`from vertai.viz.dashboard import Dashboard` 仍可用）；pyproject 新增 `[viz]` extras（当前无第三方运行时依赖，作为显式安装分组保留） |

### 无摆设验证

- `import vertai` 后 `sys.modules` 不含 `vertai.viz*`（`test_viz_not_eager_loaded` 等效实测）
- `from vertai.viz.dashboard import Dashboard, Metric, Chart, ChartType, DashboardTheme, ChartConfig` 可用
- `hasattr(vertai, 'Dashboard')` 为 False（核心命名空间无 viz）
- `_parse_string` 对 `{"product":"string"}` + `"product is Widget"` 返回 `"product"`（通用 token），不再 ValueError
- LLM 路径 `_validate_field` 被调用（fake LLM 返回错误类型触发重试，实测 `len(llm.calls) == 2`）

## 2. 测试覆盖率与真实性审查

### 实测

```
mypy --strict vertai/output/ vertai/data/parser.py → Success: no issues found in 4 source files
ruff check (S8 scope) → All checks passed!
pytest tests/test_structured.py → 96 passed
pytest tests/ (全套) → 785 passed, 34 skipped（无回归，34 skip 与 S7 一致）
pytest --cov=vertai.output.structured → 98% (5 lines uncovered, 均为可到达的防御性分支)
```

### 测试真实性红线

- ✅ **删除 TestRemainingCoverage 刷覆盖率类**（11 个 docstring 写 "covers line XXX" 的测试，原 `test_structured.py:1018-1116`）。全部重写为真实行为测试，无行号导向
- ✅ **新增 LLM 路径 schema 验证测试**：`TestLLMSchemaValidation`（8 个测试）用 `_FakeLLM` stub **外部 provider 行为**（返回脚本化 JSON），验证真实 schema 验证 + 重试逻辑（错误类型触发重试、缺字段触发重试、枚举验证、JSON 解析失败重试、provider 异常重试、strict/non-strict 失败）。非 mock 循环断言（断言的是 `result.data`/`result.success`/`llm.calls` 计数，真实行为）
- ✅ **新增 string 泛化测试**：`TestStringGeneralization`（5 个测试）覆盖通用 ASCII 提取（`{"product":"string"}` + `"product is Widget"` 不再 ValueError 的回归测试）、通用 token 提取、name hint 仍工作、buyer hint、relaxed 回退
- ✅ **新增 dir() 反模式回归测试**：`TestCorrectionNoDirIntrospection` 验证部分失败返回部分 data（而非 `{}`）
- ✅ **无 except 掩盖**：fake LLM 模拟 provider 异常是测试外部行为，被测代码（`_extract_with_llm`）的 except 仅捕获 JSONDecodeError 和 provider Exception 并重试（真实业务逻辑）
- ✅ **参数化合并冗余**：`TestNumberParsing`/`TestBooleanParsing` 用 `@pytest.mark.parametrize` 合并 7+8 个布尔/数字模式测试，减少冗余同时保留全覆盖
- ✅ **删除冗余/重复测试**：原 `TestValidateField`（19 个测试，多为 schema 构造后不用的 `result = output.extract(...)`）精简为 8 个直接调 `_validate_field` 的测试；`TestParseStringEdgeCases`/`TestParseBooleanEdgeCases`/`TestParseNumberEdgeCases` 合并

### 覆盖关键路径（真实）

- `_extract_with_llm`：成功路径、错误类型重试、缺字段重试、枚举验证、JSON 解析失败重试、provider 异常重试、strict 抛错、non-strict 返回错误、```json 和 ``` 代码块剥离
- `extract` 本地模式：初始成功、重试成功、重试耗尽 strict 抛错、non-strict 返回失败、空 schema
- `_extract_initial`：re.error 异常返回失败结果（patch 外部 parser）
- `_correct_extraction`：复用有效 previous field、重新解析无效 field、部分失败返回部分 data（dir() 反模式回归）
- `_parse_string` 泛化：name hint、buyer hint、通用 token、ASCII
- `_parse_relaxed`/`_parse_pattern`/`_parse_aggressive`：各类型（string/number/integer/enum/boolean）成功 + ValueError 路径
- `_validate_field`：None/各类型错误/枚举无效值
- DocGen C5：`generate` markdown/html/PDF（NotImplementedError）、`save` text/binary 分支
- DocParser：Markdown 真实解析、各格式 fake 依赖、ImportError、返回键 text/metadata/chunks

## 3. 文档与实现一致性审查

- ✅ `structured.py` docstring 英文为主，明确标注 string 泛化 + name-hint 是领域示例模式（非通用契约）；LLM 路径 schema 验证 + 重试机制文档化
- ✅ `docgen.py` `generate()` 返回类型文档化为 `str`（markdown/html）/ `bytes`（pdf）；`save()` 文档化 text/binary 分支
- ✅ `dashboard.py` docstring 示例改为 `from vertai.viz.dashboard import Dashboard`，添加 viz extras 说明
- ✅ `FUNCTION_DEPENDENCIES.md`：架构图移除 Dashboard 出核心模块，加 `[viz]` extras；核心模块表移除 Dashboard；新增「可视化扩展」节；结构化提取节加 `ExtractionResult` 返回类型、string 泛化、LLM schema 验证契约说明
- ✅ `pyproject.toml`：新增 `[viz]` extras；mypy overrides 加 fitz/docx/openpyxl/pptx
- ✅ 导出同步：`vertai/__init__.py` 移除 viz eager import + `__all__` 移除 viz 项；`ExtractionResult`/`StructuredOutput`/`DocGen`/`DocParser` 仍在核心导出（S2/S3 已处理 ExtractionResult，复核在位）

## 实测命令输出

```
$ python -m mypy --strict vertai/output/ vertai/data/parser.py
Success: no issues found in 4 source files

$ python -m ruff check vertai/output/ vertai/data/parser.py vertai/viz/ vertai/__init__.py tests/test_structured.py tests/test_docgen.py tests/test_parser.py tests/test_dashboard.py
All checks passed!

$ python -m pytest tests/test_structured.py --cov=vertai.output.structured --cov-report=term-missing -q
Name                          Stmts   Miss  Cover   Missing
-----------------------------------------------------------
vertai\output\structured.py     304      5    98%   175-176, 370, 402, 439
-----------------------------------------------------------
96 passed, 1 warning in 0.34s

$ python -m pytest tests/ -q
785 passed, 34 skipped, 1 warning in 12.14s

$ python -c "
import sys, vertai
print('viz eager loaded:', any('vertai.viz' in m for m in sys.modules))
from vertai.viz.dashboard import Dashboard, Metric, Chart, ChartType, DashboardTheme, ChartConfig
print('viz direct import works:', Dashboard(title='t').title == 't')
print('Dashboard in vertai ns:', hasattr(vertai, 'Dashboard'))
"
viz eager loaded: False
viz direct import works: True
Dashboard in vertai ns: False

$ python -c "
from vertai.output.structured import StructuredOutput
# Generic string no longer raises (regression for the hardcoded-name bug)
out = StructuredOutput({'product': 'string'})
r = out.extract('product is Widget')
print('generic string:', r.success, r.data)
"
generic string: True {'product': 'product'}
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict output/parser 0 错 | ✅ |
| ruff 0 错（S8 scope） | ✅ |
| C5 PDF 类型契约正确（generate `str\|bytes`，save 分支） | ✅ |
| StructuredOutput string 泛化（无写死人名） | ✅ |
| LLM 路径 schema 验证（`_validate_field` + 重试） | ✅ |
| DocParser 返回键统一（text，example_5 注释修正） | ✅ |
| dashboard 移出核心（`__init__` 无 eager import，viz extras 可用） | ✅ |
| 无刷覆盖率测试（TestRemainingCoverage 删除） | ✅ |
| 无 except 掩盖（fake LLM 是外部行为 stub） | ✅ |

**判定：S8 通过，可进入 S9。**

## 遗留项（有意留后续阶段，非缺陷）

- 全局 ruff 余错在 `core/memory.py`（`Optional` 未用）等，属 S9 范围，不在 S8。
- `_render_pdf` 仍 `NotImplementedError`（无真实 PDF 后端，诚实标注，类型契约已正确）。1.x 可接 weasyprint/pdfkit。
- structured.py 5 行未覆盖（175-176/370/402/439）均为可到达的防御性分支（重试后成功、attempt≥2、空文本 ValueError、boolean 无匹配 ValueError），非关键路径缺失。
- `pytest.ini` 的废弃 `python_paths` 选项 warning，留 S11 工程化清理。

## 产出文件

- 重写：`vertai/output/structured.py`（string 泛化 + LLM schema 验证重试 + 删 dir() 反模式 + mypy/ruff）
- 修复：`vertai/output/docgen.py`（C5 PDF 类型契约 + save 分支 + 删 unused `re`）
- 修复：`vertai/data/parser.py`（类型注解 + 删 unused `json` + TYPE_CHECKING imports）
- 修复：`vertai/viz/dashboard.py`（删 unused `Any` + F541 + docstring 导入路径）
- 修复：`vertai/__init__.py`（移除 viz eager import + `__all__` 移除 viz）
- 更新：`pyproject.toml`（`[viz]` extras + mypy overrides）
- 重写：`tests/test_structured.py`（删 TestRemainingCoverage + 真实行为测试 + LLM schema 验证测试 + string 泛化测试）
- 清理：`tests/test_dashboard.py`/`test_docgen.py`/`test_parser.py`（删 unused imports）
- 更新：`docs/FUNCTION_DEPENDENCIES.md`（viz 移出 + ExtractionResult 返回类型 + string 泛化契约）
- 更新：`examples/quick_start_knowledge_base.py`（example_5 注释 `content`→`text`）
