# S4 审查记录 — Tool 抽象 + 内置工具（0.5.0）

> 阶段完成后三重审查。基于实测，非代理自报。对标 OpenAI Agents SDK `@function_tool` 行业标准。

## 1. 代码真实实现审查

### 契约兑现（ARCHITECTURE 3.6，对标 OpenAI Agents SDK）
- ✅ `Tool` ABC：name/description/parameters(JSON Schema)/execute/aexecute/to_spec
- ✅ `FunctionTool`：从 Python 函数自动生成
- ✅ `@tool` 装饰器：inspect 签名 + get_type_hints(include_extras=True) + docstring 解析（google/numpy/sphinx，自实现无新依赖）+ pydantic 动态模型
- ✅ **Field 约束完整传播**：实测 Annotated[int, Field(ge=0,le=100)] → schema minimum/maximum=0/100，description 来自 Field（实测）
- ✅ **timeout（per-tool）**：error_as_result（默认）/ raise_exception（ToolTimeoutError）；sync daemon thread join + async asyncio.wait_for（实测 5 passed）
- ✅ **failure_error_function**：默认友好消息/自定义/None re-raise
- ✅ ToolRegistry：register/get/to_specs（确定性排序，匹配 provider.py ToolSpec）

### 内置工具（开箱即用，覆盖垂直领域集成需求）
- ✅ calculator：ast 安全求值（非 eval），白名单算子；实测拒绝 `__import__/open/os.system/__class__` 4 个危险输入
- ✅ file_read/file_write：pathlib + 路径遍历防护（resolve+relative_to+base_dir）+ max_bytes
- ✅ http_request：httpx 白名单方法 + async 真实 AsyncClient
- ✅ web_search：DuckDuckGo HTML（无 key 无新依赖），async 真实 AsyncClient

### 无摆设
- ✅ calculator 真实安全（实测拒绝危险输入）
- ✅ tool calling 端到端真实（fake LLM → tool_use → execute，9 passed）

## 2. 测试覆盖率与真实性审查

### 实测
- 全套：720 passed, 31 skipped（vs S3 基线 646 passed → 新增 74 测试全过，0 回归）
- test_tool.py：74 passed
- timeout：5 passed（error_as_result + raise_exception，sync+async 四路径）
- tool calling 端到端：9 passed
- docstring 三风格：5 passed

### 测试真实性红线
- ✅ 无 except Exception:pass 掩盖（tool.py 的 except Exception 在 get_type_hints graceful fallback，有测试验证非掩盖）
- ✅ schema 生成真实断言（Field 约束在 schema 里）
- ✅ calculator 安全性参数化测试（合法+危险输入）
- ✅ file 用 tmp_path，http/web_search 用 MockTransport stub 真实 wire 行为
- ✅ 无刷覆盖率测试类

## 3. 文档与实现一致性审查

- ✅ docstring 英文为主，内置工具有清晰 docstring（供 schema 推导）
- ✅ 导出同步（Tool/FunctionTool/ToolRegistry/tool/5 内置工具）
- ✅ ARCHITECTURE 3.6 已对标行业标准，实现兑现

## 实测命令输出

```
mypy --strict vertai/core/tool.py vertai/core/tools/ → Success: no issues (6 files)
ruff → All checks passed!
全套测试 → 720 passed, 31 skipped (0 回归)
calculator 安全 → 4 危险输入全拒
Field 约束传播 → minimum=0, maximum=100 PASS
timeout → 5 passed
tool calling 端到端 → 9 passed
docstring 三风格 → 5 passed
全局 mypy → 24 errors in 4 files (S4 文件 0 错；剩余 local/parser/output 留各自阶段)
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict S4 文件 0 错 | ✅ |
| ruff 0 错 | ✅ |
| @tool 装饰器工作（函数+docstring+Field schema） | ✅ |
| JSON Schema 正确（含 Field 约束） | ✅ |
| timeout 工作 | ✅ |
| failure_error_function 工作 | ✅ |
| tool calling 端到端 | ✅ |
| 内置工具真实可用 | ✅ |
| 无 except 掩盖、无刷覆盖率 | ✅ |

**判定：S4 通过，可进入 S5。**

## 遗留项（有意留 1.x，非缺陷）

- tool_namespace 分组、defer_loading/ToolSearch、is_enabled 条件启用、agents-as-tools、needs_approval、custom_output_extractor → 1.x（1.0 预留扩展点）
- web_search 子模块遮蔽（生产用户拿工具是预期，仅 patch 需 sys.modules）→ S10 评估
- 同步 timeout daemon thread 无法强杀（Python 固有局限，async 可取消）

## 产出文件

- `vertai/core/tool.py`（新）— Tool/FunctionTool/@tool/ToolRegistry/ToolTimeoutError
- `vertai/core/tools/`（新）— calculator/file/http/web_search + default_registry
- `tests/test_tool.py`（新，74 测试）
- `vertai/__init__.py`、`vertai/core/__init__.py` — 导出更新
