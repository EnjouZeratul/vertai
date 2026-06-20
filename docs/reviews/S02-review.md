# S2 审查记录 — LLMProvider 抽象与重构（0.3.0）

> 阶段完成后三重审查。基于实测，非代理自报。

## 阶段范围回顾

S2：把 `core/llm.py` 重构为 `LLMProvider` ABC + 4 适配器，async-first，tool calling，修 C4/chat/死代码/mypy。被 S3/S4/S5 依赖的核心。

## 1. 代码真实实现审查

### 契约兑现（ARCHITECTURE 3.1）
- ✅ `LLMProvider` ABC：generate/stream/agenerate/astream，接受 `list[ChatMessage]`，支持 `tools: list[ToolSpec]`
- ✅ 4 适配器：Ollama/Anthropic/DeepSeek/OpenAI，工厂 `create_provider(config)`
- ✅ 类型：ChatMessage（含 coerce）、GenerateResult（含 tool_calls）、StreamEvent 联合类型（TextDelta/ToolUse/Done，非裸 str）、ToolSpec、ToolCall
- ✅ async 真实 `httpx.AsyncClient`（实测：agenerate 是 async def + AsyncClient）；sync 独立 httpx.Client（非 asyncio.run 包装）
- ✅ C4 修复：OpenAIProvider 用真实 `/chat/completions` + `Bearer`（实测断言：源码含 /chat/completions + Bearer，无 x-api-key 路由错误）
- ✅ tool calling 协议层：Anthropic tools / OpenAI functions / Ollama tools 三格式 payload + tool_use/tool_calls 解析
- ✅ 向后兼容：LLMEngine facade 委托 create_provider，旧签名保留；chat dict/ChatMessage 通过 coerce 统一

### 死代码/反模式清除
- ✅ 删除 `_model_cache`、`current_block_type`（实测 grep 0 匹配）
- ✅ 哨兵默认值改 None（temperature/top_p/top_k 不硬编码比较）
- ✅ LLMConfig 改用 `model_validator(mode="before")` 注入环境变量（非 __init__ 重写反模式）
- ✅ ModelInfo core 侧重命名 LLMModelInfo（local 侧 S7 处理）

### 无"声称实现实为摆设"
- ✅ tool calling 有端到端测试（fake LLM 返回 tool_use，验证解析）
- ✅ OpenAI 路由有 MockTransport 测试验证真实 URL+Bearer

## 2. 测试覆盖率与真实性审查

### 实测
- 全套：663 passed, 17 skipped（17 skip 全为 DeepSeek 集成测试无 VERTAI_API_KEY，诚实 skip，非假装）
- S2 新测试：test_provider.py 46 passed，test_llm.py 50 passed
- tool calling + async 测试真实跑通：`pytest -k "tool_use or agenerate or astream"` → 7 passed（非 skip）
- async 测试用真实 httpx.AsyncClient + MockTransport（非同步包装）

### 测试真实性红线
- ✅ 无 `except Exception: pass` 掩盖（grep 确认 test_provider/test_llm 0 匹配）
- ✅ 修复 test_init_invalid_model_name 死代码（原 370-371 不可达）
- ✅ mock 用 MockTransport stub 真实 HTTP 行为，验证 payload/URL/headers，非"验证调了 mock"循环
- ✅ 集成测试诚实：有 VERTAI_API_KEY 时真实执行（@requires_api_key），本地无 key 诚实 skip

## 3. 文档与实现一致性审查

- ✅ docstring 英文为主
- ✅ 真实模型名：deepseek-v4-flash → deepseek-chat（docstring/示例/集成测试，实测 example_1 已改）
- ✅ sync/async API 文档明确
- ✅ 导出同步：__init__.py 导出 LLMProvider/4适配器/ToolSpec/ToolCall/StreamEvent 等
- ✅ 示例代码无崩溃（example 2-9 实测无 error/traceback）

## 实测命令输出

```
mypy --strict vertai/core/provider.py vertai/core/llm.py → Success: no issues found (2 files)
ruff check (S2 文件) → All checks passed!
全套测试 → 663 passed, 17 skipped
tool calling + async 测试 → 7 passed (真实跑通)
全局 mypy → 63 errors in 6 files (S2 文件 0 错；剩余在其他 6 文件，留各自阶段)
C4 验证 → OpenAI /chat/completions + Bearer PASS
chat coercion → dict->ChatMessage PASS
async 真实 → agenerate async def + AsyncClient PASS
死代码 → _model_cache/current_block_type 0 匹配
except 掩盖 → 0 匹配
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict 该模块 0 错 | ✅ |
| ruff 0 错 | ✅ |
| async 真实可用 | ✅ |
| OpenAI 路由正确（C4） | ✅ |
| chat 不崩溃（dict/ChatMessage 一致） | ✅ |
| tool calling 协议层工作 | ✅ |
| 集成测试有 key 时真实执行 | ✅（诚实 skip 无 key） |
| docstring 与实现一致 | ✅ |
| 无 except 掩盖、无死代码 | ✅ |

**判定：S2 通过，可进入 S3。**

## 遗留项（有意留后续阶段，非缺陷）

- C1（KnowledgeQA._get_llm 用 model=）→ S3（knowledge_qa 改依赖 LLMProvider）
- 全局 mypy 63 错在其他 6 文件 → S3(vector/scenarios)/S4-S9 各自修
- Ollama async 路径仍用同步 _is_running 探测（一次廉价 GET，async 生成本身真实）→ 可接受，非阻塞
- LLMEngine facade 访问 provider._get_sync_client（同包私有，有 docstring 说明）→ 可接受

## 产出文件

- `vertai/core/provider.py`（新）— LLMProvider ABC + 4 适配器 + 工厂 + 类型
- `vertai/core/llm.py`（重写）— LLMEngine facade
- `vertai/core/__init__.py`、`vertai/__init__.py` — 导出同步
- `tests/test_provider.py`（新，46 测试）
- `tests/test_llm.py`（重写，50 测试）
- `tests/test_deepseek_integration.py` — 真实模型名 + 去 except 掩盖
- `examples/quick_start_knowledge_base.py` — 假模型名修复
