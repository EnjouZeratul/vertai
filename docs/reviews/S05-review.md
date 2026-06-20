# S5 审查记录 — Agent 抽象 + Callbacks（0.6.0）

> 阶段完成后三重审查。基于实测，非代理自报。Agent SDK 分水岭（依赖 S2 Provider + S4 Tool）。

## 阶段范围回顾

S5：新增 `core/agent.py`（最小可用 tool-calling agent loop + max_iterations 防失控 + AgentResult）+ `core/callbacks.py`（Protocol 钩子 + LoggingCallback + TokenCountCallback）。契约在 S1 已定义。

## 1. 代码真实实现审查

### 契约兑现（ARCHITECTURE 3.7 + 3.8）
- ✅ `Agent.__init__(provider, tools, *, system_prompt, max_iterations=10, callbacks, memory)` — 签名严格兑现契约；`memory` 可选（S9 完善）
- ✅ `run(input) -> AgentResult` / `arun(input) -> AgentResult` — sync + async 真实双 API
- ✅ **真实 tool-calling loop**：generate(tools) → 若 tool_calls 非空则 registry.call/acall 执行 → assistant 消息（含 tool_calls 描述符）+ tool_result 消息追加 → 再 generate → 直至无 tool_call / max_iterations（实测：test_run_executes_multi_turn_tool_loop 断言第二次 generate 调用收到 tool 消息）
- ✅ **max_iterations 强制上界**：`for iteration in range(1, max+1)` + `for...else` 检测耗尽 → truncated=True（实测：永远返回 tool_call 的脚本在 max=3 时 iterations=3 truncated=True，不超过上界）
- ✅ **tool 失败不崩 agent**：registry.call 路由到 Tool.execute，S4 的 failure_error_function 默认返回友好字符串；agent loop 永不因 tool 异常崩溃（实测：test_tool_failure_surfaces_friendly_message_and_loop_continues，boom() 抛 ValueError → 友好消息 → loop 继续 → 产出 "recovered"）
- ✅ `AgentResult`：final_output / tool_calls_history（每轮记录 iteration+tool_call+result）/ iterations / total_tokens / elapsed_seconds / truncated / finish_reason
- ✅ **token 累计**：每轮 `result.total_tokens` 累加（fallback prompt+completion）；实测 test_token_count_callback_accumulates_via_agent 总 30 = 10+20
- ✅ Callbacks：`Callback` Protocol（on_agent_start/on_llm_start/on_llm_end/on_tool_start/on_tool_end/on_agent_end）；Protocol 用 default 空 body 允许部分实现；`dispatch` 用 getattr 调度（部分实现工作）
- ✅ LoggingCallback（记录事件顺序）+ TokenCountCallback（累计 token）
- ✅ system_prompt 前置；memory 可选（seed 历史 + 记录新消息，S9 才完善原子写/tokenizer）

### 无"声称实现实为摆设"
- ✅ loop 真实多轮（不是单次调用）：实测 echo tool_call 第一轮 → 第二轮 generate 看到 tool 消息 → 终止
- ✅ max_iterations 真实防失控：永远 tool_call 脚本被限到 max
- ✅ callbacks 真实触发且顺序正确（实测 LoggingCallback 记录 8 个事件顺序：start→llm_start→llm_end→tool_start→tool_end→llm_start→llm_end→agent_end）
- ✅ async arun 真实（agenerate + acall，实测 test_arun_runs_async_tool_loop）
- ✅ 端到端：真实 @tool calculator-like add(40,2) → "42"

## 2. 测试覆盖率与真实性审查

### 实测
- 全套：747 passed, 31 skipped（vs S4 基线 720 passed → 新增 27 S5 测试全过，0 回归；skipped 数不变）
- test_agent.py：16 测试（loop 多轮 / 即时终止 / 一轮多 tool_call / max_iterations 截断 / max<1 报错 / max=1 单轮 / tool 失败恢复 / callbacks 顺序 / token 累计 / arun 多轮 / arun max_iterations / system_prompt / memory seed+record / 无 tools 单轮 / registry 可变 / calculator 集成）
- test_callbacks.py：11 测试（Protocol runtime_checkable / 部分实现调度 / None+空 noop / 空 Protocol 子类静默 / LoggingCallback 顺序+isinstance / TokenCount 累计+fallback+reset+部分实现 / dispatch 传播异常）

### 测试真实性红线
- ✅ 无 `except Exception: pass` 掩盖（callbacks.dispatch 故意不吞异常，测试 test_dispatch_propagates_callback_errors 验证用户 callback 异常传播）
- ✅ FakeLLMProvider stub 真实行为：返回真实 GenerateResult 含 tool_calls，agent 真实驱动 generate，registry 真实执行 Tool.execute；非"验证调了 mock"循环
- ✅ max_iterations 防失控用真实"永远 tool_call"脚本验证迭代数被限制（非 mock 计数）
- ✅ tool 失败用真实 raise ValueError（非 mock raise）验证友好恢复
- ✅ 集成测试用真实 @tool add（非 mock 工具）

## 3. 文档与实现一致性审查

- ✅ docstring 英文为主（agent.py / callbacks.py）
- ✅ ARCHITECTURE 3.7 + 3.8 契约兑现（Agent 签名、Callbacks Protocol、AgentResult 字段全对齐）
- ✅ 导出同步：vertai/__init__.py + vertai/core/__init__.py 导出 Agent/AgentResult/Callback/LoggingCallback/TokenCountCallback
- ✅ Agent 使用示例（examples/agent_demo.py）：tool calling agent，无 key 时用 scripted fake provider 永远可跑，真实驱动 calculator tool_call → "42" → "The answer is 42."

## 设计决策（诚实记录）

### Callback Protocol 部分实现的 isinstance 语义
`@runtime_checkable` Protocol 的 `isinstance` 检查要求**所有**成员都在实例上。部分实现（如 TokenCountCallback 只实现 on_llm_end）的 `isinstance` 返回 False。这是 Protocol 的固有行为，不是缺陷：
- Agent 调度 hook 用 `getattr`（见 `dispatch`），**不**用 `isinstance`，所以部分实现运行时完全工作
- 测试诚实记录：test_token_count_callback_is_a_callback_protocol 断言 `not isinstance`（部分实现，by design），并验证 dispatch 真实工作
- docstring 明确说明此语义

### memory 集成（S9 前的最小可用）
memory 在 1.0 可选：seed 历史（排除已有 system 行避免 prompt drift）+ 把每条 user/assistant/tool 结果写回。SessionMemory 只接受 system/user/assistant 三种 role，tool 结果折叠为带 `[tool:name]` 前缀的 assistant 文本。完整原子写/真实 tokenizer 在 S9。

## 实测命令输出

```
mypy --strict vertai/core/agent.py vertai/core/callbacks.py → Success: no issues (2 files)
mypy --strict vertai/core/__init__.py vertai/__init__.py → Success: no issues (2 files)
ruff check vertai/core/agent.py vertai/core/callbacks.py tests/test_agent.py tests/test_callbacks.py → All checks passed!
全套测试 → 747 passed, 31 skipped (vs S4 720 passed, 0 回归, 新增 27 全过)
PYTHONPATH=. python examples/agent_demo.py → 真实驱动 calculator tool_call → "42" → "The answer is 42."
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict S5 文件 0 错 | ✅ |
| ruff 0 错 | ✅ |
| agent loop 正确执行 tool calling 序列（多轮） | ✅ |
| 终止条件工作（无 tool_call / max_iterations） | ✅ |
| tool 失败不崩 agent | ✅ |
| callbacks 触发且顺序正确 | ✅ |
| max_iterations 防失控 | ✅ |
| async arun 工作 | ✅ |
| token 统计累计 | ✅ |
| 无 except 掩盖、无刷覆盖率 | ✅ |
| 现有测试不回归 | ✅ (720→747 passed, 31 skip 不变) |

**判定：S5 通过。**

## 遗留项（有意留 1.x，非缺陷）

- multi-agent / handoff / agents-as-tools / needs_approval (Human-in-the-loop) → 1.x（1.0 预留扩展点，ARCHITECTURE 3.7 明示）
- OTel/tracing 全家桶 → 1.x（1.0 仅轻量 Callbacks，ARCHITECTURE 3.8 明示）
- streaming agent loop（astream 驱动 token 流式回传）→ 1.x（1.0 generate/agenerate 已足够）
- memory 完整可靠性（原子写、真实 tokenizer、uuid、session_id 路径遍历）→ S9
- isinstance 对部分实现返回 False（Protocol 固有，dispatch 用 getattr 兜底）→ 文档化，by design

## 产出文件

- `vertai/core/agent.py`（新）— Agent + AgentResult
- `vertai/core/callbacks.py`（新）— Callback Protocol + dispatch + LoggingCallback + TokenCountCallback
- `tests/test_agent.py`（新，16 测试）
- `tests/test_callbacks.py`（新，11 测试）
- `examples/agent_demo.py`（新）— tool calling agent 示例
- `vertai/__init__.py`、`vertai/core/__init__.py` — 导出 Agent/AgentResult/Callback/LoggingCallback/TokenCountCallback
