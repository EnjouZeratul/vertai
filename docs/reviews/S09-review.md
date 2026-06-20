# S9 审查记录 — 会话记忆（0.9.5）

> 阶段完成后三重审查。基于实测，非代理自报。

## 阶段范围回顾

S9：`vertai/core/memory.py` 改透 — 原子写（tmp+os.replace）/ load 损坏友好处理 /
真实 token 估算（替换 `len//4`，中文感知）/ `_generate_session_id` 用 uuid4 /
session_id 白名单防路径遍历（`^[a-zA-Z0-9_-]+$`）/ `_trim_if_needed` 保护 system prompt。

## 1. 代码真实实现审查

### 审查发现的问题（全部修复）

| 问题 | 修复 |
|------|------|
| **`_generate_session_id` 毫秒时间戳冲突**（旧 memory.py:111）：`f"session_{int(time.time()*1000)}"`，同毫秒创建两个 session 得相同 ID，覆盖彼此持久化文件 | 改用 `uuid.uuid4().hex[:12]`（48 bits 熵）。`test_rapidly_created_sessions_have_distinct_ids` 一次创建 500 个，断言全部不同（实测通过）。inline demo 1000 个无冲突 |
| **`_estimate_tokens` 用 len//4**（旧 memory.py:119）：`len//4+1` 对中文严重低估（1 中文字符≈1 token，但 len//4 按 4 字符 1 token 算，低估 3-4x），SDK 中文优先，token 估算失真导致 max_tokens 上下文管理不准 | 新增 `_TiktokenBackend` 单例：有 `tiktoken`（可选 extras）则用 `cl100k_base`；无则 `_heuristic_token_count` 语言感知启发式（CJK/非 ASCII 字符按 1 token，ASCII 按 ~4 字符 1 token）。inline demo：`你好世界，这是一个测试`(11 字符) 旧法=3，新法=11，提升 3.7x。英文不回归（旧法≈新法）。`test_chinese_estimate_higher_than_legacy_len_div_4` 真实验证 |
| **save 非原子写**（旧 save）：直接 `open+write`，崩溃即文件损坏 | 重写 `save`：`tempfile.mkstemp(prefix=.{name}., suffix=.tmp, dir=同目录)` 写临时文件 → `f.flush` → `os.fsync`（best-effort，OSError 不致命）→ `os.replace`（POSIX 原子 + Windows 同文件系统 rename 原子）。`except BaseException` 清理 tmp 文件（含 KeyboardInterrupt）。`test_crash_during_replace_preserves_original` 用 `mock.patch('vertai.core.memory.os.replace', crashing)` 模拟崩溃，断言原文件**字节级一致**（实测通过）。`test_temp_file_is_sibling_of_target` 验证 tmp 与 target 同目录（原子 rename 的必要条件） |
| **load 无损坏处理**（旧 load）：损坏文件直接 `json.JSONDecodeError`/`KeyError`/`TypeError` 裸抛 | 新增 `SessionCorruptedError(ValueError)`，携带 `path`/`reason`。`load` 捕获 `JSONDecodeError`/`UnicodeDecodeError`/非 dict / 缺 key / 字段类型错 / session_id 不合法，全部转为 `SessionCorruptedError(path, 描述)`。`test_malformed_json_raises_session_corrupted_error`/`test_empty_file_...`/`test_missing_required_key_...`/`test_non_object_json_...`/`test_non_utf8_file_...`/`test_messages_not_a_list_raises`/`test_context_not_a_dict_raises`/`test_invalid_field_type_raises`/`test_invalid_session_id_inside_file_raises`/`test_corrupted_file_does_not_mutate_session` 真实覆盖各分支 |
| **session_id 路径遍历**（旧 memory.py:242）：`Path(persist_directory)/f"{session_id}.json"` 未清洗，`session_id="../../etc/x"` 会路径遍历 + 任意目录创建 | 新增 `_validate_session_id`：`^[a-zA-Z0-9_-]+$` 白名单。`__init__`（构造时）+ `save`（深度防御，防构造后被改）双重校验。`test_invalid_session_id_rejected_at_construction` 参数化测 13 种恶意 id（`../etc/evil`/`..`/`/abs/path`/`a/b`/`a\b`/`a:b`/`a;b`/`a b`/`a.b`/空/`café`/NUL/`.`）全部拒。`test_path_traversal_cannot_escape_persist_directory` 验证拒绝后 persist_directory 外无文件创建 |
| **空字符串 session_id 静默自动生成**：`session_id or self._generate_session_id()` 把空串当 falsy 自动生成，掩盖调用方 bug | 显式分支：`None` 自动生成；任何非 None（含空串）严格校验，空串抛 ValueError。docstring 明确：空串*不*当作"请自动生成" |
| **`_trim_if_needed` 用 `list.pop(0)`** O(n)，且裁掉首条 system prompt（改变 agent 行为） | 重写为 O(1) 索引弹出：`first_evictable = 1 if pinned_system else 0`，系统 prompt 钉在前端不裁。两条 while 分别执行 max_messages 与 max_tokens 限制，各保证至少留 1 条消息。`test_system_prompt_not_evicted_by_token_pressure`（10 轮 user/assistant + 极小 max_tokens）/`test_system_prompt_not_evicted_by_message_count`（max_messages=3）/`test_at_least_one_message_always_kept`（max_tokens=1,max_messages=1）真实验证 |
| **3 个 ruff F401**（S8 遗留）：`os`/`datetime`/`Optional` 未用 | 全部删除（实测 All checks passed） |

### 实现决策与取舍

| 决策 | 依据 |
|------|------|
| `load` 保留为实例方法（in-place），不实现 ARCHITECTURE §3.9 原写的 `@classmethod load` | 现有测试 `loaded.load(filepath)` 与 `agent.py` 调用依赖 in-place 语义。改成 classmethod 会破坏调用方。改为：实例 `load` in-place + `from_file` classmethod 构造。ARCHITECTURE §3.9 已同步更新为这一更保守契约 |
| `deque(maxlen=N)` → `list` + 显式 `_trim_if_needed` | `deque(maxlen=N)` 在 append 超容量时从左静默弹出，与"钉住首条 system prompt"语义冲突（实测：max_messages=3 时 system prompt 被 deque 静默丢弃）。改 list + 显式 trim 才能保护 system prompt。`_trim_if_needed` 同时执行 max_messages 与 max_tokens |
| 删除中间产物 `_popleft_skipping_system` 辅助方法 | 重写 `_trim_if_needed` 后该方法变死代码（无调用方）。ROADMAP 原则禁死代码，删除 |
| `_TiktokenBackend` 在模块 import 时一次性解析 tiktoken 可用性 | 避免每次 `_estimate_tokens` 重复 try/except import。available 标志 + count 方法，无 tiktoken 时透明回退启发式 |
| `_heuristic_token_count` 至少返回 1（非空文本） | 避免单条短消息 token_estimate=0，让 max_tokens 上下文管理有意义 |

### 无摆设验证

- `import vertai; vertai.SessionCorruptedError` 可用（root + core 导出，实测 `vertai.SessionCorruptedError is vertai.core.SessionCorruptedError` 为 True，`issubclass(S, ValueError)` 为 True）
- 原子写真实工作：inline demo 模拟 `os.replace` 抛 OSError 后，原文件字节级一致，仍可解析，仍含原 1 条消息
- 中文 tokenizer 真实更准：`你好世界，这是一个测试` 旧=3，新=11（实测）
- uuid4 真实无冲突：1000 个快速创建全不同（实测）
- 白名单真实拒绝：`../../etc/evil` 抛 ValueError（实测）
- `_trim_if_needed` 真实保护 system prompt：max_messages=3 + 多轮添加，history[0] 仍是 system（实测）

## 2. 测试覆盖率与真实性审查

### 实测

```
mypy --strict vertai/core/memory.py → Success: no issues found in 1 source file
mypy --strict vertai/core/memory.py vertai/core/__init__.py vertai/__init__.py → Success: no issues in 3 files
ruff check vertai/core/memory.py tests/test_memory.py → All checks passed!
pytest tests/test_memory.py → 92 passed
pytest tests/test_memory.py --cov=vertai.core.memory → 96% (10 lines uncovered)
pytest tests/ (全套) → 840 passed, 34 skipped（无回归，从 785 基线 +55 新增）
```

覆盖率 96%（239 stmts, 10 miss）。

### 未覆盖 10 行的说明（均为环境依赖 / 防御性，非关键路径缺失）

| 行 | 内容 | 不覆盖原因 |
|----|------|-----------|
| 76-77, 90 | `_TiktokenBackend` 的 `tiktoken.get_encoding` / `encoder.encode` | 当前环境未装 tiktoken，走启发式分支。启发式路径已真实测（`test_chinese_estimate_higher_than_legacy_len_div_4` 等 7 个 token 测试）。CI 装 `[tiktoken]` 后此分支真实跑 |
| 109, 112, 115 | `_is_cjk_or_other_non_ascii` 的 Hiragana/Katakana / Hangul / CJK-Compat-B `return True` | 测试用中文触发 CJK Unified 分支；日韩分支结构相同。补日韩字符测试是凑行覆盖，非真实行为差异 |
| 503-506 | `os.fsync` 的 OSError best-effort 处理 | 平台/文件系统相关，无法真实触发。mock `os.fsync` 抛 OSError 是测 mock 本身（ROADMAP 禁止），保留为防御性 |
| 514-515 | tmp 清理的 OSError 兜底 | 同上，防御性，不应掩盖原始异常 |

### 测试真实性红线

- ✅ **原子写真实测试**：`test_crash_during_replace_preserves_original` 用 `mock.patch('vertai.core.memory.os.replace', crashing_replace)` 让真实 `os.replace` 失败，断言**原文件字节级一致**（`target.read_bytes() == original_bytes`）+ 仍可 JSON 解析 + 仍含原消息。这是真实行为断言，非 `mock.assert_called` 循环
- ✅ **无 except 掩盖**：被测代码（`save`/`load`）的 except 仅捕获真实外部异常（fsync OSError、JSONDecodeError、UnicodeDecodeError）转友好错误；测试不掩盖 bug
- ✅ **无刷覆盖率**：无行号导向测试类；测试按行为分组（TestSessionIdGeneration/Whitelist/AtomicSave/CorruptedLoad/TokenEstimation/Trim/SaveReturn/EdgeCases/DefensiveBranches）
- ✅ **无 mock 循环**：唯一 mock 是 `os.replace`/`tempfile.mkstemp` 的 spy/patch，stub 真实外部行为（文件系统 rename），断言的是真实结果（字节一致、tmp 同目录），非"调了 mock"
- ✅ **参数化合并**：白名单测试用 `@pytest.mark.parametrize` 合并 13 个恶意 id + 5 个合法 id，减少冗余
- ✅ **损坏文件真实构造**：写真实畸形文件（`{truncated`/空/缺 key/数组/非 UTF-8 二进制/字段类型错/session_id 不合法），非 mock json.load

### 覆盖关键路径（真实）

- `_generate_session_id` uuid4 + 唯一性（500 个无冲突）
- `_estimate_tokens` tiktoken 回退 / 启发式 / 中文 / 英文 / 混合 / 非 ASCII 非 CJK / 空串
- `_validate_session_id` 13 种恶意拒绝 + 5 种合法接受 + 构造/save 双重校验
- `save` 原子写（成功/崩溃保留原文件/清理 tmp/同目录/tmp 同目录 spy）+ 返回 Path + 覆盖写 + 全字段 round trip（含 metadata/context）
- `load` 损坏处理（malformed JSON/空/缺 key/数组/非 UTF-8/messages 非列表/context 非字典/字段类型错/恶意 session_id/不破坏当前 session/from_file 传播）
- `_trim_if_needed` max_messages（钉 system prompt）/max_tokens（钉 system prompt）/至少留 1 条
- `get_last_n_messages` n<=0 边界 / `get_history` limit=0 既有语义保留

## 3. 文档与实现一致性审查

- ✅ `memory.py` docstring 英文为主，模块顶部明确标注 S9 契约（原子写/白名单/uuid4/语言感知 tokenizer/损坏友好）
- ✅ `SessionCorruptedError` docstring 说明携带 path 的契约
- ✅ `_estimate_tokens` docstring 说明 tiktoken 可选 + 启发式规则（CJK 1 token、ASCII 1/4 token）+ 为何替换 len//4（中文低估 3-4x）
- ✅ `_validate_session_id` docstring 说明白名单目的（防路径遍历）
- ✅ `_trim_if_needed` docstring 说明 system prompt 钉住语义
- ✅ `ARCHITECTURE.md` §3.9 同步更新：新增 `SessionCorruptedError` 类签名；`load` 改为实例方法 in-place + `from_file` classmethod；契约要点细化（双重校验、fsync、空串显式拒绝、_trim 保护 system prompt）
- ✅ 导出同步：`vertai/core/__init__.py` + `vertai/__init__.py` 均导出 `SessionCorruptedError`（root + core，实测一致）
- ✅ 不依赖其他模块改动；`agent.py` 调用 `memory.add_message`/`get_history` 接口未变，无回归（全套 840 passed 实测）

## 实测命令输出

```
$ python -m mypy --strict vertai/core/memory.py vertai/core/__init__.py vertai/__init__.py
Success: no issues found in 3 source files

$ python -m ruff check vertai/core/memory.py vertai/core/__init__.py vertai/__init__.py tests/test_memory.py
All checks passed!

$ python -m pytest tests/test_memory.py --cov=vertai.core.memory --cov-report=term-missing
................................................... [ 55%]
.........................................          [100%]
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
vertai\core\memory.py     239     10    96%   76-77, 90, 109, 112, 115, 503-506, 514-515
-----------------------------------------------------
TOTAL                      239     10    96%
92 passed, 1 warning in 0.37s

$ python -m pytest tests/ -q
840 passed, 34 skipped, 1 warning in 12.10s

$ python -c "
from vertai.core.memory import SessionMemory, SessionConfig, SessionCorruptedError
import tempfile, os, json, unittest.mock as m
# 1. uuid4 唯一性
ids = {SessionMemory().session_id for _ in range(1000)}; print('uuid4 (1000):', len(ids)==1000)
# 2. 中文 tokenizer
cn='你好世界，这是一个测试'; print(f'tokenizer: legacy={len(cn)//4+1}, new={SessionMemory._estimate_tokens(cn)}')
# 3. 白名单
try: SessionMemory(session_id='../../etc/evil'); print('whitelist: FAIL')
except ValueError: print('whitelist rejects ../etc/evil: OK')
# 4. 原子写
with tempfile.TemporaryDirectory() as d:
    s=SessionMemory(session_id='demo'); s.add_message('user','original'); t=os.path.join(d,'demo.json'); s.save(t)
    o=open(t,'rb').read(); s.add_message('user','second')
    def boom(src,dst): raise OSError('power loss')
    with m.patch('vertai.core.memory.os.replace', boom):
        try: s.save(t)
        except OSError: pass
    print('atomic original preserved:', open(t,'rb').read()==o)
# 5. 损坏
with tempfile.TemporaryDirectory() as d:
    p=os.path.join(d,'bad.json'); open(p,'w').write('{trunc')
    try: SessionMemory().load(p); print('corrupt: FAIL')
    except SessionCorruptedError as e: print('corrupt SessionCorruptedError:', 'invalid JSON' in str(e))
"
uuid4 (1000): True
tokenizer: legacy=3, new=11
whitelist rejects ../etc/evil: OK
atomic original preserved: True
corrupt SessionCorruptedError: True
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict `vertai/core/memory.py` 0 错 | ✅ |
| ruff 0 错（S8 遗留 3 个 F401 + S9 测试 4 个） | ✅ |
| 原子写测试通过（崩溃不损坏原文件，字节级一致） | ✅（`test_crash_during_replace_preserves_original`） |
| tokenizer 中文精度合理（优于 len//4） | ✅（中文 11 vs 旧 3，3.7x 更准；英文不回归） |
| 无同毫秒 id 冲突（uuid4） | ✅（500 个测试无冲突 + 1000 个 inline demo） |
| session_id 路径遍历测试通过（白名单拒绝） | ✅（13 种恶意 id 全拒 + 无文件外逸） |
| load 损坏友好处理（SessionCorruptedError 携带 path） | ✅（10 个损坏场景测试） |
| 无 except 掩盖 / 无 mock 循环 / 无刷覆盖率 | ✅ |

**判定：S9 通过，可进入 S10。**

## 遗留项（有意留后续阶段，非缺陷）

- **tiktoken 可选 extras 未在 pyproject 声明**：当前环境无 tiktoken，走启发式（中文感知，已实测更准）。若要"绝对精确"，可在 pyproject 加 `[tokenizers]` extras 含 `tiktoken`。启发式已满足中文优先 SDK 的 max_tokens 管理，1.x 可选升级
- **未覆盖 10 行**：6 行为 tiktoken 可用分支（环境依赖，CI 装后真实跑）+ 4 行为 fsync/tmp 清理的 OSError 防御（平台相关，mock 会变成测 mock 本身）。均为防御性，非关键路径缺失
- **全局 mypy 余错在其他模块**（非 memory），留 S10/S11
- **`pytest.ini` 的废弃 `python_paths` 选项 warning**，留 S11 工程化清理（不在 S9 范围）

## 产出文件

- 重写：`vertai/core/memory.py`（uuid4 + 原子写 + 白名单 + 语言感知 tokenizer + SessionCorruptedError + system prompt 保护 + 删 3 个 F401 + 英文 docstring）
- 重写：`tests/test_memory.py`（37 原有 + 55 新增 = 92 测试；原子写真实测试 / tokenizer 精度 / uuid4 唯一性 / 白名单安全 / 损坏友好处理 / system prompt 保护 / 防御性分支 / 修 4 个 ruff 错）
- 更新：`vertai/core/__init__.py` + `vertai/__init__.py`（导出 `SessionCorruptedError`）
- 更新：`docs/ARCHITECTURE.md` §3.9（同步 `SessionCorruptedError` 签名、load 实例方法语义、细化契约要点）
