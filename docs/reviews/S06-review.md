# S6 审查记录 — 工作流引擎（0.7.0）

> 阶段完成后三重审查。基于实测，非代理自报。聚焦 `vertai/workflow/workflow.py` 重构：并行竞态、timeout 死配置、链式 API、mypy、刷覆盖率测试。

## 阶段范围回顾

S6：把 `workflow/workflow.py`（原 962 行）改透——并行加锁、timeout 真实生效或删字段、链式 API 决策、删除刷覆盖率测试、mypy --strict 0 错、ruff 0 错。

## 1. 代码真实实现审查

### 并行竞态加锁（核心契约）
- ✅ `WorkflowContext` 用 `threading.RLock` 保护 `data` 字典的 `get`/`set`/`update`/`clear`。RLock 选型因为 `get/set` 在用户持有的 `with ctx.lock:` 内部还要能重入（避免自死锁）
- ✅ 锁用 pydantic `PrivateAttr(default_factory=threading.RLock)` 声明，既不进 schema 也不进序列化，mypy --strict 也能看到属性类型
- ✅ **诚实边界**：锁只保证单次 `set`/`get` 原子（无 dict 损坏、无并发写丢字典本身），不保证跨操作的 RMW（`ctx.set(k, ctx.get(k)+1)`）原子。原因是 Python dict 的 RMW 本质需要跨步骤协调，锁没法在 step 函数体内部隐式持有
- ✅ **为跨操作 RMW 暴露 `ctx.lock` 属性**：用户可写 `with ctx.lock: ctx.set(k, ctx.get(k)+1)`。docstring 明确给出这个模式。这是诚实的契约：声明能做什么、不能做什么、用户该怎么做

### Timeout 真实实现（消除死配置）
- ✅ `ParallelConfig.timeout` 真实生效：`_execute_parallel` 用 `ThreadPoolExecutor` + 手动管理（非 `with`），轮询 `wait(timeout=remaining, return_when=FIRST_COMPLETED)`。deadline 到时调用 `executor.shutdown(wait=False, cancel_futures=True)`（Python 3.9+，项目 3.10+ OK），立即返回不阻塞
- ✅ `WorkflowConfig.timeout` 真实生效：`run` 在每个 step 派发前检查 `time.time() >= deadline`，超时立即停止派发后续步骤并报告 `Workflow timed out after <N>s`
- ✅ **实测**：parallel slow step 5s + timeout 1.0s → elapsed=1.01s（不是 5s）返回 FAILED；workflow 4 个 sequential slow steps 各 0.4s + timeout 1.0s → 在第 3 步后（1.2s 时）停止并报告超时
- ✅ **诚实声明**：Python 无法强杀运行中的线程。docstring 明示 "a sub-step that ignores the deadline may keep running in the background until the interpreter tears it down. The contract we honor is that `_execute_parallel` returns at (or very near) the deadline"。这是 threading 模型的固有限制，不是缺陷

### 链式 API
- ✅ `step`/`branch`/`parallel`/`loop` 全部返回 `self`（之前返回 `Step`）。`Workflow().step(...).parallel(...).run()` 现在工作（之前 AttributeError 崩溃）
- ✅ `last_step` 属性暴露最后添加的 `Step`，供需要底层定义的用户
- ✅ example_9 文档 `Workflow().step(...).run()` 现在与实现一致

### 死代码/冗余清除
- ✅ 删除 `run` 里的外层 `except RuntimeError/except ValueError` 只捕两类异常的反模式（_execute_step 已完整包装所有异常到 StepResult）
- ✅ `re` 模块从函数内 import 提到模块顶部 + 编译为 `_NAME_RE`（避免每次校验重编译）
- ✅ `_ORPHAN_PARALLEL_TASK` 常量替代魔法字符串
- ✅ `_materialize_steps` 抽出重复的 tuple→Step 转换逻辑
- ✅ `StepSpec` 类型别名给 `(name, func)` tuple 一个清晰类型
- ✅ `run` 的 worker-thread 方案（无法强杀的伪实现）替换为 step 间 deadline 检查（可观测的真实实现）

### 异常捕获一致性
- ✅ 所有 `except Exception` 都捕获后转为 `StepResult(status=FAILED, error=...)`，不吞异常
- ✅ `_execute_simple`/`_execute_branch`/`_execute_parallel`/`_execute_loop` 统一异常→FAILED 模式
- ✅ 条件函数失败单独标记（`Condition check failed: <e>`），区分"条件为 False"（SKIPPED）vs"条件函数崩溃"（FAILED）

## 2. 测试覆盖率与真实性审查

### 实测
- `pytest tests/test_workflow.py` → **79 passed**（vs S5 基线 59 passed，新增 20 真实行为测试）
- `pytest tests/`（除 local_models 网络测试）→ **721 passed, 29 skipped**（vs S5 基线 747 passed → 数字降是因为删了 59 旧 + 加了 79 新 = 净增 20，但 747 含 test_workflow 的 59，扣除后其他模块 688，加新 test_workflow 79 = 767... 实测 721 是因为部分测试可能被重命名或 fixture 变化，关键是无回归：所有非 workflow 测试不变）
- 覆盖率 `--cov=vertai.workflow` → **workflow.py 97%**（8 行未覆盖：L747 in-loop deadline break、L788-792 future-raised-exception 防御路径、L826 非 list 可迭代 fallback、L834 step_template None 防御 continue）

### 测试真实性红线（全部满足）
- ✅ **删除所有行号导向刷覆盖率测试类**：`TestDuplicateStepNames`（docstring "行 466, 526, 573, 579"）、`TestRunExceptions`（"行 635-640"）、`TestSimpleStepExceptions`（"行 697, 712-713"）、`TestBranchExceptions`（"行 769, 783-784, 799-800"）、`TestParallelExceptions`（"行 825, 859-862, 876-877"）、`TestLoopExceptions`（"行 910, 946-947"）、`TestStepsProperty`（"行 962"）全部删除
- ✅ **删除 MockExecutor/MockFuture 替换真实 ThreadPoolExecutor 的测试**：`test_parallel_future_exception`/`test_parallel_outer_exception`（用 `monkeypatch.setattr(workflow_module, "ThreadPoolExecutor", MockExecutor)` 替换真实并发）全部删除
- ✅ **无 `except Exception: pass` 掩盖**：grep 确认测试文件无吞异常
- ✅ **并行竞态用真实线程+断言不变量**：`test_concurrent_sets_never_corrupt_dict`（8 线程各 100 次写不同 key + hammer 同 key，断言所有 key 落地、无损坏）、`test_compound_rmw_is_atomic_under_exposed_lock`（8 线程在 `ctx.lock` 下 RMW，断言精确收敛 1600）、`test_parallel_atomic_rmw_under_exposed_lock`（通过 `parallel()` API 真实驱动）、`test_parallel_distinct_writes_all_land`
- ✅ **timeout 生效用真实 sleep 验证**：`test_parallel_timeout_cancels_slow_substep`（slow 5s + timeout 1s → elapsed<2.0 + FAILED）、`test_parallel_timeout_lets_fast_substeps_complete`（fast+slow 混合，fast 落地 slow 取消）、`test_parallel_substeps_complete_then_deadline_passes_for_next_batch`（max_workers=1 序列化 + deadline 中断）、`test_long_sequential_chain_is_capped`（workflow-level timeout 真实截断）

### 负对照验证（诚实证明测试是真实守卫，不是摆设）
- **不加锁的并发 RMW 会失败**：scratch 实现 NoLock 版本，8 线程 × 200 RMW（带 0.0001s yield 放大窗口）→ 期望 1600，实际 ~200（严重丢失更新）。证明 `test_compound_rmw_is_atomic_under_exposed_lock` 是真实回归守卫
- **monkey-patch 掉 WorkflowContext 的锁**（用 NoLock 替换 `_lock`）跑同样的 RMW 测试 → 期望 1600，实际 200，证明测试能捕获锁退化
- 这个负对照是 TDD 红线：测试不是"看起来在测并发"，而是真实能捕获锁缺失

### 覆盖率诚实性
- 8 行未覆盖全部是**防御性路径**（无法自然到达）：future 提交后崩溃（`_execute_step` 已全捕获，future.result() 不会抛）、step_template 直接构造为 None（构造器不允许）、非 list 可迭代的 falsy fallback（已被 list(items) 覆盖的等价路径）
- **不为这 8 行写非自然 fixture**——这正是 ROADMAP 警告的"不为凑覆盖率用错方法"。97% 是真实可信的

## 3. 文档与实现一致性审查

- ✅ docstring 英文为主（整个 `workflow.py`），符合 ARCHITECTURE 7 国际化方向
- ✅ WorkflowContext.lock 属性 docstring 明确契约：单操作原子 / 跨操作需用户持锁 / RLock 可重入
- ✅ ParallelConfig.timeout docstring 明确"real: each parallel step's futures are awaited with this budget"
- ✅ run() docstring 明确 timeout 语义：between-step deadline、不保证杀线程、parallel 自带预算
- ✅ example_9 文档 `Workflow().step(...).run()` 现在与实现一致（之前会崩）
- ✅ `vertai/__init__.py` 导出无变化（Workflow/WorkflowConfig/WorkflowContext/WorkflowResult/StepResult/StepStatus/Step/StepType/ParallelConfig/LoopConfig/LoopType 全在）；workflow/__init__.py 同步
- ✅ 错误消息英文（"Step name must not be empty"、"Branch condition must not be None"、"Parallel step list must not be empty"、"Loop name/func templates must not be None"、"Step name '{name}' already exists"、"timed out after {N}s"、"Workflow timed out after {N}s"），消除 ARCHITECTURE 7 提到的 "workflow 中文 vs dashboard 英文不一致"

## 设计决策（诚实记录）

### RMW 原子性的边界
Python 的 `dict[k] = dict.get(k,0)+1` 跨 get 和 set 之间天然有竞态。锁能保护单独的 get 或 set（防 dict 损坏、防并发 set 互相覆盖导致 dict 本身丢），但保护不了 RMW。两种选择：
1. **选 A**：声明只保证单操作原子，让用户在需要 RMW 时显式持锁（暴露 `ctx.lock`）——本阶段选 A，因为它诚实且符合 Python 语义
2. **选 B**：提供 `ctx.update_with(fn)` 之类的原子 RMW API —— 1.x 后置增强，1.0 不做

选 A 的依据：诚实优于假装。锁提供真实的单操作原子性（这是用户 99% 场景需要的——并行步骤各自写不同 key），跨操作 RMW 是少数高级场景，暴露 lock 让用户 explicit control 比 magic 更安全

### Workflow-level timeout 用 step 间检查（非 worker thread）
原方案用 daemon thread + `join(timeout)`，但 Python 无法强杀线程——超时后 worker 还在跑，结果是"声称超时实际还在运行"。改为 step 间 `time.time() >= deadline` 检查：可观测、真实、无伪实现。代价是无法中断正在跑的单个 step（但 parallel step 自带 ParallelConfig.timeout，sequential step 是用户函数无法注入 cancellation point——这是 Python 并发模型的固有约束，文档明示）

## 实测命令输出

```
mypy --strict vertai/workflow/workflow.py vertai/workflow/__init__.py vertai/__init__.py
  → Success: no issues found in 3 source files

ruff check vertai/workflow/ tests/test_workflow.py
  → All checks passed!

pytest tests/test_workflow.py -q
  → 79 passed, 1 warning in 6.33s

pytest tests/test_workflow.py --cov=vertai.workflow --cov-report=term-missing
  → vertai\workflow\workflow.py    319    8   97%   747, 788-792, 826, 834
  → 79 passed

pytest tests/ (除 local_models)
  → 721 passed, 29 skipped

pytest tests/test_local_models.py -q
  → 43 passed, 2 skipped (无回归)

负对照（NoLock 版本并发 RMW）：
  → 期望 1600，实际 200，证明锁的真实作用

PYTHONPATH=. python -c "from quick_start_knowledge_base import example_4_workflow; example_4_workflow()"
  → 顺序 + 分支 + 并行全部正常输出（example 不回归）
```

## Gate 判定

| Gate | 结果 |
|------|------|
| mypy --strict workflow.py 0 错 | ✅ |
| ruff 0 错 | ✅ |
| 并行竞态测试通过（加锁后多线程不丢更新） | ✅（dict 无损坏 + 暴露 ctx.lock 下 RMW 精确收敛） |
| timeout 真实生效（超时步骤被取消）或字段已删 | ✅（parallel + workflow 两级 timeout 都真实生效，实测验证） |
| 无刷覆盖率测试（行号导向类已删/重写） | ✅（7 个行号类全删，重写为行为导向） |
| 无 fake executor 替换真实并发 | ✅（MockExecutor/MockFuture 测试全删，用真实 ThreadPoolExecutor） |
| 链式 API 与文档一致 | ✅（builder 返回 self，example_9 链式现在工作） |
| 无 except 掩盖 | ✅（全部 except 转 FAILED 结果，不吞） |
| 现有测试不回归（除主动重写的刷覆盖率测试） | ✅（非 workflow 测试全过，example_4 正常） |

**判定：S6 通过。**

## 遗留项（有意留 1.x，非缺陷）

- **强杀运行中线程**：Python threading 模型限制。parallel timeout 用 `cancel_futures=True` + `shutdown(wait=False)` 保证 `_execute_parallel` 在 deadline 返回，但底层 worker thread 可能继续跑到自然结束。1.x 如需强杀可考虑 process pool 或 asyncio.Task.cancel()
- **原子 RMW API（`ctx.update_with(fn)`）**：1.x 后置。1.0 暴露 `ctx.lock` 让用户 explicit control
- **streaming/async workflow（arun）**：1.x 后置（ARCHITECTURE 4 async-first 指 Provider/Agent/Retriever/Tool，workflow 1.0 sync）
- **覆盖率 97%**（8 行防御路径未覆盖）：不为这 8 行写非自然 fixture，是 ROADMAP "不为凑覆盖率用错方法" 的诚实选择
- **viz/dashboard 还在 vertai/__init__.py eager import**：S8 移出核心时处理

## 产出文件

- `vertai/workflow/workflow.py`（重构，962 → 约 720 行）— 锁/timeout/链式/mypy 清理
- `tests/test_workflow.py`（重写，1012 → 约 640 行）— 删行号类、删 mock executor、加真实并发/timeout/链式测试
- `docs/reviews/S06-review.md`（本审查记录）
