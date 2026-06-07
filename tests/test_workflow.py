"""Workflow 工作流编排模块测试"""

import time
import pytest

from vertai.workflow import (
    Workflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowResult,
    StepResult,
    StepStatus,
    Step,
    StepType,
    ParallelConfig,
    LoopConfig,
    LoopType,
)


class TestWorkflowConfig:
    """WorkflowConfig 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = WorkflowConfig()
        assert config.max_workers == 4
        assert config.timeout == 300.0
        assert config.retry_count == 0
        assert config.retry_delay == 1.0
        assert config.continue_on_error is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = WorkflowConfig(
            max_workers=8,
            timeout=600.0,
            retry_count=2,
            retry_delay=0.5,
            continue_on_error=True,
        )
        assert config.max_workers == 8
        assert config.timeout == 600.0
        assert config.retry_count == 2
        assert config.retry_delay == 0.5
        assert config.continue_on_error is True

    def test_config_validation_bounds(self):
        """测试配置边界验证"""
        # max_workers 边界
        with pytest.raises(ValueError):
            WorkflowConfig(max_workers=0)
        with pytest.raises(ValueError):
            WorkflowConfig(max_workers=33)

        # timeout 边界
        with pytest.raises(ValueError):
            WorkflowConfig(timeout=0.5)

        # retry_count 边界
        with pytest.raises(ValueError):
            WorkflowConfig(retry_count=-1)
        with pytest.raises(ValueError):
            WorkflowConfig(retry_count=11)


class TestWorkflowContext:
    """WorkflowContext 上下文测试"""

    def test_context_get_set(self):
        """测试上下文数据存取"""
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("nonexistent") is None
        assert ctx.get("nonexistent", "default") == "default"

    def test_context_update(self):
        """测试上下文批量更新"""
        ctx = WorkflowContext()
        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1
        assert ctx.get("b") == 2

    def test_context_clear(self):
        """测试上下文清空"""
        ctx = WorkflowContext(data={"x": 1})
        ctx.clear()
        assert ctx.get("x") is None
        assert len(ctx.data) == 0


class TestWorkflowBasic:
    """Workflow 基本功能测试"""

    def test_empty_workflow(self):
        """测试空工作流"""
        wf = Workflow()
        result = wf.run()
        assert result.success is True
        assert len(result.steps) == 0
        assert result.total_duration >= 0

    def test_single_step(self):
        """测试单步骤工作流"""
        wf = Workflow()
        wf.step("step1", lambda ctx: ctx.set("done", True))

        result = wf.run()
        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].name == "step1"
        assert result.context.get("done") is True

    def test_sequential_steps(self):
        """测试顺序执行步骤"""
        wf = Workflow()
        execution_order = []

        wf.step("first", lambda ctx: execution_order.append(1))
        wf.step("second", lambda ctx: execution_order.append(2))
        wf.step("third", lambda ctx: execution_order.append(3))

        result = wf.run()
        assert result.success is True
        assert execution_order == [1, 2, 3]

    def test_initial_data(self):
        """测试初始数据"""
        wf = Workflow()
        wf.step("check", lambda ctx: ctx.set("result", ctx.get("input") * 2))

        result = wf.run(initial_data={"input": 21})
        assert result.success is True
        assert result.context.get("result") == 42

    def test_duplicate_step_name(self):
        """测试重复步骤名称"""
        wf = Workflow()
        wf.step("step1", lambda ctx: None)

        with pytest.raises(ValueError, match="已存在"):
            wf.step("step1", lambda ctx: None)


class TestWorkflowBranch:
    """Workflow 条件分支测试"""

    def test_branch_yes(self):
        """测试 yes 分支执行"""
        wf = Workflow()
        executed = []

        wf.step("setup", lambda ctx: ctx.set("flag", True))
        wf.branch(
            condition=lambda ctx: ctx.get("flag"),
            yes_steps=[
                ("yes_step", lambda ctx: executed.append("yes")),
            ],
            no_steps=[
                ("no_step", lambda ctx: executed.append("no")),
            ],
        )

        result = wf.run()
        assert result.success is True
        assert executed == ["yes"]

    def test_branch_no(self):
        """测试 no 分支执行"""
        wf = Workflow()
        executed = []

        wf.step("setup", lambda ctx: ctx.set("flag", False))
        wf.branch(
            condition=lambda ctx: ctx.get("flag"),
            yes_steps=[
                ("yes_step", lambda ctx: executed.append("yes")),
            ],
            no_steps=[
                ("no_step", lambda ctx: executed.append("no")),
            ],
        )

        result = wf.run()
        assert result.success is True
        assert executed == ["no"]

    def test_branch_condition_none(self):
        """测试条件为 None 时抛出异常"""
        wf = Workflow()

        with pytest.raises(ValueError, match="条件函数不能为空"):
            wf.branch(condition=None)


class TestWorkflowParallel:
    """Workflow 并行执行测试"""

    def test_parallel_execution(self):
        """测试并行执行"""
        wf = Workflow()
        results = []

        wf.parallel([
            ("task_a", lambda ctx: results.append("A")),
            ("task_b", lambda ctx: results.append("B")),
        ])

        result = wf.run()
        assert result.success is True
        assert "A" in results
        assert "B" in results

    def test_parallel_empty_steps(self):
        """测试空并行步骤列表"""
        wf = Workflow()

        with pytest.raises(ValueError, match="并行步骤列表不能为空"):
            wf.parallel([])

    def test_parallel_with_error(self):
        """测试并行执行中的错误处理"""
        wf = Workflow()
        results = []

        wf.parallel([
            ("task_a", lambda ctx: results.append("A")),
            ("task_fail", lambda ctx: 1 / 0),
            ("task_b", lambda ctx: results.append("B")),
        ])

        result = wf.run()
        assert result.success is False
        # 即使失败，其他任务也应该完成（并行执行）
        assert "A" in results or "B" in results


class TestWorkflowLoop:
    """Workflow 循环执行测试"""

    def test_loop_for_items(self):
        """测试 for 循环"""
        wf = Workflow()
        processed = []

        wf.loop(
            items=["a", "b", "c"],
            step_name_template=lambda item: f"process_{item}",
            step_func_template=lambda item: lambda ctx: processed.append(item),
        )

        result = wf.run()
        assert result.success is True
        assert processed == ["a", "b", "c"]

    def test_loop_dynamic_items(self):
        """测试动态迭代项"""
        wf = Workflow()
        processed = []

        wf.step("setup", lambda ctx: ctx.set("items", ["x", "y", "z"]))
        wf.loop(
            items=lambda ctx: ctx.get("items", []),
            step_name_template=lambda item: f"process_{item}",
            step_func_template=lambda item: lambda ctx: processed.append(item),
        )

        result = wf.run()
        assert result.success is True
        assert processed == ["x", "y", "z"]

    def test_loop_empty_items(self):
        """测试空循环列表"""
        wf = Workflow()

        wf.loop(
            items=[],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: None,
        )

        result = wf.run()
        assert result.success is True
        assert len(result.steps) == 1
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_loop_max_iterations(self):
        """测试循环最大迭代次数"""
        wf = Workflow()
        counter = [0]

        wf.loop(
            items=range(100),
            step_name_template=lambda i: f"item_{i}",
            step_func_template=lambda i: lambda ctx: counter.__setitem__(0, counter[0] + 1),
            config=LoopConfig(max_iterations=3),
        )

        result = wf.run()
        assert result.success is True
        assert counter[0] == 3

    def test_loop_break_on_error(self):
        """测试循环错误中断"""
        wf = Workflow()
        processed = []

        def make_func(item):
            def func(ctx):
                if item == "b":
                    raise ValueError("error on b")
                processed.append(item)
            return func

        wf.loop(
            items=["a", "b", "c"],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=make_func,
            config=LoopConfig(break_on_error=True),
        )

        result = wf.run()
        assert result.success is False
        assert processed == ["a"]


class TestWorkflowErrorHandling:
    """Workflow 错误处理测试"""

    def test_step_failure(self):
        """测试步骤失败"""
        wf = Workflow()

        def failing_step(ctx):
            raise ValueError("Intentional error")

        wf.step("fail", failing_step)
        result = wf.run()

        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Intentional error" in result.steps[0].error

    def test_continue_on_error(self):
        """测试错误后继续执行"""
        config = WorkflowConfig(continue_on_error=True)
        wf = Workflow(config=config)

        wf.step("fail", lambda ctx: 1 / 0)
        wf.step("continue", lambda ctx: ctx.set("continued", True))

        result = wf.run()
        assert result.success is False  # 整体标记为失败
        assert result.context.get("continued") is True  # 但继续执行了

    def test_retry_on_failure(self):
        """测试失败重试"""
        attempts = [0]
        config = WorkflowConfig(retry_count=2, retry_delay=0.1)

        wf = Workflow(config=config)

        def flaky_step(ctx):
            attempts[0] += 1
            if attempts[0] < 3:
                raise ValueError("Not yet")
            return "success"

        wf.step("flaky", flaky_step)
        result = wf.run()

        assert result.success is True
        assert result.steps[0].retries == 2


class TestStepNameValidation:
    """步骤名称验证测试"""

    def test_valid_step_names(self):
        """测试有效步骤名称"""
        wf = Workflow()

        # 各种有效名称 - 验证不会抛异常
        wf.step("valid_name", lambda ctx: ctx.set("a", 1))
        wf.step("valid-name", lambda ctx: ctx.set("b", 2))
        wf.step("valid123", lambda ctx: ctx.set("c", 3))
        wf.step("中文名称", lambda ctx: ctx.set("d", 4))

        # 验证步骤确实被添加
        steps = wf.steps
        assert len(steps) == 4
        step_names = [s.name for s in steps]
        assert "valid_name" in step_names
        assert "valid-name" in step_names
        assert "valid123" in step_names
        assert "中文名称" in step_names

    def test_clear_steps(self):
        """测试清空步骤"""
        wf = Workflow()
        wf.step("step1", lambda ctx: ctx.set("a", 1))
        wf.step("step2", lambda ctx: ctx.set("b", 2))

        # 验证步骤已添加
        assert len(wf.steps) == 2

        # 清空步骤
        wf.clear()
        assert len(wf.steps) == 0
        # 验证可以重新添加步骤
        wf.step("step3", lambda ctx: ctx.set("c", 3))
        assert len(wf.steps) == 1

    def test_empty_step_name(self):
        """测试空步骤名称"""
        wf = Workflow()

        with pytest.raises(ValueError, match="步骤名称不能为空"):
            wf.step("", lambda ctx: None)

    def test_long_step_name(self):
        """测试过长步骤名称"""
        wf = Workflow()
        long_name = "a" * 101

        with pytest.raises(ValueError, match="步骤名称过长"):
            wf.step(long_name, lambda ctx: None)

    def test_invalid_characters_in_step_name(self):
        """测试步骤名称中的非法字符"""
        wf = Workflow()

        with pytest.raises(ValueError, match="非法字符"):
            wf.step("invalid@name", lambda ctx: None)

        with pytest.raises(ValueError, match="非法字符"):
            wf.step("invalid name", lambda ctx: None)


class TestParallelConfig:
    """ParallelConfig 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ParallelConfig()
        assert config.max_workers == 4
        assert config.timeout == 300.0

    def test_custom_config(self):
        """测试自定义配置"""
        config = ParallelConfig(max_workers=8, timeout=60.0)
        assert config.max_workers == 8
        assert config.timeout == 60.0


class TestLoopConfig:
    """LoopConfig 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = LoopConfig()
        assert config.loop_type == LoopType.FOR
        assert config.max_iterations == 100
        assert config.break_on_error is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = LoopConfig(
            loop_type=LoopType.WHILE,
            max_iterations=50,
            break_on_error=False,
        )
        assert config.loop_type == LoopType.WHILE
        assert config.max_iterations == 50
        assert config.break_on_error is False


class TestStepResult:
    """StepResult 测试"""

    def test_result_creation(self):
        """测试结果创建"""
        result = StepResult(
            name="test",
            status=StepStatus.COMPLETED,
            output={"key": "value"},
            duration=0.5,
        )
        assert result.name == "test"
        assert result.status == StepStatus.COMPLETED
        assert result.output == {"key": "value"}
        assert result.duration == 0.5
        assert result.error is None
        assert result.retries == 0


class TestWorkflowResult:
    """WorkflowResult 测试"""

    def test_result_creation(self):
        """测试结果创建"""
        ctx = WorkflowContext(data={"result": "done"})
        result = WorkflowResult(
            success=True,
            steps=[StepResult(name="s1", status=StepStatus.COMPLETED)],
            context=ctx,
            total_duration=1.0,
        )
        assert result.success is True
        assert len(result.steps) == 1
        assert result.context.get("result") == "done"
        assert result.total_duration == 1.0


class TestStep:
    """Step 测试"""

    def test_step_creation(self):
        """测试步骤创建"""
        step = Step(
            name="test_step",
            step_type=StepType.SIMPLE,
            func=lambda ctx: ctx.set("key", "value"),
        )
        assert step.name == "test_step"
        assert step.step_type == StepType.SIMPLE
        assert step.func is not None


class TestConditionalStep:
    """条件步骤测试"""

    def test_step_with_condition_true(self):
        """测试条件为 True 时执行"""
        wf = Workflow()
        executed = []

        wf.step(
            "conditional",
            lambda ctx: executed.append("executed"),
            condition=lambda ctx: ctx.get("should_run", True),
        )

        result = wf.run(initial_data={"should_run": True})
        assert result.success is True
        assert executed == ["executed"]

    def test_step_with_condition_false(self):
        """测试条件为 False 时跳过"""
        wf = Workflow()
        executed = []

        wf.step(
            "conditional",
            lambda ctx: executed.append("executed"),
            condition=lambda ctx: ctx.get("should_run", False),
        )

        result = wf.run(initial_data={"should_run": False})
        assert result.success is True
        assert executed == []
        assert result.steps[0].status == StepStatus.SKIPPED


class TestComplexWorkflow:
    """复杂工作流测试"""

    def test_mixed_workflow(self):
        """测试混合工作流"""
        wf = Workflow()
        results = []

        # 顺序步骤
        wf.step("init", lambda ctx: ctx.set("counter", 0))

        # 循环
        wf.loop(
            items=range(3),
            step_name_template=lambda i: f"increment_{i}",
            step_func_template=lambda i: lambda ctx: ctx.set("counter", ctx.get("counter") + 1),
        )

        # 条件分支
        wf.branch(
            condition=lambda ctx: ctx.get("counter") >= 3,
            yes_steps=[
                ("success_log", lambda ctx: results.append("success")),
            ],
            no_steps=[
                ("fail_log", lambda ctx: results.append("fail")),
            ],
        )

        result = wf.run()
        assert result.success is True
        assert result.context.get("counter") == 3
        assert results == ["success"]


class TestDuplicateStepNames:
    """重复步骤名称异常测试（行 466, 526, 573, 579）"""

    def test_branch_duplicate_step_name(self):
        """测试 branch 方法重复步骤名称（行 466）"""
        wf = Workflow()
        # 添加一个步骤，计数器变为 1
        wf.step("step1", lambda ctx: None)
        # branch name=None 会生成 branch_1
        wf.branch(
            condition=lambda ctx: True,
            yes_steps=[("yes_step", lambda ctx: None)],
            name=None,
        )
        # 再添加一个步骤，计数器变为 2
        wf.step("step2", lambda ctx: None)
        # 现在尝试添加一个显式命名为 branch_1 的 branch
        with pytest.raises(ValueError, match="已存在"):
            wf.branch(
                condition=lambda ctx: True,
                yes_steps=[("yes_step2", lambda ctx: None)],
                name="branch_1",
            )

    def test_parallel_duplicate_step_name(self):
        """测试 parallel 方法重复步骤名称（行 526）"""
        wf = Workflow()
        # 添加一个步骤，计数器变为 1
        wf.step("step1", lambda ctx: None)
        # parallel name=None 会生成 parallel_1
        wf.parallel(
            steps=[("task1", lambda ctx: None)],
            name=None,
        )
        # 再添加一个步骤，计数器变为 2
        wf.step("step2", lambda ctx: None)
        # 现在尝试添加一个显式命名为 parallel_1 的 parallel
        with pytest.raises(ValueError, match="已存在"):
            wf.parallel(
                steps=[("task2", lambda ctx: None)],
                name="parallel_1",
            )

    def test_loop_duplicate_step_name(self):
        """测试 loop 方法重复步骤名称（行 573, 579）"""
        wf = Workflow()
        # 添加一个步骤，计数器变为 1
        wf.step("step1", lambda ctx: None)
        # loop name=None 会生成 loop_1
        wf.loop(
            items=["a"],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: None,
            name=None,
        )
        # 再添加一个步骤，计数器变为 2
        wf.step("step2", lambda ctx: None)
        # 现在尝试添加一个显式命名为 loop_1 的 loop
        with pytest.raises(ValueError, match="已存在"):
            wf.loop(
                items=["b"],
                step_name_template=lambda item: f"item_{item}",
                step_func_template=lambda item: lambda ctx: None,
                name="loop_1",
            )

    def test_loop_none_templates(self):
        """测试 loop 方法模板为 None（行 573）"""
        wf = Workflow()

        with pytest.raises(ValueError, match="步骤名称模板和函数模板不能为空"):
            wf.loop(
                items=["a"],
                step_name_template=None,
                step_func_template=lambda item: lambda ctx: None,
            )

        with pytest.raises(ValueError, match="步骤名称模板和函数模板不能为空"):
            wf.loop(
                items=["a"],
                step_name_template=lambda item: f"item_{item}",
                step_func_template=None,
            )


class TestRunExceptions:
    """run 方法异常处理测试（行 635-640）"""

    def test_run_runtime_error(self, monkeypatch):
        """测试 run 方法捕获 RuntimeError（行 635-637）"""
        wf = Workflow()

        def raise_runtime_error(ctx):
            raise RuntimeError("Simulated runtime error")

        # 修改 _execute_step 使其抛出 RuntimeError
        original_execute = wf._execute_step
        call_count = [0]

        def mock_execute(step, ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated runtime error")
            return original_execute(step, ctx)

        monkeypatch.setattr(wf, "_execute_step", mock_execute)

        wf.step("test", lambda ctx: None)
        result = wf.run()

        assert result.success is False
        assert "Simulated runtime error" in result.error

    def test_run_value_error(self, monkeypatch):
        """测试 run 方法捕获 ValueError（行 638-640）"""
        wf = Workflow()

        # 修改 _execute_step 使其抛出 ValueError
        original_execute = wf._execute_step
        call_count = [0]

        def mock_execute(step, ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Simulated value error")
            return original_execute(step, ctx)

        monkeypatch.setattr(wf, "_execute_step", mock_execute)

        wf.step("test", lambda ctx: None)
        result = wf.run()

        assert result.success is False
        assert "Simulated value error" in result.error


class TestSimpleStepExceptions:
    """简单步骤异常处理测试（行 697, 712-713）"""

    def test_step_with_none_func(self):
        """测试步骤 func 为 None 时跳过（行 697）"""
        wf = Workflow()

        # 直接创建一个没有 func 的 Step
        step = Step(
            name="null_step",
            step_type=StepType.SIMPLE,
            func=None,
        )
        wf._steps.append(step)

        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.SKIPPED

    def test_step_condition_raises_exception(self):
        """测试条件检查抛出异常（行 712-713）"""
        wf = Workflow()

        def bad_condition(ctx):
            raise ValueError("Condition check failed")

        wf.step(
            "conditional_step",
            lambda ctx: ctx.set("executed", True),
            condition=bad_condition,
        )

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "条件检查失败" in result.steps[0].error


class TestBranchExceptions:
    """分支步骤异常处理测试（行 769, 783-784, 799-800）"""

    def test_branch_empty_branches(self):
        """测试分支步骤没有分支列表（行 769）"""
        wf = Workflow()

        wf.branch(
            condition=lambda ctx: True,
            yes_steps=None,
            no_steps=None,
        )

        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].output == {"branch": "yes"}

    def test_branch_step_failure_no_continue(self):
        """测试分支步骤失败且不继续执行（行 783-784）"""
        wf = Workflow()

        wf.branch(
            condition=lambda ctx: True,
            yes_steps=[
                ("failing_step", lambda ctx: 1 / 0),
                ("after_step", lambda ctx: ctx.set("after", True)),
            ],
        )

        result = wf.run()
        assert result.success is False
        assert "分支步骤" in result.steps[0].error

    def test_branch_condition_raises_exception(self):
        """测试分支条件函数抛出异常（行 799-800）"""
        wf = Workflow()

        def bad_condition(ctx):
            raise ValueError("Bad condition")

        wf.branch(
            condition=bad_condition,
            yes_steps=[("yes_step", lambda ctx: None)],
        )

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Bad condition" in result.steps[0].error


class TestParallelExceptions:
    """并行执行异常处理测试（行 825, 859-862, 876-877）"""

    def test_parallel_empty_steps_list(self):
        """测试并行步骤为空列表（行 825）"""
        wf = Workflow()

        # 直接创建一个没有 parallel_steps 的 Step
        step = Step(
            name="empty_parallel",
            step_type=StepType.PARALLEL,
            parallel_steps=None,
        )
        wf._steps.append(step)

        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_parallel_future_exception(self, monkeypatch):
        """测试并行执行 future.result() 抛出异常（行 859-862）"""
        from concurrent.futures import Future

        wf = Workflow()

        # 创建一个会导致 future.result() 抛出异常的步骤
        step = Step(
            name="parallel_test",
            step_type=StepType.PARALLEL,
            parallel_steps=[
                Step(name="task1", step_type=StepType.SIMPLE, func=lambda ctx: None),
            ],
        )
        wf._steps.append(step)

        # Mock ThreadPoolExecutor 使 future.result() 抛出异常
        import vertai.workflow.workflow as workflow_module

        original_executor = workflow_module.ThreadPoolExecutor

        class MockFuture:
            def result(self):
                raise RuntimeError("Future failed")

        class MockExecutor:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def submit(self, fn, *args):
                return MockFuture()

        monkeypatch.setattr(workflow_module, "ThreadPoolExecutor", MockExecutor)

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Future failed" in result.steps[0].error

    def test_parallel_outer_exception(self, monkeypatch):
        """测试并行执行外层异常（行 876-877）"""
        wf = Workflow()

        step = Step(
            name="parallel_test",
            step_type=StepType.PARALLEL,
            parallel_steps=[
                Step(name="task1", step_type=StepType.SIMPLE, func=lambda ctx: None),
            ],
        )
        wf._steps.append(step)

        # Mock ThreadPoolExecutor 使其抛出异常
        import vertai.workflow.workflow as workflow_module

        class MockExecutor:
            def __init__(self, **kwargs):
                raise RuntimeError("Executor init failed")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr(workflow_module, "ThreadPoolExecutor", MockExecutor)

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Executor init failed" in result.steps[0].error


class TestLoopExceptions:
    """循环执行异常处理测试（行 910, 946-947）"""

    def test_loop_none_items(self):
        """测试循环 items 为 None（行 910）"""
        wf = Workflow()

        # 直接创建一个 loop_items 为 None 的 Step
        step = Step(
            name="null_loop",
            step_type=StepType.LOOP,
            loop_items=None,
            step_template=lambda item: Step(
                name=f"item_{item}",
                step_type=StepType.SIMPLE,
                func=lambda ctx: None,
            ),
            loop_config=LoopConfig(),
        )
        wf._steps.append(step)

        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_loop_step_template_raises_exception(self):
        """测试循环步骤模板抛出异常（行 946-947）"""
        wf = Workflow()

        # 创建一个 step_template 会抛出异常的 Step
        step = Step(
            name="error_loop",
            step_type=StepType.LOOP,
            loop_items=["a", "b"],
            step_template=lambda item: (_ for _ in ()).throw(ValueError("Template error")),
            loop_config=LoopConfig(),
        )
        wf._steps.append(step)

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Template error" in result.steps[0].error

    def test_loop_items_function_raises_exception(self):
        """测试获取迭代项函数抛出异常（行 946-947）"""
        wf = Workflow()

        def bad_items_func(ctx):
            raise ValueError("Items function error")

        wf.loop(
            items=bad_items_func,
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: None,
        )

        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Items function error" in result.steps[0].error


class TestStepsProperty:
    """steps 属性测试（行 962）"""

    def test_steps_property(self):
        """测试 steps 属性返回步骤列表（行 962）"""
        wf = Workflow()
        wf.step("step1", lambda ctx: None)
        wf.step("step2", lambda ctx: None)

        steps = wf.steps
        assert len(steps) == 2
        assert steps[0].name == "step1"
        assert steps[1].name == "step2"

    def test_steps_property_returns_copy(self):
        """测试 steps 属性返回副本，修改不影响原列表"""
        wf = Workflow()
        wf.step("step1", lambda ctx: None)

        steps = wf.steps
        steps.clear()

        # 原列表不应受影响
        assert len(wf.steps) == 1
