"""
Workflow 工作流编排模块

核心功能:
- 步骤编排（顺序执行）
- 条件分支（if/else）
- 循环执行（while/for）
- 并行执行
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# 配置常量
# ============================================================================

DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRY_COUNT = 0
DEFAULT_RETRY_DELAY = 1.0
MAX_STEP_NAME_LENGTH = 100
STEP_NAME_PATTERN = r'^[a-zA-Z0-9_\-一-龥]+$'


# ============================================================================
# 枚举类型
# ============================================================================

class StepStatus(str, Enum):
    """步骤状态枚举"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LoopType(str, Enum):
    """循环类型枚举"""

    FOR = "for"
    WHILE = "while"


class StepType(str, Enum):
    """步骤类型枚举"""

    SIMPLE = "simple"
    BRANCH = "branch"
    PARALLEL = "parallel"
    LOOP = "loop"


# ============================================================================
# 配置类
# ============================================================================

class WorkflowConfig(BaseModel):
    """Workflow 配置

    示例:
        # 使用默认配置
        config = WorkflowConfig()

        # 自定义配置
        config = WorkflowConfig(
            max_workers=8,
            timeout=600.0,
            retry_count=2,
        )
    """

    max_workers: int = Field(
        default=DEFAULT_MAX_WORKERS,
        ge=1,
        le=32,
        description="并行执行最大线程数"
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        ge=1.0,
        description="工作流超时时间(秒)"
    )
    retry_count: int = Field(
        default=DEFAULT_RETRY_COUNT,
        ge=0,
        le=10,
        description="失败重试次数"
    )
    retry_delay: float = Field(
        default=DEFAULT_RETRY_DELAY,
        ge=0.0,
        description="重试延迟(秒)"
    )
    continue_on_error: bool = Field(
        default=False,
        description="步骤失败时是否继续执行"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class ParallelConfig(BaseModel):
    """并行执行配置

    示例:
        config = ParallelConfig(
            max_workers=4,
            timeout=60.0,
        )
    """

    max_workers: int = Field(
        default=DEFAULT_MAX_WORKERS,
        ge=1,
        le=32,
        description="最大并行线程数"
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT,
        ge=1.0,
        description="并行执行超时时间(秒)"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


class LoopConfig(BaseModel):
    """循环执行配置

    示例:
        config = LoopConfig(
            loop_type=LoopType.WHILE,
            max_iterations=100,
        )
    """

    loop_type: LoopType = Field(
        default=LoopType.FOR,
        description="循环类型"
    )
    max_iterations: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="最大迭代次数"
    )
    break_on_error: bool = Field(
        default=True,
        description="出错时是否中断循环"
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )


# ============================================================================
# 上下文类
# ============================================================================

class WorkflowContext(BaseModel):
    """工作流执行上下文

    用于在步骤间传递数据和状态。

    示例:
        ctx = WorkflowContext()
        ctx.set("key", "value")
        value = ctx.get("key")
    """

    data: dict[str, Any] = Field(default_factory=dict, description="上下文数据")
    metadata: dict[str, Any] = Field(default_factory=dict, description="元数据")

    model_config = ConfigDict(
        extra="forbid",
    )

    def get(self, key: str, default: Any = None) -> Any:
        """获取上下文数据

        Args:
            key: 数据键名
            default: 默认值

        Returns:
            对应的数据值，如果不存在则返回默认值
        """
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """设置上下文数据

        Args:
            key: 数据键名
            value: 数据值
        """
        self.data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """批量更新上下文数据

        Args:
            data: 要更新的数据字典
        """
        self.data.update(data)

    def clear(self) -> None:
        """清空上下文数据"""
        self.data.clear()
        self.metadata.clear()


# ============================================================================
# 结果类
# ============================================================================

@dataclass
class StepResult:
    """步骤执行结果

    Attributes:
        name: 步骤名称
        status: 执行状态
        output: 输出数据
        error: 错误信息
        duration: 执行时长(秒)
        retries: 重试次数
    """

    name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0


@dataclass
class WorkflowResult:
    """工作流执行结果

    Attributes:
        success: 是否成功
        steps: 各步骤结果
        context: 最终上下文
        total_duration: 总执行时长(秒)
        error: 错误信息
    """

    success: bool
    steps: list[StepResult] = field(default_factory=list)
    context: Optional[WorkflowContext] = None
    total_duration: float = 0.0
    error: Optional[str] = None


# ============================================================================
# 步骤定义
# ============================================================================

@dataclass
class Step:
    """步骤定义

    Attributes:
        name: 步骤名称
        step_type: 步骤类型
        func: 执行函数
        condition: 执行条件
        branch_yes: 条件分支-是
        branch_no: 条件分支-否
        parallel_steps: 并行步骤列表
        loop_items: 循环项
        step_template: 步骤模板
        loop_config: 循环配置
        parallel_config: 并行配置
    """

    name: str
    step_type: StepType = StepType.SIMPLE
    func: Optional[Callable[[WorkflowContext], Any]] = None
    condition: Optional[Callable[[WorkflowContext], bool]] = None
    branch_yes: Optional[list[Step]] = None
    branch_no: Optional[list[Step]] = None
    parallel_steps: Optional[list[Step]] = None
    loop_items: Optional[Union[list[Any], Callable[[WorkflowContext], list[Any]]]] = None
    step_template: Optional[Callable[[Any], Step]] = None
    loop_config: Optional[LoopConfig] = None
    parallel_config: Optional[ParallelConfig] = None


# ============================================================================
# 验证函数
# ============================================================================

def _validate_step_name(name: str) -> str:
    """验证步骤名称

    Args:
        name: 步骤名称

    Returns:
        验证后的步骤名称

    Raises:
        ValueError: 步骤名称无效
    """
    import re

    if not name:
        raise ValueError("步骤名称不能为空")

    if len(name) > MAX_STEP_NAME_LENGTH:
        raise ValueError(f"步骤名称过长，最大 {MAX_STEP_NAME_LENGTH} 个字符")

    if not re.match(STEP_NAME_PATTERN, name):
        raise ValueError(
            f"步骤名称 '{name}' 包含非法字符。"
            "只允许字母、数字、下划线(_)、连字符(-)和中文。"
        )

    return name


# ============================================================================
# Workflow 主类
# ============================================================================

class Workflow:
    """工作流编排引擎

    支持步骤编排、条件分支、循环执行和并行执行。

    示例:
        # 基本用法
        wf = Workflow()
        wf.step("步骤1", lambda ctx: ctx.set("result", "done"))
        result = wf.run()

        # 条件分支
        wf = Workflow()
        wf.step("检查", check_fn)
        wf.branch(
            condition=lambda ctx: ctx.get("need_test"),
            yes_steps=[
                ("测试", test_fn),
            ],
            no_steps=[
                ("部署", deploy_fn),
            ]
        )

        # 并行执行
        wf = Workflow()
        wf.parallel([
            ("任务A", task_a),
            ("任务B", task_b),
        ])

        # 循环执行
        wf = Workflow()
        wf.loop(
            items=["a", "b", "c"],
            step_name_template=lambda item: f"处理{item}",
            step_func_template=lambda item: lambda ctx: process(item, ctx),
        )
    """

    def __init__(self, config: Optional[WorkflowConfig] = None):
        """初始化工作流

        Args:
            config: 工作流配置
        """
        self.config = config or WorkflowConfig()
        self._steps: list[Step] = []
        self._step_names: set[str] = set()

    def step(
        self,
        name: str,
        func: Callable[[WorkflowContext], Any],
        condition: Optional[Callable[[WorkflowContext], bool]] = None,
    ) -> Step:
        """添加顺序执行步骤

        Args:
            name: 步骤名称
            func: 执行函数，接收上下文参数
            condition: 执行条件函数，返回 True 时执行

        Returns:
            步骤对象

        Raises:
            ValueError: 步骤名称无效或重复
        """
        validated_name = _validate_step_name(name)

        if validated_name in self._step_names:
            raise ValueError(f"步骤名称 '{validated_name}' 已存在")

        step_obj = Step(
            name=validated_name,
            step_type=StepType.SIMPLE,
            func=func,
            condition=condition,
        )
        self._steps.append(step_obj)
        self._step_names.add(validated_name)

        return step_obj

    def branch(
        self,
        condition: Callable[[WorkflowContext], bool],
        yes_steps: Optional[list[tuple[str, Callable[[WorkflowContext], Any]]]] = None,
        no_steps: Optional[list[tuple[str, Callable[[WorkflowContext], Any]]]] = None,
        name: Optional[str] = None,
    ) -> Step:
        """添加条件分支

        Args:
            condition: 条件函数，返回 True 执行 yes_steps 分支
            yes_steps: 条件为 True 时执行的步骤列表，格式为 [(name, func), ...]
            no_steps: 条件为 False 时执行的步骤列表，格式为 [(name, func), ...]
            name: 分支名称（可选）

        Returns:
            分支步骤对象

        Raises:
            ValueError: 条件函数为空
        """
        if condition is None:
            raise ValueError("条件函数不能为空")

        branch_name = name or f"branch_{len(self._step_names)}"
        validated_name = _validate_step_name(branch_name)

        if validated_name in self._step_names:
            raise ValueError(f"步骤名称 '{validated_name}' 已存在")

        # 创建分支步骤
        yes_list: list[Step] = []
        if yes_steps:
            for step_name, step_func in yes_steps:
                yes_list.append(Step(
                    name=step_name,
                    step_type=StepType.SIMPLE,
                    func=step_func,
                ))

        no_list: list[Step] = []
        if no_steps:
            for step_name, step_func in no_steps:
                no_list.append(Step(
                    name=step_name,
                    step_type=StepType.SIMPLE,
                    func=step_func,
                ))

        step_obj = Step(
            name=validated_name,
            step_type=StepType.BRANCH,
            func=condition,
            branch_yes=yes_list,
            branch_no=no_list,
        )

        self._steps.append(step_obj)
        self._step_names.add(validated_name)

        return step_obj

    def parallel(
        self,
        steps: list[tuple[str, Callable[[WorkflowContext], Any]]],
        config: Optional[ParallelConfig] = None,
        name: Optional[str] = None,
    ) -> Step:
        """添加并行执行步骤

        Args:
            steps: 并行执行的步骤列表，格式为 [(name, func), ...]
            config: 并行配置
            name: 步骤名称（可选）

        Returns:
            并行步骤对象

        Raises:
            ValueError: 步骤列表为空
        """
        if not steps:
            raise ValueError("并行步骤列表不能为空")

        parallel_name = name or f"parallel_{len(self._step_names)}"
        validated_name = _validate_step_name(parallel_name)

        if validated_name in self._step_names:
            raise ValueError(f"步骤名称 '{validated_name}' 已存在")

        # 创建并行步骤
        parallel_list: list[Step] = []
        for step_name, step_func in steps:
            parallel_list.append(Step(
                name=step_name,
                step_type=StepType.SIMPLE,
                func=step_func,
            ))

        step_obj = Step(
            name=validated_name,
            step_type=StepType.PARALLEL,
            parallel_steps=parallel_list,
            parallel_config=config,
        )

        self._steps.append(step_obj)
        self._step_names.add(validated_name)

        return step_obj

    def loop(
        self,
        items: Union[list[Any], Callable[[WorkflowContext], list[Any]]],
        step_name_template: Callable[[Any], str],
        step_func_template: Callable[[Any], Callable[[WorkflowContext], Any]],
        config: Optional[LoopConfig] = None,
        name: Optional[str] = None,
    ) -> Step:
        """添加循环执行步骤

        Args:
            items: 要迭代的项列表或获取列表的函数
            step_name_template: 步骤名称模板函数，接收项参数返回步骤名称
            step_func_template: 步骤函数模板，接收项参数返回执行函数
            config: 循环配置
            name: 步骤名称（可选）

        Returns:
            循环步骤对象

        Raises:
            ValueError: 步骤模板为空
        """
        if step_name_template is None or step_func_template is None:
            raise ValueError("步骤名称模板和函数模板不能为空")

        loop_name = name or f"loop_{len(self._step_names)}"
        validated_name = _validate_step_name(loop_name)

        if validated_name in self._step_names:
            raise ValueError(f"步骤名称 '{validated_name}' 已存在")

        step_obj = Step(
            name=validated_name,
            step_type=StepType.LOOP,
            loop_items=items,
            step_template=lambda item: Step(
                name=step_name_template(item),
                step_type=StepType.SIMPLE,
                func=step_func_template(item),
            ),
            loop_config=config or LoopConfig(),
        )

        self._steps.append(step_obj)
        self._step_names.add(validated_name)

        return step_obj

    def run(
        self,
        initial_data: Optional[dict[str, Any]] = None,
    ) -> WorkflowResult:
        """执行工作流

        Args:
            initial_data: 初始上下文数据

        Returns:
            工作流执行结果
        """
        start_time = time.time()
        context = WorkflowContext(data=initial_data or {})
        results: list[StepResult] = []
        success = True
        error_msg: Optional[str] = None

        try:
            has_any_failure = False
            for step_obj in self._steps:
                result = self._execute_step(step_obj, context)
                results.append(result)

                if result.status == StepStatus.FAILED:
                    has_any_failure = True
                    if not self.config.continue_on_error:
                        success = False
                        error_msg = result.error
                        break

            # 即使 continue_on_error=True，如果有失败，整体仍标记为失败
            if has_any_failure:
                success = False
                if not error_msg:
                    error_msg = "部分步骤执行失败"

        except RuntimeError as e:
            success = False
            error_msg = str(e)
        except ValueError as e:
            success = False
            error_msg = str(e)

        total_duration = time.time() - start_time

        return WorkflowResult(
            success=success,
            steps=results,
            context=context,
            total_duration=total_duration,
            error=error_msg,
        )

    def _execute_step(
        self,
        step_obj: Step,
        context: WorkflowContext,
    ) -> StepResult:
        """执行单个步骤

        Args:
            step_obj: 步骤对象
            context: 执行上下文

        Returns:
            步骤执行结果
        """
        start_time = time.time()

        if step_obj.step_type == StepType.BRANCH:
            return self._execute_branch(step_obj, context, start_time)

        if step_obj.step_type == StepType.PARALLEL:
            return self._execute_parallel(step_obj, context, start_time)

        if step_obj.step_type == StepType.LOOP:
            return self._execute_loop(step_obj, context, start_time)

        # 普通步骤
        return self._execute_simple(step_obj, context, start_time)

    def _execute_simple(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """执行简单步骤

        Args:
            step_obj: 步骤对象
            context: 执行上下文
            start_time: 开始时间

        Returns:
            步骤执行结果
        """
        if step_obj.func is None:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.SKIPPED,
                duration=time.time() - start_time,
            )

        # 检查条件
        if step_obj.condition is not None:
            try:
                if not step_obj.condition(context):
                    return StepResult(
                        name=step_obj.name,
                        status=StepStatus.SKIPPED,
                        duration=time.time() - start_time,
                    )
            except Exception as e:
                return StepResult(
                    name=step_obj.name,
                    status=StepStatus.FAILED,
                    error=f"条件检查失败: {e}",
                    duration=time.time() - start_time,
                )

        # 执行步骤
        last_error: Optional[str] = None
        for attempt in range(self.config.retry_count + 1):
            try:
                output = step_obj.func(context)
                return StepResult(
                    name=step_obj.name,
                    status=StepStatus.COMPLETED,
                    output=output,
                    duration=time.time() - start_time,
                    retries=attempt,
                )
            except Exception as e:
                last_error = str(e)
                if attempt < self.config.retry_count:
                    time.sleep(self.config.retry_delay)

        return StepResult(
            name=step_obj.name,
            status=StepStatus.FAILED,
            error=last_error,
            duration=time.time() - start_time,
            retries=self.config.retry_count,
        )

    def _execute_branch(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """执行分支步骤

        Args:
            step_obj: 步骤对象
            context: 执行上下文
            start_time: 开始时间

        Returns:
            步骤执行结果
        """
        try:
            # func 存储条件函数
            condition_result = step_obj.func(context) if step_obj.func else False

            # 选择分支
            branches = step_obj.branch_yes if condition_result else step_obj.branch_no

            if not branches:
                return StepResult(
                    name=step_obj.name,
                    status=StepStatus.COMPLETED,
                    output={"branch": "yes" if condition_result else "no"},
                    duration=time.time() - start_time,
                )

            # 执行分支步骤
            branch_results: list[StepResult] = []
            for branch_step in branches:
                result = self._execute_step(branch_step, context)
                branch_results.append(result)

                if result.status == StepStatus.FAILED:
                    if not self.config.continue_on_error:
                        return StepResult(
                            name=step_obj.name,
                            status=StepStatus.FAILED,
                            error=f"分支步骤 '{result.name}' 失败: {result.error}",
                            output=branch_results,
                            duration=time.time() - start_time,
                        )

            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                output=branch_results,
                duration=time.time() - start_time,
            )

        except Exception as e:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )

    def _execute_parallel(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """执行并行步骤

        Args:
            step_obj: 步骤对象
            context: 执行上下文
            start_time: 开始时间

        Returns:
            步骤执行结果
        """
        steps = step_obj.parallel_steps or []
        if not steps:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                duration=time.time() - start_time,
            )

        # 获取并行配置
        parallel_config = step_obj.parallel_config
        max_workers = (
            parallel_config.max_workers
            if parallel_config
            else self.config.max_workers
        )

        futures: list[Future[StepResult]] = []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for s in steps:
                    future = executor.submit(self._execute_step, s, context)
                    futures.append(future)

            # 收集结果
            results: list[StepResult] = []
            has_error = False
            error_msg: Optional[str] = None

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    if result.status == StepStatus.FAILED:
                        has_error = True
                        error_msg = result.error
                except Exception as e:
                    has_error = True
                    error_msg = str(e)
                    results.append(StepResult(
                        name="parallel_task",
                        status=StepStatus.FAILED,
                        error=str(e),
                    ))

            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED if has_error else StepStatus.COMPLETED,
                output=results,
                error=error_msg,
                duration=time.time() - start_time,
            )

        except Exception as e:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )

    def _execute_loop(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """执行循环步骤

        Args:
            step_obj: 步骤对象
            context: 执行上下文
            start_time: 开始时间

        Returns:
            步骤执行结果
        """
        loop_config = step_obj.loop_config or LoopConfig()

        try:
            # 获取迭代项
            if step_obj.loop_items is not None:
                if callable(step_obj.loop_items):
                    items = step_obj.loop_items(context)
                else:
                    items = step_obj.loop_items
            else:
                items = []

            if not isinstance(items, list):
                items = list(items) if items else []

            results: list[StepResult] = []
            iteration = 0

            for item in items:
                if iteration >= loop_config.max_iterations:
                    break

                iteration += 1

                # 创建步骤
                if step_obj.step_template is not None:
                    step_template = step_obj.step_template(item)
                    result = self._execute_step(step_template, context)
                    results.append(result)

                    if result.status == StepStatus.FAILED and loop_config.break_on_error:
                        return StepResult(
                            name=step_obj.name,
                            status=StepStatus.FAILED,
                            error=f"循环步骤失败: {result.error}",
                            output=results,
                            duration=time.time() - start_time,
                        )

            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                output=results,
                duration=time.time() - start_time,
            )

        except Exception as e:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED,
                error=str(e),
                duration=time.time() - start_time,
            )

    def clear(self) -> None:
        """清空工作流步骤"""
        self._steps.clear()
        self._step_names.clear()

    @property
    def steps(self) -> list[Step]:
        """获取所有步骤"""
        return list(self._steps)
