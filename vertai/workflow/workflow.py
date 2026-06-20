"""Workflow orchestration engine.

Core capabilities:
- Sequential step execution
- Conditional branching (if/else)
- Loop execution (for items)
- Parallel execution with a thread-safe shared context

Design notes (S6):
- All builders (``step`` / ``branch`` / ``parallel`` / ``loop``) return ``self`` so
  the workflow is chainable: ``Workflow().step(...).parallel(...).run()``.
- The shared :class:`WorkflowContext` is guarded by a lock so parallel steps
  cannot lose updates when concurrently writing the same key.
- ``ParallelConfig.timeout`` and ``WorkflowConfig.timeout`` are real: a parallel
  step whose futures do not resolve in time is cancelled and reported FAILED;
  the whole ``run`` is bounded by the workflow timeout.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

# ============================================================================
# Configuration constants
# ============================================================================

DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT = 300.0
DEFAULT_RETRY_COUNT = 0
DEFAULT_RETRY_DELAY = 1.0
MAX_STEP_NAME_LENGTH = 100
# Letters, digits, underscore, hyphen, and CJK ideographs (U+4E00 - U+9FFF).
STEP_NAME_PATTERN = r"^[a-zA-Z0-9_\-一-龥]+$"
_NAME_RE = re.compile(STEP_NAME_PATTERN)

# Name used for a synthesized StepResult when a parallel future raised before
# producing a result of its own.
_ORPHAN_PARALLEL_TASK = "parallel_task"


# ============================================================================
# Enumerations
# ============================================================================

class StepStatus(str, Enum):
    """Status of a single step execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class LoopType(str, Enum):
    """Loop strategy."""

    FOR = "for"
    WHILE = "while"


class StepType(str, Enum):
    """Discriminator for :class:`Step`."""

    SIMPLE = "simple"
    BRANCH = "branch"
    PARALLEL = "parallel"
    LOOP = "loop"


# ============================================================================
# Configuration models
# ============================================================================

class WorkflowConfig(BaseModel):
    """Top-level workflow configuration.

    Example:
        >>> WorkflowConfig()  # defaults
        >>> WorkflowConfig(max_workers=8, timeout=600.0, retry_count=2)
    """

    max_workers: int = Field(
        default=DEFAULT_MAX_WORKERS, ge=1, le=32, description="Max threads for parallel steps"
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT, ge=1.0, description="Whole-workflow timeout in seconds"
    )
    retry_count: int = Field(default=DEFAULT_RETRY_COUNT, ge=0, le=10, description="Retries on failure")
    retry_delay: float = Field(default=DEFAULT_RETRY_DELAY, ge=0.0, description="Delay between retries (s)")
    continue_on_error: bool = Field(
        default=False, description="Whether to keep running after a step fails"
    )

    model_config = ConfigDict(extra="forbid", validate_default=True)


class ParallelConfig(BaseModel):
    """Configuration for a parallel step.

    ``timeout`` is real: each parallel step's futures are awaited with this
    budget; any future that has not resolved in time is cancelled and the step
    is reported FAILED.

    Example:
        >>> ParallelConfig(max_workers=4, timeout=60.0)
    """

    max_workers: int = Field(
        default=DEFAULT_MAX_WORKERS, ge=1, le=32, description="Max parallel threads"
    )
    timeout: float = Field(
        default=DEFAULT_TIMEOUT, ge=1.0, description="Per-step parallel timeout in seconds"
    )

    model_config = ConfigDict(extra="forbid", validate_default=True)


class LoopConfig(BaseModel):
    """Configuration for a loop step.

    Example:
        >>> LoopConfig(loop_type=LoopType.WHILE, max_iterations=100)
    """

    loop_type: LoopType = Field(default=LoopType.FOR, description="Loop strategy")
    max_iterations: int = Field(default=100, ge=1, le=10000, description="Hard iteration cap")
    break_on_error: bool = Field(default=True, description="Stop the loop on the first failure")

    model_config = ConfigDict(extra="forbid", validate_default=True)


# ============================================================================
# Shared execution context (thread-safe)
# ============================================================================

class WorkflowContext(BaseModel):
    """Execution context shared across steps.

    The ``data`` and ``metadata`` dicts are mutated by step functions, including
    from concurrent threads during a parallel step. All accessors take a lock so
    concurrent writers cannot lose updates.

    Example:
        >>> ctx = WorkflowContext()
        >>> ctx.set("key", "value")
        >>> ctx.get("key")
        'value'
    """

    data: dict[str, Any] = Field(default_factory=dict, description="Context data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")

    model_config = ConfigDict(extra="forbid")

    # Private attribute (not validated/serialized). RLock allows nested locking
    # from the same thread: ``get``/``set``/``update``/``clear`` all acquire it
    # and may be called from inside a ``with ctx.lock:`` block the user holds.
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    @property
    def lock(self) -> threading.RLock:
        """Return the context's lock.

        Each individual ``get`` / ``set`` / ``update`` / ``clear`` is atomic,
        so concurrent writers cannot corrupt the dict. A *compound*
        read-modify-write (``ctx.set(k, ctx.get(k) + 1)``) is NOT atomic by
        itself; callers that need that should hold this lock explicitly::

            with ctx.lock:
                ctx.set("counter", ctx.get("counter") + 1)

        The lock is reentrant, so ``get``/``set`` work correctly inside that
        ``with`` block.
        """
        return self._lock

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``data[key]`` if present, else ``default``."""
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` under ``key``."""
        with self._lock:
            self.data[key] = value

    def update(self, data: dict[str, Any]) -> None:
        """Merge ``data`` into the context atomically."""
        with self._lock:
            self.data.update(data)

    def clear(self) -> None:
        """Drop all data and metadata."""
        with self._lock:
            self.data.clear()
            self.metadata.clear()


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class StepResult:
    """Result of executing one step.

    Attributes:
        name: Step name.
        status: Outcome status.
        output: Optional payload produced by the step.
        error: Optional human-readable error message.
        duration: Wall-clock duration in seconds.
        retries: Number of retries that were performed.
    """

    name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0


@dataclass
class WorkflowResult:
    """Result of executing the whole workflow.

    Attributes:
        success: True iff no step failed (or failures were tolerated).
        steps: Per-step results, in execution order.
        context: Final shared context.
        total_duration: Wall-clock duration in seconds.
        error: Optional aggregated error message.
    """

    success: bool
    steps: list[StepResult] = field(default_factory=list)
    context: Optional[WorkflowContext] = None
    total_duration: float = 0.0
    error: Optional[str] = None


# ============================================================================
# Step definition
# ============================================================================

@dataclass
class Step:
    """A single workflow step.

    The meaning of the fields depends on ``step_type``:

    - ``SIMPLE``: ``func`` runs against the context (``condition`` optional).
    - ``BRANCH``: ``func`` is the predicate; ``branch_yes``/``branch_no`` are
      the two sequential sub-lists.
    - ``PARALLEL``: ``parallel_steps`` runs concurrently.
    - ``LOOP``: ``loop_items`` is a list or callable returning a list;
      ``step_template`` produces a :class:`Step` per item.
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
# Validation helpers
# ============================================================================

def _validate_step_name(name: str) -> str:
    """Validate a step name and return it on success.

    Raises:
        ValueError: If the name is empty, too long, or contains characters
            outside ``[A-Za-z0-9_-]`` and CJK ideographs.
    """
    if not name:
        raise ValueError("Step name must not be empty")

    if len(name) > MAX_STEP_NAME_LENGTH:
        raise ValueError(f"Step name too long: max {MAX_STEP_NAME_LENGTH} chars")

    if not _NAME_RE.match(name):
        raise ValueError(
            f"Step name '{name}' contains illegal characters. "
            "Allowed: letters, digits, underscore (_), hyphen (-), and CJK."
        )

    return name


# ============================================================================
# Workflow
# ============================================================================

# Type alias for the tuple form users pass to ``branch`` / ``parallel``.
StepSpec = tuple[str, Callable[[WorkflowContext], Any]]


class Workflow:
    """Workflow orchestration engine.

    Supports sequential steps, conditional branches, loops, and parallel
    execution. Builders return ``self`` so a workflow can be fluently chained
    and then run in one expression:

        >>> result = (
        ...     Workflow()
        ...     .step("setup", lambda ctx: ctx.set("ready", True))
        ...     .parallel([("a", task_a), ("b", task_b)])
        ...     .run()
        ... )

    For ergonomics, ``last_step`` exposes the most recently added :class:`Step`
    object so callers that need the underlying definition can still reach it.
    """

    def __init__(self, config: Optional[WorkflowConfig] = None) -> None:
        """Initialize with optional ``config`` (defaults are applied otherwise)."""
        self.config = config or WorkflowConfig()
        self._steps: list[Step] = []
        self._step_names: set[str] = set()
        self._last_step: Optional[Step] = None

    # ----- builders (all return self for chaining) ---------------------------

    def step(
        self,
        name: str,
        func: Callable[[WorkflowContext], Any],
        condition: Optional[Callable[[WorkflowContext], bool]] = None,
    ) -> "Workflow":
        """Append a sequential step.

        Args:
            name: Unique step name.
            func: Step body taking the shared context.
            condition: Optional predicate; when it returns False the step is
                skipped.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: Invalid or duplicate name.
        """
        validated_name = _validate_step_name(name)
        self._check_unique(validated_name)

        step_obj = Step(
            name=validated_name,
            step_type=StepType.SIMPLE,
            func=func,
            condition=condition,
        )
        self._append(step_obj)
        return self

    def branch(
        self,
        condition: Callable[[WorkflowContext], bool],
        yes_steps: Optional[list[StepSpec]] = None,
        no_steps: Optional[list[StepSpec]] = None,
        name: Optional[str] = None,
    ) -> "Workflow":
        """Append a conditional branch.

        Args:
            condition: Predicate over the context. ``True`` selects
                ``yes_steps``; ``False`` selects ``no_steps``.
            yes_steps: Steps to run when the predicate is True.
            no_steps: Steps to run when the predicate is False.
            name: Optional step name; auto-generated when omitted.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: Condition is None or name is invalid/duplicate.
        """
        if condition is None:
            raise ValueError("Branch condition must not be None")

        branch_name = name or f"branch_{len(self._step_names)}"
        validated_name = _validate_step_name(branch_name)
        self._check_unique(validated_name)

        step_obj = Step(
            name=validated_name,
            step_type=StepType.BRANCH,
            func=condition,
            branch_yes=self._materialize_steps(yes_steps),
            branch_no=self._materialize_steps(no_steps),
        )
        self._append(step_obj)
        return self

    def parallel(
        self,
        steps: list[StepSpec],
        config: Optional[ParallelConfig] = None,
        name: Optional[str] = None,
    ) -> "Workflow":
        """Append a parallel step.

        Args:
            steps: Non-empty list of ``(name, func)`` tuples run concurrently.
            config: Optional :class:`ParallelConfig` (workers + timeout).
            name: Optional step name; auto-generated when omitted.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: Empty step list, or invalid/duplicate name.
        """
        if not steps:
            raise ValueError("Parallel step list must not be empty")

        parallel_name = name or f"parallel_{len(self._step_names)}"
        validated_name = _validate_step_name(parallel_name)
        self._check_unique(validated_name)

        step_obj = Step(
            name=validated_name,
            step_type=StepType.PARALLEL,
            parallel_steps=self._materialize_steps(steps),
            parallel_config=config,
        )
        self._append(step_obj)
        return self

    def loop(
        self,
        items: Union[list[Any], Callable[[WorkflowContext], list[Any]]],
        step_name_template: Callable[[Any], str],
        step_func_template: Callable[[Any], Callable[[WorkflowContext], Any]],
        config: Optional[LoopConfig] = None,
        name: Optional[str] = None,
    ) -> "Workflow":
        """Append a loop step.

        Args:
            items: Either a literal list or a callable that resolves the list
                against the context at execution time.
            step_name_template: Maps an item to a step name.
            step_func_template: Maps an item to the step body.
            config: Optional :class:`LoopConfig`.
            name: Optional step name; auto-generated when omitted.

        Returns:
            ``self`` (for chaining).

        Raises:
            ValueError: A template is None or name is invalid/duplicate.
        """
        if step_name_template is None or step_func_template is None:
            raise ValueError("Loop name/func templates must not be None")

        loop_name = name or f"loop_{len(self._step_names)}"
        validated_name = _validate_step_name(loop_name)
        self._check_unique(validated_name)

        def _make_step(item: Any) -> Step:
            return Step(
                name=step_name_template(item),
                step_type=StepType.SIMPLE,
                func=step_func_template(item),
            )

        step_obj = Step(
            name=validated_name,
            step_type=StepType.LOOP,
            loop_items=items,
            step_template=_make_step,
            loop_config=config or LoopConfig(),
        )
        self._append(step_obj)
        return self

    # ----- run ---------------------------------------------------------------

    def run(
        self,
        initial_data: Optional[dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Execute the workflow.

        Args:
            initial_data: Optional seed for the shared context.

        Returns:
            The :class:`WorkflowResult`.

        Timeout semantics:
            ``config.timeout`` is a real wall-clock budget checked between
            steps. If the deadline has already passed before dispatching the
            next step, the run stops, is reported with ``success=False`` and
            ``error="Workflow timed out after <N>s"``, and contains the results
            gathered so far. Steps themselves are bounded by their own
            mechanisms (parallel steps by ``ParallelConfig.timeout``); the
            workflow-level timeout prevents a long chain of sequential steps
            from running indefinitely.
        """
        start_time = time.time()
        context = WorkflowContext(data=initial_data or {})
        deadline = start_time + self.config.timeout

        results: list[StepResult] = []
        success = True
        error_msg: Optional[str] = None
        has_any_failure = False
        timed_out = False

        for step_obj in self._steps:
            if time.time() >= deadline:
                timed_out = True
                break

            result = self._execute_step(step_obj, context)
            results.append(result)

            if result.status == StepStatus.FAILED:
                has_any_failure = True
                if not self.config.continue_on_error:
                    success = False
                    error_msg = result.error
                    break

        # With continue_on_error=True, the run is still marked failed if any
        # step failed, but every step had a chance to run.
        if has_any_failure:
            success = False
            if not error_msg:
                error_msg = "One or more steps failed"

        if timed_out:
            success = False
            error_msg = f"Workflow timed out after {self.config.timeout}s"

        return WorkflowResult(
            success=success,
            steps=results,
            context=context,
            total_duration=time.time() - start_time,
            error=error_msg,
        )

    # ----- per-step execution ------------------------------------------------

    def _execute_step(
        self,
        step_obj: Step,
        context: WorkflowContext,
    ) -> StepResult:
        """Dispatch a step to its type-specific executor."""
        start_time = time.time()

        if step_obj.step_type == StepType.BRANCH:
            return self._execute_branch(step_obj, context, start_time)

        if step_obj.step_type == StepType.PARALLEL:
            return self._execute_parallel(step_obj, context, start_time)

        if step_obj.step_type == StepType.LOOP:
            return self._execute_loop(step_obj, context, start_time)

        return self._execute_simple(step_obj, context, start_time)

    def _execute_simple(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """Execute a SIMPLE step, honoring ``condition`` and retries."""
        if step_obj.func is None:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.SKIPPED,
                duration=time.time() - start_time,
            )

        # Evaluate the condition first; a failing predicate is a SKIP, not a
        # failure.
        if step_obj.condition is not None:
            try:
                if not step_obj.condition(context):
                    return StepResult(
                        name=step_obj.name,
                        status=StepStatus.SKIPPED,
                        duration=time.time() - start_time,
                    )
            except Exception as exc:
                return StepResult(
                    name=step_obj.name,
                    status=StepStatus.FAILED,
                    error=f"Condition check failed: {exc}",
                    duration=time.time() - start_time,
                )

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
            except Exception as exc:
                last_error = str(exc)
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
        """Execute a BRANCH step."""
        try:
            # ``func`` stores the condition predicate.
            condition_result = bool(step_obj.func(context)) if step_obj.func else False
            branches = step_obj.branch_yes if condition_result else step_obj.branch_no

            if not branches:
                return StepResult(
                    name=step_obj.name,
                    status=StepStatus.COMPLETED,
                    output={"branch": "yes" if condition_result else "no"},
                    duration=time.time() - start_time,
                )

            branch_results: list[StepResult] = []
            for branch_step in branches:
                result = self._execute_step(branch_step, context)
                branch_results.append(result)

                if result.status == StepStatus.FAILED and not self.config.continue_on_error:
                    return StepResult(
                        name=step_obj.name,
                        status=StepStatus.FAILED,
                        error=f"Branch step '{result.name}' failed: {result.error}",
                        output=branch_results,
                        duration=time.time() - start_time,
                    )

            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                output=branch_results,
                duration=time.time() - start_time,
            )

        except Exception as exc:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED,
                error=str(exc),
                duration=time.time() - start_time,
            )

    def _execute_parallel(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """Execute a PARALLEL step with a real per-step timeout.

        Each sub-step runs on the shared thread pool. We wait for the futures
        with the configured timeout (from ``parallel_config`` or the
        workflow-level budget); any future that has not resolved in time is
        cancelled and reported FAILED, and the executor is shut down without
        blocking so the workflow returns promptly at the deadline. The shared
        :class:`WorkflowContext` is lock-protected, so concurrent writers do
        not lose updates.

        Note: Python cannot forcibly kill a running thread, so a sub-step that
        ignores the deadline may keep running in the background until the
        interpreter tears it down. The contract we honor is that
        ``_execute_parallel`` returns at (or very near) the deadline with
        FAILED results for any sub-step that has not finished.
        """
        steps = step_obj.parallel_steps or []
        if not steps:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                duration=time.time() - start_time,
            )

        parallel_config = step_obj.parallel_config
        max_workers = parallel_config.max_workers if parallel_config else self.config.max_workers
        timeout = parallel_config.timeout if parallel_config else self.config.timeout

        results: list[StepResult] = []
        has_error = False
        error_msg: Optional[str] = None

        # Manage the executor manually so that on timeout we can call
        # ``shutdown(wait=False, cancel_futures=True)`` and return immediately
        # rather than blocking on the slow worker. ``cancel_futures`` is
        # Python 3.9+; the project targets 3.10+.
        executor = ThreadPoolExecutor(max_workers=max_workers)
        future_to_step: dict[Future[StepResult], Step] = {}
        for sub_step in steps:
            future = executor.submit(self._execute_step, sub_step, context)
            future_to_step[future] = sub_step

        deadline = start_time + timeout
        pending: set[Future[StepResult]] = set(future_to_step.keys())

        # Poll until every future resolves or the deadline passes.
        while pending:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            done_now, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
            if not done_now:
                # No future finished within the remaining budget.
                break

        # Anything still in ``pending`` hit the deadline: cancel + synthesize.
        if pending:
            for future in pending:
                future.cancel()
            # Stop the pool without joining the still-running workers.
            executor.shutdown(wait=False, cancel_futures=True)
            for future in pending:
                sub_step = future_to_step[future]
                has_error = True
                if error_msg is None:
                    error_msg = f"Parallel task '{sub_step.name}' timed out after {timeout}s"
                results.append(
                    StepResult(
                        name=sub_step.name,
                        status=StepStatus.FAILED,
                        error=f"timed out after {timeout}s",
                        duration=time.time() - start_time,
                    )
                )
        else:
            executor.shutdown(wait=True)

        # Materialize results in submission order, skipping ones already added
        # above (the timed-out pending set).
        already_added: set[Future[StepResult]] = set(pending)
        for future, sub_step in future_to_step.items():
            if future in already_added:
                continue
            try:
                result = future.result()
                results.append(result)
                if result.status == StepStatus.FAILED:
                    has_error = True
                    if error_msg is None:
                        error_msg = result.error
            except Exception as exc:
                has_error = True
                if error_msg is None:
                    error_msg = str(exc)
                results.append(
                    StepResult(
                        name=sub_step.name,
                        status=StepStatus.FAILED,
                        error=str(exc),
                    )
                )

        return StepResult(
            name=step_obj.name,
            status=StepStatus.FAILED if has_error else StepStatus.COMPLETED,
            output=results,
            error=error_msg,
            duration=time.time() - start_time,
        )

    def _execute_loop(
        self,
        step_obj: Step,
        context: WorkflowContext,
        start_time: float,
    ) -> StepResult:
        """Execute a LOOP step, honoring ``max_iterations`` and ``break_on_error``."""
        loop_config = step_obj.loop_config or LoopConfig()

        try:
            if step_obj.loop_items is None:
                items: list[Any] = []
            elif callable(step_obj.loop_items):
                items = step_obj.loop_items(context)
            else:
                items = list(step_obj.loop_items)

            if not isinstance(items, list):
                items = list(items) if items else []

            results: list[StepResult] = []
            for iteration, item in enumerate(items):
                if iteration >= loop_config.max_iterations:
                    break

                if step_obj.step_template is None:
                    continue
                template_step = step_obj.step_template(item)
                result = self._execute_step(template_step, context)
                results.append(result)

                if result.status == StepStatus.FAILED and loop_config.break_on_error:
                    return StepResult(
                        name=step_obj.name,
                        status=StepStatus.FAILED,
                        error=f"Loop step failed: {result.error}",
                        output=results,
                        duration=time.time() - start_time,
                    )

            return StepResult(
                name=step_obj.name,
                status=StepStatus.COMPLETED,
                output=results,
                duration=time.time() - start_time,
            )

        except Exception as exc:
            return StepResult(
                name=step_obj.name,
                status=StepStatus.FAILED,
                error=str(exc),
                duration=time.time() - start_time,
            )

    # ----- bookkeeping -------------------------------------------------------

    def clear(self) -> None:
        """Drop every added step."""
        self._steps.clear()
        self._step_names.clear()
        self._last_step = None

    @property
    def steps(self) -> list[Step]:
        """Return a defensive copy of all added steps."""
        return list(self._steps)

    @property
    def last_step(self) -> Optional[Step]:
        """The most recently added :class:`Step`, or None."""
        return self._last_step

    # ----- internals ---------------------------------------------------------

    def _check_unique(self, name: str) -> None:
        if name in self._step_names:
            raise ValueError(f"Step name '{name}' already exists")

    def _append(self, step_obj: Step) -> None:
        self._steps.append(step_obj)
        self._step_names.add(step_obj.name)
        self._last_step = step_obj

    @staticmethod
    def _materialize_steps(specs: Optional[list[StepSpec]]) -> list[Step]:
        """Convert a list of ``(name, func)`` tuples into SIMPLE steps."""
        steps: list[Step] = []
        if not specs:
            return steps
        for step_name, step_func in specs:
            steps.append(
                Step(
                    name=step_name,
                    step_type=StepType.SIMPLE,
                    func=step_func,
                )
            )
        return steps
