"""Tests for the workflow orchestration engine.

Testing philosophy (per ROADMAP):
- Pure control-flow logic uses real assertions, no mocking.
- Concurrency is exercised with real threads and a real ThreadPoolExecutor.
- No line-number-targeted coverage tests; tests come from "how would a user
  exercise or misuse this".
- No ``except Exception: pass`` masking; failures are surfaced.
"""

import threading
import time

import pytest
from pydantic import ValidationError

from vertai.workflow import (
    LoopConfig,
    LoopType,
    ParallelConfig,
    Step,
    StepResult,
    StepStatus,
    StepType,
    Workflow,
    WorkflowConfig,
    WorkflowContext,
    WorkflowResult,
)


# ============================================================================
# Configuration models
# ============================================================================

class TestWorkflowConfig:
    """WorkflowConfig defaults and validation bounds."""

    def test_defaults(self):
        config = WorkflowConfig()
        assert config.max_workers == 4
        assert config.timeout == 300.0
        assert config.retry_count == 0
        assert config.retry_delay == 1.0
        assert config.continue_on_error is False

    def test_custom_values_round_trip(self):
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

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_workers": 0},
            {"max_workers": 33},
            {"timeout": 0.5},
            {"retry_count": -1},
            {"retry_count": 11},
        ],
    )
    def test_out_of_bounds_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            WorkflowConfig(**kwargs)

    def test_unknown_field_rejected(self):
        # ``extra="forbid"`` catches typos in the config.
        with pytest.raises(ValidationError):
            WorkflowConfig(bogus=1)  # type: ignore[call-arg]


class TestParallelConfig:
    """ParallelConfig defaults and validation bounds."""

    def test_defaults(self):
        config = ParallelConfig()
        assert config.max_workers == 4
        assert config.timeout == 300.0

    def test_custom_values(self):
        config = ParallelConfig(max_workers=8, timeout=60.0)
        assert config.max_workers == 8
        assert config.timeout == 60.0

    @pytest.mark.parametrize("kwargs", [{"max_workers": 0}, {"timeout": 0.5}])
    def test_out_of_bounds_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            ParallelConfig(**kwargs)


class TestLoopConfig:
    """LoopConfig defaults and validation bounds."""

    def test_defaults(self):
        config = LoopConfig()
        assert config.loop_type == LoopType.FOR
        assert config.max_iterations == 100
        assert config.break_on_error is True

    def test_custom_values(self):
        config = LoopConfig(loop_type=LoopType.WHILE, max_iterations=50, break_on_error=False)
        assert config.loop_type == LoopType.WHILE
        assert config.max_iterations == 50
        assert config.break_on_error is False


# ============================================================================
# WorkflowContext (thread safety lives here)
# ============================================================================

class TestWorkflowContext:
    """WorkflowContext basic API and lock behavior under concurrent access."""

    def test_get_set_update_clear(self):
        ctx = WorkflowContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"

        ctx.update({"a": 1, "b": 2})
        assert ctx.get("a") == 1 and ctx.get("b") == 2

        ctx.clear()
        assert ctx.get("a") is None
        assert ctx.data == {}

    def test_concurrent_sets_never_corrupt_dict(self):
        """Each individual ``set`` is atomic: concurrent writers writing
        distinct keys all land, and writes to the same key never corrupt
        the dict. This is the contract the internal lock provides."""
        ctx = WorkflowContext()
        n_threads = 8
        per_thread = 100

        def worker(tid):
            for i in range(per_thread):
                ctx.set(f"k_{tid}_{i}", i)
                # Also hammer a shared key; last writer wins, but the dict
                # must never be corrupted.
                ctx.set("shared", f"v_{tid}_{i}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every distinct key landed.
        assert len(ctx.data) == n_threads * per_thread + 1
        # Shared key is one of the written values (no corruption).
        assert ctx.get("shared", "").startswith("v_")
        for tid in range(n_threads):
            for i in range(per_thread):
                assert ctx.get(f"k_{tid}_{i}") == i

    def test_compound_rmw_is_atomic_under_exposed_lock(self):
        """A compound read-modify-write is atomic when the caller holds
        ``ctx.lock``. This is the documented pattern for parallel steps
        that need to update a counter; without the explicit lock the RMW
        would lose updates. We widen the race window to make the test fail
        reliably if the lock were removed or bypassed."""
        ctx = WorkflowContext()
        n_threads = 8
        per_thread = 200
        expected = n_threads * per_thread

        def worker():
            for _ in range(per_thread):
                with ctx.lock:
                    current = ctx.get("counter", 0)
                    time.sleep(0.0001)  # widen the window
                    ctx.set("counter", current + 1)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ctx.get("counter") == expected


# ============================================================================
# Chainable builders
# ============================================================================

class TestChainableBuilders:
    """All four builders return self so workflows compose fluently."""

    def test_step_returns_workflow(self):
        wf = Workflow()
        assert wf.step("s", lambda ctx: None) is wf

    def test_branch_returns_workflow(self):
        wf = Workflow()
        assert wf.branch(condition=lambda ctx: True, yes_steps=[("y", lambda ctx: None)]) is wf

    def test_parallel_returns_workflow(self):
        wf = Workflow()
        assert (
            wf.parallel(steps=[("p", lambda ctx: None)], config=ParallelConfig(timeout=2.0)) is wf
        )

    def test_loop_returns_workflow(self):
        wf = Workflow()
        assert (
            wf.loop(
                items=["a"],
                step_name_template=lambda i: f"item_{i}",
                step_func_template=lambda i: lambda ctx: None,
            )
            is wf
        )

    def test_full_chain_runs(self):
        """The fluent style from the docs must actually work end to end."""
        wf = Workflow(config=WorkflowConfig(timeout=5.0))
        result = (
            wf.step("setup", lambda ctx: ctx.set("x", 1))
            .step("double", lambda ctx: ctx.set("x", ctx.get("x") * 2))
            .run()
        )
        assert result.success is True
        assert result.context.get("x") == 2
        assert [s.name for s in result.steps] == ["setup", "double"]

    def test_last_step_exposes_underlying_definition(self):
        wf = Workflow()
        wf.step("only", lambda ctx: None)
        assert wf.last_step is not None
        assert wf.last_step.name == "only"
        assert wf.last_step.step_type == StepType.SIMPLE

    def test_clear_resets_steps_and_last_step(self):
        wf = Workflow()
        wf.step("a", lambda ctx: None)
        wf.step("b", lambda ctx: None)
        wf.clear()
        assert wf.steps == []
        assert wf.last_step is None
        # And the workflow is reusable.
        wf.step("c", lambda ctx: None)
        assert len(wf.steps) == 1
        assert wf.last_step is not None and wf.last_step.name == "c"


# ============================================================================
# Step name validation (real user errors)
# ============================================================================

class TestStepNameValidation:
    """Names users actually try, and the errors they should see."""

    @pytest.mark.parametrize("name", ["valid", "valid_name", "valid-name", "valid123", "中文名称"])
    def test_accepted_names(self, name):
        wf = Workflow()
        wf.step(name, lambda ctx: None)
        assert wf.steps[-1].name == name

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Workflow().step("", lambda ctx: None)

    def test_overlong_name_rejected(self):
        with pytest.raises(ValueError, match="too long"):
            Workflow().step("a" * 101, lambda ctx: None)

    @pytest.mark.parametrize("name", ["invalid@name", "invalid name", "tab\tname"])
    def test_illegal_characters_rejected(self, name):
        with pytest.raises(ValueError, match="illegal characters"):
            Workflow().step(name, lambda ctx: None)

    def test_duplicate_name_rejected_on_step(self):
        wf = Workflow()
        wf.step("dup", lambda ctx: None)
        with pytest.raises(ValueError, match="already exists"):
            wf.step("dup", lambda ctx: None)

    def test_auto_generated_names_are_unique(self):
        """branch/parallel/loop with name=None must not collide with each other
        or with an explicit user name."""
        wf = Workflow()
        wf.step("step1", lambda ctx: None)  # bumps the counter to 1
        wf.branch(condition=lambda ctx: True, yes_steps=[("y", lambda ctx: None)])
        # The auto-name branch_1 should now be taken; explicitly asking for it
        # must fail (rather than silently shadowing).
        with pytest.raises(ValueError, match="already exists"):
            wf.branch(
                condition=lambda ctx: True,
                yes_steps=[("y2", lambda ctx: None)],
                name="branch_1",
            )


# ============================================================================
# Sequential / conditional execution
# ============================================================================

class TestSequentialExecution:
    """The simplest workflow path: ordered steps, conditions, retries."""

    def test_empty_workflow_succeeds(self):
        result = Workflow().run()
        assert result.success is True
        assert result.steps == []
        assert result.total_duration >= 0.0

    def test_single_step_completes(self):
        wf = Workflow()
        wf.step("only", lambda ctx: ctx.set("done", True))
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.context.get("done") is True

    def test_steps_run_in_order(self):
        wf = Workflow()
        order: list[int] = []
        wf.step("a", lambda ctx: order.append(1))
        wf.step("b", lambda ctx: order.append(2))
        wf.step("c", lambda ctx: order.append(3))
        result = wf.run()
        assert result.success is True
        assert order == [1, 2, 3]

    def test_initial_data_seeds_context(self):
        wf = Workflow()
        wf.step("double", lambda ctx: ctx.set("y", ctx.get("x") * 2))
        result = wf.run(initial_data={"x": 21})
        assert result.success is True
        assert result.context.get("y") == 42

    def test_condition_true_runs_step(self):
        wf = Workflow()
        ran: list[str] = []
        wf.step(
            "guarded",
            lambda ctx: ran.append("yes"),
            condition=lambda ctx: ctx.get("go", True),
        )
        result = wf.run(initial_data={"go": True})
        assert result.success is True
        assert ran == ["yes"]
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_condition_false_skips_step(self):
        wf = Workflow()
        ran: list[str] = []
        wf.step(
            "guarded",
            lambda ctx: ran.append("yes"),
            condition=lambda ctx: ctx.get("go", False),
        )
        result = wf.run(initial_data={"go": False})
        assert result.success is True
        assert ran == []
        assert result.steps[0].status == StepStatus.SKIPPED

    def test_condition_raising_marks_step_failed(self):
        wf = Workflow()

        def bad_condition(ctx):
            raise ValueError("bad predicate")

        wf.step("guarded", lambda ctx: None, condition=bad_condition)
        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "Condition check failed" in (result.steps[0].error or "")

    def test_step_func_none_is_skipped_not_failed(self):
        """A constructed Step with no body should not crash the workflow."""
        wf = Workflow()
        wf._steps.append(Step(name="empty", step_type=StepType.SIMPLE, func=None))
        wf._step_names.add("empty")
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.SKIPPED


# ============================================================================
# Error handling and retries
# ============================================================================

class TestErrorHandling:
    """How failures propagate, and how retries behave."""

    def test_step_failure_propagates_and_stops(self):
        wf = Workflow()
        wf.step("boom", lambda ctx: 1 / 0)
        wf.step("never", lambda ctx: ctx.set("ran", True))
        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        # The second step never ran.
        assert len(result.steps) == 1
        assert result.context.get("ran") is None

    def test_continue_on_error_runs_remaining_steps(self):
        wf = Workflow(config=WorkflowConfig(continue_on_error=True, timeout=5.0))
        wf.step("boom", lambda ctx: 1 / 0)
        wf.step("ok", lambda ctx: ctx.set("ok", True))
        result = wf.run()
        # The run is still marked failed (a step failed), but the next step ran.
        assert result.success is False
        assert result.context.get("ok") is True
        assert len(result.steps) == 2

    def test_retry_eventually_succeeds(self):
        attempts = [0]
        wf = Workflow(config=WorkflowConfig(retry_count=2, retry_delay=0.01))

        def flaky(ctx):
            attempts[0] += 1
            if attempts[0] < 3:
                raise RuntimeError("not yet")
            return "ok"

        wf.step("flaky", flaky)
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].retries == 2
        assert attempts[0] == 3

    def test_retry_exhausted_reports_last_error(self):
        wf = Workflow(config=WorkflowConfig(retry_count=1, retry_delay=0.01))
        wf.step("always_boom", lambda ctx: (_ for _ in ()).throw(ValueError("always")))
        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert result.steps[0].retries == 1
        assert "always" in (result.steps[0].error or "")


# ============================================================================
# Branches
# ============================================================================

class TestBranchExecution:
    """Conditional branching: only the matching side runs."""

    def test_yes_branch_taken_when_predicate_true(self):
        ran: list[str] = []
        wf = Workflow()
        wf.step("seed", lambda ctx: ctx.set("flag", True))
        wf.branch(
            condition=lambda ctx: ctx.get("flag"),
            yes_steps=[("y", lambda ctx: ran.append("yes"))],
            no_steps=[("n", lambda ctx: ran.append("no"))],
        )
        result = wf.run()
        assert result.success is True
        assert ran == ["yes"]

    def test_no_branch_taken_when_predicate_false(self):
        ran: list[str] = []
        wf = Workflow()
        wf.step("seed", lambda ctx: ctx.set("flag", False))
        wf.branch(
            condition=lambda ctx: ctx.get("flag"),
            yes_steps=[("y", lambda ctx: ran.append("yes"))],
            no_steps=[("n", lambda ctx: ran.append("no"))],
        )
        result = wf.run()
        assert result.success is True
        assert ran == ["no"]

    def test_branch_with_no_steps_records_outcome(self):
        wf = Workflow()
        wf.branch(condition=lambda ctx: True)
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].output == {"branch": "yes"}

    def test_branch_condition_failure_marks_step_failed(self):
        wf = Workflow()

        def bad_condition(ctx):
            raise ValueError("bad branch predicate")

        wf.branch(condition=bad_condition, yes_steps=[("y", lambda ctx: None)])
        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "bad branch predicate" in (result.steps[0].error or "")

    def test_branch_with_none_condition_is_rejected(self):
        with pytest.raises(ValueError, match="condition must not be None"):
            Workflow().branch(condition=None)  # type: ignore[arg-type]

    def test_branch_step_failure_aborts_branch(self):
        wf = Workflow()
        wf.branch(
            condition=lambda ctx: True,
            yes_steps=[
                ("boom", lambda ctx: 1 / 0),
                ("after", lambda ctx: ctx.set("ran", True)),
            ],
        )
        result = wf.run()
        assert result.success is False
        assert "boom" in (result.steps[0].error or "")
        # ``after`` never ran (no continue_on_error).
        assert result.context.get("ran") is None


# ============================================================================
# Loops
# ============================================================================

class TestLoopExecution:
    """Loop semantics: items, dynamic items, max_iterations, break_on_error."""

    def test_static_items_processed_in_order(self):
        seen: list[str] = []
        wf = Workflow()
        wf.loop(
            items=["a", "b", "c"],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: seen.append(item),
        )
        result = wf.run()
        assert result.success is True
        assert seen == ["a", "b", "c"]

    def test_dynamic_items_resolved_at_runtime(self):
        seen: list[str] = []
        wf = Workflow()
        wf.step("seed", lambda ctx: ctx.set("items", ["x", "y", "z"]))
        wf.loop(
            items=lambda ctx: ctx.get("items", []),
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: seen.append(item),
        )
        result = wf.run()
        assert result.success is True
        assert seen == ["x", "y", "z"]

    def test_empty_items_completes_with_no_children(self):
        wf = Workflow()
        wf.loop(
            items=[],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: None,
        )
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].output == []

    def test_max_iterations_caps_loop(self):
        counter = [0]
        wf = Workflow()
        wf.loop(
            items=range(1000),
            step_name_template=lambda i: f"item_{i}",
            step_func_template=lambda i: lambda ctx: counter.__setitem__(0, counter[0] + 1),
            config=LoopConfig(max_iterations=5),
        )
        result = wf.run()
        assert result.success is True
        assert counter[0] == 5

    def test_break_on_error_stops_loop(self):
        seen: list[str] = []

        def make(item):
            def fn(ctx):
                if item == "b":
                    raise ValueError("stop at b")
                seen.append(item)

            return fn

        wf = Workflow()
        wf.loop(
            items=["a", "b", "c"],
            step_name_template=lambda item: f"item_{item}",
            step_func_template=make,
            config=LoopConfig(break_on_error=True),
        )
        result = wf.run()
        assert result.success is False
        assert seen == ["a"]  # c never ran

    def test_loop_with_none_template_rejected(self):
        wf = Workflow()
        with pytest.raises(ValueError, match="templates must not be None"):
            wf.loop(
                items=["a"],
                step_name_template=None,  # type: ignore[arg-type]
                step_func_template=lambda item: lambda ctx: None,
            )

    def test_loop_items_callable_failure_marks_step_failed(self):
        wf = Workflow()

        def bad_items(ctx):
            raise ValueError("items fn boom")

        wf.loop(
            items=bad_items,
            step_name_template=lambda item: f"item_{item}",
            step_func_template=lambda item: lambda ctx: None,
        )
        result = wf.run()
        assert result.success is False
        assert result.steps[0].status == StepStatus.FAILED
        assert "items fn boom" in (result.steps[0].error or "")

    def test_loop_with_iterable_non_list_items(self):
        """``items`` may be any iterable (e.g. a generator or range); the loop
        materializes it and iterates."""
        seen: list[int] = []
        wf = Workflow()
        wf.loop(
            items=range(4),
            step_name_template=lambda i: f"item_{i}",
            step_func_template=lambda i: lambda ctx: seen.append(i),
        )
        result = wf.run()
        assert result.success is True
        assert seen == [0, 1, 2, 3]

    def test_loop_step_with_null_items_completes(self):
        """A directly-constructed LOOP step with ``loop_items=None`` completes
        cleanly with no iterations (rather than crashing)."""
        wf = Workflow()
        wf._steps.append(
            Step(
                name="null_loop",
                step_type=StepType.LOOP,
                loop_items=None,
                step_template=lambda item: Step(name=f"i_{item}", func=lambda ctx: None),
                loop_config=LoopConfig(),
            )
        )
        wf._step_names.add("null_loop")
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED
        assert result.steps[0].output == []


# ============================================================================
# Parallel execution — real concurrency, no mock executors
# ============================================================================

class TestParallelExecution:
    """Parallel steps run on a real ThreadPoolExecutor.

    The two contracts under test:
    1. No lost updates to the shared context (lock works).
    2. Per-step timeout actually cancels slow tasks at the deadline.
    """

    def test_parallel_atomic_rmw_under_exposed_lock(self):
        """Parallel steps that mutate shared state must coordinate via the
        context's exposed lock. Each task performs a compound
        read-modify-write under ``ctx.lock``; the final total must equal the
        exact expected sum. The yield inside the locked region widens the
        window so the test fails reliably if a step forgets to take the
        lock."""
        n_tasks = 8
        per_task = 200
        expected = n_tasks * per_task

        def increment_many(ctx):
            for _ in range(per_task):
                with ctx.lock:
                    cur = ctx.get("counter", 0)
                    time.sleep(0.0001)
                    ctx.set("counter", cur + 1)

        wf = Workflow()
        wf.parallel(
            steps=[(f"t{i}", increment_many) for i in range(n_tasks)],
            config=ParallelConfig(max_workers=n_tasks, timeout=60.0),
        )
        result = wf.run()
        assert result.success is True
        assert result.context.get("counter") == expected

    def test_parallel_distinct_writes_all_land(self):
        """Parallel steps writing distinct keys must all land: this is the
        single-op atomicity the internal lock guarantees (no dict
        corruption, no lost writes of independent keys)."""
        n_tasks = 8
        per_task = 50

        def writer(tid):
            def fn(ctx):
                for i in range(per_task):
                    ctx.set(f"k_{tid}_{i}", i)

            return fn

        wf = Workflow()
        wf.parallel(
            steps=[(f"t{i}", writer(i)) for i in range(n_tasks)],
            config=ParallelConfig(max_workers=n_tasks, timeout=30.0),
        )
        result = wf.run()
        assert result.success is True
        for tid in range(n_tasks):
            for i in range(per_task):
                assert result.context.get(f"k_{tid}_{i}") == i

    def test_parallel_results_collected_for_all_substeps(self):
        wf = Workflow()
        wf.parallel(
            steps=[
                ("a", lambda ctx: ctx.set("a", 1)),
                ("b", lambda ctx: ctx.set("b", 2)),
                ("c", lambda ctx: ctx.set("c", 3)),
            ],
            config=ParallelConfig(timeout=10.0),
        )
        result = wf.run()
        assert result.success is True
        assert result.context.get("a") == 1
        assert result.context.get("b") == 2
        assert result.context.get("c") == 3
        inner = result.steps[0]
        names = sorted(r.name for r in inner.output)
        assert names == ["a", "b", "c"]

    def test_parallel_substep_failure_marks_step_failed_but_others_finish(self):
        """A failing sub-step fails the parallel step, but sibling sub-steps
        are still given a chance to complete (they ran concurrently)."""
        finished: list[str] = []

        def fast(name):
            def fn(ctx):
                finished.append(name)

            return fn

        wf = Workflow()
        wf.parallel(
            steps=[
                ("fast_a", fast("a")),
                ("boom", lambda ctx: 1 / 0),
                ("fast_b", fast("b")),
            ],
            config=ParallelConfig(timeout=10.0),
        )
        result = wf.run()
        assert result.success is False
        inner = result.steps[0]
        statuses = {r.name: r.status for r in inner.output}
        assert statuses["boom"] == StepStatus.FAILED
        # Both siblings had the chance to finish. (They are real threads, so
        # we assert the looser property that at least one did.)
        assert "a" in finished or "b" in finished

    def test_parallel_empty_list_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            Workflow().parallel(steps=[])

    def test_parallel_with_no_substeps_completes(self):
        """A PARALLEL step whose substep list is empty (e.g. via direct
        Step construction) should complete cleanly, not crash."""
        wf = Workflow()
        wf._steps.append(Step(name="empty_parallel", step_type=StepType.PARALLEL))
        wf._step_names.add("empty_parallel")
        result = wf.run()
        assert result.success is True
        assert result.steps[0].status == StepStatus.COMPLETED

    def test_parallel_timeout_cancels_slow_substep(self):
        """A sub-step that exceeds the ParallelConfig.timeout must be
        cancelled at the deadline and reported FAILED; the workflow must
        return promptly, not wait for the slow thread."""
        wf = Workflow()
        wf.parallel(
            steps=[("slow", lambda ctx: time.sleep(5.0))],
            config=ParallelConfig(timeout=1.0),
        )
        start = time.time()
        result = wf.run()
        elapsed = time.time() - start

        # Return well before the slow task's 5s sleep would naturally finish.
        assert elapsed < 2.0, f"parallel timeout did not bound the wait (elapsed={elapsed:.2f})"
        assert result.success is False
        inner = result.steps[0]
        assert inner.status == StepStatus.FAILED
        slow_result = next(r for r in inner.output if r.name == "slow")
        assert slow_result.status == StepStatus.FAILED
        assert "timed out" in (slow_result.error or "")

    def test_parallel_timeout_lets_fast_substeps_complete(self):
        """When a parallel batch mixes fast and slow sub-steps, the fast ones
        finish (and write to the context) while the slow one is cancelled."""

        def fast(ctx):
            ctx.set("fast_done", True)

        wf = Workflow()
        wf.parallel(
            steps=[("fast", fast), ("slow", lambda ctx: time.sleep(5.0))],
            config=ParallelConfig(timeout=1.0),
        )
        start = time.time()
        result = wf.run()
        elapsed = time.time() - start
        assert elapsed < 2.0
        assert result.success is False
        assert result.context.get("fast_done") is True

    def test_parallel_substeps_complete_then_deadline_passes_for_next_batch(self):
        """When sub-steps run serially (max_workers=1) and the second one
        straddles the per-step deadline, the deadline check between wait
        cycles (``remaining <= 0``) correctly reports a timeout. This
        exercises the in-loop deadline path."""
        wf = Workflow()

        def medium(ctx):
            time.sleep(0.6)

        wf.parallel(
            steps=[("m1", medium), ("m2", medium)],
            # max_workers=1 forces m1 then m2 serially; timeout 1.0s lets m1
            # finish (~0.6s) but fires while m2 is still running.
            config=ParallelConfig(max_workers=1, timeout=1.0),
        )
        start = time.time()
        result = wf.run()
        elapsed = time.time() - start
        # The unbounded case is 1.2s; the deadline bounds it near 1.0s.
        assert elapsed < 1.15
        assert result.success is False

    def test_parallel_may_complete_within_timeout(self):
        """A normal parallel batch finishes well before its timeout and
        reports success."""
        wf = Workflow()
        wf.parallel(
            steps=[("a", lambda ctx: ctx.set("a", 1)), ("b", lambda ctx: ctx.set("b", 2))],
            config=ParallelConfig(timeout=5.0),
        )
        result = wf.run()
        assert result.success is True


# ============================================================================
# Workflow-level timeout
# ============================================================================

class TestWorkflowTimeout:
    """``WorkflowConfig.timeout`` is a real between-step deadline."""

    def test_long_sequential_chain_is_capped(self):
        """Each step sleeps 0.4s; a 1.0s budget must stop the chain partway."""
        def slow(ctx):
            time.sleep(0.4)
            ctx.set("count", ctx.get("count", 0) + 1)

        wf = Workflow(config=WorkflowConfig(timeout=1.0))
        for i in range(5):
            wf.step(f"s{i}", slow)
        start = time.time()
        result = wf.run()
        elapsed = time.time() - start

        # The deadline triggers between steps; we should not have run all 5.
        assert elapsed < 2.5, f"workflow timeout did not bound the run (elapsed={elapsed:.2f})"
        assert result.success is False
        assert result.error is not None and "timed out" in result.error
        assert len(result.steps) < 5


# ============================================================================
# Mixed / integration scenarios
# ============================================================================

class TestMixedWorkflows:
    """Composing builders into a realistic pipeline."""

    def test_step_loop_branch_compose(self):
        wf = Workflow()
        wf.step("init", lambda ctx: ctx.set("counter", 0))
        wf.loop(
            items=range(3),
            step_name_template=lambda i: f"inc_{i}",
            step_func_template=lambda i: lambda ctx: ctx.set(
                "counter", ctx.get("counter") + 1
            ),
        )
        wf.branch(
            condition=lambda ctx: ctx.get("counter") >= 3,
            yes_steps=[("ok", lambda ctx: ctx.set("ok", True))],
            no_steps=[("ng", lambda ctx: ctx.set("ok", False))],
        )
        result = wf.run()
        assert result.success is True
        assert result.context.get("counter") == 3
        assert result.context.get("ok") is True


# ============================================================================
# Result dataclasses
# ============================================================================

class TestResultDataclasses:
    """StepResult / WorkflowResult / Step behave as plain data containers."""

    def test_step_result_defaults(self):
        r = StepResult(name="x", status=StepStatus.COMPLETED)
        assert r.output is None
        assert r.error is None
        assert r.duration == 0.0
        assert r.retries == 0

    def test_workflow_result_defaults(self):
        r = WorkflowResult(success=True)
        assert r.steps == []
        assert r.context is None
        assert r.total_duration == 0.0
        assert r.error is None

    def test_workflow_result_holds_context(self):
        ctx = WorkflowContext(data={"k": "v"})
        r = WorkflowResult(success=True, context=ctx)
        assert r.context is not None
        assert r.context.get("k") == "v"

    def test_steps_property_returns_defensive_copy(self):
        wf = Workflow()
        wf.step("a", lambda ctx: None)
        snapshot = wf.steps
        snapshot.clear()
        # Mutating the snapshot does not affect the workflow.
        assert len(wf.steps) == 1
