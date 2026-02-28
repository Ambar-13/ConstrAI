"""Tests for clampai.adapters.crewai_adapter.

Covers every public class and function in the CrewAI safety integration:

    ClampAICrewError          — base exception
    ClampAICrewBudgetError    — budget-exhausted subtype
    ClampAICrewInvariantError — invariant-violated subtype
    ClampAISafeCrewTool       — callable tool wrapper: init, call, reset,
                                 properties (budget_remaining, step_count), repr
    ClampAICrewCallback       — step/task callback: init, step_callback,
                                 task_callback, reset, properties, repr
    safe_crew_tool             — decorator factory
"""
from __future__ import annotations

import pytest

from clampai.adapters.crewai_adapter import (
    ClampAICrewBudgetError,
    ClampAICrewCallback,
    ClampAICrewError,
    ClampAICrewInvariantError,
    ClampAISafeCrewTool,
    safe_crew_tool,
)
from clampai.formal import Invariant, State
from clampai.invariants import rate_limit_invariant, string_length_invariant


def _always_block_invariant() -> Invariant:
    """An invariant whose predicate always returns False (always blocks)."""
    return Invariant(
        "always_block",
        lambda s: False,
        "Always blocked for testing",
        enforcement="blocking",
    )


def _always_pass_invariant() -> Invariant:
    """An invariant whose predicate always returns True (never blocks)."""
    return Invariant(
        "always_pass",
        lambda s: True,
        "Always passes for testing",
        enforcement="blocking",
    )


def _query_too_long_invariant(max_len: int = 5) -> Invariant:
    """Blocks when state['query'] string length exceeds max_len."""
    return string_length_invariant("query", max_len)


class TestExceptionHierarchy:
    def test_budget_error_is_crew_error(self):
        assert issubclass(ClampAICrewBudgetError, ClampAICrewError)

    def test_invariant_error_is_crew_error(self):
        assert issubclass(ClampAICrewInvariantError, ClampAICrewError)

    def test_crew_error_is_runtime_error(self):
        assert issubclass(ClampAICrewError, RuntimeError)

    def test_budget_error_can_be_raised_and_caught(self):
        with pytest.raises(ClampAICrewBudgetError):
            raise ClampAICrewBudgetError("budget gone")

    def test_invariant_error_can_be_raised_and_caught(self):
        with pytest.raises(ClampAICrewInvariantError):
            raise ClampAICrewInvariantError("invariant violated")

    def test_base_error_catches_both_subtypes(self):
        for exc_type in (ClampAICrewBudgetError, ClampAICrewInvariantError):
            with pytest.raises(ClampAICrewError):
                raise exc_type("caught as base")


class TestClampAISafeCrewToolInit:
    def test_budget_remaining_equals_budget_initially(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=50.0)
        assert tool.budget_remaining == pytest.approx(50.0)

    def test_step_count_is_zero_initially(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0)
        assert tool.step_count == 0

    def test_name_defaults_to_func_name(self):
        def my_tool():
            return "ok"

        tool = ClampAISafeCrewTool(my_tool, budget=10.0)
        assert tool.name == "my_tool"

    def test_name_can_be_overridden(self):
        tool = ClampAISafeCrewTool(lambda: None, name="custom_name", budget=10.0)
        assert tool.name == "custom_name"

    def test_description_stored(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0, description="Does a thing")
        assert tool.description == "Does a thing"

    def test_repr_contains_name(self):
        tool = ClampAISafeCrewTool(lambda: None, name="mytool", budget=10.0)
        assert "mytool" in repr(tool)

    def test_repr_contains_budget(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=99.5)
        assert "99.5" in repr(tool)

    def test_repr_contains_cost(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0, cost=3.0)
        assert "3.0" in repr(tool)


class TestClampAISafeCrewToolCall:
    def test_successful_call_returns_func_result(self):
        tool = ClampAISafeCrewTool(lambda q: f"result:{q}", budget=20.0)
        assert tool("hello") == "result:hello"

    def test_successful_call_decrements_budget_remaining(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0, cost=3.0)
        tool()
        assert tool.budget_remaining == pytest.approx(7.0)

    def test_successful_call_increments_step_count(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=20.0)
        tool()
        assert tool.step_count == 1
        tool()
        assert tool.step_count == 2

    def test_first_positional_arg_stored_as_query(self):
        received_query = []

        def my_func(q):
            received_query.append(q)
            return q

        tool = ClampAISafeCrewTool(my_func, budget=20.0)
        tool("test_input")
        assert received_query[0] == "test_input"

    def test_kwargs_stored_in_state(self):
        call_count = []

        def my_func(**kw):
            call_count.append(kw)
            return "ok"

        tool = ClampAISafeCrewTool(my_func, budget=20.0)
        tool(topic="science", depth="deep")
        assert call_count[0] == {"topic": "science", "depth": "deep"}

    def test_budget_exhaustion_raises_budget_error(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=3.0, cost=2.0)
        tool()
        with pytest.raises(ClampAICrewBudgetError):
            tool()

    def test_budget_exhaustion_func_not_called(self):
        called = []

        def expensive():
            called.append(True)

        tool = ClampAISafeCrewTool(expensive, budget=1.5, cost=1.0)
        tool()
        assert len(called) == 1
        with pytest.raises(ClampAICrewBudgetError):
            tool()
        assert len(called) == 1

    def test_invariant_violation_raises_invariant_error(self):
        tool = ClampAISafeCrewTool(
            lambda: None,
            budget=20.0,
            invariants=[_always_block_invariant()],
        )
        with pytest.raises(ClampAICrewInvariantError):
            tool()

    def test_invariant_violation_func_not_called(self):
        called = []

        def guarded():
            called.append(True)

        tool = ClampAISafeCrewTool(
            guarded,
            budget=20.0,
            invariants=[_always_block_invariant()],
        )
        with pytest.raises(ClampAICrewInvariantError):
            tool()
        assert len(called) == 0

    def test_multiple_calls_tracking(self):
        results = []
        tool = ClampAISafeCrewTool(lambda x: results.append(x), budget=100.0, cost=5.0)
        for i in range(5):
            tool(i)
        assert tool.step_count == 5
        assert tool.budget_remaining == pytest.approx(75.0)


class TestClampAISafeCrewToolReset:
    def test_reset_restores_budget_remaining(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0, cost=4.0)
        tool()
        tool()
        assert tool.budget_remaining == pytest.approx(2.0)
        tool.reset()
        assert tool.budget_remaining == pytest.approx(10.0)

    def test_reset_clears_step_count(self):
        tool = ClampAISafeCrewTool(lambda: None, budget=30.0, cost=3.0)
        tool()
        tool()
        tool()
        assert tool.step_count == 3
        tool.reset()
        assert tool.step_count == 0

    def test_new_calls_succeed_after_reset_from_exhausted_state(self):
        tool = ClampAISafeCrewTool(lambda: "ok", budget=5.0, cost=5.0)
        tool()
        with pytest.raises(ClampAICrewBudgetError):
            tool()
        tool.reset()
        assert tool() == "ok"

    def test_reset_with_invariants_kernel_re_created(self):
        invs = [_always_pass_invariant()]
        tool = ClampAISafeCrewTool(lambda: None, budget=10.0, invariants=invs)
        tool()
        tool.reset()
        tool()
        assert tool.step_count == 1
        assert tool.budget_remaining == pytest.approx(9.0)


class TestClampAICrewCallbackInit:
    def test_budget_remaining_equals_budget_initially(self):
        cb = ClampAICrewCallback(budget=100.0)
        assert cb.budget_remaining == pytest.approx(100.0)

    def test_step_count_is_zero_initially(self):
        cb = ClampAICrewCallback(budget=50.0)
        assert cb.step_count == 0

    def test_repr_contains_budget(self):
        cb = ClampAICrewCallback(budget=77.0)
        assert "77.0" in repr(cb)

    def test_repr_contains_cost_per_step(self):
        cb = ClampAICrewCallback(budget=10.0, cost_per_step=2.5)
        assert "2.5" in repr(cb)

    def test_repr_contains_invariants_count(self):
        invs = [_always_pass_invariant(), _always_pass_invariant()]
        cb = ClampAICrewCallback(budget=10.0, invariants=invs)
        assert "2" in repr(cb)


class TestClampAICrewCallbackStepCallback:
    def test_step_callback_charges_budget(self):
        cb = ClampAICrewCallback(budget=10.0, cost_per_step=3.0)
        cb.step_callback("step output")
        assert cb.budget_remaining == pytest.approx(7.0)

    def test_step_callback_increments_step_count(self):
        cb = ClampAICrewCallback(budget=20.0)
        cb.step_callback("output 1")
        assert cb.step_count == 1
        cb.step_callback("output 2")
        assert cb.step_count == 2

    def test_step_output_truncated_to_2000_chars(self):
        cb = ClampAICrewCallback(
            budget=100.0,
            invariants=[
                Invariant(
                    "check_output_len",
                    lambda s: len(s.get("step_output", "")) <= 2000,
                    "step_output must be at most 2000 chars",
                    enforcement="blocking",
                )
            ],
        )
        long_output = "x" * 5000
        cb.step_callback(long_output)
        assert cb.step_count == 1

    def test_step_number_in_state(self):
        seen_step_nums = []

        def capture_step_num(s: State) -> bool:
            seen_step_nums.append(s.get("step_number"))
            return True

        cb = ClampAICrewCallback(
            budget=20.0,
            invariants=[
                Invariant(
                    "capture",
                    capture_step_num,
                    "capture step_number",
                    enforcement="blocking",
                )
            ],
        )
        cb.step_callback("step one")
        cb.step_callback("step two")
        assert 1 in seen_step_nums
        assert 2 in seen_step_nums

    def test_none_step_output_doesnt_crash(self):
        cb = ClampAICrewCallback(budget=10.0)
        cb.step_callback(None)
        assert cb.step_count == 1

    def test_budget_exhaustion_raises_budget_error(self):
        cb = ClampAICrewCallback(budget=5.0, cost_per_step=3.0)
        cb.step_callback("ok")
        with pytest.raises(ClampAICrewBudgetError):
            cb.step_callback("blocked")

    def test_invariant_violation_raises_invariant_error(self):
        cb = ClampAICrewCallback(
            budget=20.0,
            invariants=[_always_block_invariant()],
        )
        with pytest.raises(ClampAICrewInvariantError):
            cb.step_callback("some output")

    def test_multiple_steps_tracked_correctly(self):
        cb = ClampAICrewCallback(budget=50.0, cost_per_step=2.0)
        for i in range(7):
            cb.step_callback(f"output {i}")
        assert cb.step_count == 7
        assert cb.budget_remaining == pytest.approx(36.0)


class TestClampAICrewCallbackTaskCallback:
    def test_task_callback_does_not_charge_budget(self):
        cb = ClampAICrewCallback(budget=20.0, cost_per_step=5.0)
        budget_before = cb.budget_remaining
        cb.task_callback("task done")
        assert cb.budget_remaining == pytest.approx(budget_before)

    def test_task_callback_increments_task_num(self):
        cb = ClampAICrewCallback(budget=20.0)
        assert cb._task_num == 0
        cb.task_callback("task 1")
        assert cb._task_num == 1
        cb.task_callback("task 2")
        assert cb._task_num == 2

    def test_none_task_output_doesnt_crash(self):
        cb = ClampAICrewCallback(budget=20.0)
        cb.task_callback(None)
        assert cb._task_num == 1

    def test_multiple_task_callbacks_tracked(self):
        cb = ClampAICrewCallback(budget=20.0)
        for _ in range(5):
            cb.task_callback("done")
        assert cb._task_num == 5


class TestClampAICrewCallbackReset:
    def test_reset_restores_budget_remaining(self):
        cb = ClampAICrewCallback(budget=10.0, cost_per_step=4.0)
        cb.step_callback("first")
        cb.step_callback("second")
        assert cb.budget_remaining == pytest.approx(2.0)
        cb.reset()
        assert cb.budget_remaining == pytest.approx(10.0)

    def test_reset_clears_step_count(self):
        cb = ClampAICrewCallback(budget=30.0, cost_per_step=2.0)
        cb.step_callback("a")
        cb.step_callback("b")
        cb.step_callback("c")
        assert cb.step_count == 3
        cb.reset()
        assert cb.step_count == 0

    def test_new_steps_succeed_after_reset_from_exhausted_state(self):
        cb = ClampAICrewCallback(budget=3.0, cost_per_step=3.0)
        cb.step_callback("used up")
        with pytest.raises(ClampAICrewBudgetError):
            cb.step_callback("blocked")
        cb.reset()
        cb.step_callback("fresh start")
        assert cb.step_count == 1


class TestSafeCrewToolDecorator:
    def test_decorator_wraps_function_correctly(self):
        @safe_crew_tool(budget=20.0, cost=1.0)
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert greet("World") == "Hello, World!"

    def test_result_is_clampai_safe_crew_tool(self):
        @safe_crew_tool(budget=20.0)
        def noop():
            pass

        assert isinstance(noop, ClampAISafeCrewTool)

    def test_name_defaults_to_function_name(self):
        @safe_crew_tool(budget=10.0)
        def my_search_tool(query: str) -> str:
            return query

        assert my_search_tool.name == "my_search_tool"

    def test_name_can_be_overridden(self):
        @safe_crew_tool(budget=10.0, name="web_search")
        def search(query: str) -> str:
            return query

        assert search.name == "web_search"

    def test_invariants_passed_through(self):
        invs = [_always_pass_invariant(), _always_pass_invariant()]

        @safe_crew_tool(budget=10.0, invariants=invs)
        def my_tool():
            return "ok"

        assert len(my_tool.invariants) == 2
