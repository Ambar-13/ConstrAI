"""Tests for clampai.adapters.langchain_callback.

Covers:
  ClampAICallbackError     — inheritance, raise/catch, message, str repr
  _require_langchain        — ImportError when unavailable, no-op when available
  ClampAICallbackHandler   — init state, properties, repr
  on_agent_action           — budget enforcement, invariant enforcement, state
                              construction, raise_on_block flag
  no-op callbacks           — on_tool_start, on_tool_end, on_agent_finish
  reset()                   — full state restoration
  thread safety             — concurrent calls serialised correctly
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from clampai.adapters.langchain_callback import (
    ClampAICallbackError,
    ClampAICallbackHandler,
)
from clampai.formal import Invariant, SafetyKernel
from clampai.invariants import rate_limit_invariant, resource_ceiling_invariant


def _make_handler(
    budget: float = 100.0,
    cost_per_action: float = 1.0,
    invariants: tuple = (),
    state_fn=None,
    raise_on_block: bool = True,
) -> ClampAICallbackHandler:
    """Create ClampAICallbackHandler without requiring langchain-core.

    Bypasses __init__ (which calls _require_langchain) by using __new__ and
    manually wiring the same attributes that __init__ would set.
    """
    handler = ClampAICallbackHandler.__new__(ClampAICallbackHandler)
    handler._budget = budget
    handler._cost = cost_per_action
    handler._invariants = list(invariants)
    handler._state_fn = state_fn
    handler._raise_on_block = raise_on_block
    handler._kernel = SafetyKernel(budget, list(invariants))
    handler._tool_call_count = 0
    handler._actions_blocked = 0
    handler._lock = threading.Lock()
    return handler


def _make_action(tool: str = "search", tool_input: object = "query text") -> MagicMock:
    """Return a minimal mock that looks like a LangChain AgentAction."""
    action = MagicMock()
    action.tool = tool
    action.tool_input = tool_input
    return action


class TestClampAICallbackError:
    def test_is_runtime_error(self):
        assert issubclass(ClampAICallbackError, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(ClampAICallbackError):
            raise ClampAICallbackError("blocked")

    def test_message_preserved(self):
        exc = ClampAICallbackError("something went wrong")
        assert str(exc) == "something went wrong"

    def test_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise ClampAICallbackError("caught as parent")

    def test_str_representation(self):
        exc = ClampAICallbackError("budget exhausted")
        assert "budget exhausted" in str(exc)


class TestClampAICallbackHandlerRequireLangchain:
    def test_require_langchain_raises_when_unavailable(self):
        from clampai.adapters import langchain_callback as _mod

        with patch.object(_mod, "_LANGCHAIN_AVAILABLE", False):
            with pytest.raises(ImportError):
                _mod._require_langchain()

    def test_require_langchain_no_raise_when_available(self):
        from clampai.adapters import langchain_callback as _mod

        with patch.object(_mod, "_LANGCHAIN_AVAILABLE", True):
            _mod._require_langchain()

    def test_import_error_message_mentions_extra(self):
        from clampai.adapters import langchain_callback as _mod

        with patch.object(_mod, "_LANGCHAIN_AVAILABLE", False):
            with pytest.raises(ImportError, match="clampai\\[langchain\\]"):
                _mod._require_langchain()


class TestClampAICallbackHandlerInit:
    def test_budget_remaining_equals_budget_initially(self):
        handler = _make_handler(budget=50.0)
        assert handler.budget_remaining == pytest.approx(50.0)

    def test_actions_blocked_is_zero_initially(self):
        handler = _make_handler()
        assert handler.actions_blocked == 0

    def test_tool_calls_made_is_zero_initially(self):
        handler = _make_handler()
        assert handler.tool_calls_made == 0

    def test_step_count_is_zero_initially(self):
        handler = _make_handler()
        assert handler.step_count == 0

    def test_repr_contains_budget(self):
        handler = _make_handler(budget=75.0)
        assert "75.0" in repr(handler)

    def test_repr_contains_cost_per_action(self):
        handler = _make_handler(cost_per_action=3.5)
        assert "3.5" in repr(handler)

    def test_repr_contains_invariant_count(self):
        invs = (rate_limit_invariant("tool_calls", 10),)
        handler = _make_handler(invariants=invs)
        assert "invariants=1" in repr(handler)

    def test_invariants_stored_correctly(self):
        inv1 = rate_limit_invariant("tool_calls", 10)
        inv2 = resource_ceiling_invariant("budget_spent", 80.0)
        handler = _make_handler(invariants=(inv1, inv2))
        assert len(handler._invariants) == 2


class TestOnAgentActionBudgetEnforcement:
    def test_single_action_decreases_budget_remaining(self):
        handler = _make_handler(budget=10.0, cost_per_action=3.0)
        handler.on_agent_action(_make_action())
        assert handler.budget_remaining == pytest.approx(7.0)

    def test_action_passes_increments_tool_calls_made(self):
        handler = _make_handler(budget=10.0, cost_per_action=1.0)
        handler.on_agent_action(_make_action())
        assert handler.tool_calls_made == 1

    def test_action_passes_increments_step_count(self):
        handler = _make_handler(budget=10.0, cost_per_action=1.0)
        handler.on_agent_action(_make_action())
        assert handler.step_count == 1

    def test_budget_exhaustion_raises_callback_error(self):
        handler = _make_handler(budget=2.0, cost_per_action=2.0)
        handler.on_agent_action(_make_action())
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())

    def test_budget_exhaustion_increments_actions_blocked(self):
        handler = _make_handler(budget=2.0, cost_per_action=2.0)
        handler.on_agent_action(_make_action())
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())
        assert handler.actions_blocked == 1

    def test_budget_exhaustion_no_raise_when_raise_on_block_false(self):
        handler = _make_handler(budget=1.0, cost_per_action=2.0, raise_on_block=False)
        handler.on_agent_action(_make_action())

    def test_budget_exhaustion_increments_blocked_even_when_no_raise(self):
        handler = _make_handler(budget=1.0, cost_per_action=2.0, raise_on_block=False)
        handler.on_agent_action(_make_action())
        assert handler.actions_blocked == 1

    def test_multiple_actions_across_budget_boundary(self):
        handler = _make_handler(budget=5.0, cost_per_action=2.0)
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        assert handler.tool_calls_made == 2
        assert handler.step_count == 2
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())
        assert handler.actions_blocked == 1
        assert handler.tool_calls_made == 3


class TestOnAgentActionInvariantEnforcement:
    def _blocking_invariant_that_fails(self) -> Invariant:
        """Returns a blocking invariant that always rejects."""
        return Invariant(
            "always_fail",
            lambda s: False,
            "This invariant always blocks",
            enforcement="blocking",
        )

    def _monitoring_invariant_that_fails(self) -> Invariant:
        """Returns a monitoring invariant that always triggers (but does not block)."""
        return Invariant(
            "always_monitor",
            lambda s: False,
            "This invariant monitors but never blocks",
            enforcement="monitoring",
        )

    def test_blocking_invariant_violation_raises_callback_error(self):
        handler = _make_handler(
            budget=100.0,
            invariants=(self._blocking_invariant_that_fails(),),
        )
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())

    def test_invariant_violation_increments_actions_blocked(self):
        handler = _make_handler(
            budget=100.0,
            invariants=(self._blocking_invariant_that_fails(),),
        )
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())
        assert handler.actions_blocked == 1

    def test_invariant_violation_still_increments_tool_calls_made(self):
        handler = _make_handler(
            budget=100.0,
            invariants=(self._blocking_invariant_that_fails(),),
        )
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())
        assert handler.tool_calls_made == 1

    def test_passing_invariant_allows_step_count_to_increment(self):
        passing_inv = Invariant(
            "always_pass",
            lambda s: True,
            "This invariant always passes",
            enforcement="blocking",
        )
        handler = _make_handler(budget=100.0, invariants=(passing_inv,))
        handler.on_agent_action(_make_action())
        assert handler.step_count == 1

    def test_invariant_violation_no_raise_when_raise_on_block_false(self):
        handler = _make_handler(
            budget=100.0,
            invariants=(self._blocking_invariant_that_fails(),),
            raise_on_block=False,
        )
        handler.on_agent_action(_make_action())

    def test_state_fn_extra_fields_exposed_to_invariants(self):
        sentinel_value: list = []

        def capturing_state_fn(action):
            return {"custom_key": "custom_value"}

        saw_custom: list = []

        custom_inv = Invariant(
            "sees_custom",
            lambda s: (saw_custom.append(s.get("custom_key")) or True),
            "Captures custom_key from state",
            enforcement="monitoring",
        )

        handler = _make_handler(
            budget=100.0,
            invariants=(custom_inv,),
            state_fn=capturing_state_fn,
        )
        handler.on_agent_action(_make_action())
        assert saw_custom == ["custom_value"]


class TestOnAgentActionState:
    def test_tool_name_extracted_from_action_tool(self):
        captured: list = []
        inv = Invariant(
            "capture_tool",
            lambda s: (captured.append(s.get("tool")) or True),
            "Captures tool name",
            enforcement="monitoring",
        )
        handler = _make_handler(budget=100.0, invariants=(inv,))
        handler.on_agent_action(_make_action(tool="my_tool"))
        assert captured == ["my_tool"]

    def test_tool_input_extracted_from_action_tool_input_str(self):
        captured: list = []
        inv = Invariant(
            "capture_input",
            lambda s: (captured.append(s.get("tool_input")) or True),
            "Captures tool_input",
            enforcement="monitoring",
        )
        handler = _make_handler(budget=100.0, invariants=(inv,))
        handler.on_agent_action(_make_action(tool_input="my input string"))
        assert captured == ["my input string"]

    def test_tool_input_coerced_to_str_when_dict(self):
        captured: list = []
        inv = Invariant(
            "capture_input_type",
            lambda s: (captured.append(type(s.get("tool_input"))) or True),
            "Captures type of tool_input",
            enforcement="monitoring",
        )
        handler = _make_handler(budget=100.0, invariants=(inv,))
        handler.on_agent_action(_make_action(tool_input={"key": "val"}))
        assert captured == [str]

    def test_state_includes_tool_calls_counter(self):
        captured: list = []
        inv = Invariant(
            "capture_count",
            lambda s: (captured.append(s.get("tool_calls")) or True),
            "Captures tool_calls counter",
            enforcement="monitoring",
        )
        handler = _make_handler(budget=100.0, invariants=(inv,))
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        assert captured == [1, 2]

    def test_state_includes_budget_spent(self):
        captured: list = []
        inv = Invariant(
            "capture_spent",
            lambda s: (captured.append(s.get("budget_spent")) or True),
            "Captures budget_spent",
            enforcement="monitoring",
        )
        handler = _make_handler(budget=100.0, cost_per_action=5.0, invariants=(inv,))
        handler.on_agent_action(_make_action())
        assert captured[0] == pytest.approx(0.0)
        handler.on_agent_action(_make_action())
        assert captured[1] == pytest.approx(5.0)


class TestNoOpCallbacks:
    def test_on_tool_start_returns_none(self):
        handler = _make_handler()
        result = handler.on_tool_start({"name": "tool"}, "input text")
        assert result is None

    def test_on_tool_end_returns_none(self):
        handler = _make_handler()
        result = handler.on_tool_end("tool output")
        assert result is None

    def test_on_agent_finish_returns_none(self):
        handler = _make_handler()
        finish = MagicMock()
        result = handler.on_agent_finish(finish)
        assert result is None


class TestResetMethod:
    def test_reset_restores_budget_remaining_to_original(self):
        handler = _make_handler(budget=10.0, cost_per_action=3.0)
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        handler.reset()
        assert handler.budget_remaining == pytest.approx(10.0)

    def test_reset_clears_tool_calls_made(self):
        handler = _make_handler(budget=50.0)
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        handler.reset()
        assert handler.tool_calls_made == 0

    def test_reset_clears_actions_blocked(self):
        handler = _make_handler(budget=2.0, cost_per_action=2.0, raise_on_block=False)
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        assert handler.actions_blocked > 0
        handler.reset()
        assert handler.actions_blocked == 0

    def test_reset_clears_step_count(self):
        handler = _make_handler(budget=50.0)
        handler.on_agent_action(_make_action())
        handler.on_agent_action(_make_action())
        handler.reset()
        assert handler.step_count == 0

    def test_reset_allows_new_actions_after_budget_exhaustion(self):
        handler = _make_handler(budget=2.0, cost_per_action=2.0)
        handler.on_agent_action(_make_action())
        with pytest.raises(ClampAICallbackError):
            handler.on_agent_action(_make_action())
        handler.reset()
        handler.on_agent_action(_make_action())
        assert handler.step_count == 1
        assert handler.budget_remaining == pytest.approx(0.0)


class TestThreadSafety:
    def test_concurrent_calls_all_charge_budget(self):
        n_threads = 10
        cost = 1.0
        handler = _make_handler(budget=float(n_threads) * cost, cost_per_action=cost)

        errors: list = []

        def call_action():
            try:
                handler.on_agent_action(_make_action())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call_action) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert handler.tool_calls_made == n_threads

    def test_final_budget_remaining_is_consistent_after_concurrent_calls(self):
        n_threads = 5
        cost = 2.0
        budget = float(n_threads) * cost
        handler = _make_handler(budget=budget, cost_per_action=cost)

        def call_action():
            try:
                handler.on_agent_action(_make_action())
            except ClampAICallbackError:
                pass

        threads = [threading.Thread(target=call_action) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert handler.budget_remaining == pytest.approx(0.0)
        assert handler.step_count == n_threads
