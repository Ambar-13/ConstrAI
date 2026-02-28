"""Tests for clampai.adapters.langgraph_adapter.

Covers SafetyNode, @clampai_node, budget_guard, invariant_guard, and the
ClampAISafetyError / ClampAIBudgetError / ClampAIInvariantError hierarchy.

No LangGraph package required — the adapter is pure Python / clampai.formal.
"""
from __future__ import annotations

import functools
from typing import Any, Dict

import pytest

from clampai.adapters.langgraph_adapter import (
    ClampAIBudgetError,
    ClampAIInvariantError,
    ClampAISafetyError,
    SafetyNode,
    budget_guard,
    clampai_node,
    invariant_guard,
)
from clampai.formal import Invariant, State
from clampai.invariants import (
    no_delete_invariant,
    rate_limit_invariant,
    value_range_invariant,
)

# Helpers

def identity_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple pass-through node."""
    return {}


def echo_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Returns whatever is in state under 'input'."""
    return {"output": state.get("input")}


def failing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    raise RuntimeError("inner error")


# TestExceptionHierarchy

class TestExceptionHierarchy:
    def test_budget_error_is_safety_error(self):
        assert issubclass(ClampAIBudgetError, ClampAISafetyError)

    def test_invariant_error_is_safety_error(self):
        assert issubclass(ClampAIInvariantError, ClampAISafetyError)

    def test_safety_error_is_runtime_error(self):
        assert issubclass(ClampAISafetyError, RuntimeError)

    def test_budget_error_can_be_raised(self):
        with pytest.raises(ClampAIBudgetError, match="budget"):
            raise ClampAIBudgetError("budget gone")

    def test_invariant_error_can_be_raised(self):
        with pytest.raises(ClampAIInvariantError, match="invariant"):
            raise ClampAIInvariantError("invariant failed")

    def test_safety_error_catches_budget(self):
        with pytest.raises(ClampAISafetyError):
            raise ClampAIBudgetError("caught as base")

    def test_safety_error_catches_invariant(self):
        with pytest.raises(ClampAISafetyError):
            raise ClampAIInvariantError("caught as base")


# TestSafetyNodeConstruction

class TestSafetyNodeConstruction:
    def test_basic_construction(self):
        node = SafetyNode(identity_node, budget=10.0)
        assert node.budget == 10.0
        assert node.cost == 1.0
        assert node.action_id == "identity_node"

    def test_custom_cost(self):
        node = SafetyNode(identity_node, budget=50.0, cost=5.0)
        assert node.cost == 5.0

    def test_custom_action_id(self):
        node = SafetyNode(identity_node, budget=10.0, action_id="custom_id")
        assert node.action_id == "custom_id"

    def test_invariants_stored(self):
        invs = [no_delete_invariant("k"), rate_limit_invariant("c", 10)]
        node = SafetyNode(identity_node, budget=10.0, invariants=invs)
        assert len(node.invariants) == 2

    def test_empty_invariants(self):
        node = SafetyNode(identity_node, budget=10.0)
        assert node.invariants == []

    def test_repr_contains_fn_name(self):
        node = SafetyNode(identity_node, budget=10.0)
        assert "identity_node" in repr(node)

    def test_repr_contains_budget(self):
        node = SafetyNode(identity_node, budget=42.0)
        assert "42" in repr(node)

    def test_repr_contains_cost(self):
        node = SafetyNode(identity_node, budget=10.0, cost=3.0)
        assert "3.0" in repr(node)


# TestSafetyNodeCall

class TestSafetyNodeCall:
    def test_returns_function_result(self):
        node = SafetyNode(echo_node, budget=10.0)
        result = node({"input": "hello"})
        assert result == {"output": "hello"}

    def test_empty_state(self):
        node = SafetyNode(identity_node, budget=10.0)
        result = node({})
        assert result == {}

    def test_budget_decreases_per_call(self):
        node = SafetyNode(identity_node, budget=10.0, cost=3.0)
        node({})
        assert node.budget_remaining == pytest.approx(7.0, abs=0.01)

    def test_step_count_increments(self):
        node = SafetyNode(identity_node, budget=50.0)
        node({})
        node({})
        assert node.step_count == 2

    def test_budget_exhaustion_raises_budget_error(self):
        node = SafetyNode(identity_node, budget=2.0, cost=1.0)
        node({})
        node({})
        with pytest.raises(ClampAIBudgetError):
            node({})

    def test_budget_exhaustion_does_not_call_fn(self):
        calls = []

        def tracking_node(state):
            calls.append(1)
            return {}

        node = SafetyNode(tracking_node, budget=1.0, cost=1.0)
        node({})
        with pytest.raises(ClampAIBudgetError):
            node({})
        assert len(calls) == 1  # Only called once

    def test_inner_function_exception_propagates(self):
        node = SafetyNode(failing_node, budget=10.0)
        with pytest.raises(RuntimeError, match="inner error"):
            node({})

    def test_blocking_invariant_raises_invariant_error(self):
        inv = no_delete_invariant("must_exist")
        node = SafetyNode(identity_node, budget=10.0, invariants=[inv])
        # 'must_exist' is absent (falsy) → invariant violated
        with pytest.raises(ClampAIInvariantError):
            node({})

    def test_blocking_invariant_passes_when_satisfied(self):
        inv = no_delete_invariant("must_exist")
        node = SafetyNode(identity_node, budget=10.0, invariants=[inv])
        result = node({"must_exist": "yes"})
        assert result == {}

    def test_multiple_invariants_all_checked(self):
        inv1 = no_delete_invariant("k1")
        inv2 = rate_limit_invariant("calls", 5)
        node = SafetyNode(identity_node, budget=10.0, invariants=[inv1, inv2])
        # k1 absent → violation
        with pytest.raises(ClampAIInvariantError):
            node({"calls": 3})


# TestSafetyNodeReset

class TestSafetyNodeReset:
    def test_reset_restores_budget(self):
        node = SafetyNode(identity_node, budget=2.0, cost=1.0)
        node({})
        node({})
        assert node.budget_remaining == pytest.approx(0.0, abs=0.01)
        node.reset()
        assert node.budget_remaining == pytest.approx(2.0, abs=0.01)

    def test_reset_restores_step_count(self):
        node = SafetyNode(identity_node, budget=10.0)
        node({})
        node({})
        node.reset()
        assert node.step_count == 0

    def test_after_reset_can_call_again(self):
        node = SafetyNode(identity_node, budget=1.0, cost=1.0)
        node({})
        with pytest.raises(ClampAIBudgetError):
            node({})
        node.reset()
        result = node({})
        assert result == {}


# TestClampAINodeDecorator

class TestClampAINodeDecorator:
    def test_returns_safety_node(self):
        @clampai_node(budget=10.0)
        def my_node(state):
            return {}

        assert isinstance(my_node, SafetyNode)

    def test_function_name_preserved(self):
        @clampai_node(budget=10.0)
        def my_named_node(state):
            return {}

        assert my_named_node.__name__ == "my_named_node"

    def test_docstring_preserved(self):
        @clampai_node(budget=10.0)
        def documented_node(state):
            """This is a docstring."""
            return {}

        assert documented_node.__doc__ == "This is a docstring."

    def test_custom_cost_per_step(self):
        @clampai_node(budget=10.0, cost_per_step=2.5)
        def node(state):
            return {}

        assert node.cost == 2.5

    def test_custom_action_id(self):
        @clampai_node(budget=10.0, action_id="my_action")
        def node(state):
            return {}

        assert node.action_id == "my_action"

    def test_invariants_passed(self):
        @clampai_node(budget=10.0, invariants=[no_delete_invariant("x")])
        def node(state):
            return {}

        assert len(node.invariants) == 1

    def test_calls_wrapped_function(self):
        @clampai_node(budget=50.0)
        def node(state):
            return {"tagged": True}

        result = node({"x": 1})
        assert result == {"tagged": True}

    def test_budget_enforced_by_decorator(self):
        @clampai_node(budget=1.0, cost_per_step=1.0)
        def node(state):
            return {}

        node({})
        with pytest.raises(ClampAIBudgetError):
            node({})


# TestBudgetGuard

class TestBudgetGuard:
    def test_returns_callable(self):
        guard = budget_guard(budget=10.0)
        assert callable(guard)

    def test_returns_empty_dict(self):
        guard = budget_guard(budget=10.0)
        result = guard({"x": 1})
        assert result == {}

    def test_does_not_modify_state(self):
        guard = budget_guard(budget=10.0)
        state = {"x": 99}
        guard(state)
        assert state == {"x": 99}

    def test_raises_budget_error_when_exhausted(self):
        guard = budget_guard(budget=2.0, cost_per_step=1.0)
        guard({})
        guard({})
        with pytest.raises(ClampAIBudgetError):
            guard({})

    def test_custom_cost(self):
        guard = budget_guard(budget=5.0, cost_per_step=2.0)
        guard({})
        guard({})
        with pytest.raises(ClampAIBudgetError):
            guard({})

    def test_name_attribute(self):
        guard = budget_guard(budget=10.0)
        assert guard.__name__ == "budget_guard"

    def test_doc_contains_budget(self):
        guard = budget_guard(budget=77.0)
        assert "77" in (guard.__doc__ or "")

    def test_multiple_calls_track_budget(self):
        guard = budget_guard(budget=10.0, cost_per_step=3.0)
        guard({})
        guard({})
        guard({})
        with pytest.raises(ClampAIBudgetError):
            guard({})


# TestInvariantGuard

class TestInvariantGuard:
    def test_returns_callable(self):
        guard = invariant_guard([])
        assert callable(guard)

    def test_empty_invariants_always_passes(self):
        guard = invariant_guard([])
        result = guard({"anything": "here"})
        assert result == {}

    def test_returns_empty_dict(self):
        guard = invariant_guard([no_delete_invariant("k")])
        result = guard({"k": "present"})
        assert result == {}

    def test_raises_on_blocking_violation(self):
        guard = invariant_guard([no_delete_invariant("required_field")])
        with pytest.raises(ClampAIInvariantError):
            guard({})

    def test_passes_when_invariant_satisfied(self):
        guard = invariant_guard([rate_limit_invariant("calls", 10)])
        guard({"calls": 5})

    def test_multiple_invariants_first_violation_raises(self):
        guard = invariant_guard([
            no_delete_invariant("k1"),
            no_delete_invariant("k2"),
        ])
        with pytest.raises(ClampAIInvariantError):
            guard({"k2": "ok"})  # k1 absent

    def test_no_budget_charge(self):
        # invariant_guard has no budget — can call unlimited times
        inv = rate_limit_invariant("calls", 100)
        guard = invariant_guard([inv])
        for _ in range(200):
            guard({"calls": 0})

    def test_name_attribute(self):
        guard = invariant_guard([])
        assert guard.__name__ == "invariant_guard"

    def test_doc_contains_invariant_count(self):
        guard = invariant_guard([no_delete_invariant("x"), rate_limit_invariant("y", 5)])
        assert "2" in (guard.__doc__ or "")

    def test_value_range_invariant(self):
        guard = invariant_guard([value_range_invariant("score", 0.0, 1.0)])
        guard({"score": 0.5})
        with pytest.raises(ClampAIInvariantError):
            guard({"score": 2.0})

    def test_monitoring_mode_invariant_does_not_raise(self):
        inv = Invariant(
            "mon_inv",
            lambda s: False,  # always violates
            description="always fails",
            enforcement="monitoring",
        )
        guard = invariant_guard([inv])
        # Should NOT raise — monitoring mode only
        result = guard({})
        assert result == {}
